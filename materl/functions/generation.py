# materl.functions.generation

# This file contains pure functions for generating text completions from a model.
# This logic was extracted from the old monolithic GRPOEngine to promote
# a more modular, functional, and reusable design.

import os
import logging
import warnings
from contextlib import contextmanager

from typing import List, Dict, Any, Union, TYPE_CHECKING

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from max.entrypoints.llm import LLM

from .masks import create_completion_masks

# Reduce tokenizer parallelism warnings by setting environment variable
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Configure logging to reduce verbosity
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)
logging.getLogger("max").setLevel(logging.WARNING)

@contextmanager
def suppress_tokenizer_warnings():
    """Context manager to suppress tokenizer-related warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        warnings.filterwarnings("ignore", message=".*tokenizer.*", category=UserWarning)
        yield

if TYPE_CHECKING:
    from max.graph import Graph


def build_model_graph(
    model_path: str,
    device_id: int = 0,
) -> "Graph":
    """
    Builds a MAX Graph that is functionally equivalent to a HuggingFace model.

    This function reads the model's configuration from HuggingFace and builds
    the corresponding computational graph from first principles using MAX nn
    and graph primitives. It does NOT load the weights, only defines the
    architecture.

    Args:
        model_path: The HuggingFace model ID or path to a local model.
        device_id: The GPU device ID to build the graph for.

    Returns:
        A MAX Graph representing the model's architecture.
    """
    import os
    from pathlib import Path
    from max.graph import Graph, TensorType, DeviceRef, ops, TensorValue
    from max.dtype import DType
    from max.nn import Embedding, Linear, LayerNorm, Module
    from max.nn.layer.layer_list import LayerList
    from transformers import AutoConfig
    from max.graph.weights.load import load_weights

    config = AutoConfig.from_pretrained(model_path)
    device = DeviceRef.GPU(device_id)
    dtype = DType.float32  # Using float32 for simplicity

    # Define the graph's input signature
    input_ids_type = TensorType(DType.int64, shape=["batch_size", "seq_len"], device=device)

    # Use a top-level Module to manage weights
    class TransformerModel(Module):
        def __init__(self):
            super().__init__()
            self.embedding = Embedding(
                vocab_size=config.vocab_size,
                hidden_dim=config.hidden_size,
                dtype=dtype,
                device=device
            )
            
            self.layers = LayerList([
                self.make_layer() for _ in range(config.num_hidden_layers)
            ])
            
            self.final_ln = LayerNorm(dims=config.hidden_size, device=device)
            self.lm_head = Linear(
                in_dim=config.hidden_size, out_dim=config.vocab_size, device=device, dtype=dtype
            )

        def make_layer(self):
            class TransformerLayer(Module):
                def __init__(self):
                    super().__init__()
                    self.ln1 = LayerNorm(dims=config.hidden_size, device=device)
                    # Simplified attention - in a real scenario, you'd implement multi-head attention
                    self.attn = Linear(
                        in_dim=config.hidden_size,
                        out_dim=config.hidden_size,
                        device=device,
                        dtype=dtype
                    )
                    self.ln2 = LayerNorm(dims=config.hidden_size, device=device)
                    ff_dim = getattr(config, 'n_inner', config.hidden_size * 4)
                    self.ffn_up = Linear(in_dim=config.hidden_size, out_dim=ff_dim, device=device, dtype=dtype)
                    self.ffn_down = Linear(in_dim=ff_dim, out_dim=config.hidden_size, device=device, dtype=dtype)

                def __call__(self, x: TensorValue) -> TensorValue:
                    # Simplified feedforward, no multi-head attention
                    # Apply attention with residual connection
                    x = x + self.attn(self.ln1(x))
                    # Apply feedforward network with residual connection
                    x = x + self.ffn_down(ops.gelu(self.ffn_up(self.ln2(x))))
                    return x
            return TransformerLayer()

        def __call__(self, tokens: TensorValue) -> TensorValue:
            # Convert token IDs to embeddings
            h = self.embedding(tokens)
            # Process through transformer layers sequentially
            for layer in self.layers:
                h = layer(h)
                # Ensure h is always a TensorValue (type assertion for type checker)
                assert isinstance(h, TensorValue), "Layer must return TensorValue"
            # Apply final layer normalization before output projection
            h = self.final_ln(h)
            # Project to vocabulary size for logits
            return self.lm_head(h)

    # Instantiate the model architecture
    transformer_model = TransformerModel()
    
    # Build the graph
    with Graph(
        name=f"graph_{model_path.replace('/', '_')}",
        input_types=[input_ids_type]
    ) as graph:
        input_ids = graph.inputs[0].tensor
        
        # This is where the magic happens: we find the weight files
        # and load them into our graph structure.
        weight_files = [Path(model_path) / f for f in os.listdir(model_path) if f.endswith(('.safetensors', '.bin'))]
        if not weight_files:
            raise FileNotFoundError(f"No weight files (.safetensors or .bin) found in {model_path}")
            
        loaded_weights = load_weights(weight_files)
        transformer_model.load_state_dict({
            name: weight.data() for name, weight in loaded_weights.items()
        })
        
        logits = transformer_model(input_ids)
        graph.output(logits)
        
    return graph


def generate_with_graph(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    max_new_tokens: int = 10,
    device_id: int = 0,
) -> List[str]:
    """
    Generates text by building a MAX graph from first principles, loading
    the model's weights, and running autoregressive inference.
    """
    from max.engine.api import InferenceSession
    from max.driver import Tensor
    import numpy as np
    
    graph = build_model_graph(model_path, device_id)
    
    session = InferenceSession()
    model = session.load(graph)
    
    completions = []
    for prompt in prompts:
        # Use tokenizer.__call__ method for better efficiency with fast tokenizers
        with suppress_tokenizer_warnings():
            tokenized = tokenizer(prompt, return_tensors="np", padding=False, truncation=False)
        input_ids_array = tokenized.input_ids
        # Ensure proper shape (add batch dimension if needed)
        if input_ids_array.ndim == 1:
            input_ids_array = input_ids_array.reshape(1, -1)
        
        current_ids = input_ids_array
        for _ in range(max_new_tokens):
            input_tensor = Tensor.from_numpy(current_ids)
            outputs = model.execute(input_tensor)
            logits_val = outputs[0]
            # Convert MojoValue to numpy array directly
            logits = np.array(logits_val)

            next_token_id = np.argmax(logits[:, -1, :], axis=-1)
            
            current_ids = np.concatenate([current_ids, np.array([[next_token_id[0]]])], axis=1)

            if next_token_id[0] == tokenizer.eos_token_id:
                break
        
        original_len = input_ids_array.shape[1]
        new_tokens = current_ids[0, original_len:]
        
        with suppress_tokenizer_warnings():
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(completion)
    
    return completions


def generate_completions(
    model: Union[PreTrainedModel, LLM, str],
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    max_prompt_length: int,
    max_completion_length: int,
    num_generations: int,
    backend: str = "torch",
    **generation_kwargs,
) -> Dict[str, Any]:
    """
    Generates completions for a batch of prompts and prepares all necessary tensors.

    Args:
        model: The policy model to use for generation. 
               For 'torch' backend, this should be a PreTrainedModel.
               For 'max' backend, this should be an `LLM` instance.
               For 'graph' backend, this should be a model path string.
        tokenizer: The tokenizer associated with the model.
        prompts: A list of string prompts.
        max_prompt_length: The maximum length for tokenized prompts.
        max_completion_length: The maximum number of new tokens to generate.
        num_generations: The number of completions to generate per prompt.
        backend: The backend to use for generation ('torch', 'max', or 'graph').
        **generation_kwargs: Additional arguments for the model's `generate` method
                             (e.g., temperature, top_p, top_k).

    Returns:
        A dictionary containing all tensors needed for subsequent pipeline steps,
        including completions, prompts, and their corresponding masks.
    """
    if backend == "graph":
        # Use the MAX LLM entrypoint for automated graph construction
        # This leverages the same infrastructure as the 'max' backend but with direct model path
        from max.entrypoints.llm import LLM
        from max.pipelines.lib.config import PipelineConfig
        
        if not isinstance(model, str):
            raise ValueError("For 'graph' backend, model must be a model path string.")
        
        expanded_prompts = [p for p in prompts for _ in range(num_generations)]
        
        # Create LLM instance with pipeline config
        # This automatically handles model detection, weight loading, and graph construction
        pipeline_config = PipelineConfig(model_path=model)
        llm = LLM(pipeline_config=pipeline_config) # type: ignore
        
        # Generate completions using the LLM - same as 'max' backend but with automatic model loading
        responses = llm.generate(
            prompts=expanded_prompts, 
            max_new_tokens=max_completion_length,
            use_tqdm=False
        )
        
        completions_text = list(responses)
        
        # The rest of this is for compatibility with the RL pipeline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eos_token_id = tokenizer.eos_token_id
        assert isinstance(eos_token_id, int)

        with suppress_tokenizer_warnings():
            prompt_inputs = tokenizer(
                expanded_prompts, return_tensors="pt", padding="max_length", 
                max_length=max_prompt_length, truncation=True, add_special_tokens=False
            ).to(device)

            completion_tokens = tokenizer(
                completions_text, return_tensors="pt", padding="max_length", 
                max_length=max_completion_length, truncation=True, add_special_tokens=False
            ).to(device)

        completions_ids = completion_tokens.input_ids
        completion_masks = create_completion_masks(completions_ids, eos_token_id, device)
        
        full_input_ids = torch.cat([prompt_inputs.input_ids, completions_ids], dim=1)
        full_attention_mask = torch.cat([prompt_inputs.attention_mask, completion_masks], dim=1)
        
        return {
            "prompts_text": expanded_prompts,
            "completions_text": completions_text,
            "prompts_input_ids": prompt_inputs.input_ids,
            "prompt_masks": prompt_inputs.attention_mask,
            "completions_ids": completions_ids,
            "completion_masks": completion_masks,
            "full_input_ids": full_input_ids,
            "full_attention_mask": full_attention_mask,
        }
    
    elif backend == "torch":
        if not isinstance(model, PreTrainedModel):
            raise ValueError("For 'torch' backend, model must be a PreTrainedModel instance.")
        
        # Explicit type assertion to resolve scoping issues with type checker
        policy_model: PreTrainedModel = model
        device = policy_model.device
        eos_token_id = tokenizer.eos_token_id
        assert isinstance(eos_token_id, int), "Tokenizer must have an integer `eos_token_id`."

        expanded_prompts_text = [p for p in prompts for _ in range(num_generations)]

        with suppress_tokenizer_warnings():
            prompt_inputs = tokenizer(
                expanded_prompts_text,
                return_tensors="pt", padding="max_length", max_length=max_prompt_length,
                truncation=True, add_special_tokens=False,
            ).to(device)

        generation_kwargs.pop("max_seq_length", None)
        if 'do_sample' not in generation_kwargs:
            generation_kwargs['do_sample'] = True

        with torch.no_grad():
            # Use explicitly typed variable to avoid type inference issues
            output_sequences = policy_model.generate( # type: ignore
                input_ids=prompt_inputs.input_ids,
                attention_mask=prompt_inputs.attention_mask,
                max_new_tokens=max_completion_length,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                **generation_kwargs,
            )

        completions_ids = output_sequences[:, prompt_inputs.input_ids.shape[1]:]
        completion_masks = create_completion_masks(completions_ids, eos_token_id, device)
        
        with suppress_tokenizer_warnings():
            completions_text = tokenizer.batch_decode(completions_ids, skip_special_tokens=True)

        full_input_ids = torch.cat([prompt_inputs.input_ids, completions_ids], dim=1)
        full_attention_mask = torch.cat([prompt_inputs.attention_mask, completion_masks], dim=1)

        return {
            "prompts_text": expanded_prompts_text,
            "completions_text": completions_text,
            "prompts_input_ids": prompt_inputs.input_ids,
            "prompt_masks": prompt_inputs.attention_mask,
            "completions_ids": completions_ids,
            "completion_masks": completion_masks,
            "full_input_ids": full_input_ids,
            "full_attention_mask": full_attention_mask,
        }
    
    elif backend == "max":
        # Import LLM locally to avoid scoping issues
        from max.entrypoints.llm import LLM as MaxLLM
        
        if not isinstance(model, MaxLLM):
            raise ValueError(
                "For 'max' backend, model must be a pre-initialized `max.entrypoints.llm.LLM` instance."
            )
        
        # Explicit type assertion to resolve scoping issues with type checker
        llm_model: MaxLLM = model
        expanded_prompts_text = [p for p in prompts for _ in range(num_generations)]

        responses = llm_model.generate(
            prompts=expanded_prompts_text, 
            max_new_tokens=max_completion_length,
            use_tqdm=False
        )
        
        completions_text = list(responses)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eos_token_id = tokenizer.eos_token_id
        assert isinstance(eos_token_id, int), "Tokenizer must have an integer `eos_token_id`."
        
        with suppress_tokenizer_warnings():
            prompt_inputs = tokenizer(
                expanded_prompts_text,
                return_tensors="pt", padding="max_length", max_length=max_prompt_length,
                truncation=True, add_special_tokens=False,
            ).to(device)
            
            completions_tokens = tokenizer(
                completions_text, return_tensors="pt", padding="max_length", 
                max_length=max_completion_length, truncation=True, add_special_tokens=False
            ).to(device)
        
        completions_ids = completions_tokens.input_ids
        completion_masks = create_completion_masks(completions_ids, eos_token_id, device)
        
        full_input_ids = torch.cat([prompt_inputs.input_ids, completions_ids], dim=1)
        full_attention_mask = torch.cat([prompt_inputs.attention_mask, completion_masks], dim=1)

        return {
            "prompts_text": expanded_prompts_text,
            "completions_text": completions_text,
            "prompts_input_ids": prompt_inputs.input_ids,
            "prompt_masks": prompt_inputs.attention_mask,
            "completions_ids": completions_ids,
            "completion_masks": completion_masks,
            "full_input_ids": full_input_ids,
            "full_attention_mask": full_attention_mask,
        }
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported backends are 'torch', 'max', and 'graph'.")
