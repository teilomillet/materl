import os
from pathlib import Path
from typing import List, TYPE_CHECKING
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from max.graph import Graph

def build_model_graph(model_path: str, device_id: int = 0) -> "Graph":
    """
    Builds a MAX Graph from first principles using the MAX nn API.
    This function constructs a transformer model architecture that mirrors
    the HuggingFace model configuration but uses MAX's statically-typed
    graph construction approach.
    
    Args:
        model_path: Path to the HuggingFace model directory
        device_id: GPU device ID to use for graph construction
        
    Returns:
        MAX Graph representing the transformer model
    """
    from max.graph import Graph, TensorType, DeviceRef, ops
    from max.dtype import DType
    from max.nn import Embedding, Linear, LayerNorm, Module
    from max.nn.layer.layer_list import LayerList
    from transformers import AutoConfig
    from max.graph.weights.load import load_weights

    from transformers import AutoConfig
    from huggingface_hub import snapshot_download
    
    # Handle both local paths and HuggingFace model names
    if os.path.exists(model_path):
        # Local path - use as-is
        local_model_path = model_path
    else:
        # HuggingFace model name - download to cache
        local_model_path = snapshot_download(model_path)
    
    config = AutoConfig.from_pretrained(model_path)
    device = DeviceRef.GPU(device_id)
    dtype = DType.float32  # Using float32 for compatibility

    input_ids_type = TensorType(DType.int64, shape=["batch_size", "seq_len"], device=device)

    class TransformerModel(Module):
        """
        Top-level transformer model that mirrors HuggingFace architecture.
        Uses MAX nn components with explicit type specifications.
        """
        def __init__(self):
            super().__init__()
            # Use unique names for each component to avoid weight name collisions
            self.embedding = Embedding(
                vocab_size=config.vocab_size, 
                hidden_dim=config.hidden_size, 
                dtype=dtype, 
                device=device
            )
            # Create layers with unique names
            self.layers = LayerList([
                self.make_layer(layer_idx) for layer_idx in range(config.num_hidden_layers)
            ])
            # Fix: Add dtype parameter to final LayerNorm
            self.final_ln = LayerNorm(dims=config.hidden_size, device=device, dtype=dtype)
            self.lm_head = Linear(
                in_dim=config.hidden_size, 
                out_dim=config.vocab_size, 
                device=device, 
                dtype=dtype
            )

        def make_layer(self, layer_idx):
            """
            Creates a single transformer layer with attention and feedforward.
            This is a simplified version focusing on the core architecture.
            """
            class TransformerLayer(Module):
                def __init__(self):
                    super().__init__()
                    # Fix: Add dtype parameter to both LayerNorm instances
                    self.ln1 = LayerNorm(dims=config.hidden_size, device=device, dtype=dtype)
                    self.attn = Linear(
                        in_dim=config.hidden_size, 
                        out_dim=config.hidden_size, 
                        device=device, 
                        dtype=dtype
                    )
                    self.ln2 = LayerNorm(dims=config.hidden_size, device=device, dtype=dtype)
                    ff_dim = getattr(config, 'n_inner', config.hidden_size * 4)
                    self.ffn_up = Linear(
                        in_dim=config.hidden_size, 
                        out_dim=ff_dim, 
                        device=device, 
                        dtype=dtype
                    )
                    self.ffn_down = Linear(
                        in_dim=ff_dim, 
                        out_dim=config.hidden_size, 
                        device=device, 
                        dtype=dtype
                    )

                def __call__(self, x):
                    """
                    Forward pass through transformer layer.
                    Simplified: attention + feedforward with residual connections.
                    """
                    # Simplified attention (not multi-head)
                    x = x + self.attn(self.ln1(x))
                    # Feedforward with GELU activation
                    x = x + self.ffn_down(ops.gelu(self.ffn_up(self.ln2(x))))
                    return x
            return TransformerLayer()

        def __call__(self, tokens):
            """
            Forward pass through entire transformer model.
            """
            h = self.embedding(tokens)
            for layer in self.layers:
                h = layer(h)
            h = self.final_ln(h)
            return self.lm_head(h)

    transformer_model = TransformerModel()
    
    with Graph(name=f"graph_{model_path.replace('/', '_')}", input_types=[input_ids_type]) as graph:
        input_ids = graph.inputs[0].tensor
        
        # Load weights from HuggingFace model files
        # Priority: .safetensors > .bin to avoid ambiguity
        safetensors_files = [Path(local_model_path) / f for f in os.listdir(local_model_path) if f.endswith('.safetensors')]
        bin_files = [Path(local_model_path) / f for f in os.listdir(local_model_path) if f.endswith('.bin')]
        
        if safetensors_files:
            weight_files = safetensors_files
        elif bin_files:
            weight_files = bin_files
        else:
            raise FileNotFoundError(f"No weight files (.safetensors or .bin) found in {local_model_path}")
            
        loaded_weights = load_weights(weight_files)
        # Debug: Print available weight names to understand the mapping
        print("Available weight names:", [name for name, _ in loaded_weights.items()])
        
        # For now, skip weight loading to test the graph structure
        # TODO: Implement proper weight name mapping for HuggingFace <-> MAX nn
        # transformer_model.load_state_dict({
        #     name: weight.data() for name, weight in loaded_weights.items()
        # })
        
        logits = transformer_model(input_ids)
        graph.output(logits)
        
    return graph

def generate_with_graph(
    model_path: str, 
    tokenizer: PreTrainedTokenizerBase, 
    prompts: List[str], 
    max_new_tokens: int = 10, 
    device_id: int = 0
) -> List[str]:
    """
    Generates text using the MAX graph built from first principles.
    This function demonstrates autoregressive text generation using
    the MAX InferenceSession API with our custom-built graph.
    
    Args:
        model_path: Path to the HuggingFace model directory
        tokenizer: HuggingFace tokenizer for the model
        prompts: List of input text prompts
        max_new_tokens: Maximum number of tokens to generate
        device_id: GPU device ID to use for inference
        
    Returns:
        List of generated text completions
    """
    from max.engine.api import InferenceSession
    from max.driver import Tensor
    
    graph = build_model_graph(model_path, device_id)
    
    session = InferenceSession()
    model = session.load(graph)
    
    completions = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        
        current_ids = input_ids
        for _ in range(max_new_tokens):
            input_tensor = Tensor.from_numpy(current_ids)
            outputs = model.execute(input_tensor)
            logits = outputs[0].to_numpy()
            
            # Simple greedy decoding (argmax)
            next_token_id = np.argmax(logits[:, -1, :], axis=-1)
            
            current_ids = np.concatenate([current_ids, [next_token_id]], axis=1)

            # Stop if we hit EOS token
            if next_token_id[0] == tokenizer.eos_token_id:
                break
        
        # Extract only the newly generated tokens
        new_tokens = current_ids[0, len(input_ids[0]):]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(completion)
    
    return completions 