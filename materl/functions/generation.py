# materl.functions.generation

# This file contains pure functions for generating text completions from a model.
# This logic was extracted from the old monolithic GRPOEngine to promote
# a more modular, functional, and reusable design.

from typing import List, Dict, Any

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .masks import create_completion_masks


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    max_prompt_length: int,
    max_completion_length: int,
    num_generations: int,
    **generation_kwargs,
) -> Dict[str, Any]:
    """
    Generates completions for a batch of prompts and prepares all necessary tensors.

    Args:
        model: The policy model to use for generation.
        tokenizer: The tokenizer associated with the model.
        prompts: A list of string prompts.
        max_prompt_length: The maximum length for tokenized prompts.
        max_completion_length: The maximum number of new tokens to generate.
        num_generations: The number of completions to generate per prompt.
        **generation_kwargs: Additional arguments for the model's `generate` method
                             (e.g., temperature, top_p, top_k).

    Returns:
        A dictionary containing all tensors needed for subsequent pipeline steps,
        including completions, prompts, and their corresponding masks.
    """
    device = model.device
    eos_token_id = tokenizer.eos_token_id
    assert isinstance(eos_token_id, int), "Tokenizer must have an integer `eos_token_id`."

    # Duplicate prompts for the number of generations required
    expanded_prompts_text = [p for p in prompts for _ in range(num_generations)]

    # Tokenize prompts for the model
    prompt_inputs = tokenizer(
        expanded_prompts_text,
        return_tensors="pt",
        padding="max_length",
        max_length=max_prompt_length,
        truncation=True,
        add_special_tokens=False,
    ).to(device)

    # --- Generation ---
    # The 'max_seq_length' is a config for our pipeline, not for Hugging Face's `generate`.
    # We remove it to avoid passing an unexpected keyword argument.
    generation_kwargs.pop("max_seq_length", None)
    
    with torch.no_grad():
        output_sequences = model.generate(  # type: ignore[attr-defined]
            input_ids=prompt_inputs.input_ids,
            attention_mask=prompt_inputs.attention_mask,
            max_new_tokens=max_completion_length,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,  # Essential for exploration in RL
            **generation_kwargs,
        )

    # Extract completions and create masks
    completions_ids = output_sequences[:, prompt_inputs.input_ids.shape[1]:]
    completion_masks = create_completion_masks(completions_ids, eos_token_id, device)
    
    # Decode completions for reward functions that operate on text
    completions_text = tokenizer.batch_decode(completions_ids, skip_special_tokens=True)

    # Combine prompts and completions for logprob calculation
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