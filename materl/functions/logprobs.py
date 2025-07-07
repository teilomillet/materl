# materl.functions.logprobs

"""
Stateless log probability calculation functions.

This module provides a pure function for a core operation in policy-based RL:
calculating the log probability of a sequence of tokens given a model. This
logic is used for both the "old" policy (for the PPO ratio) and the "current"
policy (for the loss gradient), so centralizing it here removes redundancy
and improves clarity.
"""
from typing import Dict, Any, Optional
import torch
from transformers.modeling_utils import PreTrainedModel

def _compute_logprobs_from_logits(
    logits: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor
) -> torch.Tensor:
    """Helper to compute logprobs from logits, completions, and masks."""
    all_logprobs = torch.log_softmax(logits, dim=-1)
    gathered_logprobs = torch.gather(
        all_logprobs, 
        2, 
        completion_ids.unsqueeze(-1)
    ).squeeze(-1)
    return gathered_logprobs * completion_mask

def compute_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the log probabilities of completion tokens given a single model.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    prompt_seq_len = input_ids.shape[1] - completion_ids.shape[1]
    completion_logits = logits[:, prompt_seq_len - 1 : -1, :]
    return _compute_logprobs_from_logits(completion_logits, completion_ids, completion_mask)

def compute_policy_and_ref_logprobs(
    policy_model: PreTrainedModel,
    full_input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    completions_ids: torch.Tensor,
    completion_masks: torch.Tensor,
    ref_model: Optional[PreTrainedModel] = None,
    **kwargs, # To absorb other unused kwargs from the engine
) -> Dict[str, Any]:
    """
    Computes log probabilities for both the policy and an optional reference model.

    This is the primary logprob function designed to be injected into the RLEngine.
    It returns a dictionary, as expected by the engine's orchestration logic.
    """
    # The caller is responsible for the gradient context (e.g., `torch.no_grad()`).
    policy_logits = policy_model(
        input_ids=full_input_ids, 
        attention_mask=full_attention_mask
    ).logits
    
    prompt_seq_len = full_input_ids.shape[1] - completions_ids.shape[1]
    completion_logits = policy_logits[:, prompt_seq_len - 1 : -1, :]
    
    policy_logprobs = _compute_logprobs_from_logits(
        completion_logits, completions_ids, completion_masks
    )

    output = {"policy_logprobs": policy_logprobs}

    if ref_model:
        ref_logits = ref_model(
            input_ids=full_input_ids, 
            attention_mask=full_attention_mask
        ).logits
        ref_completion_logits = ref_logits[:, prompt_seq_len - 1 : -1, :]
        ref_logprobs = _compute_logprobs_from_logits(
            ref_completion_logits, completions_ids, completion_masks
        )
        output["ref_logprobs"] = ref_logprobs
        
    return output 