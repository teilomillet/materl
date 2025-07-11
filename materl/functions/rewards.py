# materl.functions.rewards

"""
Stateless reward computation functions.

This module provides pure functions for calculating rewards, which are a critical
component of the RL feedback loop. Each function takes the necessary data
(prompts, completions, etc.) and returns a list or tensor of scalar rewards.

A registry (`_get_reward_function`) is used to map reward function names from the
config to their implementations, making the system extensible. The main entry
point, `compute_rewards`, orchestrates the calling of these functions based on
the provided configuration.
"""
from typing import List, Callable, Dict, Any
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from materl.config.reward_config import RewardFunctionConfig, RewardFunctionType


def length_reward(
    completion_ids: torch.Tensor,
    completion_masks: torch.Tensor,
    eos_token_id: int,
    max_completion_length: int,
    **kwargs,  # To absorb any other unused kwargs
) -> List[float]:
    """
    Computes a reward based on the length of the generated text, normalized
    to a [0, 1] range.

    The length is determined by the number of non-padding tokens, capped
    by the first occurrence of the EOS token. The raw length is then scaled
    by `max_completion_length` to prevent reward explosion.

    Args:
        completion_ids: The tensor of completion tokens.
        completion_masks: The attention mask for the completions.
        eos_token_id: The ID of the end-of-sentence token.
        max_completion_length: The maximum possible length for normalization.

    Returns:
        A list of scalar rewards, one for each completion in the batch.
    """
    rewards = []
    max_len = float(max_completion_length)

    for i in range(completion_ids.shape[0]):
        # Find the position of the first EOS token
        eos_indices = (completion_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]

        if len(eos_indices) > 0:
            # Length is the index of the first EOS token
            length = eos_indices[0].item()
        else:
            # If no EOS, length is the total number of non-padding tokens
            length = completion_masks[i].sum().item()

        # Scale the reward to a [0, 1] range.
        rewards.append(float(length) / max_len)
    return rewards


def outcome_reward(
    completions_text: List[str],
    **kwargs,
) -> List[float]:
    """
    A simple placeholder for an outcome-based reward.

    It returns 1.0 if the completion contains "success", otherwise -1.0.
    This simulates a reward based on the final outcome of a trajectory,
    which is a core concept for the REINFORCE algorithm.
    """
    rewards = []
    for text in completions_text:
        if "success" in text.lower():
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards



REWARD_FUNCTION_REGISTRY: Dict[str, Callable] = {
    "length_reward": length_reward,
    "outcome_reward": outcome_reward,
    # New Python reward functions can be registered here
}


def _get_reward_function(name: str) -> Callable:
    """
    Retrieves a reward function from the registry.

    Args:
        name: The name of the reward function.

    Returns:
        The callable reward function.

    Raises:
        ValueError: If the function name is not in the registry.
    """
    if name not in REWARD_FUNCTION_REGISTRY:
        raise ValueError(f"Reward function '{name}' not found in registry.")
    return REWARD_FUNCTION_REGISTRY[name]


def compute_reward(
    # This function computes a single, weighted reward.
    name: str,
    weight: float,
    prompts_text: List[str],
    completions_text: List[str],
    completions_ids: torch.Tensor,
    completion_masks: torch.Tensor,
    tokenizer: "PreTrainedTokenizerBase",
    device: torch.device,
    max_completion_length: int,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Computes a single reward function and returns both per-sequence and
    per-token tensors.
    """
    reward_fn = _get_reward_function(name)
    
    # Prepare args for the specific reward function
    reward_fn_args = {
        "prompts_text": prompts_text,
        "completions_text": completions_text,
        "completion_ids": completions_ids,
        "completion_masks": completion_masks,
        "max_completion_length": max_completion_length,
        "eos_token_id": tokenizer.eos_token_id,
        **kwargs,
    }

    py_rewards = reward_fn(**reward_fn_args)
    
    # The final rewards are per-sequence
    rewards_tensor = torch.tensor(py_rewards, dtype=torch.float32, device=device)
    
    # We also create a per-token reward tensor, where the reward is assigned
    # to the last token of the completion. This is used by GAE.
    per_token_rewards = torch.zeros_like(completions_ids, dtype=torch.float32, device=device)
    sequence_lengths = completion_masks.sum(dim=1) - 1
    batch_indices = torch.arange(rewards_tensor.shape[0], device=device)
    per_token_rewards[batch_indices, sequence_lengths] = rewards_tensor
    
    return {
        "rewards_tensor": rewards_tensor * weight,
        "rewards_per_token": per_token_rewards * weight
    }

def compute_rewards(
    reward_configs: List[Dict[str, Any]],
    prompts_text: List[str],
    completions_text: List[str],
    completions_ids: torch.Tensor,
    completion_masks: torch.Tensor,
    tokenizer: "PreTrainedTokenizerBase", # Fwd reference
    device: torch.device,
    reward_baseline: float,
    max_completion_length: int,
    **kwargs, # To absorb other unused kwargs from the engine
) -> Dict[str, torch.Tensor]:
    """
    Computes the total reward for each completion by aggregating rewards from
    all configured reward functions.

    This function iterates through the provided reward configurations, dispatches
    to the appropriate Python or Mojo reward function, applies the specified

    weight, and aggregates the results into a single tensor.

    Args:
        reward_configs: A list of `RewardFunctionConfig` objects.
        prompts_text: The raw text of the prompts.
        completions_text: The raw text of the generated completions.
        completions_ids: The tensor of completion tokens.
        completion_masks: The attention mask for the completions.
        tokenizer: The tokenizer used for tokenization.
        device: The torch device to place the final tensor on.
        reward_baseline: A baseline value to subtract from all rewards.
        max_completion_length: The max completion length, needed by some functions.

    Returns:
        A tensor of total rewards for each completion.
    """
    num_total_gens = completions_ids.shape[0]
    total_rewards = torch.zeros(num_total_gens, dtype=torch.float32, device=device)
    total_rewards_per_token = torch.zeros_like(completions_ids, dtype=torch.float32, device=device)

    for config in reward_configs:
        name = config.get("name")
        if not name:
            print(f"Warning: Skipping reward config with no name: {config}")
            continue

        # We now call the single reward function for each config
        reward_result = compute_reward(
            name=name,
            weight=config.get("weight", 1.0),
            prompts_text=prompts_text,
            completions_text=completions_text,
            completions_ids=completions_ids,
            completion_masks=completion_masks,
            tokenizer=tokenizer,
            device=device,
            max_completion_length=max_completion_length,
            **config.get("kwargs", {}),
        )
        total_rewards += reward_result["rewards_tensor"]
        total_rewards_per_token += reward_result["rewards_per_token"]

    # The final rewards are per-sequence, but for GAE we need per-token rewards.
    # The simplest approach is to assign the total reward to the last token of the completion.
    per_token_rewards = torch.zeros_like(completions_ids, dtype=torch.float32, device=device)
    
    # Get the index of the last non-padding token for each sequence
    sequence_lengths = completion_masks.sum(dim=1) - 1
    
    # Create an index tensor for scatter_
    batch_indices = torch.arange(num_total_gens, device=device)
    
    # Place the total reward at the position of the last token
    per_token_rewards[batch_indices, sequence_lengths] = total_rewards
    
    # Apply the reward baseline to the final sequence reward, not per-token
    total_rewards -= reward_baseline
    
    # Return rewards in a dictionary to maintain a consistent interface
    # This now serves both value-free (GRPO/DAPO) and value-based (VAPO) methods.
    return {
        "rewards_tensor": total_rewards,       # For value-free advantage calculation
        "rewards_per_token": total_rewards_per_token # For GAE advantage calculation
    } 
# Note: Adding a comment to force linter re-evaluation. 