# materl.functions.advantages

"""
Stateless advantage calculation functions.

This module provides pure functions for computing advantages, a key component
in policy gradient algorithms. The primary method is group-wise normalization,
which stabilizes training by standardizing rewards within a group of generated
samples.
"""
import torch
from typing import Dict, Any, List, Optional
from . import rewards as reward_fns

def _compute_groupwise_advantages(
    rewards_tensor: torch.Tensor,
    num_generations: int,
    scale_advantages: bool = True,
    **kwargs, # To absorb other unused kwargs from the engine
) -> Dict[str, torch.Tensor]:
    """
    Computes group-wise normalized advantages from rewards.

    This function first shapes the flat rewards tensor into groups based on the
    number of generations per prompt. It then normalizes the rewards for each
    group (prompt) by subtracting the mean and optionally dividing by the
    standard deviation. This process, often called "whitening," stabilizes training.

    Args:
        rewards_tensor: A 1D tensor of rewards for each generated completion.
                 Shape: (num_prompts * num_generations,)
        num_generations: The number of completions generated per prompt.
        scale_advantages: If True, scale advantages by the standard deviation of
                          the rewards within each group.

    Returns:
        A dictionary containing the computed advantages tensor.
    """
    num_total_gens = rewards_tensor.shape[0]
    if num_total_gens == 0:
        return {"advantages": torch.tensor([], device=rewards_tensor.device)}
    
    if num_total_gens % num_generations != 0:
        raise ValueError("Total number of rewards is not divisible by num_generations.")
    
    num_unique_prompts = num_total_gens // num_generations
    
    # Reshape rewards to (num_unique_prompts, num_generations) for group-wise stats
    grouped_rewards = rewards_tensor.view(num_unique_prompts, num_generations)
    
    # Compute mean and std dev per group
    mean_grouped_rewards = grouped_rewards.mean(dim=1, keepdim=True)
    std_grouped_rewards = grouped_rewards.std(dim=1, keepdim=True)
    
    # Center the advantages by subtracting the mean reward of the group
    advantages = grouped_rewards - mean_grouped_rewards
    
    # Stabilization: Only scale advantages if the standard deviation is meaningful.
    if scale_advantages:
        is_std_meaningful_mask = std_grouped_rewards > 1e-8
        rows_to_scale = is_std_meaningful_mask.squeeze(1)
        if torch.any(rows_to_scale):
             advantages[rows_to_scale] = advantages[rows_to_scale] / (std_grouped_rewards[rows_to_scale] + 1e-8)
            
    # Return the flattened advantages tensor. The key is now standardized.
    return {"advantages": advantages.view(-1)} 


def compute_advantages(
    # This function now accepts the raw inputs for reward computation,
    # encapsulating the dependency and simplifying the compiler.
    reward_configs: List[Dict[str, Any]],
    prompts_text: List[str],
    completions_text: List[str],
    completions_ids: torch.Tensor,
    completion_masks: torch.Tensor,
    tokenizer: Any,
    device: Any,
    max_completion_length: int,
    num_generations: int,
    # The presence of `values` determines whether we use GAE or not.
    values: Optional[torch.Tensor] = None,
    full_input_ids: Optional[torch.Tensor] = None,
    # GAE-specific parameters
    gamma: float = 0.99,
    lam: float = 0.95,
    # kwargs can absorb other context from the compiler.
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Orchestrates reward and advantage computation.

    This function acts as a dispatcher. If a `values` tensor is provided, it
    computes advantages using Generalized Advantage Estimation (GAE). Otherwise,
    it computes group-wise normalized advantages, a value-free method.
    """
    # 1. Compute rewards. This is common to both GAE and value-free methods.
    # The reward function returns both per-sequence and per-token rewards.
    reward_result = reward_fns.compute_rewards(
        reward_configs=reward_configs,
        prompts_text=prompts_text,
        completions_text=completions_text,
        completions_ids=completions_ids,
        completion_masks=completion_masks,
        tokenizer=tokenizer,
        device=device,
        reward_baseline=0.0,
        max_completion_length=max_completion_length,
    )

    # 2. Dispatch to the appropriate advantage calculation method.
    if values is not None:
        # Value-based path (GAE)
        if full_input_ids is None:
            raise ValueError("`full_input_ids` must be provided for GAE computation.")
        # We need the prompt length to correctly slice the `values` tensor, which
        # is computed on the full sequence (prompt + completion).
        # We can infer prompt length from the difference between the full sequence
        # and the completion sequence.
        # Note: This assumes all prompts in the batch are padded to the same length.
        prompt_length = full_input_ids.shape[1] - completions_ids.shape[1]
        
        return compute_gae_advantages(
            rewards_per_token=reward_result["rewards_per_token"],
            values=values,
            completion_masks=completion_masks,
            prompts_input_ids=full_input_ids[:, :prompt_length],
            gamma=gamma,
            lam=lam,
        )
    else:
        # Value-free path (group-wise normalization)
        return _compute_groupwise_advantages(
            rewards_tensor=reward_result["rewards_tensor"],
            num_generations=num_generations
        )


def compute_gae_advantages(
    rewards_per_token: torch.Tensor,
    values: torch.Tensor,
    completion_masks: torch.Tensor,
    prompts_input_ids: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    adaptive_lam: bool = False,
    **kwargs, # Absorb unused arguments from the engine's generic call
) -> Dict[str, torch.Tensor]:
    """
    Computes Generalized Advantage Estimation (GAE).

    Args:
        rewards_per_token: Tensor of rewards for each token.
        values: Tensor of value estimates for the *full sequence* (prompt + completion).
        completion_masks: Mask indicating the completion part of the sequence.
        prompts_input_ids: The tensor of prompt tokens, used to determine where completions begin.
        gamma: Discount factor.
        lam: GAE lambda parameter.
        adaptive_lam: Whether to use length-adaptive lambda.
        **kwargs: Absorbs other arguments passed by the engine.

    Returns:
        A dictionary containing 2D "advantages" and "returns" (value targets).
    """
    
    # The `values` tensor is for the full sequence. We must slice it to get
    # the values for only the completion part, which aligns with the shape of
    # `completion_masks` and `rewards_per_token`.
    prompt_length = prompts_input_ids.shape[1]
    completion_values = values[:, prompt_length:]

    # Ensure inputs are properly masked. The advantage calculation should only
    # consider the rewards and values of the generated completion tokens.
    rewards = rewards_per_token * completion_masks
    values_masked = completion_values * completion_masks
    
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    sequence_lengths = completion_masks.sum(dim=1)

    for t in reversed(range(rewards.size(1))):
        
        current_mask = completion_masks[:, t]
        
        # Determine the lambda for the current batch
        if adaptive_lam:
            # Simple linear scaling of lambda with sequence length.
            # Shorter sequences get smaller lambda (more bias, less variance)
            # Longer sequences get larger lambda (less bias, more variance)
            max_len = rewards.size(1)
            # Ensure we don't divide by zero for empty sequences
            safe_max_len = max_len if max_len > 0 else 1.0
            current_lam = lam * (sequence_lengths / safe_max_len)
        else:
            current_lam = lam

        # The value of the next state is the value of the next token,
        # or 0 if we are at the end of the sequence.
        next_values = values_masked[:, t + 1] if t < rewards.size(1) - 1 else torch.zeros_like(values_masked[:, t])
        
        # TD-error (delta)
        delta = rewards[:, t] + gamma * next_values - values_masked[:, t]
        
        # GAE advantage estimation: if mask is 0, last_advantage becomes 0.
        last_advantage = delta + gamma * current_lam * last_advantage * current_mask
        advantages[:, t] = last_advantage

    returns = advantages + values_masked
    
    return {"advantages": advantages.detach(), "returns": returns.detach()} 