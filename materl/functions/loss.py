# materl.functions.loss

"""
Stateless loss computation functions for policy gradient algorithms.

This module provides pure functions for calculating the loss, such as the
clipped surrogate objective used in GRPO and PPO. By breaking down the loss
calculation into smaller, "first principles" functions, we can easily compose
them to create different RL objectives.
"""
import torch
from typing import Dict, Any, Optional


def _compute_ppo_policy_loss(
    log_ratio: torch.Tensor, 
    advantages: torch.Tensor, 
    epsilon: float,
    completion_masks: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes the PPO clipped surrogate objective."""
    ratio = torch.exp(log_ratio)
    policy_loss_unclipped = -advantages * ratio
    policy_loss_clipped = -advantages * torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    
    loss_per_element = torch.maximum(policy_loss_unclipped, policy_loss_clipped)
    
    if completion_masks is not None:
        # Per-token loss
        return (loss_per_element * completion_masks).sum() / completion_masks.sum().clamp(min=1)
    else:
        # Per-sequence loss
        return torch.mean(loss_per_element)


def _compute_kl_penalty(
    current_policy_logprobs: torch.Tensor, 
    ref_logprobs: torch.Tensor, 
    completion_masks: torch.Tensor
) -> torch.Tensor:
    """Computes the KL divergence penalty."""
    kl_div_per_token = (current_policy_logprobs - ref_logprobs)
    # Average KL divergence only over completion tokens for each sequence
    mean_kl_per_sequence = (kl_div_per_token * completion_masks).sum(dim=1) / completion_masks.sum(dim=1).clamp(min=1)
    # Average over the batch
    return mean_kl_per_sequence.mean()


def _compute_value_function_loss(
    current_values: torch.Tensor, 
    returns: torch.Tensor, 
    completion_masks: torch.Tensor,
    clip_range_vf: Optional[float] = None,
    old_values: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes the mean squared error for the value function."""
    masked_current_values = current_values[completion_masks.bool()]
    masked_returns = returns[completion_masks.bool()]

    # PPO's value function clipping
    # This stabilizes training by restricting how much the value function can change at each step.
    if clip_range_vf is not None and old_values is not None:
        masked_old_values = old_values[completion_masks.bool()]
        
        # Clip the difference between new and old values
        values_clipped = masked_old_values + torch.clamp(
            masked_current_values - masked_old_values, 
            -clip_range_vf, 
            clip_range_vf
        )
        
        # Two versions of the loss
        vf_loss_unclipped = (masked_current_values - masked_returns) ** 2
        vf_loss_clipped = (values_clipped - masked_returns) ** 2
        
        # Return the maximum of the two losses
        return 0.5 * torch.mean(torch.maximum(vf_loss_unclipped, vf_loss_clipped))
    
    # Standard, unclipped value loss
    return torch.mean((masked_current_values - masked_returns) ** 2)


def compute_policy_gradient_loss(
    advantages: torch.Tensor,
    current_policy_logprobs: torch.Tensor,
    old_policy_logprobs: torch.Tensor,
    completion_masks: torch.Tensor,
    ref_logprobs: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    beta: float = 0.1,
    vf_coef: float = 0.1,
    # The following parameters are for DAPO-style loss calculation
    average_logprobs_by_token: bool = False,
    clip_ratio_lower: float = 0.2,
    clip_ratio_upper: float = 0.2,
    **kwargs, # To absorb other unused kwargs from the engine
) -> Dict[str, torch.Tensor]:
    """
    Computes a policy gradient loss, supporting both GRPO and DAPO-style
    calculations. This function is a generalization of the PPO loss.

    Args:
        advantages: The advantages tensor.
        current_policy_logprobs: Log probabilities from the current policy.
        old_policy_logprobs: Log probabilities from the policy before the update.
        completion_masks: Mask for the completion part of the sequences.
        ref_logprobs: Log probabilities from the reference policy for KL penalty.
        values: Predicted values from the policy.
        returns: Returns from the environment.
        beta: The KL divergence penalty coefficient.
        vf_coef: The value function loss coefficient.
        average_logprobs_by_token: If True, average loss over all tokens in the
                                   batch (DAPO-style). If False, average first
                                   by sequence length, then by batch size (GRPO-style).
        clip_ratio_lower: The lower bound for clipping the policy ratio.
        clip_ratio_upper: The upper bound for clipping the policy ratio.

    Returns:
        A dictionary containing the total loss and other metrics.
    """
    # Policy Loss
    old_policy_logprobs = old_policy_logprobs.detach()
    is_per_token = advantages.dim() > 1

    if is_per_token:
        # Per-token advantages (from GAE for value-based models like VAPO)
        # require per-token importance ratios.
        log_ratio = current_policy_logprobs - old_policy_logprobs
        ratio = torch.exp(log_ratio)
        
        unclipped_loss = -advantages * ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_lower, 1.0 + clip_ratio_upper)
        clipped_loss = -advantages * clipped_ratio
        
        # Mask the loss to only include completion tokens and average over all tokens.
        policy_loss_per_token = torch.maximum(unclipped_loss, clipped_loss) * completion_masks
        policy_loss = policy_loss_per_token.sum() / completion_masks.sum()
        
    else:
        # Per-sequence advantages (from value-free models like GRPO/DAPO).
        summed_current_logprobs = (current_policy_logprobs * completion_masks).sum(dim=1)
        summed_old_logprobs = (old_policy_logprobs * completion_masks).sum(dim=1)
        ratio = torch.exp(summed_current_logprobs - summed_old_logprobs)

        unclipped_loss = -advantages * ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_lower, 1.0 + clip_ratio_upper)
        clipped_loss = -advantages * clipped_ratio
        policy_loss_per_sequence = torch.maximum(unclipped_loss, clipped_loss)

        if average_logprobs_by_token:
            # DAPO-style: Average the loss over all tokens in the batch.
            policy_loss = policy_loss_per_sequence.sum() / completion_masks.sum()
        else:
            # GRPO-style: Average the loss per sequence, giving each equal weight.
            policy_loss = policy_loss_per_sequence.mean()

    # KL Divergence Penalty (optional)
    kl_divergence_loss = torch.tensor(0.0, device=policy_loss.device)
    if ref_logprobs is not None and beta > 0:
        ref_logprobs = ref_logprobs.detach()

        # The KL divergence is calculated on a per-sequence basis, so we sum
        # the logprobs over the sequence length for the policy and ref models.
        summed_current_logprobs = (current_policy_logprobs * completion_masks).sum(dim=1)
        summed_ref_logprobs = (ref_logprobs * completion_masks).sum(dim=1)
        
        # The KL divergence is the difference between the logprobs of the current
        # policy and the reference policy.
        kl_per_sequence = summed_current_logprobs - summed_ref_logprobs
        
        # Average the KL divergence across the batch
        kl_divergence_loss = kl_per_sequence.mean()

    # Value Loss (optional)
    value_loss = torch.tensor(0.0, device=policy_loss.device)
    if values is not None and returns is not None:
        # `values` is for the full sequence, but `returns` is for the completion only.
        # We must slice `values` to align with `returns` before computing the loss.
        prompt_len = values.shape[1] - returns.shape[1]
        completion_values = values[:, prompt_len:]

        # The value loss is the mean squared error between the predicted values
        # and the empirical returns from the GAE calculation. We only compute
        # the loss over the actual (unmasked) completion tokens.
        value_loss_unmasked = (completion_values - returns).pow(2)
        value_loss_masked = value_loss_unmasked * completion_masks
        value_loss = value_loss_masked.sum() / completion_masks.sum()

    # Total loss is the policy loss plus scaled penalties
    total_loss = policy_loss + (beta * kl_divergence_loss) + (vf_coef * value_loss)

    return {
        "loss": total_loss,
        "policy_loss_scalar": torch.tensor(policy_loss.item()),
        "kl_divergence_scalar": torch.tensor(kl_divergence_loss.item()),
        "value_loss_scalar": torch.tensor(value_loss.item()),
    }


def compute_reinforce_loss(
    current_policy_logprobs: torch.Tensor,
    returns: torch.Tensor,
    completion_masks: torch.Tensor,
    **kwargs, # Absorb unused arguments
) -> Dict[str, torch.Tensor]:
    """
    Computes the REINFORCE policy gradient loss.

    The loss is the negative sum of log probabilities of actions multiplied by
    the discounted future returns. This function is the concrete implementation
    for the "reinforce_loss" operation.

    Loss = - (G_t * log(pi(a_t | s_t)))

    Args:
        current_policy_logprobs: Log probabilities from the current policy.
        returns: The discounted returns (Gt) for each token.
        completion_masks: Mask for the completion part of the sequences.

    Returns:
        A dictionary containing the total loss and other metrics.
    """
    # The policy gradient is -(returns * logprobs).
    # We want to maximize this, so we minimize its negative.
    policy_loss_per_token = -returns * current_policy_logprobs

    # We only consider the loss for the tokens that were actually generated.
    masked_loss = policy_loss_per_token * completion_masks

    # We average the loss over all tokens in the batch.
    loss = masked_loss.sum() / completion_masks.sum().clamp(min=1)

    return {
        "loss": loss,
        "policy_loss_scalar": torch.tensor(loss.item()),
    }


def compute_kimi_reinforce_loss(
    current_policy_logprobs: torch.Tensor,
    returns: torch.Tensor,
    rewards_tensor: torch.Tensor,
    completion_masks: torch.Tensor,
    discard_negative_samples: bool = True,
    **kwargs, # Absorb unused arguments
) -> Dict[str, torch.Tensor]:
    """
    Computes the REINFORCE loss with strategic negative sample control,
    inspired by the Kimi-Researcher paper.

    If `discard_negative_samples` is True, it masks out any trajectories
    that received a negative final reward.
    """
    policy_loss_per_token = -returns * current_policy_logprobs
    masked_loss = policy_loss_per_token * completion_masks

    if discard_negative_samples:
        # Create a mask for samples with non-negative rewards.
        # rewards_tensor has shape (batch_size,), we need it to be (batch_size, seq_len)
        # to correctly mask the loss.
        positive_sample_mask = (rewards_tensor >= 0).unsqueeze(1)
        masked_loss *= positive_sample_mask
    
    # Average the loss over all tokens in the batch.
    # The denominator remains the same to not artificially inflate the loss
    # when samples are discarded.
    loss = masked_loss.sum() / completion_masks.sum().clamp(min=1)

    return {
        "loss": loss,
        "policy_loss_scalar": torch.tensor(loss.item()),
    }


def compute_vapo_loss(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes the VAPO loss from a unified data dictionary.

    This function unpacks the necessary tensors and hyperparameters from the data
    dictionary, adhering to a standardized interface for loss computation. It
    handles both the policy and value loss components.
    """
    # Unpack required tensors from the data dictionary
    current_policy_logprobs = data["current_policy_logprobs"]
    old_policy_logprobs = data["old_policy_logprobs"]
    advantages = data["advantages"]
    returns = data["returns"]
    current_values = data["current_values"]
    old_values = data["old_values"]
    completion_masks = data["completion_masks"]
    ref_logprobs = data.get("ref_logprobs")

    # Unpack hyperparameters
    beta = data["beta"]
    epsilon = data["epsilon"]
    vf_coef = data["vf_coef"]
    clip_range_vf = data.get("clip_range_vf")

    # VAPO uses per-token advantages, so we compute a per-token log ratio
    log_ratio = (current_policy_logprobs - old_policy_logprobs)
    
    policy_loss = _compute_ppo_policy_loss(log_ratio, advantages, epsilon, completion_masks)
    value_loss = _compute_value_function_loss(
        current_values, returns, completion_masks, clip_range_vf, old_values
    )

    total_loss = policy_loss + vf_coef * value_loss
    
    kl_div_scalar = 0.0
    if beta > 0 and ref_logprobs is not None:
        kl_div_scalar_tensor = _compute_kl_penalty(current_policy_logprobs, ref_logprobs, completion_masks)
        total_loss += beta * kl_div_scalar_tensor
        kl_div_scalar = kl_div_scalar_tensor.item()

    metrics: Dict[str, Any] = {
        "loss": total_loss,
        "policy_loss_scalar": policy_loss.item(),
        "value_loss_scalar": value_loss.item(),
        "kl_divergence_scalar": kl_div_scalar
    }
    
    return metrics 