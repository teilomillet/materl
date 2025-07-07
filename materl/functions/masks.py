# materl.functions.masks

"""
Stateless mask creation functions.

This module provides pure functions for creating various types of masks
(e.g., for attention, completions) that are used throughout the training
process. Centralizing this logic improves clarity and reusability.
"""
import torch

def create_completion_masks(
    completions_ids: torch.Tensor,
    eos_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Creates a completion mask based on the position of the EOS token.

    The mask is a tensor of ones with the same shape as `completions_ids`.
    For each sequence in the batch, all elements after the first occurrence
    of the `eos_token_id` (inclusive) are set to zero.

    Args:
        completions_ids: The tensor of completion tokens.
        eos_token_id: The ID of the end-of-sentence token.
        device: The torch device to create the tensor on.

    Returns:
        A long tensor representing the completion mask.
    """
    # Start with a mask of all ones
    completion_masks = torch.ones_like(completions_ids, dtype=torch.long, device=device)
    
    # Find the first EOS token in each sequence and zero out the mask from that point on
    for i, seq in enumerate(completions_ids):
        eos_indices = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_indices) > 0:
            first_eos_idx = eos_indices[0]
            # Mask tokens *after* the first EOS token.
            # The +1 is crucial to ensure the EOS token itself is included
            # in the loss calculation, but subsequent tokens are not.
            completion_masks[i, first_eos_idx + 1:] = 0
            
    return completion_masks 