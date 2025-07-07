# materl.functions.values

"""
This module contains the functions for computing value estimates from a
value model. This is a key component of actor-critic algorithms like VAPO.
"""
import torch
from transformers.modeling_utils import PreTrainedModel
from typing import Dict, Any

def compute_values(
    value_model: PreTrainedModel,
    full_input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    **kwargs, # To absorb other unused kwargs from the engine
) -> Dict[str, torch.Tensor]:
    """
    Computes value estimates for each token in the sequence.

    Args:
        value_model: The model to use for computing values. This is typically a
                     model with a scalar value head.
        full_input_ids: The full sequence of tokens (prompt + completion).
        full_attention_mask: The attention mask for the full sequence.

    Returns:
        A dictionary containing the value estimates tensor.
    """
    # We use no_grad as the value model is not trained during this step;
    # its loss is computed separately from the gradient of the value estimates.
    with torch.no_grad():
        outputs = value_model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
        )
    
    # Assuming the value model outputs a 'value' field in its outputs,
    # which is common for models with a custom value head. If not, this
    # would need to be adapted (e.g., use the logits of a specific token).
    values = outputs.get("value", outputs.logits)

    # If the model returns a full logit distribution (e.g., when using a
    # base CausalLM as a value model), its shape will be (B, S, V). We must
    # reduce this to a scalar value per token, (B, S), to be used in GAE.
    if values.dim() == 3 and values.shape[-1] > 1:
        # We project the logits onto a fixed, randomly-initialized vector to
        # simulate a value head. This is a common practice for making examples
        # runnable without requiring a custom model architecture.
        vocab_size = values.shape[-1]
        torch.manual_seed(0) # for reproducibility
        rand_vector = torch.randn(
            vocab_size, 1, device=values.device, dtype=values.dtype
        )
        values = torch.matmul(values, rand_vector)

    # Squeeze the last dimension if it's 1 to get shape (B, S).
    values = values.squeeze(-1)

    return {"values": values} 