# materl.functions.step

# This module provides pure functions that define a single training step
# for a given RL algorithm. By encapsulating the optimization logic here,
# we make the RLTrainer a generic loop runner and allow for full algorithmic
# customization by simply swapping the step function.

from typing import Dict, Any, Optional, Callable, Union
import torch
from transformers.modeling_utils import PreTrainedModel
from dataclasses import asdict

from ..config import GRPOConfig, DAPOConfig, VAPOConfig
from .logprobs import compute_logprobs
from .values import compute_values

def _perform_gradient_update(
    loss: torch.Tensor,
    optimizers: list,
    lr_schedulers: Optional[list] = None
) -> None:
    """
    Performs the backward pass and optimizer step for a list of optimizers.
    This is a core "first principle" of training that is shared across algorithms.
    """
    # Reset gradients for all optimizers
    for opt in optimizers:
        opt.zero_grad()

    # Perform the backward pass once for the combined loss
    loss.backward()

    # Step each optimizer
    for opt in optimizers:
        opt.step()

    # Step each learning rate scheduler
    if lr_schedulers:
        for scheduler in lr_schedulers:
            if scheduler:
                scheduler.step()
                
def unified_training_step(
    processed_data: Dict[str, Any],
    models: Dict[str, PreTrainedModel],
    optimizers: Dict[str, torch.optim.Optimizer],
    lr_schedulers: Dict[str, Any],
    loss_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    algorithm_config: Union[GRPOConfig, DAPOConfig, VAPOConfig],
) -> Dict[str, Any]:
    """
    Performs a single, unified training step for any configured RL algorithm.

    This function is the heart of the "first principles" training process. It
    is a generic orchestrator that:
    1.  Performs the necessary forward passes for the given models (policy, value).
    2.  Assembles a standardized data dictionary with all inputs for the loss function.
    3.  Calls the injected, algorithm-specific `loss_fn`.
    4.  Performs a gradient update for all provided optimizers.

    This design decouples the training loop from the specifics of any single
    RL algorithm, allowing for maximum composability.
    """
    # 1. Prepare data for forward passes
    # Detach tensors from the engine output that should not have gradients flowing back
    data_for_loss = {
        key: val.detach() if isinstance(val, torch.Tensor) else val
        for key, val in processed_data.items()
    }

    # 2. Perform forward passes to enable gradient computation
    # Re-calculate logprobs with the current (training) policy model
    policy_model = models['policy']
    current_policy_logprobs = compute_logprobs(
        model=policy_model,
        input_ids=data_for_loss["full_input_ids"],
        attention_mask=data_for_loss["full_attention_mask"],
        completion_ids=data_for_loss["completions_ids"],
        completion_mask=data_for_loss["completion_masks"],
    )
    data_for_loss["current_policy_logprobs"] = current_policy_logprobs
    data_for_loss["old_policy_logprobs"] = processed_data["policy_logprobs"].detach() # Ensure it is there

    # If a value model is part of the algorithm, run its forward pass
    if 'value' in models and models['value'] is not None:
        value_model = models['value']
        current_values_dict = compute_values(
            value_model=value_model,
            full_input_ids=data_for_loss["full_input_ids"],
            full_attention_mask=data_for_loss["full_attention_mask"],
        )
        current_values_full = current_values_dict["values"]

        # Slice to get completion-only values, which is what the loss function expects
        prompt_length = data_for_loss["prompts_input_ids"].shape[1]
        data_for_loss["current_values"] = current_values_full[:, prompt_length:]
        data_for_loss["old_values"] = processed_data["values"].detach() # Ensure it is there

    # 3. Assemble final data and compute loss
    # Add algorithm hyperparameters to the data dictionary for the loss function
    data_for_loss.update(asdict(algorithm_config))

    # Compute the loss using the injected, pure loss function
    loss_metrics = loss_fn(data_for_loss)
    loss = loss_metrics["loss"]

    # 4. Perform Gradient Update
    # This step is delegated to our shared, first-principle gradient update function,
    # which can handle multiple optimizers and schedulers simultaneously.
    _perform_gradient_update(
        loss,
        optimizers=list(optimizers.values()),
        lr_schedulers=list(lr_schedulers.values())
    )

    # 5. Return metrics for logging
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_metrics.items()} 