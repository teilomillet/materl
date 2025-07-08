"""
materl.recipes.grpo

The canonical recipe for the GRPO algorithm.
"""
from typing import TYPE_CHECKING, Optional
from .decorator import recipe
from .steps import (
    generation_step,
    logprobs_step,
    rewards_step,
    advantages_step,
    loss_step,
)

# This block is only for static analysis. It's ignored at runtime.
if TYPE_CHECKING:
    from ..agents import Agent
    from .builder import GraphBuilder


@recipe
def grpo(
    ml: "GraphBuilder",
    policy: "Agent",
    ref_policy: "Agent",
    prompts: list[str],
    beta: float = 0.04,  # KL divergence penalty coefficient (GRPO default)
    loss_kwargs: Optional[dict] = None,
):
    """
    The GRPO recipe, composed from individual, reusable steps.
    This recipe can be configured with different loss parameters.
    """
    # 1. Generation
    completions = generation_step(ml, policy=policy, prompts=prompts)

    # 2. Log-Probabilities
    policy_logprobs, ref_logprobs = logprobs_step(
        ml, policy=policy, ref_policy=ref_policy, completions=completions
    )

    # 3. Rewards
    rewards = rewards_step(ml)

    # 4. Advantages
    advantages = advantages_step(
        ml,
        rewards_tensor=rewards,
        prompts=prompts,
        completions=completions,
        policy=policy,
    )

    # 5. Loss - merge beta with any additional loss_kwargs
    final_loss_kwargs = {"beta": beta}
    if loss_kwargs:
        final_loss_kwargs.update(loss_kwargs)
    
    loss_step(
        ml,
        advantages=advantages,
        policy_logprobs=policy_logprobs,
        ref_logprobs=ref_logprobs,
        completions=completions,
        loss_kwargs=final_loss_kwargs,
    )

    return ml.get_graph() 