"""
materl.recipes.reinforce

The recipe for the REINFORCE algorithm, composed from modular steps.
"""
from typing import TYPE_CHECKING, Optional
from .decorator import recipe
from .steps import (
    generation_step,
    rewards_step,
    returns_step,
    reinforce_loss_step,
)

if TYPE_CHECKING:
    from ..agents import Agent
    from .builder import GraphBuilder


@recipe
def reinforce(
    ml: "GraphBuilder",
    policy: "Agent",
    prompts: list[str],
    gamma: float,
    reward_fn: str = "outcome_reward",
    loss_fn: str = "reinforce_loss",
    loss_kwargs: Optional[dict] = None,
):
    """
    The REINFORCE recipe, composed from individual, reusable steps.
    This recipe can be configured with different reward and loss functions.
    """
    # 1. Generation
    completions = generation_step(ml, policy=policy, prompts=prompts)

    # 2. Log-Probabilities
    # For REINFORCE, we only need the policy's logprobs.
    policy_logprobs = ml.logprobs(model=policy, completions=completions)

    # 3. Rewards
    rewards = rewards_step(ml, reward_name=reward_fn, reward_weight=1.0)

    # 4. Discounted Returns
    returns = returns_step(
        ml,
        rewards=rewards,
        completions=completions,
        gamma=gamma,
    )

    # 5. Loss
    reinforce_loss_step(
        ml,
        policy_logprobs=policy_logprobs,
        returns=returns,
        completions=completions,
        loss_fn=loss_fn,
        loss_kwargs=loss_kwargs,
    )

    return ml.get_graph() 