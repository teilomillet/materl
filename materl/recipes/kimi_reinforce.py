"""
materl.recipes.kimi_reinforce

The recipe for the Kimi-style REINFORCE algorithm, which is a composition
of the base REINFORCE recipe with a custom loss function.
"""
from typing import TYPE_CHECKING
from .reinforce import reinforce

if TYPE_CHECKING:
    from ..agents import Agent


def kimi_reinforce(
    policy: "Agent",
    prompts: list[str],
    max_completion_length: int,
    gamma: float,
):
    """
    The Kimi-style REINFORCE recipe.

    This is the standard REINFORCE algorithm but with a different loss function
    that strategically discards negative samples. This demonstrates how to
    create a new algorithm by reconfiguring the components of an existing one.
    """
    kimi_loss_kwargs = {
        "discard_negative_samples": True,
    }

    return reinforce(
        policy=policy,
        prompts=prompts,
        max_completion_length=max_completion_length,
        gamma=gamma,
        loss_fn="kimi_reinforce_loss",
        loss_kwargs=kimi_loss_kwargs,
    ) 