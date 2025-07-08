"""
materl.recipes.dapo

The recipe for the DAPO algorithm, implemented as a composition of the GRPO recipe.
"""
from typing import TYPE_CHECKING
from .grpo import grpo

if TYPE_CHECKING:
    from ..agents import Agent


def dapo(
    policy: "Agent",
    ref_policy: "Agent",
    prompts: list[str],
    max_completion_length: int,
):
    """
    The DAPO recipe, which is GRPO with a different loss configuration.
    This demonstrates how to create a new recipe by composing and configuring
    an existing one.
    """
    dapo_loss_kwargs = {
        "average_logprobs_by_token": True,
        "clip_ratio_lower": 0.2,
        "clip_ratio_upper": 0.28,
    }

    return grpo(
        policy=policy,
        ref_policy=ref_policy,
        prompts=prompts,
        max_completion_length=max_completion_length,
        loss_kwargs=dapo_loss_kwargs,
    ) 