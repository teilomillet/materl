"""
materl.recipes.steps

This module provides the building blocks for creating reinforcement learning recipes.
Each function in this module corresponds to a logical step in a typical RL
algorithm, such as generation, calculating log-probabilities, computing rewards,
and defining the loss function.

By composing these steps, you can create new recipes with minimal code duplication.
"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..agents import Agent
    from .builder import GraphBuilder
    from ..graph import SymbolicNode


def generation_step(
    ml: "GraphBuilder", 
    policy: "Agent", 
    prompts: list[str]
) -> "SymbolicNode":
    """Adds a generation node to the computation graph."""
    return ml.generate(policy=policy, prompts=prompts)


def logprobs_step(
    ml: "GraphBuilder",
    policy: "Agent",
    ref_policy: "Agent",
    completions: "SymbolicNode",
) -> tuple["SymbolicNode", "SymbolicNode"]:
    """Adds log-probability nodes for the policy and reference models."""
    policy_logprobs = ml.logprobs(model=policy, completions=completions)
    ref_logprobs = ml.logprobs(model=ref_policy, completions=completions, is_ref=True)
    return policy_logprobs, ref_logprobs


def rewards_step(
    ml: "GraphBuilder", 
    reward_name: str = "length_reward", 
    reward_weight: float = 0.1
) -> "SymbolicNode":
    """Adds a reward computation node to the graph."""
    return ml.reward(name=reward_name, weight=reward_weight)


def advantages_step(
    ml: "GraphBuilder",
    rewards_tensor: "SymbolicNode",
    prompts: list[str],
    completions: "SymbolicNode",
    policy: "Agent",
) -> "SymbolicNode":
    """Adds an advantage calculation node to the graph."""
    return ml.advantages(
        rewards_tensor=rewards_tensor,
        prompts_text=prompts,
        completions_text=completions,
        completions_ids=completions,
        completion_masks=completions,
        tokenizer=policy,
        device=policy,
    )


def loss_step(
    ml: "GraphBuilder",
    advantages: "SymbolicNode",
    policy_logprobs: "SymbolicNode",
    ref_logprobs: "SymbolicNode",
    completions: "SymbolicNode",
    loss_kwargs: Optional[dict] = None,
) -> None:
    """Adds the loss node to the graph, with optional overrides."""
    kwargs = {
        "advantages": advantages,
        "current_policy_logprobs": policy_logprobs,
        "old_policy_logprobs": policy_logprobs,  # Note: old and current are the same here
        "ref_logprobs": ref_logprobs,
        "completion_masks": completions,
    }
    if loss_kwargs:
        kwargs.update(loss_kwargs)
    
    ml.set_loss(**kwargs)


def returns_step(
    ml: "GraphBuilder",
    rewards: "SymbolicNode",
    completions: "SymbolicNode",
    gamma: float,
) -> "SymbolicNode":
    """Adds a discounted returns calculation node to the graph."""
    return ml.returns(
        rewards_per_token=rewards,
        completion_masks=completions,
        gamma=gamma,
    )


def reinforce_loss_step(
    ml: "GraphBuilder",
    policy_logprobs: "SymbolicNode",
    returns: "SymbolicNode",
    completions: "SymbolicNode",
    loss_fn: str = "reinforce_loss",
    loss_kwargs: Optional[dict] = None,
) -> None:
    """Adds the REINFORCE loss node to the graph."""
    kwargs = {
        "loss_fn": loss_fn,
        "current_policy_logprobs": policy_logprobs,
        "returns": returns,
        "completion_masks": completions,
    }
    if loss_kwargs:
        kwargs.update(loss_kwargs)

    ml.set_loss(**kwargs) 