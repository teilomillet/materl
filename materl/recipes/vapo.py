"""
materl.recipes.vapo

The recipe for the VAPO algorithm, a value-based PPO method.
"""
from typing import TYPE_CHECKING
from .decorator import recipe

if TYPE_CHECKING:
    from ..agents import Agent
    from .builder import GraphBuilder

@recipe
def vapo(
    ml: "GraphBuilder",
    policy: "Agent",
    ref_policy: "Agent",
    value_model: "Agent",
    prompts: list[str],
    max_completion_length: int,
):
    """
    The VAPO recipe explicitly composes the building blocks for a value-based
    algorithm that uses Generalized Advantage Estimation (GAE).

    It demonstrates how to introduce a value model, compute values, and then
    pipe those values into the advantage and loss calculations.
    """
    # 1. Generation
    completions = ml.generate(policy=policy, prompts=prompts)

    # 2. Log-Probabilities
    policy_logprobs = ml.logprobs(model=policy, completions=completions)
    ref_logprobs = ml.logprobs(model=ref_policy, completions=completions, is_ref=True)
    
    # 3. Rewards
    rewards = ml.reward(name="length_reward", weight=0.1)

    # 4. Values
    # This is the first key step of VAPO: explicitly creating a node to
    # compute state values from the dedicated value model.
    values = ml.values(model=value_model, completions=completions)

    # 5. Advantages (using GAE)
    # The advantages node is now passed the `values` tensor. The underlying
    # advantage function will detect this and automatically use GAE.
    advantages = ml.advantages(
        rewards_per_token=rewards, # GAE uses per-token rewards
        values=values,             # Pass the values node to enable GAE
        prompts_text=prompts,
        completions_text=completions,
        completions_ids=completions,
        completion_masks=completions,
        tokenizer=policy,
        device=policy,
        max_completion_length=max_completion_length,
    )
    
    # 6. Loss
    # The loss function is also passed the `values`. The underlying loss
    # function will detect this and add the required value loss component.
    ml.set_loss(
        advantages=advantages,
        current_policy_logprobs=policy_logprobs,
        old_policy_logprobs=policy_logprobs,
        ref_logprobs=ref_logprobs,
        values=values, # Pass values to enable the value loss term
        completion_masks=completions,
    )

    return ml.get_graph() 