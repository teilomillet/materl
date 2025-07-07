"""
materl.recipes.vapo

The recipe for the VAPO algorithm, a value-based PPO method.
"""
from typing import TYPE_CHECKING
from . import recipe

if TYPE_CHECKING:
    from ..agents import Agent
    from . import GraphBuilder

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
    The VAPO recipe defines the symbolic steps for a value-based algorithm
    that uses Generalized Advantage Estimation (GAE).
    """
    # 1. Generation
    completions = ml.generate(policy=policy, prompts=prompts)

    # 2. Log-Probabilities
    policy_logprobs = ml.logprobs(model=policy, completions=completions)
    ref_logprobs = ml.logprobs(model=ref_policy, completions=completions, is_ref=True)
    
    # 3. Rewards
    length = ml.reward(name="length_reward", weight=0.1)

    # 4. Values
    # The value model estimates the expected return from each state.
    values = ml.values(model=value_model, completions=completions)

    # 5. Advantages (GAE)
    # The advantages node will detect the presence of `values` and automatically
    # use GAE for the calculation.
    advantages = ml.advantages(
        reward_configs=[length],
        values=values, # Pass the values node here
        prompts_text=prompts,
        completions_text=completions,
        completions_ids=completions,
        completion_masks=completions,
        full_input_ids=completions,
        tokenizer=policy,
        device=policy,
        max_completion_length=max_completion_length,
    )
    
    # 6. Loss
    # The loss function will detect the presence of `values` and `returns`
    # (from GAE) and add the value loss component automatically.
    ml.set_loss(
        advantages=advantages,
        current_policy_logprobs=policy_logprobs,
        old_policy_logprobs=policy_logprobs,
        ref_logprobs=ref_logprobs,
        values=values,
        completion_masks=completions,
    )

    return ml.get_graph() 