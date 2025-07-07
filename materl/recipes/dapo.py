"""
materl.recipes.dapo

The recipe for the DAPO algorithm, based on the GRPO recipe.
"""
from typing import TYPE_CHECKING
from . import recipe

# This block is only for static analysis. It's ignored at runtime.
if TYPE_CHECKING:
    from ..agents import Agent
    from . import GraphBuilder

@recipe
def dapo(
    ml: "GraphBuilder",
    policy: "Agent",
    ref_policy: "Agent",
    prompts: list[str],
    max_completion_length: int,
):
    """
    The DAPO recipe defines the symbolic steps for the algorithm, which is an
    enhancement of GRPO with a focus on token-level loss and decoupled clipping.
    """
    # 1. Generation
    completions = ml.generate(policy=policy, prompts=prompts)

    # 2. Log-Probabilities
    policy_logprobs = ml.logprobs(model=policy, completions=completions)
    ref_logprobs = ml.logprobs(model=ref_policy, completions=completions, is_ref=True)
    
    # 3. Rewards
    length = ml.reward(name="length_reward", weight=0.1)

    # 4. Advantages
    advantages = ml.advantages(
        reward_configs=[length],
        prompts_text=prompts,
        completions_text=completions,
        completions_ids=completions,
        completion_masks=completions,
        tokenizer=policy,
        device=policy,
        max_completion_length=max_completion_length,
    )
    
    # 5. Loss (DAPO specific)
    # DAPO modifies the loss calculation by averaging at the token-level
    # across the entire batch, rather than per-sample.
    ml.set_loss(
        advantages=advantages,
        current_policy_logprobs=policy_logprobs,
        old_policy_logprobs=policy_logprobs,
        ref_logprobs=ref_logprobs,
        completion_masks=completions,
        # DAPO-specific parameters for the loss function
        average_logprobs_by_token=True,
        clip_ratio_lower=0.2, # Corresponds to epsilon
        clip_ratio_upper=0.28, # Decoupled clipping (Clip-Higher)
    )

    return ml.get_graph() 