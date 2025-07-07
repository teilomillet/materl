"""
materl.recipes.grpo

The canonical recipe for the GRPO algorithm.
"""
from typing import TYPE_CHECKING
from . import recipe

# This block is only for static analysis. It's ignored at runtime.
if TYPE_CHECKING:
    from ..agents import Agent
    from . import GraphBuilder

# The `ml` object is now passed as the first argument by the @recipe
# decorator, making the dependency explicit and analyzable.
@recipe
def grpo(
    ml: "GraphBuilder", 
    policy: "Agent", 
    ref_policy: "Agent", 
    prompts: list[str],
    max_completion_length: int
):
    """
    The GRPO recipe defines the symbolic steps for the algorithm.
    """
    # 1. Generation
    completions = ml.generate(policy=policy, prompts=prompts)

    # 2. Log-Probabilities
    policy_logprobs = ml.logprobs(model=policy, completions=completions)
    ref_logprobs = ml.logprobs(model=ref_policy, completions=completions, is_ref=True)
    
    # 3. Rewards
    # The names here must match the keys in the REWARD_FUNCTION_REGISTRY
    length = ml.reward(name="length_reward", weight=0.1)
    # mojo_diversity = ml.reward(name="mojo_diversity_reward", weight=0.2) # We'll add this back later

    # 4. Advantages
    # The `advantages` op now takes the reward functions directly, as well as all
    # the necessary context to compute them internally. This encapsulates the
    # logic and removes the hidden dependency from the compiler.
    advantages = ml.advantages(
        reward_configs=[length], 
        prompts_text=prompts,
        completions_text=completions, # This will be resolved by the compiler from the generate node
        completions_ids=completions,
        completion_masks=completions,
        tokenizer=policy, # The compiler will extract the tokenizer from the agent
        device=policy, # The compiler will extract the device
        max_completion_length=max_completion_length,
    )
    
    # 5. Loss
    # The loss function now takes its arguments directly, removing the need
    # for a special 'data' dictionary in the compiler.
    ml.set_loss(
        advantages=advantages, 
        current_policy_logprobs=policy_logprobs, 
        old_policy_logprobs=policy_logprobs, # In a real PPO loop, this would be from the previous step
        ref_logprobs=ref_logprobs,
        completion_masks=completions,
    )

    # By explicitly returning the graph, we make the recipe's output clear
    # to static analysis tools. We accept the resulting linter errors as a
    # known limitation of static analysis on this type of dynamic code.
    return ml.get_graph()
# Note: Adding a comment to force linter re-evaluation. 