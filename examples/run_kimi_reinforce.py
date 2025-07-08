"""
This script demonstrates the Kimi-style REINFORCE algorithm.

This version showcases the composable, recipe-based approach to building
algorithms in materl, where Kimi-style REINFORCE is a direct, configured
instantiation of the more general REINFORCE recipe.
"""

import torch
from materl.agents import Agent
from materl.compiler import compile
from materl.config import GenerationConfig
from materl.recipes.kimi_reinforce import kimi_reinforce

def main():
    """
    Sets up and runs a Kimi-style REINFORCE experiment.
    """
    print("--- Starting Kimi-style REINFORCE Test (Recipe-based) ---")

    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Agent(model_name, trainable=True, device=device)

    # We use prompts where some should succeed and some should fail to test
    # the negative sample discarding.
    prompts = [
        "To solve this problem, the first step is success", # Should get reward +1.0
        "The plan is destined for failure", # Should get reward -1.0
    ]

    # This is the core of the new, composable design.
    # We directly call the `kimi_reinforce` recipe, which is a pre-configured
    # version of the standard `reinforce` recipe. This is clean, explicit,
    # and requires no decorators or complex setup.
    symbolic_graph = kimi_reinforce(
        policy=policy,
        prompts=prompts,
        max_completion_length=10,
        gamma=0.99,
    )

    print(f"\nSuccessfully created symbolic graph: {symbolic_graph.name}")

    compiled_graph = compile(symbolic_graph)

    # The configuration for generation is kept separate from the algorithm
    # definition, as it controls the execution environment.
    gen_config = GenerationConfig(max_completion_length=10)

    print("\nExecuting graph...")
    final_context = compiled_graph.run(
        policy=policy,
        prompts=prompts,
        generation_config=gen_config,
    )

    print("\n✅ Kimi-style REINFORCE experiment finished successfully!")
    print(f"Final context contains {len(final_context)} keys.")


if __name__ == "__main__":
    main() 