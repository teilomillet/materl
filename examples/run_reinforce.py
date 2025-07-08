"""
This script demonstrates the end-to-end functionality of the REINFORCE
algorithm, using the declarative API.
"""

import torch
from materl.agents import Agent
from materl.compiler import compile
from materl.config import GenerationConfig, REINFORCEConfig
from materl.recipes import reinforce

def main():
    """
    Sets up and runs a REINFORCE experiment to train a language model based
    on an outcome-based reward signal.
    """
    print("--- Starting Declarative REINFORCE Test ---")

    # 1. Instantiate the policy model using the Agent wrapper.
    # REINFORCE is an on-policy algorithm, so it only requires a single policy model.
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Agent(model_name, trainable=True, device=device)

    # 2. Define the initial inputs.
    # The prompts are designed to guide the model. The `outcome_reward` function
    # will give a positive reward if the model's completion includes "success".
    prompts = [
        "To solve this problem, the first step is success",
        "The plan is destined for success",
    ]

    # 3. Get the symbolic graph from the REINFORCE recipe.
    symbolic_graph = reinforce(
        policy=policy,
        prompts=prompts,
        max_completion_length=10,
        gamma=0.99,  # This is passed directly to the recipe
    ) # type: ignore

    print(f"\nSuccessfully created symbolic graph: {symbolic_graph.name}")
    print("Nodes in graph:")
    for node in symbolic_graph.nodes:
        print(f"  - {node.name} (op: {node.op}), depends on: {[dep.name for dep in node.dependencies]}")

    # 4. Compile the graph into an executable plan.
    print("\nCompiling graph...")
    compiled_graph = compile(symbolic_graph)

    # 5. Create Configuration objects for generation and the algorithm.
    gen_config = GenerationConfig(max_completion_length=10)
    reinforce_config = REINFORCEConfig(gamma=0.99, learning_rate=1e-5)

    # 6. Run the compiled graph.
    print("\nExecuting graph...")
    final_context = compiled_graph.run(
        policy=policy,
        prompts=prompts,
        generation_config=gen_config,
        reinforce_config=reinforce_config,
    )

    print("\n✅ REINFORCE experiment finished successfully!")
    print(f"Final context contains {len(final_context)} keys.")


if __name__ == "__main__":
    main() 