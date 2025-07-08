"""
This script demonstrates running the VAPO algorithm by calling the explicit
`vapo` recipe, which clearly shows its value-based composition.
"""

import torch
from materl.agents import Agent
from materl.compiler import compile
from materl.config import GenerationConfig
from materl.recipes import vapo

def main():
    """
    Sets up and runs a VAPO experiment using the explicit VAPO recipe.
    """
    print("--- Starting VAPO Test (Explicit Recipe) ---")

    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # In VAPO, we have three models: a policy, a reference policy, and a
    # value model for Generalized Advantage Estimation (GAE).
    policy = Agent(model_name, trainable=True, device=device)
    ref_policy = Agent(model_name, trainable=False, device=device)
    value_model = Agent(model_name, trainable=True, device=device)

    prompts = ["The VAPO algorithm is a"]

    # We now directly call the `vapo` recipe. This is more transparent, as
    # the recipe itself contains the specific composition of building blocks
    # (including the `values` node) that defines the VAPO algorithm.
    symbolic_graph = vapo(
        policy=policy,
        ref_policy=ref_policy,
        value_model=value_model,
        prompts=prompts,
        max_completion_length=20,
    ) # type: ignore

    print(f"\nSuccessfully created symbolic graph: {symbolic_graph.name}")
    print("Graph constructed using the explicit VAPO recipe.")

    compiled_graph = compile(symbolic_graph)

    gen_config = GenerationConfig(max_completion_length=20)

    print("\nExecuting graph...")
    final_context = compiled_graph.run(
        policy=policy,
        ref_policy=ref_policy,
        value_model=value_model,
        prompts=prompts,
        generation_config=gen_config,
    )

    print("\n✅ VAPO experiment finished successfully!")
    print(f"Final context contains {len(final_context)} keys.")


if __name__ == "__main__":
    main() 