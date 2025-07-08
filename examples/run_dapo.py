"""
This script demonstrates running the DAPO algorithm by calling the explicit
`dapo` recipe, which clearly shows its composition.
"""

import torch
from materl.agents import Agent
from materl.compiler import compile
from materl.config import GenerationConfig
from materl.recipes import dapo

def main():
    """
    Sets up and runs a DAPO experiment.
    """
    print("--- Starting DAPO Test (Explicit Recipe) ---")

    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    policy = Agent(model_name, trainable=True, device=device)
    ref_policy = Agent(model_name, trainable=False, device=device)

    prompts = ["The DAPO algorithm is a"]

    # We now directly call the `dapo` recipe. This is more transparent, as
    # the recipe itself contains the specific composition of building blocks
    # that defines the DAPO algorithm.
    symbolic_graph = dapo(
        policy=policy,
        ref_policy=ref_policy,
        prompts=prompts,
        max_completion_length=20,
    ) # type: ignore

    print(f"\nSuccessfully created symbolic graph: {symbolic_graph.name}")
    print("Graph constructed using the explicit DAPO recipe.")

    compiled_graph = compile(symbolic_graph)

    gen_config = GenerationConfig(max_completion_length=20)

    print("\nExecuting graph...")
    final_context = compiled_graph.run(
        policy=policy,
        ref_policy=ref_policy,
        prompts=prompts,
        generation_config=gen_config,
    )

    print("\n✅ DAPO experiment finished successfully!")
    print(f"Final context contains {len(final_context)} keys.")


if __name__ == "__main__":
    main() 