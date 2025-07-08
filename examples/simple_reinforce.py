"""
This script demonstrates the end-to-end functionality of the base REINFORCE
algorithm using the simplified `materl` API.
"""
import torch
import materl
from materl.agents import Agent
from materl.config import GenerationConfig

def main():
    """
    Sets up and runs a REINFORCE experiment.
    """
    print("--- Starting Simple REINFORCE Test ---")

    # 1. Setup Agent and Prompts
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Agent(model_name, trainable=True, device=device)

    prompts = [
        "To solve this problem, the first step is success",
        "The plan is destined for success",
    ]

    # 2. Define Generation Config
    gen_config = GenerationConfig(
        max_completion_length=10, 
        max_prompt_length=20, 
        num_generations=1
    )

    # 3. Run the 'reinforce' algorithm from the registry
    print("\nExecuting graph...")
    result = materl.run(
        materl.get_algorithm("reinforce"),
        policy=policy,
        prompts=prompts,
        gamma=0.99,
        generation_config=gen_config,
    )

    print("\n✅ REINFORCE experiment finished successfully!")
    print(f"✅ Run completed. Final context keys: {list(result.keys())}")


if __name__ == "__main__":
    main() 