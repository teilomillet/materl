"""
A simple example of running the base GRPO algorithm.
"""
import torch
import materl
from materl.agents import Agent
from materl.config import GenerationConfig

def main():
    """
    Sets up and runs a GRPO experiment.
    """
    print("--- Starting Simple GRPO Test ---")

    # 1. Configuration
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Instantiate Agents
    policy_agent = Agent(model_name, trainable=True, device=device)
    ref_agent = Agent(model_name, trainable=False, device=device)

    # 3. Define initial inputs and configurations
    prompts = [
        "Hello, my name is",
        "The quick brown fox jumps over the",
        "What is the capital of France?",
    ]

    gen_config = GenerationConfig(
        num_generations=2,
        temperature=0.7,
        top_p=1.0,
        top_k=50,
        max_prompt_length=128,
        max_completion_length=50,
    )

    # 4. Run the 'grpo' algorithm from the registry
    print("Executing graph...")
    result = materl.run(
        materl.get_algorithm("grpo"),
        policy=policy_agent,
        ref_policy=ref_agent,
        prompts=prompts,
        beta=0.1,  # Pass beta directly
        generation_config=gen_config,
    )

    print("\nGRPO training test finished successfully!")
    print(f"✅ Run completed. Final context keys: {list(result.keys())}")


if __name__ == "__main__":
    main() 