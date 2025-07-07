from materl.agents import Agent
from materl.compiler import compile
from materl.config import GenerationConfig, GRPOConfig
from materl.recipes import grpo
import torch

def main():
    """
    Main function to set up and run a test of the declarative API with GRPO.
    """
    print("--- Starting Declarative GRPO Test ---")

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

    algorithm_config = GRPOConfig(
        beta=0.1,
    )

    # 4. Get the symbolic graph from the GRPO recipe
    print("Creating symbolic graph for GRPO...")
    symbolic_graph = grpo(  # type: ignore
        policy=policy_agent,
        ref_policy=ref_agent,
        prompts=prompts,
        max_completion_length=gen_config.max_completion_length,
    )

    # 5. Compile and run the graph
    print("Compiling and executing graph...")
    compiled_graph = compile(symbolic_graph)
    
    final_context = compiled_graph.run(
        policy=policy_agent,
        ref_policy=ref_agent,
        prompts=prompts,
        generation_config=gen_config,
        grpo_config=algorithm_config,
    )

    print("\nGRPO training test finished successfully!")
    print("\n--- Final Execution Context Summary ---")
    for key, value in final_context.items():
        if hasattr(value, "shape"):
            print(f"  {key}: Tensor with shape {value.shape}")
        else:
            if isinstance(value, list) and len(value) > 5:
                print(f"  {key}: list of {len(value)} items")
            elif not isinstance(value, (dict, list)):
                 print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 