import torch
from materl.agents import Agent
from materl.compiler import compile
from materl.config import GRPOConfig, GenerationConfig
from materl.recipes.grpo import grpo


def main():
    """
    This script demonstrates the declarative API by running a test of the GRPO
    recipe, which has been updated to use the new Agent and compiler abstractions.
    """
    print("--- Starting Declarative GRPO Scaffold Test ---")

    # 1. Configuration
    model_name = "gpt2"  # A small model for quick testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Instantiate Agents for policy and reference models
    # The Agent class now handles tokenizer setup, including left-padding.
    policy_agent = Agent(model_name, trainable=True, device=device)
    ref_agent = Agent(model_name, trainable=False, device=device)

    # 3. Define initial inputs
    prompts = [
        "Hello, what is your name?",
        "The quick brown fox jumps over the lazy",
    ]
    
    gen_config = GenerationConfig(
        max_prompt_length=32,
        max_completion_length=20,
        num_generations=2,
    )
    
    grpo_config = GRPOConfig(
        beta=0.1,
        epsilon=0.2,
    )

    # 4. Get the symbolic graph from the GRPO recipe
    # The recipe encapsulates the sequence of operations (generate, reward, loss).
    symbolic_graph = grpo(  # type: ignore
        policy=policy_agent,
        ref_policy=ref_agent,
        prompts=prompts,
        max_completion_length=gen_config.max_completion_length,
    )

    # 5. Compile the graph into an executable plan
    compiled_graph = compile(symbolic_graph)
    
    # 6. Run the compiled graph
    # The compiler orchestrates the execution, passing context and configs
    # to the underlying functions.
    print("\nExecuting graph...")
    try:
        final_context = compiled_graph.run(
            policy=policy_agent,
            ref_policy=ref_agent,
            prompts=prompts,
            generation_config=gen_config,
            grpo_config=grpo_config,
        )
        print("\nGraph execution finished successfully!")
        
        # Print a summary of the final context
        print("\n--- Final Execution Context Summary ---")
        for key, value in final_context.items():
            if hasattr(value, "shape"):
                print(f"  {key}: Tensor with shape {value.shape}")
            elif isinstance(value, (Agent, GenerationConfig, GRPOConfig)):
                 print(f"  {key}: {value}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
            elif isinstance(value, list) and len(value) > 3:
                print(f"  {key}: list of {len(value)} items")
            else:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"\nAn error occurred during graph execution: {e}")
        import traceback
        traceback.print_exc()

    print("\nScaffold test script finished.")


if __name__ == "__main__":
    main() 