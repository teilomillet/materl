"""
This script demonstrates the end-to-end functionality of the new
declarative API by compiling and running the GRPO recipe.
"""

from materl.recipes.grpo import grpo
from materl.compiler import compile
from materl.config import GenerationConfig, GRPOConfig
from materl.agents import Agent

def main():
    print("--- Starting Declarative GRPO Test ---")

    # 1. Instantiate the stateful models ("Agents")
    # Using a small, fast model for this test.
    model_name = "gpt2"
    policy = Agent(model_name, trainable=True, device="cpu")
    # For this test, ref_policy can be the same. In a real run, it might be a separate model.
    ref_policy = Agent(model_name, device="cpu")

    # 2. Define the initial inputs for the graph
    prompts = ["Hello, my name is", "The capital of France is"]

    # 3. Get the symbolic graph from the recipe
    symbolic_graph = grpo( # type: ignore
        policy=policy, 
        ref_policy=ref_policy, 
        prompts=prompts,
        max_completion_length=10, # This must match the GenerationConfig
    )
    
    print(f"\nSuccessfully created symbolic graph: {symbolic_graph.name}")
    print("Nodes in graph:")
    for node in symbolic_graph.nodes:
        print(f"  - {node.name} (op: {node.op}), depends on: {[dep.name for dep in node.dependencies]}")

    # 4. Compile the graph
    print("\nCompiling graph...")
    compiled_graph = compile(symbolic_graph)

    # 5. Create Configuration Objects
    gen_config = GenerationConfig(
        max_prompt_length=10, 
        max_completion_length=10,
        num_generations=2
    )
    grpo_config = GRPOConfig()

    # 6. Run the compiled graph
    # The ExecutableCompiler will now use its intelligent argument passing
    # to orchestrate the entire workflow.
    print("\nExecuting graph...")
    final_context = compiled_graph.run(
        # Initial inputs
        policy=policy, 
        ref_policy=ref_policy, 
        prompts=prompts,
        # Configs
        generation_config=gen_config,
        grpo_config=grpo_config
    )
    
    print("\n--- Final Execution Context ---")
    for key, value in final_context.items():
        # Don't print large tensors or models
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


if __name__ == "__main__":
    main() 