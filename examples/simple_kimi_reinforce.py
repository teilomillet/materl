"""
Simple example showing how easy it is to create a new algorithm with materl.

This is what the user experience should be like.
"""

import materl
from materl import algorithm
from materl.config import GenerationConfig

# Create a new algorithm in 5 lines of code!
@algorithm(using=materl.reinforce)
def kimi_reinforce(graph, policy, prompts, gamma):
    """Kimi-style REINFORCE with negative sample discarding."""
    return graph.replace_loss("kimi_reinforce_loss", discard_negative_samples=True)

# Register it so others can use it
materl.register_algorithm("kimi_reinforce", kimi_reinforce)

def main():
    """Run the new algorithm."""
    print("Available algorithms:", materl.list_algorithms())
    
    # Create agent and data
    policy = materl.Agent("gpt2", trainable=True)
    prompts = ["To solve this problem", "The plan is destined for failure"]
    
    # Configuration for the generation process
    gen_config = GenerationConfig(
        max_completion_length=10, 
        max_prompt_length=20, 
        num_generations=1
    )
    
    # Get the algorithm by name and run it
    algo = materl.get_algorithm("kimi_reinforce")
    result = materl.run(
        algo,
        policy=policy, 
        prompts=prompts, 
        gamma=0.99,
        generation_config=gen_config
    )
    
    print("✅ Success! Created and ran a new algorithm in ~10 lines of code")
    print(f"✅ Run completed. Final context keys: {list(result.keys())}")

if __name__ == "__main__":
    main() 