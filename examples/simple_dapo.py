"""
Creating DAPO from GRPO - the way it should be.

This shows how simple it is to create algorithm variants.
"""

import materl
from materl import algorithm
from materl.config import GenerationConfig

# DAPO is just GRPO with different loss parameters
@algorithm(using=materl.grpo)
def dapo(graph, policy, ref_policy, prompts):
    """DAPO algorithm - GRPO with token-averaged loss and asymmetric clipping."""
    return graph.replace_loss(
        "default",  # Use the default policy gradient loss function
        average_logprobs_by_token=True,
        clip_ratio_lower=0.2,
        clip_ratio_upper=0.28,
    )

# Register it
materl.register_algorithm("dapo", dapo)

def main():
    """Demonstrate DAPO creation and usage."""
    
    # Create agents
    policy = materl.Agent("gpt2", trainable=True)
    ref_policy = materl.Agent("gpt2", trainable=False)
    prompts = ["Write a function to", "Implement an algorithm for"]
    
    gen_config = GenerationConfig(
        max_completion_length=20, max_prompt_length=20, num_generations=1
    )
    
    print("Running DAPO...")
    result = materl.run(
        dapo,
        policy=policy,
        ref_policy=ref_policy,
        prompts=prompts,
        generation_config=gen_config,
    )
    
    print("✅ DAPO ran successfully!")
    print(f"✅ Run completed. Final context keys: {list(result.keys())}")

if __name__ == "__main__":
    main() 