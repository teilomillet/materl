"""
This script demonstrates creating the VAPO algorithm by applying a
transformation to the base GRPO recipe.
"""
import materl
from materl import algorithm
from materl.config import GenerationConfig

# VAPO is a transformation of GRPO that adds a value model and uses GAE.
@algorithm(using=materl.grpo)
def vapo(graph, policy, ref_policy, value_model, prompts):
    """Standard VAPO: GRPO + value model + GAE + value loss."""
    graph.add_values_node(value_model)  # Default: enable_gae=True, enable_value_loss=True
    return graph

materl.register_algorithm("vapo", vapo)

# Example: Custom VAPO variant that only uses GAE but no value loss
@algorithm(using=materl.grpo)
def vapo_gae_only(graph, policy, ref_policy, value_model, prompts):
    """VAPO variant: GAE for advantages but no value loss."""
    graph.add_values_node(value_model, enable_gae=True, enable_value_loss=False)
    return graph

materl.register_algorithm("vapo_gae_only", vapo_gae_only)

# Example: Value loss only (no GAE)
@algorithm(using=materl.grpo)
def vapo_value_loss_only(graph, policy, ref_policy, value_model, prompts):
    """VAPO variant: value loss but no GAE."""
    graph.add_values_node(value_model, enable_gae=False, enable_value_loss=True)
    return graph

materl.register_algorithm("vapo_value_loss_only", vapo_value_loss_only)

def main():
    """Sets up and runs a VAPO experiment."""
    print("--- Starting VAPO Test (Transformation-based) ---")

    # VAPO requires a policy, a reference model, and a value model
    policy = materl.Agent("gpt2", trainable=True)
    ref_policy = materl.Agent("gpt2", trainable=False)
    value_model = materl.Agent("gpt2", trainable=True)
    prompts = ["The VAPO algorithm is a"]

    gen_config = GenerationConfig(
        max_completion_length=20, max_prompt_length=20, num_generations=1
    )

    result = materl.run(
        vapo,
        policy=policy,
        ref_policy=ref_policy,
        value_model=value_model,
        prompts=prompts,
        generation_config=gen_config,
    )

    print("\n✅ VAPO experiment finished successfully!")
    print(f"✅ Run completed. Final context keys: {list(result.keys())}")


if __name__ == "__main__":
    main() 