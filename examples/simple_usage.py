#!/usr/bin/env python3
"""
Simple Usage Example: Torch vs MAX Backends

This example shows the basic usage differences between PyTorch and MAX backends
in materl for text generation.

Requirements:
- torch
- transformers
- max (Modular MAX SDK)
- materl
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from max.entrypoints.llm import LLM
from max.pipelines.lib.config import PipelineConfig

# Import materl generation function
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from materl.functions.generation import generate_completions


def torch_backend_example():
    """Example using PyTorch backend."""
    print("🔥 PyTorch Backend Example")
    print("-" * 40)
    
    # Load model and tokenizer - use Phi-4 mini for 16GB RAM
    model_name = "microsoft/Phi-3.5-mini-instruct"  # 3.8B parameters
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Sample prompts
    prompts = [
        "The key to success is",
        "Artificial intelligence will"
    ]
    
    # Generate completions using PyTorch backend
    results = generate_completions(
        model=model,  # Pass the actual PyTorch model object
        tokenizer=tokenizer,
        prompts=prompts,
        max_prompt_length=128,
        max_completion_length=30,
        num_generations=2,
        backend="torch",  # Specify torch backend
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )
    
    # Print results
    print(f"Generated {len(results['completions_text'])} completions:")
    for i, (prompt, completion) in enumerate(zip(results['prompts_text'], results['completions_text'])):
        print(f"{i+1}. '{prompt}' → '{completion.strip()}'")
    
    print("\nTensor shapes:")
    print(f"- Full input IDs: {results['full_input_ids'].shape}")
    print(f"- Completion IDs: {results['completions_ids'].shape}")
    
    return results


def max_backend_example():
    """Example using MAX backend."""
    print("⚡ MAX Backend Example")
    print("-" * 40)
    
    # For MAX backend, we need to create an LLM instance
    model_path = "microsoft/Phi-3.5-mini-instruct"
    
    # Create MAX LLM instance
    config = PipelineConfig()
    llm = LLM(model_path, config=config)
    
    # Load tokenizer separately (MAX LLM uses its own tokenizer internally)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Sample prompts (same as PyTorch example)
    prompts = [
        "The key to success is",
        "Artificial intelligence will"
    ]
    
    # Generate completions using MAX backend
    results = generate_completions(
        model=llm,  # Pass the LLM instance (not a string path!)
        tokenizer=tokenizer,
        prompts=prompts,
        max_prompt_length=128,
        max_completion_length=30,
        num_generations=2,
        backend="max",  # Specify MAX backend
        # Note: MAX has different parameter support, so some torch params may be ignored
    )
    
    # Print results
    print(f"Generated {len(results['completions_text'])} completions:")
    for i, (prompt, completion) in enumerate(zip(results['prompts_text'], results['completions_text'])):
        print(f"{i+1}. '{prompt}' → '{completion.strip()}'")
    
    print("\nTensor shapes:")
    print(f"- Full input IDs: {results['full_input_ids'].shape}")
    print(f"- Completion IDs: {results['completions_ids'].shape}")
    
    return results


def main():
    """Main function demonstrating both backends."""
    print("🎯 materl Backend Usage Examples")
    print("=" * 50)
    
    # Example 1: PyTorch Backend
    try:
        torch_results = torch_backend_example()
        print("✅ PyTorch backend completed successfully!")
    except Exception as e:
        print(f"❌ PyTorch backend failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: MAX Backend  
    try:
        max_results = max_backend_example()
        print("✅ MAX backend completed successfully!")
    except Exception as e:
        print(f"❌ MAX backend failed: {e}")
        print("💡 Make sure MAX SDK is installed and configured properly")
    
    print("\n📋 Key Differences:")
    print("PyTorch Backend:")
    print("  • model: Pass PreTrainedModel object")
    print("  • Supports all HuggingFace generation parameters")
    print("  • Standard PyTorch/Transformers performance")
    
    print("\nMAX Backend:")
    print("  • model: Pass pre-initialized LLM instance")
    print("  • Optimized for speed and memory efficiency")
    print("  • May have different parameter support")
    print("  • Typically 2-5x faster for inference")
    print("  • Handles tokenization internally but we provide external tokenizer for compatibility")
    
    print("\n🔗 Both backends return identical tensor structures!")
    print("   This ensures compatibility with materl's RL training pipeline.")


if __name__ == "__main__":
    main() 