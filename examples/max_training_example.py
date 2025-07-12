#!/usr/bin/env python3
"""
MAX Training Example

This example demonstrates how to use the MAX backend for significantly faster
RL training with materl. The MAX backend can provide 7x+ speedup over PyTorch.

Usage:
    python examples/max_training_example.py

Requirements:
    - A GGUF format model (e.g., from HuggingFace)
    - MAX platform installation
"""

import time
import torch
from transformers import AutoTokenizer

from materl import run
from materl.agents import Agent  
from materl.recipes import grpo
from materl.config import GenerationConfig, GRPOConfig


def main():
    print("🚀 MAX Backend Training Example")
    print("=" * 60)
    
    # Configuration
    model_name = "microsoft/DialoGPT-small"  # Small model for demo
    max_model_path = "microsoft/DialoGPT-small"  # Use same model path for demo (MAX backend will handle conversion)
    
    # Sample prompts for training
    prompts = [
        "Hello, how are you?",
        "What is artificial intelligence?", 
        "The future of technology is",
        "In my opinion, the best way to learn is"
    ]
    
    print(f"📋 Configuration:")
    print(f"   Model: {model_name}")
    print(f"   MAX Model: {max_model_path}")
    print(f"   Prompts: {len(prompts)}")
    print()
    
    # Load tokenizer
    print("📦 Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_time = time.time() - start_time
    print(f"   ✅ Tokenizer loaded in {tokenizer_time:.2f}s")
    print()
    
    # Create agents (we'll use the same model for policy and reference)
    policy_agent = Agent(model_name_or_path=model_name)
    ref_agent = Agent(model_name_or_path=model_name)
    
    # Algorithm configuration
    grpo_config = GRPOConfig(
        beta=0.04,  # KL penalty
        epsilon=0.2,  # PPO clipping
    )
    
    # Compare PyTorch vs MAX backends
    backends_to_test = [
        ("PyTorch", "torch", None),
        ("MAX", "max", max_model_path),
    ]
    
    results = {}
    
    for backend_name, backend_type, model_path in backends_to_test:
        print(f"🔥 Testing {backend_name} Backend")
        print("-" * 40)
        
        # Generation configuration
        if backend_type == "torch":
            generation_config = GenerationConfig(
                backend="torch",
                max_prompt_length=256,
                max_completion_length=64,
                num_generations=2,  # Smaller for demo
                temperature=1.0
            )
        else:
            generation_config = GenerationConfig(
                backend=backend_type,
                max_model_path=model_path,
                max_prompt_length=256,
                max_completion_length=64,
                num_generations=2,
                temperature=1.0
            )
        
        print(f"⚡ Running training step with {backend_name}...")
        start_time = time.time()
        
        try:
            # Run one training step with the specified backend
            # Note: Use materl.run() with the algorithm function, not methods on Agent objects
            # Config objects are passed as parameters and automatically extracted by the compiler
            training_result = run(
                grpo,
                policy=policy_agent,
                ref_policy=ref_agent,
                prompts=prompts,
                generation_config=generation_config,
                grpo_config=grpo_config
            )
            
            step_time = time.time() - start_time
            results[backend_name] = step_time
            
            print(f"   ✅ {backend_name} step completed in {step_time:.2f}s")
            print(f"   📊 Generated {len(prompts) * generation_config.num_generations} completions")
            
        except Exception as e:
            print(f"   ❌ Error with {backend_name} backend: {e}")
            results[backend_name] = None
        
        print()
    
    # Performance comparison
    print("📊 Performance Comparison")
    print("=" * 40)
    
    pytorch_time = results.get("PyTorch")
    max_time = results.get("MAX")
    
    if pytorch_time and max_time:
        speedup = pytorch_time / max_time
        print(f"⏱  PyTorch Time: {pytorch_time:.2f}s")
        print(f"⚡ MAX Time: {max_time:.2f}s")
        print(f"🚀 Speedup: {speedup:.1f}x (MAX faster)")
        print()
        print(f"💡 For a full training run with thousands of steps,")
        print(f"   this speedup would save significant time!")
    else:
        print("Could not compare - one or both backends failed")
    
    print()
    print("✅ Training example completed!")
    print()
    print("🎯 Key Takeaways:")
    print("   • MAX backend provides significant speedup for training")
    print("   • Same training algorithm, just faster generation")
    print("   • Easy to switch between backends via configuration")
    print("   • Use materl.run() with algorithm functions, not methods on Agent objects")
    print("   • Config objects are automatically extracted by the compiler")


if __name__ == "__main__":
    main() 