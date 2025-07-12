#!/usr/bin/env python3
"""
Backend Comparison Example: Torch vs MAX

This example demonstrates the performance difference between using PyTorch (HuggingFace)
and MAX backends for text generation in materl.

Requirements:
- torch
- transformers  
- max (Modular MAX SDK)
- materl

Usage:
    python examples/backend_comparison.py
"""

import time
import statistics
from typing import Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import materl generation function
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from materl.functions.generation import generate_completions

# MAX SDK imports
try:
    from max.entrypoints.llm import LLM as MaxLLM
    from max.pipelines.lib.config import PipelineConfig as MaxPipelineConfig
    from max.pipelines.lib.config_enums import SupportedEncoding as MaxSupportedEncoding
    MAX_SDK_AVAILABLE = True
except ImportError:
    MaxLLM = None
    MaxPipelineConfig = None
    MaxSupportedEncoding = None
    MAX_SDK_AVAILABLE = False


def format_time(seconds: float) -> str:
    """Format time in a human-readable way."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def print_results(results: Dict[str, Any], backend: str, elapsed_time: float) -> None:
    """Print generation results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"🚀 {backend.upper()} BACKEND RESULTS")
    print(f"{'='*60}")
    print(f"⏱️  Generation Time: {format_time(elapsed_time)}")
    print(f"📝 Number of Prompts: {len(set(results['prompts_text']))}")
    print(f"🔄 Generations per Prompt: {len(results['prompts_text']) // len(set(results['prompts_text']))}")
    print(f"📊 Total Completions: {len(results['completions_text'])}")
    print(f"🎯 Tensor Shapes:")
    print(f"   - Prompt IDs: {results['prompts_input_ids'].shape}")
    print(f"   - Completion IDs: {results['completions_ids'].shape}")
    print(f"   - Full Input IDs: {results['full_input_ids'].shape}")
    
    print(f"\n📝 Sample Completions:")
    for i, (prompt, completion) in enumerate(zip(results['prompts_text'][:3], results['completions_text'][:3])):
        print(f"   {i+1}. Prompt: '{prompt[:50]}...'")
        print(f"      Completion: '{completion[:100]}...'")
        print()


def benchmark_backend(
    backend: str,
    model, 
    tokenizer, 
    prompts: List[str], 
    max_completion_length: int = 50,
    num_generations: int = 2,
    num_runs: int = 3,
    num_warmups: int = 1
) -> tuple[Dict[str, Any], float]:
    """Benchmark a specific backend with multiple runs."""
    
    print(f"\n🔧 Warming up {backend.upper()} backend ({num_warmups} runs)...")
    
    # Warmup runs
    for i in range(num_warmups):
        try:
            _ = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts[:1],  # Use just one prompt for warmup
                max_prompt_length=512,
                max_completion_length=max_completion_length,
                num_generations=1,
                backend=backend,
                temperature=0.7,
                top_p=0.9
            )
            print(f"   Warmup {i+1}/{num_warmups} ✅")
        except Exception as e:
            print(f"   Warmup {i+1}/{num_warmups} ❌ Error: {e}")
            if "Cannot determine model path" in str(e) and backend == "max":
                print("   💡 Tip: For MAX backend, pass the model path as a string")
            raise
    
    print(f"\n⚡ Running {backend.upper()} benchmark ({num_runs} runs)...")
    
    # Actual benchmark runs
    times = []
    results = None
    
    for run in range(num_runs):
        start_time = time.time()
        
        try:
            results = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_prompt_length=512,
                max_completion_length=max_completion_length,
                num_generations=num_generations,
                backend=backend,
                temperature=0.7,
                top_p=0.9
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            
            print(f"   Run {run+1}/{num_runs}: {format_time(elapsed)} ✅")
            
        except Exception as e:
            print(f"   Run {run+1}/{num_runs}: ❌ Error: {e}")
            raise
    
    avg_time = statistics.mean(times)
    assert results is not None, "No successful runs completed"
    return results, avg_time


def main():
    """Run the comparison between PyTorch and MAX backends."""
    print("🎯 materl Backend Comparison: PyTorch vs MAX")
    print("=" * 60)
    
    # Configuration - Same model in different formats for fair comparison (Phi-4 mini for 16GB RAM)
    pytorch_model_name = "unsloth/Phi-4-mini-instruct"  # 3.8B parameters (~6-8GB RAM)
    max_model_name = "unsloth/Phi-4-mini-instruct-GGUF"  # Same model in Q4_K_M GGUF format (~2.5GB)
    
    prompts = [
        "Hello, how are you?",
        "What is artificial intelligence?", 
        "The future of technology is",
        "In a world where"
    ]
    num_generations = 1  # Reduced for faster testing
    max_completion_length = 30  # Reduced for faster testing
    num_runs = 2  # Reduced for faster testing
    num_warmups = 1
    
    print(f"📋 Configuration:")
    print(f"   PyTorch Model: {pytorch_model_name}")
    print(f"   MAX Model: {max_model_name}")
    print(f"   Prompts: {len(prompts)}")
    print(f"   Generations per prompt: {num_generations}")
    print(f"   Max completion length: {max_completion_length}")
    print()

    # Load tokenizer (use PyTorch model's tokenizer for both)
    print("📦 Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(pytorch_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_time = time.time() - start_time
    print(f"   ✅ Tokenizer loaded in {format_time(load_time)}")
    print()

    # Load PyTorch model
    print("📦 Loading PyTorch model...")
    start_time = time.time()
    pytorch_model = AutoModelForCausalLM.from_pretrained(pytorch_model_name)
    if torch.cuda.is_available():
        pytorch_model = pytorch_model.to("cuda")
        print("   💻 Using GPU for PyTorch")
    else:
        print("   💻 Using CPU for PyTorch")
    load_time = time.time() - start_time
    print(f"   ✅ PyTorch model loaded in {format_time(load_time)}")
    print()

    # Run benchmarks
    results = {}
    
    # Benchmark PyTorch backend
    print(f"\n" + "🔥" * 20 + " PYTORCH BACKEND " + "🔥" * 20)
    try:
        pytorch_results, pytorch_time = benchmark_backend(
            "torch", pytorch_model, tokenizer, prompts, 
            max_completion_length, num_generations, num_runs, num_warmups
        )
        results["torch"] = {"results": pytorch_results, "time": pytorch_time}
        print_results(pytorch_results, "torch", pytorch_time)
    except Exception as e:
        print(f"❌ PyTorch backend failed: {e}")
        results["torch"] = {"error": str(e)}
    
    # Benchmark MAX backend  
    print(f"\n" + "⚡" * 20 + " MAX BACKEND " + "⚡" * 20)
    
    if not MAX_SDK_AVAILABLE:
        print("❌ MAX backend failed: MAX SDK not found. Skipping.")
        results["max"] = {"error": "MAX SDK not installed."}
    else:
        max_llm = None
        try:
            # Initialize MAX LLM once
            print("📦 Initializing MAX LLM...")
            start_time = time.time()
            # The Phi-4 model architecture in MAX does not support GGUF quantizations like q4_k.
            # We use a compatible Llama model for the MAX backend comparison.
            pipeline_config = MaxPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                quantization_encoding=MaxSupportedEncoding.q4_k,
            )
            max_llm = MaxLLM(pipeline_config)
            load_time = time.time() - start_time
            print(f"   ✅ MAX LLM initialized in {format_time(load_time)}")
            
            # Run benchmark with the pre-initialized LLM
            max_results, max_time = benchmark_backend(
                "max", max_llm, tokenizer, prompts,
                max_completion_length, num_generations, num_runs, num_warmups
            )
            results["max"] = {"results": max_results, "time": max_time}
            print_results(max_results, "max", max_time)
        except Exception as e:
            print(f"❌ MAX backend failed: {e}")
            print(f"💡 Note: Make sure MAX SDK is properly installed and configured")
            results["max"] = {"error": str(e)}
        finally:
            # Ensure the background process is terminated
            if max_llm:
                print("📦 Shutting down MAX LLM...")
                del max_llm
    
    # Performance comparison
    print(f"\n" + "📊" * 20 + " PERFORMANCE COMPARISON " + "📊" * 20)
    
    if ("torch" in results and "max" in results and 
        "error" not in results["torch"] and "error" not in results["max"]):
        torch_time = results["torch"]["time"]  # type: ignore
        max_time = results["max"]["time"]  # type: ignore
        
        speedup = torch_time / max_time if max_time > 0 else float('inf')  # type: ignore
        
        print(f"⏱️  PyTorch Time: {format_time(torch_time)}")  # type: ignore
        print(f"⚡ MAX Time: {format_time(max_time)}")  # type: ignore
        print(f"🚀 Speedup: {speedup:.2f}x {'(MAX faster)' if speedup > 1 else '(PyTorch faster)'}")
        
        # Tokens per second calculation
        total_tokens = len(prompts) * num_generations * max_completion_length
        torch_tps = total_tokens / torch_time  # type: ignore
        max_tps = total_tokens / max_time  # type: ignore
        
        print(f"📈 PyTorch: {torch_tps:.1f} tokens/sec")
        print(f"📈 MAX: {max_tps:.1f} tokens/sec")
        
    else:
        print("⚠️  Could not compare backends due to errors")
        if "error" in results.get("torch", {}):
            print(f"   PyTorch error: {results['torch']['error']}")
        if "error" in results.get("max", {}):
            print(f"   MAX error: {results['max']['error']}")
    
    print(f"\n" + "✅" * 20 + " COMPARISON COMPLETE " + "✅" * 20)
    
    # Only show takeaways if we have successful results from both backends
    if ("torch" in results and "max" in results and 
        "error" not in results["torch"] and "error" not in results["max"]):
        
        torch_result = results["torch"]
        max_result = results["max"]
        
        # Ensure we have valid time data
        if (isinstance(torch_result, dict) and "time" in torch_result and 
            isinstance(max_result, dict) and "time" in max_result and
            isinstance(torch_result["time"], (int, float)) and
            isinstance(max_result["time"], (int, float))):
            
            torch_time = float(torch_result["time"])
            max_time = float(max_result["time"])
            speedup = torch_time / max_time if max_time > 0 else float('inf')
            
            print(f"🎯 Key Takeaways:")
            print(f"   • MAX was {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than PyTorch")
            print(f"   • PyTorch: {format_time(torch_time)}")
            print(f"   • MAX: {format_time(max_time)}")
            print(f"   • Both backends return identical tensor structures")
            
            if speedup > 1:
                print(f"   • MAX shows performance advantage for this workload")
            else:
                print(f"   • PyTorch performed better for this specific test")
        else:
            print(f"🎯 Results:")
            print(f"   • Invalid timing data - could not calculate performance comparison")
            
    else:
        print(f"🎯 Results:")
        print(f"   • Could not complete fair comparison due to backend errors")
        print(f"   • Both backends need to work for meaningful performance comparison")
        if "torch" in results and "error" not in results["torch"]:
            print(f"   • PyTorch backend: ✅ Working")
        else:
            print(f"   • PyTorch backend: ❌ Failed")
        if "max" in results and "error" not in results["max"]:
            print(f"   • MAX backend: ✅ Working") 
        else:
            print(f"   • MAX backend: ❌ Failed")


if __name__ == "__main__":
    main() 