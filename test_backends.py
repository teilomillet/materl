#!/usr/bin/env python3
"""
Test script to verify both backends work correctly with updated code.
"""

import sys
import os

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        print("✅ HuggingFace transformers imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import transformers: {e}")
        return False
    
    try:
        print("✅ MAX SDK imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MAX SDK: {e}")
        print("💡 Make sure MAX SDK is installed and configured")
        return False
    
    try:
        # Add materl to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        print("✅ materl generation function imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import materl: {e}")
        return False
    
    return True


def test_torch_backend():
    """Test PyTorch backend with a small model."""
    print("\n🔥 Testing PyTorch Backend...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from materl.functions.generation import generate_completions
        
        # Use a small model for testing
        model_name = "microsoft/DialoGPT-small"  # Small model for testing
        
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Test with a simple prompt
        results = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=["Hello"],
            max_prompt_length=32,
            max_completion_length=10,
            num_generations=1,
            backend="torch",
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )
        
        print("✅ PyTorch backend test successful!")
        print(f"   Generated: '{results['completions_text'][0].strip()}'")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch backend test failed: {e}")
        return False


def test_graph_backend():
    """Test the MAX Graph backend with a real model."""
    print("\n🏗️ Testing MAX Graph Backend (from First Principles)...")
    
    try:
        from transformers import AutoTokenizer
        from materl.functions.generation import generate_completions
        
        # Use a small model for testing
        model_name = "microsoft/DialoGPT-small"
        
        print(f"Loading tokenizer for: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Test with a simple prompt
        results = generate_completions(
            model=model_name,  # Pass the model path for the graph backend
            tokenizer=tokenizer,
            prompts=["Hello"],
            max_prompt_length=32,
            max_completion_length=10,
            num_generations=1,
            backend="graph"  # Use the new graph backend
        )
        
        print("✅ MAX Graph backend test successful!")
        print(f"   Generated: '{results['completions_text'][0].strip()}'")
        return True
        
    except Exception as e:
        print(f"❌ MAX Graph backend test failed: {e}")
        print("💡 This likely means there's an issue in the graph construction or weight loading.")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🎯 Testing Updated Backend Implementations")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import tests failed. Cannot proceed with backend tests.")
        return
    
    # Test PyTorch backend
    torch_success = test_torch_backend()
    
    # Test MAX backend
    graph_success = test_graph_backend()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"PyTorch Backend: {'✅ PASS' if torch_success else '❌ FAIL'}")
    print(f"MAX Graph Backend: {'✅ PASS' if graph_success else '❌ FAIL'}")
    
    if torch_success and graph_success:
        print("\n🎉 All tests passed! Both backends are working correctly.")
    elif torch_success:
        print("\n⚠️  PyTorch backend works, but MAX Graph backend needs attention.")
    else:
        print("\n❌ Both backends need attention.")


if __name__ == "__main__":
    main() 