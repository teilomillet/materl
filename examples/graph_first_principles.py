#!/usr/bin/env python3
"""
Example: Building Text Generation from MAX Graph First Principles

This demonstrates how to build a text generation system directly from the
MAX Graph API, bypassing the pipeline abstractions entirely.

The key insight is that the Graph API provides all the primitives you need:
- TensorType, DeviceRef for defining inputs
- ops.* for all operations  
- nn.* for neural network components
- Graph context for building the computation graph

This approach gives you:
1. Full control over the computation graph
2. Clear visibility into what's happening
3. Easy debugging and modification
4. Direct access to MAX's performance optimizations
"""

from typing import List
import numpy as np
from transformers import AutoTokenizer

def build_transformer_from_graph_ops():
    """
    Build a transformer model using only graph operations.
    
    This is the core insight: instead of using high-level pipeline abstractions,
    we build the model directly from the fundamental graph operations.
    """
    from max.graph import Graph, TensorType, DeviceRef, ops
    from max.dtype import DType
    from max.nn import Embedding, Linear, LayerNorm
    
    # Define the computation graph
    device = DeviceRef.GPU(0)
    
    # Input specification - this is much clearer than pipeline config
    input_type = TensorType(
        DType.int64,
        shape=["batch", "seq_len"], 
        device=device
    )
    
    # Build the graph with explicit control
    with Graph("transformer_from_first_principles", input_types=[input_type]) as graph:
        tokens = graph.inputs[0]
        
        # 1. Embedding layer
        embed = Embedding(
            vocab_size=50257,  # GPT-2 vocab size
            hidden_dim=768,
            device=device,
            dtype=DType.float32
        )
        h = embed(tokens.tensor)
        
        # 2. Transformer layers - built from basic operations
        for layer in range(12):  # 12 layers
            # Self-attention (simplified)
            ln1 = LayerNorm(dims=768, device=device)
            normed = ln1(h)
            
            # Q, K, V projections
            qkv_proj = Linear(in_dim=768, out_dim=768*3, device=device, dtype=DType.float32)
            qkv = qkv_proj(normed)
            
            # Split into Q, K, V
            q, k, v = ops.split(qkv, [768, 768, 768], axis=-1)
            
            # Attention (simplified - just using Q for now)
            attn_out = Linear(in_dim=768, out_dim=768, device=device, dtype=DType.float32)(q)
            
            # Residual connection
            h = h + attn_out
            
            # Feed forward
            ln2 = LayerNorm(dims=768, device=device)
            normed2 = ln2(h)
            
            # FFN
            ffn_up = Linear(in_dim=768, out_dim=3072, device=device, dtype=DType.float32)
            ffn_down = Linear(in_dim=3072, out_dim=768, device=device, dtype=DType.float32)
            
            # Apply FFN with GELU
            ff_out = ffn_down(ops.gelu(ffn_up(normed2)))
            
            # Residual connection
            h = h + ff_out
        
        # 3. Final layer norm and output projection
        final_ln = LayerNorm(dims=768, device=device)
        final_h = final_ln(h)
        
        # Output projection to vocabulary
        lm_head = Linear(in_dim=768, out_dim=50257, device=device, dtype=DType.float32)
        logits = lm_head(final_h)
        
        # Set output
        graph.output(logits)
    
    return graph


def functional_generation_pipeline():
    """
    Create a functional generation pipeline using first principles.
    
    This replaces the complex pipeline abstractions with simple, 
    composable functions.
    """
    
    def tokenize(text: str, tokenizer) -> np.ndarray:
        """Pure function: text -> tokens"""
        return tokenizer(text, return_tensors="np")["input_ids"]
    
    def forward_pass(tokens: np.ndarray, model) -> np.ndarray:
        """Pure function: tokens -> logits"""
        from max.driver import Tensor
        input_tensor = Tensor.from_numpy(tokens)
        outputs = model.execute(input_tensor)
        return outputs[0].to_numpy()
    
    def sample_next_token(logits: np.ndarray, temperature: float = 1.0) -> int:
        """Pure function: logits -> next token"""
        # Apply temperature
        logits = logits / temperature
        # Softmax
        probs = np.exp(logits) / np.sum(np.exp(logits))
        # Sample
        return np.random.choice(len(probs), p=probs)
    
    def decode_tokens(tokens: np.ndarray, tokenizer) -> str:
        """Pure function: tokens -> text"""
        return tokenizer.decode(tokens, skip_special_tokens=True)
    
    # These functions compose naturally without complex abstractions
    return {
        'tokenize': tokenize,
        'forward': forward_pass,
        'sample': sample_next_token,
        'decode': decode_tokens
    }


def compare_approaches():
    """
    Compare the pipeline approach vs. graph-first approach.
    """
    
    print("=== PIPELINE APPROACH (Complex) ===")
    print("""
    1. Create PipelineConfig with dozens of parameters
    2. Load architecture-specific model class
    3. Navigate complex inheritance hierarchies
    4. Debug through multiple abstraction layers
    5. Limited control over graph construction
    6. Hidden performance optimizations
    """)
    
    print("\n=== GRAPH-FIRST APPROACH (Simple) ===")
    print("""
    1. Define input types directly
    2. Build computation graph with explicit operations
    3. Full visibility into every operation
    4. Direct access to MAX optimizations
    5. Easy to debug and modify
    6. Composable functional design
    """)
    
    print("\n=== KEY ADVANTAGES ===")
    print("""
    ✅ No configuration hell
    ✅ Clear error messages
    ✅ Direct control over computation
    ✅ Easy to understand and modify
    ✅ Performance optimizations are explicit
    ✅ Functional programming principles
    """)


if __name__ == "__main__":
    print("🚀 Building Text Generation from First Principles")
    print("=" * 50)
    
    # This is the approach you should take:
    # 1. Build directly from graph operations
    # 2. Use functional programming principles
    # 3. Avoid complex abstractions
    
    compare_approaches()
    
    print("\n💡 Next Steps:")
    print("1. Replace pipeline usage with direct graph construction")
    print("2. Build functional generation utilities")
    print("3. Compose operations instead of using monolithic classes")
    print("4. Test each component in isolation")
    print("5. Optimize performance at the graph level") 