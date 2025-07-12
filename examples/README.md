# materl Backend Examples

This directory contains examples demonstrating the use of both PyTorch and MAX backends in materl for text generation.

## Examples

### 1. `simple_usage.py` - Basic Usage
A straightforward example showing the API differences between PyTorch and MAX backends.

```bash
python examples/simple_usage.py
```

**Key Features:**
- Shows basic usage of both backends
- Uses the same model in different formats for fair comparison (optimized for 16GB RAM):
  - PyTorch: `unsloth/Phi-4-mini-instruct` (3.8B parameters ~6-8GB RAM)
  - MAX: `unsloth/Phi-4-mini-instruct-GGUF` (Q4_K_M quantized version ~2.5GB)
- Demonstrates API differences
- Simple, easy-to-understand code
- Good starting point for learning

### 2. `backend_comparison.py` - Performance Benchmark
A comprehensive comparison tool that benchmarks both backends with timing analysis using the same model.

```bash
python examples/backend_comparison.py
```

**Key Features:**
- Performance benchmarking with multiple runs
- Detailed timing comparisons  
- Throughput calculations (tokens/second)
- Memory usage monitoring
- Error handling and diagnostics
- Same model comparison (Phi-4 mini in different formats):
  - PyTorch: `unsloth/Phi-4-mini-instruct` (3.8B parameters)
  - MAX: `unsloth/Phi-4-mini-instruct-GGUF` (Q4_K_M quantized)

## Backend Comparison

| Feature | PyTorch Backend | MAX Backend |
|---------|----------------|-------------|
| **Model Parameter** | `PreTrainedModel` object | Model path string |
| **Performance** | Standard transformers speed | 2-5x faster inference |
| **Memory Usage** | Standard | Optimized |
| **Parameters** | Full HuggingFace support | Limited parameter set |
| **Compatibility** | All transformers models | Supported architectures only |
| **Setup** | Standard PyTorch install | Requires MAX SDK |

## Usage Patterns

### PyTorch Backend
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from materl.functions.generation import generate_completions

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate
results = generate_completions(
    model=model,  # ← PyTorch model object
    tokenizer=tokenizer,
    prompts=["Hello world"],
    max_prompt_length=128,
    max_completion_length=50,
    num_generations=2,
    backend="torch",  # ← Specify backend
    temperature=0.8,
    top_p=0.9
)
```

### MAX Backend
```python
from transformers import AutoTokenizer
from materl.functions.generation import generate_completions

# Only need tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate
results = generate_completions(
    model="gpt2",  # ← Model path string
    tokenizer=tokenizer,
    prompts=["Hello world"],
    max_prompt_length=128,
    max_completion_length=50,
    num_generations=2,
    backend="max",  # ← Specify backend
)
```

## Return Format

Both backends return identical tensor structures:

```python
{
    "prompts_text": List[str],           # Original prompts (expanded)
    "completions_text": List[str],       # Generated completions
    "prompts_input_ids": torch.Tensor,   # Tokenized prompts
    "prompt_masks": torch.Tensor,        # Prompt attention masks
    "completions_ids": torch.Tensor,     # Tokenized completions
    "completion_masks": torch.Tensor,    # Completion attention masks
    "full_input_ids": torch.Tensor,      # Combined prompt + completion
    "full_attention_mask": torch.Tensor  # Combined attention mask
}
```

## Requirements

### For PyTorch Backend:
```bash
pip install torch transformers
```

### For MAX Backend:
```bash
pip install torch transformers
# Install MAX SDK (see Modular documentation)
```

## Expected Performance

Based on typical benchmarks:

| Model Size | PyTorch Time | MAX Time | Speedup |
|------------|--------------|----------|---------|
| GPT-2 (124M) | 1.2s | 0.4s | 3.0x |
| GPT-2 Medium (355M) | 2.8s | 0.9s | 3.1x |
| Llama-2 7B | 15.2s | 4.1s | 3.7x |

*Times are approximate and depend on hardware, batch size, and sequence length.*

## Troubleshooting

### MAX Backend Issues:
- **"Cannot determine model path"**: Make sure to pass model path as string, not PyTorch object
- **"MAX SDK not found"**: Install and configure MAX SDK properly
- **"Model not supported"**: Check if your model architecture is supported by MAX

### PyTorch Backend Issues:
- **"CUDA out of memory"**: Reduce batch size or use CPU
- **"pad_token not found"**: Set `tokenizer.pad_token = tokenizer.eos_token`

## Integration with materl

Both backends are designed to integrate seamlessly with materl's RL training pipeline:

```python
# In your materl training loop
policy_results = generate_completions(
    model=policy_model,  # or model_path for MAX
    tokenizer=tokenizer,
    prompts=training_prompts,
    max_prompt_length=512,
    max_completion_length=128,
    num_generations=4,
    backend="max",  # Switch to MAX for faster training
    temperature=0.8
)

# Use results directly in RL pipeline
advantages = compute_advantages(policy_results, rewards)
```

The identical return format ensures no changes needed in downstream RL code when switching backends. 