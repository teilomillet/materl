# MAX Backend for Accelerated Training

The MAX backend provides **significant speedup** for RL training by accelerating the text generation step, which is often the bottleneck in policy-based RL algorithms like GRPO, DAPO, and REINFORCE.

## 🚀 Performance Benefits

Based on benchmarks with Phi-4-mini-instruct:
- **7.14x speedup** over PyTorch backend
- **39.8 tokens/sec** vs **5.6 tokens/sec** 
- Same training algorithms, just faster generation

## 📋 Requirements

1. **MAX Platform**: Install the MAX platform from Modular
2. **GGUF Models**: Use GGUF format models (available on HuggingFace)
3. **Model Compatibility**: Ensure your training model has a GGUF equivalent

## 🔧 Configuration

### Basic Setup

```python
from materl.config import GenerationConfig

# Configure MAX backend
generation_config = GenerationConfig(
    backend="max",  # or "graph" for automatic model loading
    max_model_path="microsoft/DialoGPT-small-GGUF",  # GGUF model path
    max_prompt_length=512,
    max_completion_length=128,
    num_generations=8,
    temperature=1.0
)
```

### Training Example

```python
from materl import run
from materl.agents import Agent
from materl.recipes import grpo
from materl.config import GenerationConfig, GRPOConfig

# Create agents with PyTorch models (for gradient computation)
policy_agent = Agent(model_name_or_path="microsoft/DialoGPT-small")
ref_agent = Agent(model_name_or_path="microsoft/DialoGPT-small")

# Configure MAX backend for fast generation
generation_config = GenerationConfig(
    backend="max",
    max_model_path="microsoft/DialoGPT-small-GGUF"
)

# Algorithm configuration  
grpo_config = GRPOConfig(beta=0.04, epsilon=0.2)

# Run training with MAX acceleration
result = run(
    grpo,
    policy=policy_agent,
    ref_policy=ref_agent,
    prompts=["Hello, how are you?", "What is AI?"],
    generation_config=generation_config,
    grpo_config=grpo_config
)
```

## 🔄 Backend Options

### 1. torch (Default)
- Uses PyTorch models directly
- Standard HuggingFace integration
- Good for development and debugging

### 2. max
- Requires pre-initialized LLM instance
- Best for inference workloads
- Reuses model between calls

### 3. graph
- **Recommended for training**
- Takes model path, creates LLM automatically
- Caches LLM instances for efficiency

## 🏃‍♂️ Training Workflow

The training workflow with MAX backend:

1. **Policy/Reference Models**: PyTorch models for gradient computation
2. **Generation**: MAX backend for fast text generation  
3. **Logprobs**: PyTorch models compute log probabilities
4. **Loss**: Standard PyTorch loss computation and backprop

This hybrid approach maximizes both speed and compatibility.

## 🎯 Best Practices

### Model Selection
```python
# Choose GGUF models that match your PyTorch models
pytorch_model = "microsoft/DialoGPT-small"
gguf_model = "microsoft/DialoGPT-small-GGUF"  # Equivalent GGUF version
```

### Memory Management
```python
from materl.functions.generation import clear_llm_cache

# Clear LLM cache periodically during long training runs
clear_llm_cache()
```

### Error Handling
```python
try:
    generation_config = GenerationConfig(
        backend="max",
        max_model_path="model-gguf"
    )
except ValueError as e:
    # Fallback to PyTorch if MAX setup fails
    generation_config = GenerationConfig(backend="torch")
```

## 🔍 Troubleshooting

### Common Issues

1. **Missing GGUF Model**
   ```
   ValueError: max_model_path is required when using backend='max'
   ```
   **Solution**: Provide valid GGUF model path

2. **Model Compatibility**
   ```
   Error loading GGUF model
   ```
   **Solution**: Ensure GGUF model matches PyTorch model architecture

3. **MAX Platform Issues**
   ```
   ImportError: No module named 'max'
   ```
   **Solution**: Install MAX platform from Modular

### Performance Monitoring

```python
import time

start_time = time.time()
result = run(grpo, ...)
end_time = time.time()

print(f"Training step completed in {end_time - start_time:.2f}s")
```

## 📊 Expected Speedups

| Model Size | PyTorch | MAX | Speedup |
|------------|---------|-----|---------|
| Small (117M) | 21.5s | 3.0s | 7.1x |
| Medium (345M) | ~60s | ~8s | ~7.5x |
| Large (774M) | ~120s | ~16s | ~7.5x |

*Results may vary based on hardware and model configuration*

## 🚀 Getting Started

1. **Install Requirements**
   ```bash
   # Install MAX platform (follow Modular's installation guide)
   # Ensure materl is installed with MAX support
   ```

2. **Try the Example**
   ```bash
   python examples/max_training_example.py
   ```

3. **Adapt Your Training**
   - Update `GenerationConfig` to use MAX backend
   - Ensure GGUF model availability
   - Monitor performance improvements

The MAX backend integration makes materl training significantly faster while maintaining full algorithm compatibility! 