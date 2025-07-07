# 🔥 materl - A Declarative RL Library for Fast Experimentation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

```bash
uv add materl
```

**materl** is a Reinforcement Learning library designed for rapid experimentation with language models. It combines a clean, declarative API with an accelerated backend, allowing you to focus on algorithm logic instead of boilerplate.

It's built for researchers who want to test a new reward function, tweak a loss calculation, or implement a novel algorithm quickly and efficiently.

## ✨ Philosophy: From Idea to Result, Faster

`materl` is built for iterating quickly. The design is centered on simplicity and performance at the point of experimentation.

-   **Declarative & Functional**: Define your entire RL workflow as a series of functional steps. This makes experiments easy to read, modify, and reproduce.
-   **Performant by Default**: The library is designed to be fast. Performance-critical sections are handled by an optimized backend, so you get great speed without writing low-level code.
-   **Minimalist API**: The API is intentionally simple. Core concepts like `Agent`, `Recipe`, and `compile` are all you need to get started, reducing cognitive overhead.

## 🏗️ Architecture: Your Experiment as a Graph

The core of `materl` is its declarative, graph-based paradigm. A "recipe" is a Python function that defines the sequence of operations in your experiment.

1.  **Agents**: Simple wrappers around your models (e.g., from Hugging Face Transformers).
2.  **Recipe**: A function that describes the steps: generate text, calculate log-probabilities, compute rewards, and define the loss.
3.  **Symbolic Graph**: The recipe returns a lightweight data structure that represents your entire workflow.
4.  **Compiler**: The `compile()` function processes this graph and prepares it for execution.
5.  **Execution**: Calling `.run()` on the compiled graph executes the experiment.

## 🚀 A Simple DAPO Experiment

This example shows how to set up and run a DAPO experiment. The code reads like a description of the experimental procedure itself.

```python
from materl.agents import Agent
from materl.compiler import compile
from materl.config import GenerationConfig, DAPOConfig
from materl.recipes import dapo
import torch

# 1. Set up your models using the Agent wrapper
model_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
policy_agent = Agent(model_name, trainable=True, device=device)
ref_agent = Agent(model_name, trainable=False, device=device)

# 2. Define your inputs and configurations
prompts = ["Hello, my name is", "What is the capital of France?"]
gen_config = GenerationConfig(max_completion_length=50)
algorithm_config = DAPOConfig(beta=0.1)

# 3. Use a recipe to create a symbolic graph of your experiment
symbolic_graph = dapo(
    policy=policy_agent,
    ref_policy=ref_agent,
    prompts=prompts,
    max_completion_length=gen_config.max_completion_length,
)

# 4. Compile the graph and run the experiment
compiled_graph = compile(symbolic_graph)
final_context = compiled_graph.run(
    policy=policy_agent,
    ref_policy=ref_agent,
    prompts=prompts,
    generation_config=gen_config,
    dapo_config=algorithm_config,
)

print("✅ DAPO experiment finished successfully!")
print(f"Final context keys: {list(final_context.keys())}")
```

## 🧪 Included Recipes

`materl` comes with several pre-built recipes to get you started:

-   **GRPO** (Group Relative Policy Optimization)
-   **DAPO** (Decoupled Advantage Policy Optimization)
-   **VAPO** (Value-Aligned Policy Optimization)

You can find these in `materl/recipes` and see them in action in the `examples/` directory. Creating your own recipe is as simple as writing a new Python function.

## 🔭 Future Direction

Our goal is to make `materl` the best tool for **applied RL research and fast prototyping**. We plan to:

-   **Expand the Recipe Book**: Add more state-of-the-art algorithms.
-   **Enhance Debugging Tools**: Provide tools to inspect and visualize the computational graph.
-   **Broaden Hardware Support**: Continue to optimize performance across a wider range of GPUs.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Ready to accelerate your RL training? Get started with materl today!** 🚀 