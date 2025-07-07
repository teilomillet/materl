# `materl`: A Declarative RL Algorithm Composition Framework

## 1. Core Philosophy

`materl` is a framework for building and experimenting with Generative AI reinforcement learning algorithms. Our guiding philosophy is to move beyond monolithic, imperative trainers and embrace a **declarative, compositional, and high-performance** paradigm.

An algorithm should be expressed as what it *is*, not the series of steps required to execute it. New algorithms are often transformations of existing ones. `materl` provides a clean, beautiful domain-specific language (DSL) to express these relationships directly, enabling researchers and engineers to move from idea to optimized execution in minutes.

The core principles are:
*   **Declarative:** Users define algorithms by describing their structure and components, not by writing explicit training loops.
*   **Compositional:** New algorithms are built by transforming, replacing, or adding to the components of existing "recipes."
*   **Performant by Default:** The declarative nature allows a powerful backend compiler to perform deep optimizations like Mojo Kernel Fusion and CUDA Graph Capture, delivering bleeding-edge performance automatically.

---

## 2. The "Recipe" Abstraction

The foundation of `materl` is the `@recipe`. A recipe is a declarative template that defines the essential computational steps of a canonical algorithm. It is not a function that runs eagerly; it's a symbolic blueprint.

**Example: A GRPO Recipe**
```python
# file: materl/recipes/grpo.py
import materl as ml

@ml.recipe
def grpo(policy, ref_policy, prompts):
    """
    This defines the symbolic structure of the GRPO algorithm.
    Each line declares a named computational node.
    """
    # Generation & Scoring
    completions = ml.generate(policy, prompts, name="completions")
    
    policy_logprobs = ml.logprobs(policy, prompts, completions, name="policy_logprobs")
    ref_logprobs = ml.logprobs(ref_policy, prompts, completions, name="ref_logprobs")
    
    # Rewards
    # Rewards are composable by default. The framework sums their outputs.
    rewards = ml.reward("length", weight=0.1) + ml.reward("mojo_diversity", weight=0.9)
    
    # Advantages & Loss
    advantages = ml.advantages(
        rewards, ref_logprobs, policy_logprobs, name="advantages"
    )
    
    # Define the final loss objective for this recipe.
    ml.set_loss(ml.grpo_loss(advantages, policy_logprobs, ref_logprobs))
```

This recipe is simple, readable, and captures the abstract structure of GRPO.

---

## 3. Algorithm Transformation

The true power of `materl` lies in creating new algorithms by applying transformations to a base recipe. This is done with the `@ml.algorithm` decorator.

**Example: Creating VAPO from the GRPO Recipe**
```python
# file: my_experiments/vapo.py
import materl as ml
from materl.recipes import grpo # Import the base recipe

# The `using` decorator applies the GRPO recipe as a starting point.
# The function body describes the *modifications* to that recipe's graph.
@ml.algorithm(using=grpo)
def vapo(graph, value_model):
    """
    This function defines VAPO by transforming the GRPO graph.
    The 'graph' object provides symbolic access to the nodes of the base recipe.
    """
    
    # 1. Add a new component: a value function pass.
    values = ml.values(value_model, graph.prompts, graph.completions)

    # 2. Replace a component: Swap the advantage calculation with GAE.
    #    The framework automatically re-wires the data dependencies.
    graph.advantages.replace_with(
        ml.gae_advantages(graph.rewards, values)
    )

    # 3. Add a new loss term. The framework knows to sum loss terms.
    ml.add_loss(ml.value_loss(values, graph.rewards))
```
This is the core aesthetic: new algorithms are defined by how they *differ* from established ones. This makes experimentation fast, intuitive, and highly readable.

---

## 4. The Compilation and Execution Model

`materl` separates algorithm *definition* from *execution*. The `ml.compile` function is the bridge. It takes a high-level algorithm specification and transforms it into a highly optimized, replayable execution graph.

```python
# 1. Instantiate the stateful models ("Agents").
policy = ml.Agent("mistral-7b", trainable=True)
ref_policy = ml.Agent("mistral-7b")
value_model = ml.Agent("value-head.max", trainable=True)

# 2. Compile the final algorithm specification.
compiled_vapo = ml.compile(
    vapo, # The algorithm function from the previous step
    policy=policy, 
    ref_policy=ref_policy, 
    value_model=value_model
)

# 3. Run the optimized graph in a loop.
for batch in dataset:
    # This call has near-zero CPU overhead.
    metrics = compiled_vapo.run(prompts=batch['prompts'])
    print(f"Loss: {metrics['loss']:.4f}")
```
The `ml.compile` step is where the magic happens:
*   **Mojo Kernel Fusion:** It automatically finds sequential compute kernels written in Mojo and fuses their source code into a single, monolithic kernel, eliminating intermediate memory traffic.
*   **CUDA Graph Capture:** It unrolls the entire sequence of GPU operations (model inference, fused kernel calls) for one turn and captures it into a static CUDA Graph, which can be replayed with minimal CPU overhead.

This ensures that even complex, multi-stage algorithms run with maximum performance without requiring manual optimization from the user. 

---

## 5. Path to Implementation

To make this vision a reality, we will follow a phased approach. This allows us to build the user-facing API first and then progressively replace the backend with the high-performance compiler.

### Phase 1: Build the Declarative Frontend

**Goal:** Allow a user to write and "run" a declarative algorithm. The execution will not be optimized yet, but the user experience and API will be in place.

**Where to Start: Create the core API and symbolic graph.**

1.  **Core API (`materl/__init__.py`):** Define the primary user-facing decorators and functions: `@recipe`, `@algorithm`, `compile`, `Agent`. This creates the new public interface.
2.  **Symbolic Graph (`materl/graph.py`):** Implement the `Graph` and `SymbolicNode` classes. When a `@recipe` function is called, it will not execute but rather populate a `Graph` instance with symbolic nodes representing the declared operations.
3.  **Recipe Decorator (`@materl/recipes.py`):** Implement the `@recipe` decorator. It will wrap a user's function and inject a `GraphBuilder` that translates the function body into the static graph representation. We will start by creating `materl/recipes/grpo.py` as our first test case.
4.  **"Naive" Compiler (`materl/compiler.py`):** Create the `compile` function. Initially, this function will perform a topological sort on the graph's nodes and execute them sequentially by calling the existing Python functions in `materl/functions`. This provides a working end-to-end system using the new API without the complex optimizations.
5.  **Algorithm Transformation (`materl/algorithm.py`):** Implement the `@algorithm` decorator. It will take a base `recipe`, generate its graph, and then pass a "graph controller" object to the user's transformation function, allowing modifications like `graph.advantages.replace_with(...)`.

**Outcome of Phase 1:** A user can write and run the `vapo` algorithm exactly as designed. `ml.compile(vapo, ...)` will return a runnable object that executes the correct logic, laying the foundation for future performance optimizations.

---

### Phase 2: Implement the High-Performance Compiler Backend

**Goal:** Make it fast. Replace the "naive" compiler backend with the optimizing one without changing the user-facing API.

**Where to Start: Focus on Mojo Kernel Fusion first.**

1.  **Enhance `materl/compiler.py`:**
    *   **Mojo Fusion:** Add logic to scan the graph for adjacent `MojoKernel` nodes. The compiler will read their source, merge them into a single temporary `.mojo` file, compile it into a shared library (`.so`), and update the graph to use a single `FusedMojoKernel` node.
    *   **CUDA Graph Capture:** After fusion, the compiler will trace the entire sequence of GPU operations (agent inference, fused kernel calls) using `torch.cuda.graph()`. The `compile` function will now return a `CompiledGraph` object containing this replayable graph.

**Outcome of Phase 2:** The `compile` function now produces a highly optimized executable. The `.run()` method's performance increases dramatically, with no changes required to the user's algorithm definition code.

---

### Phase 3: Finalize Integration and Deprecate Old Code

**Goal:** Make the declarative system the official, documented way to use `materl` and clean up the legacy codebase.

1.  **Refactor `materl/functions`:** The original pure functions (`compute_grpo_loss`, etc.) become internal implementation details for the default recipe nodes. They will be moved to a private `materl/_ops/` directory.
2.  **Deprecate `RLTrainer` and `RLEngine`:** Mark `materl/trainer/rl_trainer.py` and `materl/engine/rl_engine.py` as deprecated. Their docstrings will be updated to guide users to the new `materl.compile` workflow.
3.  **Update All Documentation and Examples:** Rewrite all examples in the `examples/` directory to showcase the new declarative paradigm. Update the `README.md` and other documentation to reflect the new philosophy and API.

This phased plan provides a clear and actionable roadmap to realize the vision for `materl`. 

---

## 6. Extensibility: User-Defined Operations

A core principle of `materl` is that it is an **extensible framework, not a walled garden.** While it provides a rich standard library of common components (`ml.reward`, `ml.advantages`), users must be able to define their own custom logic as first-class citizens of the ecosystem.

This is achieved through simple, declarative decorators that register user functions with the `materl` compiler.

### The `@ml.op` Decorator for Custom Python Logic

For rapid prototyping or for components that involve external systems, users can define their own operations in plain Python.

```python
import materl as ml
import torch

@ml.op
def sentiment_reward(completions_text: list[str]) -> {"rewards": torch.Tensor}:
    """
    A custom reward function that uses an external sentiment analysis model.
    The type hints are used by the compiler to understand the op's inputs and outputs.
    """
    scores = my_sentiment_analyzer.predict(completions_text)
    return {"rewards": torch.tensor(scores, dtype=torch.float32)}

# The user's custom op can now be used directly inside any recipe.
@ml.recipe
def sentiment_rl(policy, ref_policy, prompts):
    # ...
    completions = ml.generate(policy, prompts)
    rewards = sentiment_reward(completions.text) # Use like a built-in op
    # ...
```
The compiler will see the `@ml.op` decorator, inspect the function signature to determine its place in the computation graph, and integrate it seamlessly.

### The `@ml.mojo_op` Decorator for Custom High-Performance Kernels

For performance-critical custom logic, users can write their own Mojo kernels and register them with the framework.

```python
import materl as ml
import torch

@ml.mojo_op(path="kernels/repetition_penalty.mojo")
def repetition_penalty(completions: torch.Tensor) -> {"penalty_scores": torch.Tensor}:
    """
    This op is implemented in Mojo for maximum performance.
    The 'path' argument points to the source file, and the body is a pass-through declaration.
    """
    pass

# Use the custom Mojo op in a recipe, even combining it with standard library functions.
@ml.recipe
def custom_penalty_rl(policy, ref_policy, prompts):
    # ...
    # The compiler will automatically handle the data flow.
    rewards = ml.reward("length") - repetition_penalty(graph.completions)
    # ...
```

Crucially, the `materl` compiler will treat this user-defined Mojo op as a candidate for **Kernel Fusion**. If it appears sequentially with other Mojo ops (either built-in or user-defined), it will be automatically fused into the same high-performance super-kernel.

This ensures that there is no performance penalty for extending the framework. User-defined code can be just as fast and efficient as the core library components, making `materl` a truly powerful tool for both established and novel research. 

---

## 7. Operational Vision: Handling Asynchronous & Structured RL Workflows

The primary purpose of `materl` is to provide a powerful and elegant framework for **Reinforcement Learning from Tool Use**. Modern AI agents often operate in complex, multi-turn environments where they must interact with external tools, APIs, or provers. This introduces asynchronicity and creates structured, multi-step rollouts. `materl` is explicitly designed to handle this reality as a first-class citizen.

### Pluggable Generation Backends

The core generation engine is not monolithic. Acknowledging the rapid innovation in inference serving, `materl` allows the user to declaratively select the backend that drives the `ml.generate` and `ml.generate_step` operations.

```python
# The user can configure the generation engine at the Agent level.
# This could be 'vllm', 'sglang', or the default built-in 'mojo' engine.
policy = ml.Agent("mistral-7b", generation_backend="sglang")

# The recipe code remains clean and abstract, independent of the backend.
@ml.recipe
def tool_use_recipe(policy, prompts):
    rollouts = ml.generate(policy, prompts)
    ...
```
The `materl` compiler is responsible for dispatching to the correct backend, allowing users to leverage state-of-the-art inference engines without altering their core algorithm logic.

### Modeling Asynchronous Tool Calls (e.g., MCP)

The computation graph is designed to natively understand asynchronous operations. This is critical for modeling interactions with external tools, which are inherently I/O-bound.

This is achieved by making `@ml.op` decorators `async`-aware and by structuring recipes to reflect the multi-step generation process.

```python
import materl as ml

@ml.op
async def execute_mcp_tools(tool_requests: list[str]) -> {"tool_responses": list[str]}:
    """
    An async op that calls external tools and awaits their responses.
    The materl compiler will handle this I/O boundary efficiently.
    """
    responses = await my_tool_client.batch_execute(tool_requests)
    return {"tool_responses": responses}

@ml.recipe
def grpo_with_tools(policy, prompts):
    # Step 1: Generate the initial thought process and tool calls.
    initial_generation = ml.generate_step(policy, prompts, name="initial_step")
    
    # Step 2: Execute the tool calls asynchronously.
    tool_responses = execute_mcp_tools(initial_generation.tool_requests)
    
    # Step 3: Generate the final answer using the tool responses.
    final_generation = ml.generate_step(
        policy, 
        initial_generation.history, 
        tool_responses,
        name="final_step"
    )

    # The full 'rollout' is now a structured object containing all steps.
    full_rollout = ml.structure_rollout(initial_generation, tool_responses, final_generation)

    # Rewards can now be applied to different parts of the structured rollout.
    rewards = (
        ml.reward("good_tool_use", rollout=full_rollout) +
        ml.reward("final_answer_quality", rollout=full_rollout)
    )
    
    # ... The rest of the algorithm ...
```

This design makes the asynchronous, multi-step nature of the workflow explicit and clear within the recipe. The compiler can leverage this information to optimize execution, for example, by preparing GPU resources for the final generation step while waiting for the tool calls to complete.

### Flexible, Step-wise Optimization

By representing a rollout as a structured object, `materl` provides the flexibility to apply rewards and optimization pressure to any part of the process.

*   **Whole Sequence Optimization:** A reward function can analyze the `full_rollout` object to score the final answer's quality in the context of the tools used.
*   **Step-Level Optimization:** A reward function can be applied directly to the `initial_generation` step to reward the model for choosing the correct tool, or to the `final_generation` step to focus only on the quality of the concluding response.

This capability is a natural consequence of the declarative graph. The user simply "wires" their custom reward function to the part of the data structure they wish to optimize, giving them fine-grained control over the agent's learning process. 