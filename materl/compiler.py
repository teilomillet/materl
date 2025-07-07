"""
materl.compiler

This module defines the NaiveCompiler, which translates a symbolic Graph
into an executable plan using existing Python functions.
"""
import inspect
from typing import Dict, Any, List
from .graph import Graph, SymbolicNode

# Mapping from symbolic op names to the actual Python functions
from materl.functions import (
    generation, 
    logprobs, 
    rewards, 
    advantages, 
    loss as loss_fns,
    values as values_fns
)

# Configuration classes are now imported from the central config module,
# ensuring a single source of truth for algorithm and generation parameters.
from materl.config import GenerationConfig, GRPOConfig


OP_MAP = {
    "generate": generation.generate_completions,
    "logprobs": logprobs.compute_policy_and_ref_logprobs,
    "advantages": advantages.compute_advantages,
    "loss": loss_fns.compute_policy_gradient_loss,
    "values": values_fns.compute_values,
}

class ExecutableCompiler:
    """
    A compiler that executes a materl graph by mapping symbolic ops
    to concrete Python functions and managing data flow.
    """
    def __init__(self, graph: Graph):
        self.graph = graph
        self.execution_plan = self._topological_sort()

    def _topological_sort(self) -> List[SymbolicNode]:
        """
        Performs a topological sort on the graph nodes to determine execution order.
        """
        sorted_nodes = []
        visited = set()
        node_deps = {node: set(node.dependencies) for node in self.graph.nodes}

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for dep in node_deps.get(node, []):
                visit(dep)
            sorted_nodes.append(node)

        for node in self.graph.nodes:
            visit(node)
            
        return sorted_nodes

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the compiled plan with intelligent argument passing.
        """
        print("--- Running ExecutableCompiler ---")
        execution_context: Dict[str, Any] = {"configs": {}}
        
        # Separate initial inputs from configs
        initial_inputs = {}
        for key, value in kwargs.items():
            if "config" in key.lower():
                # Store configs in a dedicated 'configs' dictionary within the context.
                execution_context["configs"][key] = value
            else:
                initial_inputs[key] = value
        execution_context.update(initial_inputs)

        # The device is a core component of the execution context, as all
        # tensor operations will be performed on it. We extract it from the
        # policy agent's model and add it to the context, making it available
        # to all downstream operations.
        if "policy" in execution_context:
            execution_context["device"] = execution_context["policy"].model.device

        reward_configs = []

        for node in self.execution_plan:
            print(f"Executing node: {node.name} (op: {node.op})")
            
            if node.op == "reward":
                reward_configs.append({"name": node.name, **node.op_kwargs})
                continue
            if node.op == "add":
                continue

            op_func = OP_MAP.get(node.op)
            if not op_func:
                print(f"  [Warning] No op for '{node.op}', skipping.")
                continue

            # Before executing an operation, we inject the collected reward
            # configurations into the execution context. This makes them
            # available to functions like `compute_advantages` that depend on them.
            if reward_configs:
                execution_context["reward_configs"] = reward_configs

            # Build dependencies
            func_kwargs = self._build_kwargs(op_func, execution_context)

            # The special dependency injection logic has been removed.
            # The compiler is now a generic engine that simply maps context
            # variables to function arguments. All operation-specific logic
            # is now encapsulated within the functions themselves (e.g., in
            # `materl.functions.advantages`). This greatly simplifies the
            # compiler and makes the system more modular and scalable.

            result_dict = op_func(**func_kwargs)
            
            # Map the output of the logprobs nodes to the expected input names
            # for the loss function. This is a temporary solution to align the
            # generic compiler with the specific signature of `compute_grpo_loss`.
            if node.name == "logprobs":
                execution_context["current_policy_logprobs"] = result_dict["policy_logprobs"]
            elif node.name == "logprobs_2":
                execution_context["old_policy_logprobs"] = result_dict["policy_logprobs"]
            
            execution_context[node.name] = result_dict
            if isinstance(result_dict, dict):
                execution_context.update(result_dict)

        print("--- ExecutableCompiler run finished ---")
        return execution_context

    def _build_kwargs(self, func, context, keys_to_ignore=None) -> Dict[str, Any]:
        if keys_to_ignore is None:
            keys_to_ignore = []
        
        kwargs = {}
        func_sig = inspect.signature(func)

        for param_name in func_sig.parameters:
            if param_name in keys_to_ignore:
                continue

            # We must handle the specific case of agent models first, as their
            # parameter names (`value_model`, `policy`, etc.) also exist as
            # top-level keys in the context. Checking for them first ensures we
            # extract the underlying `.model` attribute instead of passing the
            # entire Agent object, which is not callable.
            if param_name == 'model' and 'policy' in context:
                kwargs['model'] = context['policy'].model
                continue
            if param_name == 'policy_model' and 'policy' in context:
                kwargs['policy_model'] = context['policy'].model
                continue
            if param_name == 'ref_model' and 'ref_policy' in context:
                kwargs['ref_model'] = context['ref_policy'].model
                continue
            if param_name == 'value_model' and 'value_model' in context:
                kwargs['value_model'] = context['value_model'].model
                continue
            if param_name == 'tokenizer' and 'policy' in context:
                kwargs['tokenizer'] = context['policy'].tokenizer
                continue

            # Now, handle the generic case where the parameter name is a key
            # in the execution context.
            if param_name in context:
                kwargs[param_name] = context[param_name]
                continue
            
            # Finally, check inside the configuration objects for the parameter.
            found_in_config = False
            for config in context["configs"].values():
                if hasattr(config, param_name):
                    kwargs[param_name] = getattr(config, param_name)
                    found_in_config = True
                    break
            if found_in_config:
                continue

        return kwargs

def compile(graph: Graph) -> ExecutableCompiler:
    """
    The main entry point for the compiler.
    """
    return ExecutableCompiler(graph) 