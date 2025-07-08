"""
materl.compiler

This module defines the NaiveCompiler, which translates a symbolic Graph
into an executable plan using existing Python functions.
"""
import inspect
import torch
from typing import Dict, Any, List
from .graph import Graph, SymbolicNode

# Mapping from symbolic op names to the actual Python functions
from materl.functions import (
    generation, 
    logprobs, 
    advantages, 
    loss as loss_fns,
    values as values_fns
)
from materl.functions.rewards import compute_reward


OP_MAP = {
    "generate": generation.generate_completions,
    "logprobs": logprobs.compute_policy_and_ref_logprobs,
    "advantages": advantages.compute_advantages,
    "returns": advantages.compute_discounted_returns,
    "reward": compute_reward, # The op for a single reward
    "values": values_fns.compute_values,
    # The "loss" op is now a generic dispatcher, see run method.
    "loss": {
        "default": loss_fns.compute_policy_gradient_loss,
        "reinforce_loss": loss_fns.compute_reinforce_loss,
        "kimi_reinforce_loss": loss_fns.compute_kimi_reinforce_loss,
    }
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

        for node in self.execution_plan:
            print(f"Executing node: {node.name} (op: {node.op})")
            
            if node.op == "add":
                continue

            op_func = OP_MAP.get(node.op)
            if not op_func:
                print(f"  [Warning] No op for '{node.op}', skipping.")
                continue

            # Dynamic Loss Function Dispatch
            # This is the key to making the system extensible. It allows the loss
            # function to be passed as a callable object directly, bypassing the
            # static OP_MAP.
            if node.op == "loss":
                loss_fn_or_key = node.op_kwargs.get("loss_fn", "default")
                
                # If loss_fn is a callable function, use it directly.
                if callable(loss_fn_or_key):
                    op_func = loss_fn_or_key
                # Otherwise, fall back to the OP_MAP lookup for backward compatibility.
                else:
                    op_func = op_func.get(loss_fn_or_key)
                    if not op_func:
                        raise ValueError(f"Loss function '{loss_fn_or_key}' not found in OP_MAP.")

            # Argument Resolution
            # This is the core logic for dependency injection. It inspects the
            # function signature and resolves arguments from the execution context,
            # including the outputs of previous nodes.
            func_kwargs = {}
            func_sig = inspect.signature(op_func)

            for param_name in func_sig.parameters:
                found_arg = False
                # 1. Check if the argument is defined on the node itself
                if param_name in node.op_kwargs:
                    value = node.op_kwargs[param_name]
                    if isinstance(value, SymbolicNode):
                        # If the value is a node, it's a dependency. We first check
                        # its direct output dictionary for the required key.
                        dependency_output = execution_context.get(value.name, {})
                        if param_name in dependency_output:
                            func_kwargs[param_name] = dependency_output[param_name]
                            found_arg = True
                    else:
                        # It's a literal value (e.g., gamma, weight)
                        func_kwargs[param_name] = value
                        found_arg = True
                
                if found_arg:
                    continue
                
                # 2. If not found in the node's specific output, check the global context.
                #    This handles aliases like `current_policy_logprobs`.
                if param_name in execution_context:
                    func_kwargs[param_name] = execution_context[param_name]
                    continue
                
                # 3. Check configs for hyperparameters
                found_in_config = False
                for config in execution_context["configs"].values():
                    if hasattr(config, param_name):
                        func_kwargs[param_name] = getattr(config, param_name)
                        found_in_config = True
                        break
                if found_in_config:
                    continue

                # 4. Handle special cases for Agent properties, extracting the
                #    underlying model or tokenizer from the Agent wrapper.
                if param_name == 'model' and 'policy' in execution_context:
                    func_kwargs['model'] = execution_context['policy'].model
                elif param_name == 'policy_model' and 'policy' in execution_context:
                    func_kwargs['policy_model'] = execution_context['policy'].model
                elif param_name == 'tokenizer' and 'policy' in execution_context:
                    func_kwargs['tokenizer'] = execution_context['policy'].tokenizer

            result_dict = op_func(**func_kwargs)

            # All op functions are expected to return a dictionary. This assertion
            # helps the linter understand the type of `result_dict` and also
            # serves as a robust runtime check.
            assert isinstance(result_dict, dict), f"Op '{node.op}' returned type {type(result_dict)}, but expected a dict."

            # Aggregate rewards into the context
            if node.op == "reward":
                if "rewards_tensor" not in execution_context:
                    execution_context["rewards_tensor"] = torch.zeros_like(result_dict["rewards_tensor"])
                execution_context["rewards_tensor"] += result_dict["rewards_tensor"]
                
                if "rewards_per_token" not in execution_context:
                    execution_context["rewards_per_token"] = torch.zeros_like(result_dict["rewards_per_token"])
                execution_context["rewards_per_token"] += result_dict["rewards_per_token"]

            
            # The order of these operations is critical. We unpack the dictionary
            # into the general context first, then set the specific entry for
            # the node's output. This prevents name collisions where a node's
            # name is the same as a key in its output dict (e.g., 'returns').
            if isinstance(result_dict, dict):
                execution_context.update(result_dict)
            execution_context[node.name] = result_dict

            # Map the output of the logprobs nodes to the expected input names
            # for the loss function. This is a temporary solution to align the
            # generic compiler with the specific signature of `compute_grpo_loss`.
            if node.name == "logprobs":
                execution_context["current_policy_logprobs"] = result_dict["policy_logprobs"]
            elif node.name == "logprobs_2":
                execution_context["old_policy_logprobs"] = result_dict["policy_logprobs"]
            
        print("--- ExecutableCompiler run finished ---")
        return execution_context


def compile(graph: Graph) -> ExecutableCompiler:
    """
    The main entry point for the compiler.
    """
    return ExecutableCompiler(graph) 