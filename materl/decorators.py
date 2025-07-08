"""
materl.decorators

This module provides a set of decorators that can be used to modify
materl computation graphs on the fly. This allows for a composable
and extensible way to build and experiment with RL algorithms without
needing to create monolithic, duplicative recipes.
"""

from functools import wraps
from typing import Callable, Any, Dict

def use_loss(loss_fn: Callable, **loss_kwargs: Any) -> Callable:
    """
    A decorator factory that replaces the loss function in a materl graph.

    This is the core of the composable recipe system. It allows a user to
    take an existing recipe and hot-swap the loss function with a custom
    implementation, without modifying the original recipe code.

    Args:
        loss_fn: The new, callable loss function to use.
        **loss_kwargs: Additional keyword arguments to be passed to the loss
                     function during execution.

    Returns:
        A decorator that, when applied to a recipe, will modify its graph.
    """
    def decorator(recipe_fn: Callable) -> Callable:
        @wraps(recipe_fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 1. First, call the original recipe function to get the base
            #    symbolic graph that it produces.
            graph = recipe_fn(*args, **kwargs)

            # 2. Find the 'loss' node within the graph's list of nodes.
            #    We assume there is only one loss node per graph.
            loss_node = None
            for node in graph.nodes:
                if node.op == "loss":
                    loss_node = node
                    break
            
            if not loss_node:
                raise ValueError("Could not find a 'loss' node in the graph to replace.")

            # 3. This is the critical step: modify the graph in place.
            #    We update the 'op_kwargs' of the loss node. The 'loss_fn'
            #    key will now hold a direct reference to the new callable
            #    function. The compiler has been updated to handle this.
            #    We also merge in any new kwargs for the new loss function.
            loss_node.op_kwargs["loss_fn"] = loss_fn
            loss_node.op_kwargs.update(loss_kwargs)

            # 4. Return the modified graph.
            return graph
        return wrapper
    return decorator 