"""
materl.algorithms

This module defines the @algorithm decorator for transforming recipes.
"""
from functools import wraps

def algorithm(using):
    """
    A decorator that transforms a base recipe into a new algorithm.
    """
    base_recipe_func = using
    
    def decorator(transform_func):
        @wraps(transform_func)
        def wrapper(*args, **kwargs):
            # 1. Generate the base graph from the recipe
            base_graph = base_recipe_func(*args, **kwargs)
            
            # 2. Apply the user's transformation function to the graph
            # The 'transform_func' is the user's function (e.g., vapo)
            # that contains the logic to modify the graph.
            print(f"Transforming recipe '{base_graph.name}' with '{transform_func.__name__}'")
            
            # Here, the user's function would be called with the graph
            # and any other necessary arguments to modify it.
            # For now, this is a placeholder.
            # transform_func(base_graph, ...)
            
            return base_graph # Return the modified graph
        return wrapper
    return decorator 