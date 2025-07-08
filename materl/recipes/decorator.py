"""
materl.recipes.decorator

Defines the @recipe decorator.
"""
import functools
from typing import Callable
from ..graph import Graph
from .builder import GraphBuilder

def recipe(func: Callable) -> Callable:
    """
    A decorator that simplifies writing recipes. It automatically creates a
    GraphBuilder instance and passes it to the decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Graph:
        # By using functools.wraps, the __name__ attribute of the original
        # function (`func`) is preserved on the `wrapper`, allowing us to
        # correctly name the graph.
        builder = GraphBuilder(func.__name__) # type: ignore
        
        # The 'ml' object (the GraphBuilder) is passed as the first argument,
        # followed by the rest of the original arguments.
        func(builder, *args, **kwargs)
        
        # The recipe is expected to have called `set_loss`, so we can now
        # safely return the completed graph.
        return builder.get_graph()
    return wrapper 