"""
materl.ops

This module defines the decorators for creating user-defined operations.
"""

def op(func):
    """
    Decorator to register a Python function as a custom operation
    in the materl graph.
    """
    # In a real implementation, this would register the function
    # with a central registry and inspect its signature.
    print(f"Registered Python op: {func.__name__}")
    return func

def mojo_op(path):
    """
    Decorator to register a Mojo kernel as a custom operation.
    """
    def decorator(func):
        # Here, we would store the path to the Mojo file and associate
        # it with the decorated (placeholder) function.
        print(f"Registered Mojo op: {func.__name__} from '{path}'")
        return func
    return decorator 