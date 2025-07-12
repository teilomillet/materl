"""
materl - A declarative machine learning framework

The simple way to define and run RL algorithms.
"""

from .algorithm import algorithm
from .agents import Agent
from .compiler import compile
from .recipes import grpo, reinforce

# Simple registry for algorithms
_algorithms = {}

def register_algorithm(name: str, algo_func):
    """Register an algorithm so users can get it by name."""
    _algorithms[name] = algo_func

def get_algorithm(name: str):
    """Get an algorithm by name."""
    if name not in _algorithms:
        raise ValueError(f"Algorithm '{name}' not found. Available: {list(_algorithms.keys())}")
    return _algorithms[name]

def list_algorithms():
    """List all available algorithms."""
    return list(_algorithms.keys())

def run(algorithm_func, **kwargs):
    """Simple one-liner to run any algorithm."""
    # Separate configs from algorithm arguments
    # Config objects should be passed to the compiled graph's run() method,
    # not to the algorithm function itself
    configs = {}
    algorithm_kwargs = {}
    
    for key, value in kwargs.items():
        if "config" in key.lower():
            configs[key] = value
        else:
            algorithm_kwargs[key] = value
    
    # Create the symbolic graph with only algorithm parameters
    graph = algorithm_func(**algorithm_kwargs)
    compiled = compile(graph)
    
    # Pass all original kwargs (including configs) to the compiled graph runner
    # The compiler will handle extracting individual parameters from config objects
    return compiled.run(**kwargs)

# Register built-in algorithms
register_algorithm("grpo", grpo)
register_algorithm("reinforce", reinforce)

# Export the main API
__all__ = [
    "algorithm", 
    "Agent", 
    "compile", 
    "run",
    "get_algorithm",
    "list_algorithms",
    "grpo", 
    "reinforce"
]