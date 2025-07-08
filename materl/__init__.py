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
    # Separate generation config from other args
    gen_config = kwargs.pop("generation_config", None)
    
    graph = algorithm_func(**kwargs)
    compiled = compile(graph)
    
    # Pass config to the compiled graph runner
    if gen_config:
        kwargs["generation_config"] = gen_config
        
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