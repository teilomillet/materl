"""
materl.recipes

This package contains the high-level recipes for different RL algorithms.
Each recipe defines a symbolic graph of operations that can be compiled
and executed by the materl engine.
"""
# Import all recipes in the package to make them accessible
from .decorator import recipe
from .grpo import grpo
from .dapo import dapo
from .vapo import vapo
from .reinforce import reinforce

# The GraphBuilder is now imported from its own module to avoid circular dependencies.
from .builder import GraphBuilder

__all__ = ["grpo", "dapo", "vapo", "reinforce", "recipe", "GraphBuilder"]

