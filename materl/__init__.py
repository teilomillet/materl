"""
materl

A Declarative RL Algorithm Composition Framework.
"""

# Public API for the materl framework.

# Core Decorators
from .recipes import recipe
from .algorithm import algorithm
from .ops import op, mojo_op

# Core Classes
from .graph import Graph
from .agents import Agent

# Placeholder symbolic operations for static analysis.
# These are not the actual implementations but are used for type hinting
# and graph construction before the compiler replaces them.
from . import recipes

__all__ = [
    "Graph",
    "Agent",
    "recipes"
]

# materl Python Package

__version__ = "0.0.1.dev0"

# Import key modules and classes for easier access.
from . import config
from . import functions
from . import recipes
from .compiler import compile

# Direct access to primary classes
from .config import (
    DAPOConfig,
    VAPOConfig,
    GRPOConfig,
    GenerationConfig,
)

# Update __all__ to a more explicit and maintainable list.
__all__ = [
    "Agent",
    "Graph",
    "compile",
    "config",
    "functions",
    "recipes",
    "DAPOConfig",
    "VAPOConfig",
    "GRPOConfig",
    "GenerationConfig",
]