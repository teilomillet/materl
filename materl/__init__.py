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
# At runtime, the @recipe decorator will replace these with
# the methods from a GraphBuilder instance.
def generate(*args, **kwargs): pass
def logprobs(*args, **kwargs): pass
def advantages(*args, **kwargs): pass
def reward(*args, **kwargs): pass
def loss(*args, **kwargs): pass
def set_loss(*args, **kwargs): pass

__all__ = [
    "recipe",
    "algorithm",
    "op",
    "mojo_op",
    "Graph",
    "Agent",
    "generate",
    "logprobs",
    "advantages",
    "reward",
    "loss",
    "set_loss",
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
    TrainingConfig,
    GenerationConfig,
)

# Update __all__ to expose the public API.
__all__.extend([
    "config",
    "functions",
    "recipes",
    "compile",
    "DAPOConfig",
    "VAPOConfig",
    "GRPOConfig",
    "TrainingConfig",
    "GenerationConfig",
])