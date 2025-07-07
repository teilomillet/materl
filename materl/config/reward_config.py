# materl.config.reward_config

from dataclasses import dataclass, field
from typing import Dict, Any
from enum import Enum

class RewardFunctionType(str, Enum):
    """Specifies the implementation type of a reward function."""
    MOJO = "mojo"
    PYTHON = "python"

@dataclass
class RewardFunctionConfig:
    """Configuration for a single reward function."""
    name: str = field(metadata={"help": "Unique name for the reward function. If type is MOJO, this should match the kernel name."})
    type: RewardFunctionType = field(default=RewardFunctionType.PYTHON, metadata={"help": "Type of reward function implementation ('mojo' or 'python')."})
    weight: float = field(default=1.0, metadata={"help": "Weighting factor for this reward in the total reward calculation."})
    kwargs: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Dictionary of keyword arguments to pass to the reward function's constructor or call."}) 