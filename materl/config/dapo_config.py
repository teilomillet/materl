# materl.config.dapo_config

from dataclasses import dataclass, field
from typing import List, Optional

from .reward_config import RewardFunctionConfig

@dataclass
class DAPOConfig:
    """
    Configuration class for DAPO-specific algorithm parameters.
    DAPO (Direct Advantage Policy Optimization) is a value-free algorithm
    similar to GRPO.
    """
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Optional name or path of the base model, useful for logging and tracking."})

    # Core DAPO/PPO Parameters
    beta: float = field(default=0.04, metadata={"help": "KL coefficient for DAPO loss."})
    clip_ratio_lower: float = field(default=0.2, metadata={"help": "The lower bound for clipping the policy ratio."})
    clip_ratio_upper: float = field(default=0.2, metadata={"help": "The upper bound for clipping the policy ratio."})
    num_optimization_epochs: int = field(default=4, metadata={"help": "Number of optimization epochs to perform on each batch of generated data."})
    target_kl: Optional[float] = field(default=None, metadata={"help": "Target KL divergence for adaptive KL control."})

    # Materl-Specific Parameters for rewards
    reward_function_configs: List[RewardFunctionConfig] = field(
        default_factory=list,
        metadata={"help": "List of RewardFunctionConfig objects defining the reward functions."}
    )
    scale_rewards: bool = field(default=True, metadata={"help": "If true, scale rewards by their standard deviation."})
    reward_baseline: float = field(default=0.0, metadata={"help": "A fixed baseline value to subtract from rewards."})

    def __post_init__(self):
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be > 0 if set.")

        print("materl.DAPOConfig initialized and validated.") 