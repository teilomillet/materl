# materl.config.vapo_config

from dataclasses import dataclass, field
from typing import List, Optional

from .reward_config import RewardFunctionConfig

@dataclass
class VAPOConfig:
    """
    Configuration class for VAPO-specific algorithm parameters.
    VAPO (Value-based Augmented Proximal Policy Optimization) is a value-based
    RL algorithm for LLMs.
    """
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Optional name or path of the base model, useful for logging and tracking."})

    # Core VAPO/PPO Parameters
    beta: float = field(default=0.1, metadata={"help": "KL coefficient for VAPO loss."})
    epsilon: float = field(default=0.2, metadata={"help": "Clipping parameter for the VAPO surrogate objective."})
    num_optimization_epochs: int = field(default=4, metadata={"help": "Number of optimization epochs to perform on each batch of generated data."})
    target_kl: Optional[float] = field(default=None, metadata={"help": "Target KL divergence for adaptive KL control."})

    # GAE and Value Function Parameters
    gamma: float = field(default=1.0, metadata={"help": "Discount factor for GAE."})
    lam: float = field(default=0.95, metadata={"help": "Lambda parameter for GAE."})
    adaptive_lam: bool = field(default=True, metadata={"help": "Whether to use length-adaptive lambda for GAE."})
    clip_range_vf: Optional[float] = field(default=None, metadata={"help": "Clipping range for the value function loss."})
    vf_coef: float = field(default=0.1, metadata={"help": "Coefficient for the value function loss."})

    # Materl-Specific Parameters for rewards
    reward_function_configs: List[RewardFunctionConfig] = field(
        default_factory=list,
        metadata={"help": "List of RewardFunctionConfig objects defining the reward functions."}
    )
    scale_rewards: bool = field(default=False, metadata={"help": "If true, scale rewards by their standard deviation (not typical for GAE)."})
    reward_baseline: float = field(default=0.0, metadata={"help": "A fixed baseline value to subtract from rewards."})

    def __post_init__(self):
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be > 0 if set.")
        if not (0 <= self.gamma <= 1):
            raise ValueError("gamma must be between 0 and 1.")
        if not (0 <= self.lam <= 1) and not self.adaptive_lam:
            raise ValueError("lam must be between 0 and 1.")

        print("materl.VAPOConfig initialized and validated.") 