# materl.config.grpo_config

from dataclasses import dataclass, field
from typing import List, Optional

from .reward_config import RewardFunctionConfig

# from transformers import TrainingArguments # Could potentially inherit later

@dataclass
class GRPOConfig:
    """
    Configuration class for GRPO-specific algorithm parameters.

    This class holds all hyperparameters necessary to configure the GRPO
    algorithm's loss function and advantage estimation.
    """
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Optional name or path of the base model, useful for logging and tracking."})

    # Core GRPO/PPO Parameters
    beta: float = field(default=0.04, metadata={"help": "KL coefficient for GRPO loss."})
    epsilon: float = field(default=0.2, metadata={"help": "Clipping parameter for the GRPO surrogate objective."})
    loss_type: str = field(default="bnpo", metadata={"help": "Type of loss function to use. Options: 'grpo', 'bnpo', 'dr_grpo'."})
    num_optimization_epochs: int = field(default=4, metadata={"help": "Number of optimization epochs to perform on each batch of generated data."})
    gamma: float = field(default=0.99, metadata={"help": "Discount factor for GAE."})
    lam: float = field(default=0.95, metadata={"help": "Lambda parameter for GAE."})
    clip_range_vf: Optional[float] = field(default=None, metadata={"help": "Clipping range for the value function loss."})
    vf_coef: float = field(default=0.1, metadata={"help": "Coefficient for the value function loss."})
    target_kl: Optional[float] = field(default=None, metadata={"help": "Target KL divergence for adaptive KL control."})

    # Materl-Specific Parameters for rewards and Mojo
    reward_function_configs: List[RewardFunctionConfig] = field(
        default_factory=list,
        metadata={"help": "List of RewardFunctionConfig objects defining the reward functions."}
    )
    scale_rewards: bool = field(default=True, metadata={"help": "If true, scale rewards by their standard deviation."})
    reward_baseline: float = field(default=0.0, metadata={"help": "A fixed baseline value to subtract from rewards."})
    mojo_kernel_library_path: Optional[str] = field(default=None, metadata={"help": "Path to custom Mojo kernels."})
    mojo_loss_kernel_name: Optional[str] = field(default=None, metadata={"help": "Name of the Mojo kernel for GRPO loss."})

    def __post_init__(self):
        if self.loss_type not in ["grpo", "bnpo", "dr_grpo"]:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}.")
        
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be > 0 if set.")
        if not (0 <= self.gamma <= 1):
            raise ValueError("gamma must be between 0 and 1.")
        if not (0 <= self.lam <= 1):
            raise ValueError("lam must be between 0 and 1.")

        print("materl.GRPOConfig initialized and validated.") 