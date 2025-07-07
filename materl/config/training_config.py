# materl.config.training_config

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    """
    Configuration for the generic training loop.
    """
    # Core Training Parameters
    output_dir: str = field(default="./materl_output", metadata={"help": "Directory to save model checkpoints and logs."})
    num_epochs: int = field(default=1, metadata={"help": "Total number of training epochs."})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/CPU for training."})
    per_device_eval_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients before performing an optimizer step."})
    learning_rate: float = field(default=1e-5, metadata={"help": "Initial learning rate for the optimizer."})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "Max gradient norm for clipping, None to disable"})

    # Optimizer and Scheduler
    optimizer_type: str = field(default="adamw", metadata={"help": "Optimizer to use (e.g., 'adamw', 'sgd')."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type."})
    warmup_steps: int = field(default=0, metadata={"help": "Number of warmup steps for the learning rate scheduler."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay to apply."})

    # Logging and Saving
    logging_steps: int = field(default=10, metadata={"help": "Log training metrics every N steps."})
    save_steps: int = field(default=500, metadata={"help": "Save a model checkpoint every N steps."})
    use_wandb: bool = field(default=True, metadata={"help": "Whether to use Weights & Biases for logging."})
    wandb_project_name: str = field(default="materl-rl-run", metadata={"help": "W&B project name."})

    # Miscellaneous
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})
    shuffle_dataset: bool = field(default=True, metadata={"help": "Whether to shuffle the training dataset."}) 