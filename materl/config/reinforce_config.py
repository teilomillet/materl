from dataclasses import dataclass, field

@dataclass
class REINFORCEConfig:
    """
    Configuration class for REINFORCE algorithm parameters.
    """
    gamma: float = field(
        default=0.99,
        metadata={"help": "Discount factor for future rewards. The Kimi-Researcher paper uses this to encourage shorter trajectories."}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "The learning rate for the optimizer."}
    )
    num_optimization_epochs: int = field(
        default=4,
        metadata={"help": "Number of optimization epochs to perform on each batch of generated data."}
    ) 