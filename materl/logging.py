# materl.logging

# This module provides a flexible logging system for the RLTrainer.
# By abstracting the logger, we can easily support different logging backends
# like Weights & Biases, TensorBoard, or a simple console logger. This aligns
# with the framework's core philosophy of composability.

from typing import Dict, Any, Optional

class WandbLogger:
    """
    A logger that sends metrics to Weights & Biases.

    This class handles the initialization of a wandb run and logs a dictionary
    of metrics at each logging step. It's designed to be injected into the
    RLTrainer.
    """
    def __init__(self, project_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the wandb run.

        Args:
            project_name: The name of the wandb project to log to.
            config: A dictionary of configuration parameters to save with the run.
        """
        self.project_name = project_name
        self.config = config
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            print("Warning: `wandb` is not installed. Please `pip install wandb` to use WandbLogger.")
            self._wandb = None

        if self._wandb:
            self._wandb.init(project=self.project_name, config=self.config)
            print(f"WandbLogger initialized. Logging to project: '{self.project_name}'")
        else:
            print("WandbLogger disabled because the `wandb` package is not found.")


    def log(self, metrics: Dict[str, Any], step: int):
        """
        Logs a dictionary of metrics to wandb.

        Args:
            metrics: A dictionary where keys are metric names and values are
                     the values to log.
            step: The current training step.
        """
        if self._wandb:
            self._wandb.log(metrics, step=step) 