# materl.config.generation_config

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GenerationConfig:
    """
    Configuration for the text generation process.
    """
    # Backend Configuration
    backend: str = field(
        default="torch", 
        metadata={"help": "Backend for text generation: 'torch' (default), 'max', or 'graph'. MAX provides significant speedup."}
    )
    max_model_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Model path for MAX backend (GGUF format). Required when backend='max' or 'graph'."}
    )
    
    # Generation Control
    max_prompt_length: int = field(default=512, metadata={"help": "Maximum token length for input prompts."})
    max_completion_length: int = field(default=128, metadata={"help": "Maximum token length for generated completions."})
    max_seq_length: Optional[int] = field(default=None, metadata={"help": "Overall maximum sequence length (prompt + completion). If None, defaults to max_prompt_length + max_completion_length."})
    num_generations: int = field(default=8, metadata={"help": "Number of completions generated per prompt during training."})
    
    # Sampling Parameters
    temperature: float = field(default=1.0, metadata={"help": "Temperature for sampling during generation. 1.0 means no scaling."})
    top_p: Optional[float] = field(default=None, metadata={"help": "Nucleus sampling probability."})
    top_k: Optional[int] = field(default=None, metadata={"help": "Top-k filtering."})
    repetition_penalty: float = field(default=1.0, metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."})
    
    def __post_init__(self):
        if self.max_seq_length is None:
            self.max_seq_length = self.max_prompt_length + self.max_completion_length
        elif self.max_seq_length < self.max_prompt_length + self.max_completion_length:
            raise ValueError(f"max_seq_length ({self.max_seq_length}) must be >= max_prompt_length ({self.max_prompt_length}) + max_completion_length ({self.max_completion_length}).")
        
        # Validate MAX backend configuration
        if self.backend in ["max", "graph"] and not self.max_model_path:
            raise ValueError(f"max_model_path is required when using backend='{self.backend}'. Please provide path to GGUF model.") 