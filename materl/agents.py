"""
materl.agents

This module will define the Agent class, which is a stateful wrapper
around a model (e.g., a Hugging Face model or a MAX engine).
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

class Agent:
    """
    A stateful wrapper for a model and its tokenizer.
    """
    def __init__(
        self, 
        model_name_or_path: str, 
        trainable: bool = False,
        device: Optional[str] = None
    ):
        self.name = model_name_or_path
        self.trainable = trainable
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            padding_side='left'
        )

        # Set pad token if it's not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        if device:
            self.model.to(device)
    
    def __repr__(self):
        return f"Agent(name='{self.name}', trainable={self.trainable}, device='{self.model.device}')" 