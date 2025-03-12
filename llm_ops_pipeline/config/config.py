"""Configuration management module for LLM Ops Pipeline."""

from typing import Dict, Any, Optional
import os
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for model settings."""
    
    model_name: str = Field(..., description="Name of the model to use")
    model_size: str = Field(..., description="Size/variant of the model")
    revision: Optional[str] = Field(None, description="Specific model revision")
    tokenizer: Optional[str] = Field(None, description="Tokenizer to use")
    max_length: int = Field(512, description="Maximum sequence length")
    

class TrainingConfig(BaseModel):
    """Configuration for training settings."""
    
    batch_size: int = Field(16, description="Batch size for training")
    learning_rate: float = Field(5e-5, description="Learning rate")
    num_epochs: int = Field(3, description="Number of training epochs")
    warmup_steps: int = Field(500, description="Warmup steps for lr scheduler")
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation steps")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm")
    fp16: bool = Field(False, description="Whether to use fp16 precision")
    seed: int = Field(42, description="Random seed")


class DataConfig(BaseModel):
    """Configuration for data settings."""
    
    train_path: str = Field(..., description="Path to training data")
    eval_path: str = Field(..., description="Path to evaluation data")
    test_path: Optional[str] = Field(None, description="Path to test data")
    data_cache_dir: Optional[str] = Field("./cache", description="Directory to cache processed data")


class Config(BaseModel):
    """Main configuration class."""
    
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    experiment_name: str = Field("default_experiment", description="Name of the experiment")
    output_dir: str = Field("./outputs", description="Directory to save outputs")
    logging_steps: int = Field(100, description="Log every X steps")
    eval_steps: int = Field(500, description="Evaluate every X steps")
    save_steps: int = Field(1000, description="Save checkpoint every X steps")
    track_with: Optional[str] = Field("wandb", description="Tracking tool (wandb, mlflow, etc.)")


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Config object with loaded configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    return Config(**config_dict)


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Config object to save
        config_path: Path to save the configuration YAML file
    """
    config_dict = config.dict()
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)