"""Tests for configuration module."""

import os
import tempfile
import pytest
import yaml

from llm_ops_pipeline.config.config import Config, ModelConfig, TrainingConfig, DataConfig, load_config, save_config


def test_config_creation():
    """Test creation of config objects."""
    # Create model config
    model_config = ModelConfig(
        model_name="gpt2",
        model_size="base",
        max_length=512
    )
    
    # Create training config
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=5e-5,
        num_epochs=3
    )
    
    # Create data config
    data_config = DataConfig(
        train_path="data/train.jsonl",
        eval_path="data/eval.jsonl",
        test_path="data/test.jsonl"
    )
    
    # Create main config
    config = Config(
        model=model_config,
        training=training_config,
        data=data_config,
        experiment_name="test_experiment"
    )
    
    # Check values
    assert config.model.model_name == "gpt2"
    assert config.training.batch_size == 16
    assert config.data.train_path == "data/train.jsonl"
    assert config.experiment_name == "test_experiment"


def test_config_save_load():
    """Test saving and loading configuration."""
    # Create a config
    config = Config(
        model=ModelConfig(
            model_name="gpt2",
            model_size="base"
        ),
        training=TrainingConfig(
            batch_size=16,
            learning_rate=5e-5
        ),
        data=DataConfig(
            train_path="data/train.jsonl",
            eval_path="data/eval.jsonl"
        )
    )
    
    # Save and load the config
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # Save config
        save_config(config, tmp_path)
        
        # Verify file exists
        assert os.path.exists(tmp_path)
        
        # Load config
        loaded_config = load_config(tmp_path)
        
        # Verify values
        assert loaded_config.model.model_name == "gpt2"
        assert loaded_config.training.batch_size == 16
        assert loaded_config.data.train_path == "data/train.jsonl"
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)