"""Training utilities for LLM Ops Pipeline."""

import os
from typing import Dict, Optional, Union, Any, List

import torch
from datasets import Dataset
from transformers import (
    Trainer, 
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    EarlyStoppingCallback
)
import wandb
import mlflow

from llm_ops_pipeline.config.config import TrainingConfig
from llm_ops_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def setup_tracking(tracking_tool: str, experiment_name: str, config: Dict[str, Any]) -> None:
    """Setup experiment tracking using the specified tool.
    
    Args:
        tracking_tool: Name of the tracking tool (wandb, mlflow)
        experiment_name: Name of the experiment
        config: Configuration dictionary to log
    """
    if tracking_tool.lower() == "wandb":
        wandb.init(
            project="llm_ops_pipeline",
            name=experiment_name,
            config=config
        )
        logger.info(f"Initialized wandb tracking for experiment {experiment_name}")
    
    elif tracking_tool.lower() == "mlflow":
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=experiment_name)
        mlflow.log_params(config)
        logger.info(f"Initialized mlflow tracking for experiment {experiment_name}")
    
    else:
        logger.warning(f"Unsupported tracking tool: {tracking_tool}, proceeding without tracking")


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    training_config: TrainingConfig = None,
    output_dir: str = "./outputs",
    experiment_name: str = "default_experiment",
    track_with: Optional[str] = "wandb",
    callbacks: List[Any] = None
) -> PreTrainedModel:
    """Train a model using the Hugging Face Trainer.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer to use
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        training_config: Training configuration
        output_dir: Directory to save outputs
        experiment_name: Name of the experiment
        track_with: Tracking tool to use (wandb, mlflow)
        callbacks: Optional list of callbacks
        
    Returns:
        Trained model
    """
    logger.info(f"Setting up training for experiment {experiment_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup tracking if specified
    if track_with:
        setup_tracking(
            track_with, 
            experiment_name,
            training_config.dict() if training_config else {}
        )
    
    # Set default training arguments
    if not training_config:
        training_config = TrainingConfig()
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        eval_steps=500,
        save_steps=1000,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=training_config.fp16,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        max_grad_norm=training_config.max_grad_norm,
        report_to=track_with if track_with else "none",
        seed=training_config.seed
    )
    
    # Setup default callbacks if none provided
    if callbacks is None:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    logger.info("Starting training")
    trainer.train()
    
    # Save model and tokenizer
    model_output_dir = os.path.join(output_dir, "final_model")
    os.makedirs(model_output_dir, exist_ok=True)
    
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    logger.info(f"Training completed, model saved to {model_output_dir}")
    
    # End tracking session if active
    if track_with == "wandb" and wandb.run is not None:
        wandb.finish()
    elif track_with == "mlflow":
        mlflow.end_run()
    
    return model