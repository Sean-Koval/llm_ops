#!/usr/bin/env python
"""Script for fine-tuning a language model."""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_ops_pipeline.config.config import load_config
from llm_ops_pipeline.data.dataset import load_data, preprocess_for_llm
from llm_ops_pipeline.model.model_loader import load_model_and_tokenizer
from llm_ops_pipeline.training.trainer import train_model
from llm_ops_pipeline.evaluation.metrics import compute_perplexity
from llm_ops_pipeline.utils.logging import get_logger

logger = get_logger("fine_tune")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the configuration file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory (overrides config if specified)"
    )
    
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default=None,
        help="Experiment name (overrides config if specified)"
    )
    
    return parser.parse_args()


def main():
    """Main function for fine-tuning a language model."""
    args = parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override output_dir and experiment_name if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model.model_name,
        model_type="causal_lm",  # Assuming causal language model for fine-tuning
        revision=config.model.revision
    )
    
    # Load datasets
    train_dataset = load_data(config.data.train_path, cache_dir=config.data.data_cache_dir)
    eval_dataset = load_data(config.data.eval_path, cache_dir=config.data.data_cache_dir)
    
    # Preprocess datasets
    train_dataset = preprocess_for_llm(
        train_dataset,
        tokenizer,
        max_length=config.model.max_length,
        is_train=True
    )
    
    eval_dataset = preprocess_for_llm(
        eval_dataset,
        tokenizer,
        max_length=config.model.max_length,
        is_train=False
    )
    
    # Fine-tune model
    logger.info(f"Starting fine-tuning for experiment {config.experiment_name}")
    model = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=config.training,
        output_dir=config.output_dir,
        experiment_name=config.experiment_name,
        track_with=config.track_with
    )
    
    # Compute final perplexity
    perplexity = compute_perplexity(model, eval_dataset)
    logger.info(f"Final evaluation perplexity: {perplexity:.4f}")
    
    logger.info(f"Fine-tuning complete. Model saved to {os.path.join(config.output_dir, 'final_model')}")


if __name__ == "__main__":
    main()