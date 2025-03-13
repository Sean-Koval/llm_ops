#!/usr/bin/env python
"""
Evaluation script for comparing different prompts with Gemini models.
Tests different prompt formulations against a dataset to find the most effective prompt.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Local imports
from utils import (
    load_config, 
    load_dataset, 
    load_prompts, 
    setup_gemini_evaluation,
    create_sentiment_metrics_fn,
    create_entity_extraction_metrics_fn,
    format_results_summary
)

# Required for Gemini
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from llm_ops_pipeline.utils.logging import setup_logger

# Set up logger
logger = setup_logger(name="evaluate_prompts")

def gemini_sentiment_inference(prompt: str, example: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """
    Inference function for sentiment classification with Gemini.
    
    Args:
        prompt: Prompt template
        example: Input example
        model_id: Gemini model ID
        
    Returns:
        Prediction with token usage
    """
    # Get text field from example
    content_field = "text"  # Default field name
    content = example.get(content_field, "")
    
    # Create Gemini model
    model = GenerativeModel(model_name=model_id)
    
    # Generate content
    try:
        # The autologging will automatically log this call to MLflow
        response = model.generate_content(
            [prompt, content],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "top_p": 0.95,
                "top_k": 40,
            }
        )
        
        prediction = response.text.strip().lower()
        
        # Normalize prediction
        if "positive" in prediction:
            prediction = "positive"
        elif "negative" in prediction:
            prediction = "negative"
        elif "neutral" in prediction:
            prediction = "neutral"
        
        # Extract token usage if available
        token_usage = {}
        if hasattr(response, "usage_metadata"):
            token_usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0)
            }
        
        return {
            "prediction": prediction,
            "token_usage": token_usage
        }
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None

def gemini_entity_extraction_inference(prompt: str, example: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """
    Inference function for entity extraction with Gemini.
    
    Args:
        prompt: Prompt template
        example: Input example
        model_id: Gemini model ID
        
    Returns:
        Prediction with token usage
    """
    # Get text field from example
    content_field = "text"  # Default field name
    content = example.get(content_field, "")
    
    # Create Gemini model
    model = GenerativeModel(model_name=model_id)
    
    # Generate content
    try:
        # The autologging will automatically log this call to MLflow
        response = model.generate_content(
            [prompt, content],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 2048,
                "top_p": 0.95,
                "top_k": 40,
            }
        )
        
        result_text = response.text.strip()
        
        # Try to parse as JSON
        try:
            result_json = json.loads(result_text)
            prediction = result_json
        except json.JSONDecodeError:
            # If not valid JSON, return the raw text
            prediction = result_text
        
        # Extract token usage if available
        token_usage = {}
        if hasattr(response, "usage_metadata"):
            token_usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0)
            }
        
        return {
            "prediction": prediction,
            "token_usage": token_usage
        }
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None

def evaluate_prompts(args):
    """Run the prompt evaluation."""
    # Check if Gemini is available
    if not GEMINI_AVAILABLE:
        logger.error("Google Generative AI package not found. Please install with: pip install google-generativeai")
        sys.exit(1)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    if not dataset:
        logger.error("Failed to load dataset")
        sys.exit(1)
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Load prompts
    logger.info(f"Loading prompts from {args.prompts}")
    prompts = load_prompts(args.prompts)
    if not prompts:
        logger.error("Failed to load prompts")
        sys.exit(1)
    logger.info(f"Loaded {len(prompts)} prompts: {list(prompts.keys())}")
    
    # Determine task type and set up appropriate inference function
    if args.task_type == "entity_extraction":
        inference_fn = gemini_entity_extraction_inference
        metrics_fn = create_entity_extraction_metrics_fn(entities_field=args.entities_field)
        primary_metric = args.primary_metric or "entity_f1"
    else:  # Default to sentiment classification
        inference_fn = gemini_sentiment_inference
        metrics_fn = create_sentiment_metrics_fn(label_field=args.label_field, labels=args.labels.split(",") if args.labels else None)
        primary_metric = args.primary_metric or "accuracy"
    
    # Set up evaluation framework
    logger.info("Setting up evaluation framework")
    config, eval_framework, gemini_logger = setup_gemini_evaluation(
        config_path=args.config,
        api_key=args.api_key,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name or f"gemini_prompt_evaluation_{args.model_id}",
        workspace_dir=args.output_dir,
        use_mlflow=not args.disable_mlflow,
        use_langfuse=args.use_langfuse,
        use_dvc=args.use_dvc
    )
    
    # Create experiment for tracking
    logger.info("Creating evaluation experiment")
    experiment_id = eval_framework.create_experiment(
        name=f"prompt_evaluation_{args.model_id}",
        description=f"Evaluation of different prompts with {args.model_id}",
        tags={
            "model_id": args.model_id,
            "dataset": args.dataset,
            "task_type": args.task_type,
            "primary_metric": primary_metric
        }
    )
    
    # Apply dataset sampling if specified
    sample_size = args.sample_size or config.get("evaluation", {}).get("sample_size")
    if sample_size and sample_size < len(dataset):
        import random
        random.seed(args.seed or 42)
        dataset = random.sample(dataset, sample_size)
        logger.info(f"Sampled {sample_size} examples from dataset")
    
    # Run evaluation
    logger.info(f"Starting prompt evaluation with model {args.model_id}")
    results = eval_framework.evaluate_prompts(
        task_id=f"{args.task_type}_prompt_evaluation",
        prompts=prompts,
        dataset=dataset,
        inference_fn=inference_fn,
        metrics_fn=metrics_fn,
        model_id=args.model_id,
        experiment_id=experiment_id,
        batch_size=args.batch_size or config.get("evaluation", {}).get("batch_size", 1)
    )
    
    # Generate report
    logger.info("Generating evaluation report")
    report = eval_framework.generate_metrics_report(
        results,
        output_dir=os.path.join(args.output_dir, f"prompt_evaluation_{args.model_id}"),
        include_plots=True
    )
    
    # Find best prompt
    try:
        best_prompt_id, best_result = eval_framework.find_best_result(
            results,
            metric=primary_metric,
            higher_is_better=args.higher_is_better if args.higher_is_better is not None else config.get("evaluation", {}).get("higher_is_better", True)
        )
        
        logger.info(f"Best prompt: {best_prompt_id}, Score: {best_result.metrics.get(primary_metric):.4f}")
        
        # Register best prompt if using Langfuse
        if args.use_langfuse and eval_framework.prompt_manager:
            production_prompt_id = eval_framework.register_best_prompt(
                results,
                metric=primary_metric,
                higher_is_better=args.higher_is_better if args.higher_is_better is not None else config.get("evaluation", {}).get("higher_is_better", True),
                environment="development"
            )
            logger.info(f"Registered best prompt as {production_prompt_id}")
    except Exception as e:
        logger.warning(f"Failed to find best prompt: {e}")
    
    # Print results summary
    summary = format_results_summary(results, primary_metric=primary_metric)
    print("\n" + summary)
    
    # Save summary to file
    summary_path = os.path.join(args.output_dir, f"prompt_evaluation_{args.model_id}", "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")
    
    # Disable autologging
    if gemini_logger:
        gemini_logger.disable_autologging()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate different prompts with Gemini models")
    
    # Configuration
    parser.add_argument("--config", type=str, default="../configs/evaluation_config.yaml",
                        help="Path to evaluation configuration file")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset file (JSON)")
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to prompts file (JSON)")
    
    # Gemini and model configuration
    parser.add_argument("--model-id", type=str, default="gemini-1.5-flash",
                        help="Gemini model ID to use")
    parser.add_argument("--api-key", type=str,
                        help="Google API key for Gemini API")
    
    # Experiment tracking
    parser.add_argument("--tracking-uri", type=str,
                        help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str,
                        help="MLflow experiment name")
    parser.add_argument("--disable-mlflow", action="store_true",
                        help="Disable MLflow tracking")
    parser.add_argument("--use-langfuse", action="store_true",
                        help="Use Langfuse for prompt management")
    parser.add_argument("--use-dvc", action="store_true",
                        help="Use DVC for dataset versioning")
    
    # Task configuration
    parser.add_argument("--task-type", type=str, default="sentiment",
                        choices=["sentiment", "entity_extraction"],
                        help="Type of evaluation task")
    parser.add_argument("--label-field", type=str, default="label",
                        help="Field in dataset containing the label (for sentiment)")
    parser.add_argument("--entities-field", type=str, default="entities",
                        help="Field in dataset containing the entities (for entity extraction)")
    parser.add_argument("--labels", type=str,
                        help="Comma-separated list of classification labels")
    
    # Evaluation parameters
    parser.add_argument("--sample-size", type=int,
                        help="Number of examples to sample from the dataset")
    parser.add_argument("--batch-size", type=int,
                        help="Batch size for inference")
    parser.add_argument("--primary-metric", type=str,
                        help="Primary metric to use for comparison")
    parser.add_argument("--higher-is-better", type=bool,
                        help="Whether higher metric values are better")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="../results",
                        help="Directory for evaluation results")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_prompts(args)

if __name__ == "__main__":
    main()