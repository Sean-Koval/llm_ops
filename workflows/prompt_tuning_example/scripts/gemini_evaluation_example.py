#!/usr/bin/env python
"""
Example script demonstrating how to use the Gemini MLflow integration with the evaluation framework.
Shows how to leverage MLflow autologging with Gemini models for comprehensive experiment tracking.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from llm_ops_pipeline.utils.logging import setup_logger
from llm_ops_pipeline.utils.gemini_mlflow import setup_gemini_autologging
from llm_ops_pipeline.evaluation.llm_evaluation_framework import LLMEvaluationFramework, calculate_classification_metrics

# Required for Gemini
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Set up logger
logger = setup_logger(name="gemini_evaluation_example")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            return yaml.safe_load(f)
        elif config_path.endswith(".json"):
            return json.load(f)
        else:
            logger.error(f"Unsupported configuration file format: {config_path}")
            sys.exit(1)

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from file."""
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    with open(dataset_path, "r") as f:
        if dataset_path.endswith(".json"):
            return json.load(f)
        else:
            logger.error(f"Unsupported dataset format: {dataset_path}")
            sys.exit(1)

def load_prompts(prompts_path: str) -> Dict[str, str]:
    """Load prompts from file."""
    if not os.path.exists(prompts_path):
        logger.error(f"Prompts file not found: {prompts_path}")
        sys.exit(1)
    
    with open(prompts_path, "r") as f:
        if prompts_path.endswith(".json"):
            return json.load(f)
        else:
            logger.error(f"Unsupported prompts format: {prompts_path}")
            sys.exit(1)

def gemini_inference_fn(prompt, example, model_id):
    """Inference function for Gemini models with autologging."""
    if not GEMINI_AVAILABLE:
        logger.error("Google Generative AI package not found. Please install with: pip install google-generativeai")
        sys.exit(1)
    
    # Get content field from example
    content_field = "text"  # Default field name
    content = example.get(content_field, "")
    
    # Create Gemini model - autologging will capture this activity
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
        
        prediction = response.text.strip()
        
        # Extract token usage if available
        token_usage = {}
        if hasattr(response, "usage_metadata"):
            token_usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0)
            }
        
        # Calculate latency (note: accurate latency tracking is already done by autologging)
        return {
            "prediction": prediction,
            "token_usage": token_usage
        }
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None

def evaluate_gemini_with_autologging(args):
    """Run evaluation with Gemini and MLflow autologging."""
    # Load configuration
    config = load_config(args.config)
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    logger.info(f"Loaded {len(dataset)} examples from dataset")
    
    # Load prompts
    prompts = load_prompts(args.prompts)
    logger.info(f"Loaded {len(prompts)} prompts for evaluation")
    
    # Initialize Gemini autologging
    gemini_logger = setup_gemini_autologging(
        api_key=args.api_key,
        project_id=args.project_id,
        location=args.location,
        tracking_uri=config.get("mlflow", {}).get("tracking_uri"),
        experiment_name=f"{config.get('experiment_name', 'gemini_evaluation')}",
        log_inputs=True,
        log_outputs=True,
        log_metrics=True,
        log_parameters=True,
        auto_end_run=False  # Let the evaluation framework handle runs
    )
    
    # Initialize evaluation framework
    eval_framework = LLMEvaluationFramework(
        config=config,
        experiment_name=f"{config.get('experiment_name', 'gemini_evaluation')}",
        workspace_dir=args.output_dir,
        use_mlflow=True,
        use_langfuse=args.use_langfuse,
        use_dvc=args.use_dvc
    )
    
    # Create metrics function
    def metrics_fn(examples, predictions):
        true_values = [ex.get(args.label_field, "") for ex in examples]
        return calculate_classification_metrics(true_values, predictions, labels=args.labels.split(",") if args.labels else None)
    
    # Create experiment
    experiment_id = eval_framework.create_experiment(
        name=f"gemini_{args.model_id}_evaluation",
        description=f"Evaluation of {args.model_id} with autologging",
        tags={
            "model_type": "gemini",
            "model_id": args.model_id,
            "dataset": args.dataset,
            "autologging": "enabled"
        }
    )
    
    # Run evaluation - this will use the autologging as the model is called
    logger.info(f"Starting prompt evaluation with model {args.model_id}")
    results = eval_framework.evaluate_prompts(
        task_id=f"gemini_prompt_evaluation",
        prompts=prompts,
        dataset=dataset[:args.sample_size] if args.sample_size else dataset,
        inference_fn=gemini_inference_fn,
        metrics_fn=metrics_fn,
        model_id=args.model_id,
        experiment_id=experiment_id
    )
    
    # Generate report
    report = eval_framework.generate_metrics_report(
        results,
        output_dir=os.path.join(args.output_dir, f"gemini_evaluation_{args.model_id}"),
        include_plots=True
    )
    
    # Find best prompt
    try:
        best_prompt_id, best_result = eval_framework.find_best_result(
            results,
            metric="f1_macro",
            higher_is_better=True
        )
        
        logger.info(f"Best prompt: {best_prompt_id}, Score: {best_result.metrics.get('f1_macro'):.4f}")
        
        # Register best prompt if using Langfuse
        if args.use_langfuse and eval_framework.prompt_manager:
            production_prompt_id = eval_framework.register_best_prompt(
                results,
                metric="f1_macro",
                higher_is_better=True,
                environment="development"
            )
            logger.info(f"Registered best prompt as {production_prompt_id}")
    except Exception as e:
        logger.warning(f"Failed to find best prompt: {e}")
    
    # Disable autologging when done
    gemini_logger.disable_autologging()

def main():
    parser = argparse.ArgumentParser(description="Gemini evaluation with MLflow autologging")
    parser.add_argument("--config", type=str, default="../configs/evaluation_config.yaml",
                        help="Path to evaluation configuration file")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset file (JSON)")
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to prompts file (JSON)")
    parser.add_argument("--model-id", type=str, default="gemini-1.0-pro",
                        help="Gemini model ID to evaluate")
    parser.add_argument("--api-key", type=str,
                        help="Google API key for Gemini API")
    parser.add_argument("--project-id", type=str,
                        help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="us-central1",
                        help="Google Cloud location")
    parser.add_argument("--label-field", type=str, default="label",
                        help="Field in dataset containing the label")
    parser.add_argument("--labels", type=str,
                        help="Comma-separated list of classification labels")
    parser.add_argument("--sample-size", type=int,
                        help="Number of examples to sample from the dataset")
    parser.add_argument("--output-dir", type=str, default="../results",
                        help="Directory for evaluation results")
    parser.add_argument("--use-langfuse", action="store_true",
                        help="Use Langfuse for prompt management")
    parser.add_argument("--use-dvc", action="store_true",
                        help="Use DVC for dataset versioning")
    
    args = parser.parse_args()
    
    # Check Gemini availability
    if not GEMINI_AVAILABLE:
        logger.error("Google Generative AI package not found. Please install with: pip install google-generativeai")
        sys.exit(1)
    
    # Run evaluation
    evaluate_gemini_with_autologging(args)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()