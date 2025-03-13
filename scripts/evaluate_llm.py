#!/usr/bin/env python
"""
CLI tool for running LLM evaluations using the evaluation framework.
Provides a simple interface for common evaluation tasks.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_ops_pipeline.evaluation.llm_evaluation_framework import (
    LLMEvaluationFramework, 
    calculate_classification_metrics,
    calculate_generation_metrics
)
from llm_ops_pipeline.utils.logging import setup_logger

# Set up logger
logger = setup_logger(name="evaluate_llm_cli")

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from file."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    if dataset_path.endswith(".json"):
        with open(dataset_path, "r") as f:
            return json.load(f)
    elif dataset_path.endswith(".csv"):
        import pandas as pd
        return pd.read_csv(dataset_path).to_dict("records")
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")

def load_prompts(prompts_path: str) -> Dict[str, str]:
    """Load prompts from file."""
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    if prompts_path.endswith(".json"):
        with open(prompts_path, "r") as f:
            data = json.load(f)
        
        # Handle different prompt formats
        if isinstance(data, dict):
            if all(isinstance(v, str) for v in data.values()):
                return data
            elif all(isinstance(v, dict) and "system_prompt" in v for v in data.values()):
                return {k: v["system_prompt"] for k, v in data.items()}
    
        raise ValueError("Unsupported prompts JSON format. Expected {prompt_id: prompt_text}")
    
    elif prompts_path.endswith(".yaml") or prompts_path.endswith(".yml"):
        with open(prompts_path, "r") as f:
            data = yaml.safe_load(f)
        
        if isinstance(data, dict):
            if all(isinstance(v, str) for v in data.values()):
                return data
            elif all(isinstance(v, dict) and "system_prompt" in v for v in data.values()):
                return {k: v["system_prompt"] for k, v in data.items()}
        
        raise ValueError("Unsupported prompts YAML format. Expected {prompt_id: prompt_text}")
    
    else:
        raise ValueError(f"Unsupported prompts file format: {prompts_path}")

def load_models(models_path: str) -> Dict[str, Dict[str, Any]]:
    """Load models configuration from file."""
    if not os.path.exists(models_path):
        raise FileNotFoundError(f"Models file not found: {models_path}")
    
    if models_path.endswith(".json"):
        with open(models_path, "r") as f:
            data = json.load(f)
    elif models_path.endswith(".yaml") or models_path.endswith(".yml"):
        with open(models_path, "r") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported models file format: {models_path}")
    
    # Validate models format
    if not isinstance(data, dict):
        raise ValueError("Models file must contain a dictionary")
    
    for model_id, model_config in data.items():
        if not isinstance(model_config, dict):
            data[model_id] = {"model_id": model_id, "config": {}}
    
    return data

def create_inference_function(args):
    """Create the appropriate inference function based on args."""
    if args.model_type == "openai":
        import openai
        
        def inference_fn(prompt, example, model_id):
            try:
                content_field = args.content_field or "text"
                content = example.get(content_field, "")
                
                response = openai.ChatCompletion.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": content}
                    ],
                    temperature=args.temperature
                )
                
                prediction = response.choices[0].message.content.strip()
                
                return {
                    "prediction": prediction,
                    "token_usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return None
        
        return inference_fn
    
    elif args.model_type == "vertexai":
        try:
            import google.cloud.aiplatform as vertexai
            from google.cloud.aiplatform.prediction import PredictionServiceClient
            
            # Initialize Vertex AI
            project_id = args.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = args.location or "us-central1"
            
            if not project_id:
                raise ValueError("Project ID not provided. Set --project-id or GOOGLE_CLOUD_PROJECT environment variable.")
            
            vertexai.init(project=project_id, location=location)
            
            def inference_fn(prompt, example, model_id):
                try:
                    content_field = args.content_field or "text"
                    content = example.get(content_field, "")
                    
                    model = vertexai.GenerativeModel(model_name=f"models/{model_id}")
                    
                    response = model.generate_content(
                        [prompt, content],
                        generation_config={
                            "temperature": args.temperature,
                            "max_output_tokens": args.max_tokens or 1024,
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
                    
                    return {
                        "prediction": prediction,
                        "token_usage": token_usage
                    }
                except Exception as e:
                    logger.error(f"Vertex AI API error: {e}")
                    return None
            
            return inference_fn
        except ImportError:
            logger.error("google-cloud-aiplatform not installed. Please install it with: pip install google-cloud-aiplatform")
            sys.exit(1)
    
    elif args.model_type == "api":
        import requests
        
        def inference_fn(prompt, example, model_id):
            try:
                content_field = args.content_field or "text"
                content = example.get(content_field, "")
                
                response = requests.post(
                    args.api_url,
                    json={
                        "model": model_id,
                        "prompt": prompt,
                        "input": content,
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens
                    },
                    headers={"Authorization": f"Bearer {args.api_key}"}
                )
                
                response.raise_for_status()
                result = response.json()
                
                prediction = result.get("prediction") or result.get("output") or result.get("text")
                
                # Extract token usage if available
                token_usage = result.get("token_usage") or result.get("usage") or {}
                
                return {
                    "prediction": prediction,
                    "token_usage": token_usage
                }
            except Exception as e:
                logger.error(f"API error: {e}")
                return None
        
        return inference_fn
    
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

def create_metrics_function(args):
    """Create the appropriate metrics function based on args."""
    if args.task_type == "classification":
        labels = None
        if args.labels:
            labels = args.labels.split(",")
        
        def metrics_fn(examples, predictions):
            true_values = [ex.get(args.label_field, "") for ex in examples]
            return calculate_classification_metrics(true_values, predictions, labels=labels)
        
        return metrics_fn
    
    elif args.task_type == "generation":
        def metrics_fn(examples, predictions):
            references = [ex.get(args.reference_field, "") for ex in examples]
            return calculate_generation_metrics(
                references, 
                predictions, 
                use_bleu=args.use_bleu, 
                use_rouge=args.use_rouge, 
                use_bert_score=args.use_bert_score
            )
        
        return metrics_fn
    
    elif args.task_type == "custom" and args.metrics_file:
        if not os.path.exists(args.metrics_file):
            raise FileNotFoundError(f"Metrics file not found: {args.metrics_file}")
        
        # Import custom metrics function
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_metrics", args.metrics_file)
        custom_metrics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_metrics_module)
        
        if not hasattr(custom_metrics_module, "calculate_metrics"):
            raise ValueError(f"Metrics file must define a calculate_metrics(examples, predictions) function")
        
        return custom_metrics_module.calculate_metrics
    
    else:
        raise ValueError(f"Unsupported task type: {args.task_type}")

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation CLI")
    
    # Basic configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--experiment-name", type=str, default="llm_evaluation", help="Name of the MLflow experiment")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory for evaluation results")
    
    # Evaluation type
    parser.add_argument("--mode", type=str, choices=["prompt", "model", "pipeline"], required=True, 
                        help="Evaluation mode: prompt (evaluate multiple prompts), model (evaluate multiple models), or pipeline (evaluate full pipeline)")
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file (JSON or CSV)")
    parser.add_argument("--sample-size", type=int, help="Number of examples to sample from the dataset")
    
    # Task type
    parser.add_argument("--task-type", type=str, choices=["classification", "generation", "custom"], default="classification",
                        help="Type of task: classification, generation, or custom")
    parser.add_argument("--content-field", type=str, default="text", help="Field in dataset containing the input text")
    parser.add_argument("--label-field", type=str, default="label", help="Field in dataset containing the label (for classification)")
    parser.add_argument("--reference-field", type=str, default="reference", help="Field in dataset containing the reference text (for generation)")
    parser.add_argument("--labels", type=str, help="Comma-separated list of classification labels")
    parser.add_argument("--metrics-file", type=str, help="Path to Python file with custom metrics function (for custom task type)")
    
    # Generation metrics options
    parser.add_argument("--use-bleu", action="store_true", help="Use BLEU score for generation tasks")
    parser.add_argument("--use-rouge", action="store_true", help="Use ROUGE score for generation tasks")
    parser.add_argument("--use-bert-score", action="store_true", help="Use BERTScore for generation tasks")
    
    # Prompts and models
    parser.add_argument("--prompts", type=str, help="Path to prompts file (JSON or YAML)")
    parser.add_argument("--models", type=str, help="Path to models file (JSON or YAML)")
    parser.add_argument("--prompt-id", type=str, help="ID of prompt to use (for model evaluation)")
    parser.add_argument("--model-id", type=str, help="ID of model to use (for prompt evaluation)")
    
    # Model configuration
    parser.add_argument("--model-type", type=str, choices=["openai", "vertexai", "api"], default="openai",
                        help="Type of model API to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for model generation")
    parser.add_argument("--max-tokens", type=int, help="Maximum number of tokens to generate")
    parser.add_argument("--api-url", type=str, help="URL for custom API endpoint")
    parser.add_argument("--api-key", type=str, help="API key for model API")
    parser.add_argument("--project-id", type=str, help="Google Cloud project ID (for VertexAI)")
    parser.add_argument("--location", type=str, help="Google Cloud location (for VertexAI)")
    
    # Tracking options
    parser.add_argument("--use-mlflow", action="store_true", help="Use MLflow for experiment tracking")
    parser.add_argument("--use-langfuse", action="store_true", help="Use Langfuse for prompt management")
    parser.add_argument("--use-dvc", action="store_true", help="Use DVC for dataset versioning")
    parser.add_argument("--enable-gemini-autologging", action="store_true", help="Enable Gemini autologging with MLflow")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration from file if provided
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        
        if args.config.endswith(".json"):
            with open(args.config, "r") as f:
                config = json.load(f)
        elif args.config.endswith(".yaml") or args.config.endswith(".yml"):
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
        else:
            logger.error(f"Unsupported configuration file format: {args.config}")
            sys.exit(1)
    
    # Set up evaluation framework
    try:
        eval_framework = LLMEvaluationFramework(
            config=config,
            experiment_name=args.experiment_name,
            workspace_dir=args.output_dir,
            use_mlflow=args.use_mlflow,
            use_langfuse=args.use_langfuse,
            use_dvc=args.use_dvc,
            enable_gemini_autologging=args.enable_gemini_autologging and args.model_type == "vertexai"
        )
    except Exception as e:
        logger.error(f"Failed to initialize evaluation framework: {e}")
        sys.exit(1)
    
    # Load dataset
    try:
        dataset = load_dataset(args.dataset)
        logger.info(f"Loaded {len(dataset)} examples from dataset")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Create inference function
    try:
        inference_fn = create_inference_function(args)
    except Exception as e:
        logger.error(f"Failed to create inference function: {e}")
        sys.exit(1)
    
    # Create metrics function
    try:
        metrics_fn = create_metrics_function(args)
    except Exception as e:
        logger.error(f"Failed to create metrics function: {e}")
        sys.exit(1)
    
    # Create experiment
    experiment_id = eval_framework.create_experiment(
        name=f"{args.experiment_name}_{args.mode}",
        description=f"{args.task_type.capitalize()} evaluation using {args.model_type} in {args.mode} mode",
        tags={
            "task_type": args.task_type,
            "model_type": args.model_type,
            "mode": args.mode,
            "dataset": os.path.basename(args.dataset)
        }
    )
    
    # Run appropriate evaluation based on mode
    if args.mode == "prompt":
        # Load prompts
        if not args.prompts:
            logger.error("Prompts file required for prompt evaluation mode")
            sys.exit(1)
        
        try:
            prompts = load_prompts(args.prompts)
            logger.info(f"Loaded {len(prompts)} prompts for evaluation")
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            sys.exit(1)
        
        if not args.model_id:
            logger.error("Model ID required for prompt evaluation mode")
            sys.exit(1)
        
        # Run prompt evaluation
        logger.info(f"Starting prompt evaluation with model {args.model_id}")
        results = eval_framework.evaluate_prompts(
            task_id=f"prompt_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            prompts=prompts,
            dataset=dataset[:args.sample_size] if args.sample_size else dataset,
            inference_fn=inference_fn,
            metrics_fn=metrics_fn,
            model_id=args.model_id,
            experiment_id=experiment_id
        )
        
        # Generate report
        report = eval_framework.generate_metrics_report(
            results,
            output_dir=os.path.join(args.output_dir, f"prompt_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            include_plots=True
        )
        
        # Find best prompt
        try:
            best_prompt_id, best_result = eval_framework.find_best_result(
                results,
                metric="f1_macro" if args.task_type == "classification" else "rouge1_f",
                higher_is_better=True
            )
            
            logger.info(f"Best prompt: {best_prompt_id}, Score: {best_result.metrics.get('f1_macro') or best_result.metrics.get('rouge1_f'):.4f}")
            
            # Register best prompt if using Langfuse
            if args.use_langfuse:
                production_prompt_id = eval_framework.register_best_prompt(
                    results,
                    metric="f1_macro" if args.task_type == "classification" else "rouge1_f",
                    higher_is_better=True,
                    environment="development"
                )
                logger.info(f"Registered best prompt as {production_prompt_id}")
        except Exception as e:
            logger.warning(f"Failed to find best prompt: {e}")
    
    elif args.mode == "model":
        # Check prompt ID
        if not args.prompt_id:
            logger.error("Prompt ID required for model evaluation mode")
            sys.exit(1)
        
        # Load models
        if not args.models:
            logger.error("Models file required for model evaluation mode")
            sys.exit(1)
        
        try:
            models = load_models(args.models)
            logger.info(f"Loaded {len(models)} models for evaluation")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            sys.exit(1)
        
        # Run model evaluation
        logger.info(f"Starting model evaluation with prompt {args.prompt_id}")
        results = eval_framework.evaluate_models(
            task_id=f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            prompt_id=args.prompt_id,
            models=models,
            dataset=dataset[:args.sample_size] if args.sample_size else dataset,
            inference_fn=inference_fn,
            metrics_fn=metrics_fn,
            experiment_id=experiment_id
        )
        
        # Generate report
        report = eval_framework.generate_metrics_report(
            results,
            output_dir=os.path.join(args.output_dir, f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            include_plots=True
        )
        
        # Find best model
        try:
            best_model_id, best_result = eval_framework.find_best_result(
                results,
                metric="f1_macro" if args.task_type == "classification" else "rouge1_f",
                higher_is_better=True
            )
            
            logger.info(f"Best model: {best_model_id}, Score: {best_result.metrics.get('f1_macro') or best_result.metrics.get('rouge1_f'):.4f}")
        except Exception as e:
            logger.warning(f"Failed to find best model: {e}")
    
    elif args.mode == "pipeline":
        # For pipeline mode, we need both prompt and model
        if not args.prompt_id:
            logger.error("Prompt ID required for pipeline evaluation mode")
            sys.exit(1)
        
        if not args.model_id:
            logger.error("Model ID required for pipeline evaluation mode")
            sys.exit(1)
        
        # Define pipeline function
        def pipeline_fn(example, config):
            prompt = config.get("prompt", "")
            model_id = config.get("model_id", args.model_id)
            return inference_fn(prompt, example, model_id)
        
        # Get prompt text
        prompt_text = ""
        if args.use_langfuse and eval_framework.prompt_manager:
            try:
                prompt_data = eval_framework.prompt_manager.get_prompt(args.prompt_id)
                prompt_text = prompt_data["content"]
            except Exception as e:
                logger.warning(f"Failed to get prompt from Langfuse: {e}")
                
                # Try to load from prompts file if provided
                if args.prompts:
                    try:
                        prompts = load_prompts(args.prompts)
                        if args.prompt_id in prompts:
                            prompt_text = prompts[args.prompt_id]
                    except Exception as e:
                        logger.warning(f"Failed to load prompt from file: {e}")
        elif args.prompts:
            try:
                prompts = load_prompts(args.prompts)
                if args.prompt_id in prompts:
                    prompt_text = prompts[args.prompt_id]
            except Exception as e:
                logger.warning(f"Failed to load prompt from file: {e}")
        
        if not prompt_text:
            logger.error(f"Could not find prompt text for {args.prompt_id}")
            sys.exit(1)
        
        # Run pipeline evaluation
        logger.info(f"Starting pipeline evaluation with prompt {args.prompt_id} and model {args.model_id}")
        result = eval_framework.evaluate_pipeline(
            task_id=f"pipeline_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pipeline_fn=pipeline_fn,
            dataset=dataset[:args.sample_size] if args.sample_size else dataset,
            metrics_fn=metrics_fn,
            pipeline_config={"prompt": prompt_text, "model_id": args.model_id},
            experiment_id=experiment_id
        )
        
        # Generate report
        report = eval_framework.generate_metrics_report(
            result,
            output_dir=os.path.join(args.output_dir, f"pipeline_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            include_plots=True
        )
        
        # Log results
        metric_name = "f1_macro" if args.task_type == "classification" else "rouge1_f"
        metric_value = result.metrics.get(metric_name)
        logger.info(f"Pipeline evaluation complete. {metric_name}: {metric_value:.4f}")
        logger.info(f"Average latency: {result.latency_ms:.2f} ms")
        
        # Log token usage
        for usage_type, usage_value in result.token_usage.items():
            logger.info(f"{usage_type}: {usage_value}")
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()