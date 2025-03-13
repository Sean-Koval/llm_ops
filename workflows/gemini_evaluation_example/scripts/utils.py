"""
Utility functions for the Gemini evaluation workflow.
Contains helper functions for loading data, models, and reporting.
"""

import os
import sys
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import evaluation framework
from llm_ops_pipeline.utils.logging import setup_logger
from llm_ops_pipeline.utils.gemini_mlflow import setup_gemini_autologging
from llm_ops_pipeline.evaluation.llm_evaluation_framework import (
    LLMEvaluationFramework, 
    EvaluationResult,
    calculate_classification_metrics
)

# Set up logger
logger = setup_logger(name="gemini_eval_utils")

# Constants
DEFAULT_CONFIG_PATH = "../configs/evaluation_config.yaml"
DEFAULT_MODELS_CONFIG_PATH = "../configs/models_config.yaml"

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                return yaml.safe_load(f)
            elif config_path.endswith(".json"):
                return json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path}")
                return {}
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        return {}

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON file.
    
    Args:
        dataset_path: Path to dataset file
        
    Returns:
        List of examples
    """
    try:
        with open(dataset_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
        return []

def load_prompts(prompts_path: str) -> Dict[str, str]:
    """
    Load prompts from JSON file.
    
    Args:
        prompts_path: Path to prompts file
        
    Returns:
        Dictionary of prompt_id -> prompt_text
    """
    try:
        with open(prompts_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load prompts from {prompts_path}: {e}")
        return {}

def load_models_config(config_path: str = DEFAULT_MODELS_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load model configurations.
    
    Args:
        config_path: Path to models configuration file
        
    Returns:
        Dictionary of model_id -> model_config
    """
    return load_config(config_path)

def setup_gemini_evaluation(
    config_path: str = DEFAULT_CONFIG_PATH,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    use_mlflow: bool = True,
    use_langfuse: bool = False,
    use_dvc: bool = False
) -> tuple:
    """
    Set up the Gemini evaluation environment.
    
    Args:
        config_path: Path to configuration file
        api_key: Google API key for Gemini
        project_id: Google Cloud project ID
        location: Google Cloud location
        tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
        workspace_dir: Directory for evaluation artifacts
        use_mlflow: Whether to use MLflow
        use_langfuse: Whether to use Langfuse
        use_dvc: Whether to use DVC
        
    Returns:
        Tuple of (config, eval_framework, gemini_logger)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override configuration with provided values
    if tracking_uri:
        if "mlflow" not in config:
            config["mlflow"] = {}
        config["mlflow"]["tracking_uri"] = tracking_uri
    
    if experiment_name:
        config["experiment_name"] = experiment_name
        
    if workspace_dir:
        config["workspace_dir"] = workspace_dir
    
    # Set up Gemini autologging
    gemini_logger = None
    try:
        gemini_logger = setup_gemini_autologging(
            api_key=api_key,
            project_id=project_id,
            location=location,
            tracking_uri=config.get("mlflow", {}).get("tracking_uri"),
            experiment_name=config.get("experiment_name", "gemini_evaluation"),
            log_inputs=config.get("gemini", {}).get("log_inputs", True),
            log_outputs=config.get("gemini", {}).get("log_outputs", True),
            log_metrics=config.get("gemini", {}).get("log_metrics", True),
            log_parameters=config.get("gemini", {}).get("log_parameters", True),
            auto_end_run=False
        )
    except Exception as e:
        logger.warning(f"Failed to set up Gemini autologging: {e}")
    
    # Initialize evaluation framework
    try:
        eval_framework = LLMEvaluationFramework(
            config=config,
            experiment_name=config.get("experiment_name", "gemini_evaluation"),
            workspace_dir=config.get("workspace_dir", "../results"),
            use_mlflow=use_mlflow,
            use_langfuse=use_langfuse,
            use_dvc=use_dvc,
            enable_gemini_autologging=gemini_logger is not None
        )
    except Exception as e:
        logger.error(f"Failed to initialize evaluation framework: {e}")
        raise
    
    return config, eval_framework, gemini_logger

def create_sentiment_metrics_fn(label_field: str = "label", labels: Optional[List[str]] = None) -> Callable:
    """
    Create a metrics function for sentiment analysis.
    
    Args:
        label_field: Field in examples containing the true label
        labels: List of possible label values
        
    Returns:
        Metrics function for the evaluation framework
    """
    if labels is None:
        labels = ["positive", "negative", "neutral"]
    
    def metrics_fn(examples, predictions):
        true_values = [ex.get(label_field, "") for ex in examples]
        return calculate_classification_metrics(true_values, predictions, labels=labels)
    
    return metrics_fn

def create_entity_extraction_metrics_fn(entities_field: str = "entities") -> Callable:
    """
    Create a metrics function for entity extraction.
    
    Args:
        entities_field: Field in examples containing the true entities
        
    Returns:
        Metrics function for the evaluation framework
    """
    def metrics_fn(examples, predictions):
        metrics = {
            "entity_f1": 0.0,
            "entity_precision": 0.0,
            "entity_recall": 0.0,
            "entity_count_accuracy": 0.0
        }
        
        total_examples = len(examples)
        if total_examples == 0:
            return metrics
        
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_count_accuracy = 0.0
        
        for ex, pred in zip(examples, predictions):
            # Skip examples with missing predictions
            if pred is None:
                continue
                
            true_entities = ex.get(entities_field, {})
            
            # Convert prediction to dict if it's a string (JSON)
            if isinstance(pred, str):
                try:
                    pred_entities = json.loads(pred)
                except:
                    pred_entities = {}
            else:
                pred_entities = pred
            
            # Calculate entity-level metrics
            all_entity_types = set(true_entities.keys()) | set(pred_entities.keys())
            
            # Per example metrics
            example_precision = 0.0
            example_recall = 0.0
            example_f1 = 0.0
            type_count = 0
            
            for entity_type in all_entity_types:
                true_values = set(true_entities.get(entity_type, []))
                pred_values = set(pred_entities.get(entity_type, []))
                
                if not true_values and not pred_values:
                    continue
                
                # Calculate precision, recall, F1 for this entity type
                tp = len(true_values & pred_values)
                fp = len(pred_values - true_values)
                fn = len(true_values - pred_values)
                
                type_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                type_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0
                
                example_precision += type_precision
                example_recall += type_recall
                example_f1 += type_f1
                type_count += 1
            
            # Average across entity types
            if type_count > 0:
                example_precision /= type_count
                example_recall /= type_count
                example_f1 /= type_count
            
            # Add to totals
            total_precision += example_precision
            total_recall += example_recall
            total_f1 += example_f1
            
            # Entity count accuracy
            true_count = sum(len(entities) for entities in true_entities.values())
            pred_count = sum(len(entities) for entities in pred_entities.values())
            count_accuracy = 1.0 - min(abs(true_count - pred_count) / max(true_count, 1), 1.0)
            total_count_accuracy += count_accuracy
        
        # Calculate final metrics
        metrics["entity_precision"] = total_precision / total_examples
        metrics["entity_recall"] = total_recall / total_examples
        metrics["entity_f1"] = total_f1 / total_examples
        metrics["entity_count_accuracy"] = total_count_accuracy / total_examples
        
        return metrics
    
    return metrics_fn

def format_results_summary(results: Dict[str, EvaluationResult], primary_metric: str = "accuracy") -> str:
    """
    Generate a formatted summary of evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        primary_metric: Primary metric to display
        
    Returns:
        Formatted summary string
    """
    summary = ["Evaluation Results Summary:", ""]
    summary.append(f"Primary Metric: {primary_metric}")
    summary.append("-" * 60)
    summary.append(f"{'Model/Prompt':<25} | {primary_metric:<10} | {'Latency (ms)':<15} | {'Total Tokens':<15}")
    summary.append("-" * 60)
    
    # Sort results by primary metric
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1].metrics.get(primary_metric, 0), 
        reverse=True
    )
    
    for name, result in sorted_results:
        metric_value = result.metrics.get(primary_metric, 0)
        latency = result.latency_ms or 0
        total_tokens = result.token_usage.get("total_tokens", 0)
        
        summary.append(f"{name:<25} | {metric_value:<10.4f} | {latency:<15.2f} | {total_tokens:<15}")
    
    summary.append("-" * 60)
    summary.append("")
    
    # Add best result
    if sorted_results:
        best_name, best_result = sorted_results[0]
        summary.append(f"Best Result: {best_name}")
        summary.append(f"  {primary_metric}: {best_result.metrics.get(primary_metric, 0):.4f}")
        
        # Add token usage and cost details
        if best_result.token_usage:
            cost_estimate = 0.0
            if "cost_estimate_usd" in best_result.metrics:
                cost_estimate = best_result.metrics["cost_estimate_usd"]
            
            summary.append(f"  Token Usage:")
            summary.append(f"    Prompt Tokens: {best_result.token_usage.get('prompt_tokens', 0)}")
            summary.append(f"    Completion Tokens: {best_result.token_usage.get('completion_tokens', 0)}")
            summary.append(f"    Total Tokens: {best_result.token_usage.get('total_tokens', 0)}")
            summary.append(f"    Estimated Cost: ${cost_estimate:.6f}")
    
    return "\n".join(summary)