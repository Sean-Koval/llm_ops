"""
Prompt experimentation and A/B testing utilities.

This module provides tools for A/B testing prompt variants and tracking results in MLflow.
"""

import os
import json
import random
import hashlib
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
import time
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from llm_ops_pipeline.utils.prompt_management import PromptManager

logger = logging.getLogger(__name__)


class PromptExperiment:
    """
    Manages A/B testing of prompt variants with MLflow tracking.
    
    This class allows you to define multiple prompt variants and run experiments
    to compare their performance on the same inputs.
    """
    
    def __init__(
        self,
        experiment_name: str,
        prompt_manager: PromptManager,
        mlflow_tracking_uri: Optional[str] = None,
        default_metrics: Optional[List[str]] = None,
    ):
        """
        Initialize a prompt experiment.
        
        Args:
            experiment_name: Name of the MLflow experiment
            prompt_manager: PromptManager instance for prompt handling
            mlflow_tracking_uri: Optional MLflow tracking URI (uses env var if not provided)
            default_metrics: Optional list of default metrics to track
        """
        self.experiment_name = experiment_name
        self.prompt_manager = prompt_manager
        self.default_metrics = default_metrics or ["accuracy", "latency"]
        
        # Set up MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if not self.experiment:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow experiment: {e}")
            self.experiment_id = None
        
        # Initialize variants
        self.variants = {}
        self.variant_weights = {}
        
    def add_variant(self, 
                  name: str, 
                  prompt_id: str, 
                  weight: float = 1.0,
                  description: Optional[str] = None) -> None:
        """
        Add a prompt variant to the experiment.
        
        Args:
            name: Unique name for this variant
            prompt_id: Prompt ID as used in PromptManager
            weight: Optional sampling weight (for weighted random selection)
            description: Optional description of this variant
        """
        self.variants[name] = {
            "prompt_id": prompt_id,
            "description": description or f"Variant: {name}",
        }
        self.variant_weights[name] = weight
        
    def get_random_variant(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get a random variant based on weights.
        
        Returns:
            Tuple of (variant_name, variant_info)
        """
        if not self.variants:
            raise ValueError("No variants defined for this experiment")
        
        # Weighted random selection
        variant_names = list(self.variants.keys())
        weights = [self.variant_weights[name] for name in variant_names]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        selected_name = random.choices(variant_names, weights=weights, k=1)[0]
        return selected_name, self.variants[selected_name]
    
    def get_variant_by_name(self, name: str) -> Dict[str, Any]:
        """
        Get a specific variant by name.
        
        Args:
            name: Variant name
            
        Returns:
            Variant info dictionary
            
        Raises:
            ValueError: If variant not found
        """
        if name not in self.variants:
            raise ValueError(f"Variant '{name}' not found")
        
        return self.variants[name]
    
    def get_deterministic_variant(self, user_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Get a deterministic variant based on user ID.
        
        This ensures the same user always sees the same variant.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Tuple of (variant_name, variant_info)
        """
        if not self.variants:
            raise ValueError("No variants defined for this experiment")
        
        # Hash the user ID to get a deterministic selection
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        variant_names = list(self.variants.keys())
        selected_index = hash_val % len(variant_names)
        selected_name = variant_names[selected_index]
        
        return selected_name, self.variants[selected_name]
    
    def run_inference(self, 
                     variant_name: str, 
                     inputs: Dict[str, Any],
                     inference_fn: Callable,
                     **inference_kwargs) -> Dict[str, Any]:
        """
        Run inference with a specific prompt variant and track in MLflow.
        
        Args:
            variant_name: Name of the variant to use
            inputs: Input data for the inference
            inference_fn: Function that takes prompt and inputs and returns results
            **inference_kwargs: Additional keyword args for the inference function
            
        Returns:
            Dictionary with inference results and metadata
        """
        if variant_name not in self.variants:
            raise ValueError(f"Variant '{variant_name}' not found")
        
        variant = self.variants[variant_name]
        prompt_id = variant["prompt_id"]
        
        try:
            # Get the prompt from the manager
            prompt = self.prompt_manager.get_prompt(prompt_id)
            
            # Track start time for latency measurement
            start_time = time.time()
            
            # Run inference
            results = inference_fn(prompt, inputs, **inference_kwargs)
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Prepare metadata for logging
            metadata = {
                "variant": variant_name,
                "prompt_id": prompt_id,
                "prompt_version": prompt.get("metadata", {}).get("version", "unknown"),
                "latency": latency,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add metrics from results if they exist
            if isinstance(results, dict) and "metrics" in results:
                metadata["metrics"] = results["metrics"]
            
            # Log to prompt manager
            self.prompt_manager.log_prompt_usage(
                prompt_id=prompt_id,
                inputs=inputs,
                completion=results.get("completion", str(results)),
                metadata=metadata
            )
            
            # Combine results with metadata
            full_results = {
                "results": results,
                "metadata": metadata
            }
            
            return full_results
            
        except Exception as e:
            logger.error(f"Error running inference with variant {variant_name}: {e}")
            raise
    
    def run_ab_test(self,
                   inputs_batch: List[Dict[str, Any]],
                   inference_fn: Callable,
                   sampling_method: str = "random",
                   user_ids: Optional[List[str]] = None,
                   evaluation_fn: Optional[Callable] = None,
                   **inference_kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run an A/B test on a batch of inputs across multiple variants.
        
        Args:
            inputs_batch: List of input data dictionaries
            inference_fn: Function to run inference with a prompt and inputs
            sampling_method: How to assign variants ('random', 'deterministic', or 'all')
            user_ids: List of user IDs (required for deterministic assignment)
            evaluation_fn: Optional function to evaluate results and compute metrics
            **inference_kwargs: Additional keyword args for the inference function
            
        Returns:
            Dictionary mapping variant names to lists of results
        """
        # Validate arguments
        if sampling_method == "deterministic" and (not user_ids or len(user_ids) != len(inputs_batch)):
            raise ValueError("For deterministic sampling, user_ids must be provided and match inputs_batch length")
        
        all_results = {variant: [] for variant in self.variants}
        
        # Start MLflow run if available
        mlflow_run = None
        if self.experiment_id:
            try:
                mlflow_run = mlflow.start_run(experiment_id=self.experiment_id)
                
                # Log experiment parameters
                mlflow.log_param("sampling_method", sampling_method)
                mlflow.log_param("variants", list(self.variants.keys()))
                mlflow.log_param("batch_size", len(inputs_batch))
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}")
        
        try:
            # Process each input
            for i, inputs in enumerate(inputs_batch):
                # Select variant based on sampling method
                if sampling_method == "all":
                    # Run all variants for this input
                    for variant_name in self.variants:
                        result = self.run_inference(
                            variant_name=variant_name,
                            inputs=inputs,
                            inference_fn=inference_fn,
                            **inference_kwargs
                        )
                        result["input_index"] = i
                        all_results[variant_name].append(result)
                
                elif sampling_method == "deterministic":
                    # Select variant based on user ID
                    user_id = user_ids[i]
                    variant_name, _ = self.get_deterministic_variant(user_id)
                    result = self.run_inference(
                        variant_name=variant_name,
                        inputs=inputs,
                        inference_fn=inference_fn,
                        **inference_kwargs
                    )
                    result["input_index"] = i
                    result["user_id"] = user_id
                    all_results[variant_name].append(result)
                
                else:  # Default to random
                    # Select random variant
                    variant_name, _ = self.get_random_variant()
                    result = self.run_inference(
                        variant_name=variant_name,
                        inputs=inputs,
                        inference_fn=inference_fn,
                        **inference_kwargs
                    )
                    result["input_index"] = i
                    all_results[variant_name].append(result)
            
            # Run evaluation if provided
            if evaluation_fn:
                for variant_name, results in all_results.items():
                    metrics = evaluation_fn(results)
                    
                    # Log metrics to MLflow if available
                    if mlflow_run:
                        try:
                            for metric_name, metric_value in metrics.items():
                                mlflow.log_metric(f"{variant_name}_{metric_name}", metric_value)
                        except Exception as e:
                            logger.warning(f"Failed to log metrics to MLflow: {e}")
            
            return all_results
            
        finally:
            # End MLflow run if it was started
            if mlflow_run:
                try:
                    # Save results as artifact
                    with open("ab_test_results.json", "w") as f:
                        # Convert to serializable format
                        serializable_results = {}
                        for variant, results in all_results.items():
                            serializable_results[variant] = []
                            for result in results:
                                # Clean up any non-serializable items
                                clean_result = {
                                    "input_index": result.get("input_index"),
                                    "metadata": result.get("metadata", {}),
                                }
                                
                                # Add results, handling non-serializable objects
                                if "results" in result:
                                    r = result["results"]
                                    if isinstance(r, dict):
                                        clean_result["results"] = r
                                    else:
                                        clean_result["results"] = str(r)
                                
                                serializable_results[variant].append(clean_result)
                        
                        json.dump(serializable_results, f, indent=2, default=str)
                    
                    mlflow.log_artifact("ab_test_results.json")
                    mlflow.end_run()
                except Exception as e:
                    logger.warning(f"Error finalizing MLflow run: {e}")
    
    def analyze_experiment(self, min_samples: int = 10) -> Dict[str, Any]:
        """
        Analyze experiment results from MLflow.
        
        Args:
            min_samples: Minimum number of samples required for analysis
            
        Returns:
            Dictionary with analysis results
        """
        if not self.experiment_id:
            raise ValueError("MLflow experiment not initialized")
        
        client = MlflowClient()
        
        # Get all runs for this experiment
        runs = client.search_runs(experiment_ids=[self.experiment_id])
        
        if not runs:
            return {"status": "No runs found for this experiment"}
        
        # Group metrics by variant
        variant_metrics = {}
        
        for run in runs:
            metrics = run.data.metrics
            params = run.data.params
            
            for metric_name, metric_value in metrics.items():
                # Extract variant from metric name (format: variant_metric)
                if "_" in metric_name:
                    variant, base_metric = metric_name.split("_", 1)
                    
                    if variant not in variant_metrics:
                        variant_metrics[variant] = {}
                    
                    if base_metric not in variant_metrics[variant]:
                        variant_metrics[variant][base_metric] = []
                    
                    variant_metrics[variant][base_metric].append(metric_value)
        
        # Calculate statistics
        analysis = {"variants": {}}
        
        for variant, metrics in variant_metrics.items():
            analysis["variants"][variant] = {}
            
            for metric_name, values in metrics.items():
                if len(values) < min_samples:
                    continue
                
                import numpy as np
                
                analysis["variants"][variant][metric_name] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "samples": len(values)
                }
        
        # Determine best variant for each metric
        analysis["best_variants"] = {}
        
        for metric_name in self.default_metrics:
            variants_with_metric = []
            
            for variant, metrics in analysis["variants"].items():
                if metric_name in metrics:
                    variants_with_metric.append((variant, metrics[metric_name]["mean"]))
            
            if variants_with_metric:
                # For accuracy, higher is better
                if metric_name in ["accuracy", "precision", "recall", "f1"]:
                    best_variant = max(variants_with_metric, key=lambda x: x[1])
                # For latency and error rates, lower is better
                else:
                    best_variant = min(variants_with_metric, key=lambda x: x[1])
                
                analysis["best_variants"][metric_name] = {
                    "variant": best_variant[0],
                    "value": best_variant[1]
                }
        
        return analysis


def load_few_shot_examples(examples_path: str) -> List[Dict[str, Any]]:
    """
    Load few-shot examples from a JSON file.
    
    Args:
        examples_path: Path to the examples JSON file
        
    Returns:
        List of examples as dictionaries
    """
    if not os.path.exists(examples_path):
        raise ValueError(f"Examples file not found: {examples_path}")
    
    with open(examples_path, "r") as f:
        examples = json.load(f)
    
    return examples