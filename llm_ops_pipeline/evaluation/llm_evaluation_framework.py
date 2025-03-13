"""
LLM Evaluation Framework - A comprehensive system for evaluating LLM performance,
tracking experiments, and optimizing prompts and models for specific use cases.
"""

import os
import json
import yaml
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
from langfuse import Langfuse
from langfuse.api.resources.experiments import Experiment
import dvc.api

# Import Gemini integration if available
try:
    from llm_ops_pipeline.utils.gemini_mlflow import setup_gemini_autologging, GeminiMLflowLogger
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from llm_ops_pipeline.utils.prompt_management import PromptManager
from llm_ops_pipeline.utils.logging import setup_logger
from llm_ops_pipeline.config.config import load_config

# Initialize logger
logger = setup_logger(name="llm_evaluation")

@dataclass
class EvaluationResult:
    """Data structure for storing evaluation results"""
    task_id: str
    prompt_id: str = None
    prompt_version: str = None
    model_id: str = None
    model_version: str = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_params: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = None
    latency_ms: float = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class LLMEvaluationFramework:
    """
    Comprehensive framework for evaluating LLM performance across different tasks,
    prompt variants, and models. Integrates with MLflow, Langfuse, and DVC for
    experiment tracking and versioning.
    """
    
    def __init__(
        self, 
        config_path: str = None,
        config: Dict[str, Any] = None,
        experiment_name: str = None,
        workspace_dir: str = None,
        use_mlflow: bool = True,
        use_langfuse: bool = True,
        use_dvc: bool = True,
        enable_gemini_autologging: bool = False,
    ):
        """
        Initialize the evaluation framework.
        
        Args:
            config_path: Path to the configuration file
            config: Configuration dictionary (alternative to config_path)
            experiment_name: Name of the experiment for tracking
            workspace_dir: Directory for storing evaluation artifacts
            use_mlflow: Whether to use MLflow for experiment tracking
            use_langfuse: Whether to use Langfuse for prompt management and tracking
            use_dvc: Whether to use DVC for dataset versioning
            enable_gemini_autologging: Whether to enable autologging for Gemini models
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = load_config(config_path)
        else:
            self.config = {}
        
        # Set up workspace
        self.workspace_dir = workspace_dir or os.path.join(os.getcwd(), "evaluation_workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Initialize experiment tracking
        self.use_mlflow = use_mlflow
        self.use_langfuse = use_langfuse
        self.use_dvc = use_dvc
        self.experiment_name = experiment_name or self.config.get("experiment_name", "llm_evaluation")
        
        # Set up MLflow
        if self.use_mlflow:
            mlflow_uri = self.config.get("mlflow", {}).get("tracking_uri")
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(self.experiment_name)
            
            # Setup Gemini autologging if requested
            self.gemini_autologging = None
            if enable_gemini_autologging and GEMINI_AVAILABLE:
                try:
                    # Get Gemini configuration
                    gemini_config = self.config.get("gemini", {})
                    api_key = gemini_config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
                    project_id = gemini_config.get("project_id") or os.environ.get("GOOGLE_CLOUD_PROJECT")
                    location = gemini_config.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
                    
                    self.gemini_autologging = setup_gemini_autologging(
                        api_key=api_key,
                        project_id=project_id,
                        location=location,
                        tracking_uri=mlflow_uri,
                        experiment_name=self.experiment_name,
                        log_inputs=gemini_config.get("log_inputs", True),
                        log_outputs=gemini_config.get("log_outputs", True),
                        log_metrics=gemini_config.get("log_metrics", True),
                        log_parameters=gemini_config.get("log_parameters", True),
                        auto_end_run=False  # Let framework handle runs
                    )
                    logger.info("Gemini autologging enabled")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini autologging: {e}")
                    self.gemini_autologging = None
        
        # Initialize Langfuse for prompt tracking
        if self.use_langfuse:
            langfuse_api_key = self.config.get("langfuse", {}).get("api_key") or os.environ.get("LANGFUSE_API_KEY")
            langfuse_secret_key = self.config.get("langfuse", {}).get("secret_key") or os.environ.get("LANGFUSE_SECRET_KEY")
            langfuse_host = self.config.get("langfuse", {}).get("host") or os.environ.get("LANGFUSE_HOST")
            
            if langfuse_api_key and langfuse_secret_key:
                self.langfuse = Langfuse(
                    api_key=langfuse_api_key,
                    secret_key=langfuse_secret_key,
                    host=langfuse_host
                )
                self.prompt_manager = PromptManager(
                    langfuse_api_key=langfuse_api_key,
                    langfuse_secret_key=langfuse_secret_key,
                    langfuse_host=langfuse_host,
                    environment=self.config.get("environment", "development")
                )
            else:
                logger.warning("Langfuse credentials not found. Langfuse features disabled.")
                self.langfuse = None
                self.prompt_manager = None
                self.use_langfuse = False
        else:
            self.langfuse = None
            self.prompt_manager = None
        
        # Set up DVC
        if self.use_dvc:
            try:
                self.dvc_repo_url = self.config.get("dvc", {}).get("repo") or os.getcwd()
            except Exception as e:
                logger.warning(f"DVC initialization failed: {e}. DVC features disabled.")
                self.use_dvc = False
        
        # Initialize experiment tracking state
        self.active_run = None
        self.experiments = {}
    
    def create_experiment(
        self, 
        name: str,
        description: str = None,
        tags: Dict[str, str] = None,
    ) -> str:
        """
        Create a new experiment for evaluation.
        
        Args:
            name: Name of the experiment
            description: Description of the experiment
            tags: Tags for the experiment
            
        Returns:
            experiment_id: ID of the created experiment
        """
        experiment_id = f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Create experiment directory
        experiment_dir = os.path.join(self.workspace_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create experiment metadata
        metadata = {
            "id": experiment_id,
            "name": name,
            "description": description or "",
            "tags": tags or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "runs": []
        }
        
        # Save metadata
        with open(os.path.join(experiment_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Track in Langfuse if available
        if self.use_langfuse and self.langfuse:
            try:
                langfuse_experiment = self.langfuse.experiments.create(
                    name=name,
                    description=description or ""
                )
                metadata["langfuse_id"] = langfuse_experiment.id
                
                # Update metadata with Langfuse ID
                with open(os.path.join(experiment_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to create Langfuse experiment: {e}")
        
        # Store experiment metadata
        self.experiments[experiment_id] = metadata
        
        return experiment_id
    
    def start_run(
        self,
        run_name: str = None,
        experiment_id: str = None,
        tags: Dict[str, str] = None,
        parameters: Dict[str, Any] = None,
    ) -> str:
        """
        Start a new evaluation run.
        
        Args:
            run_name: Name of the run
            experiment_id: ID of the experiment to associate with the run
            tags: Tags for the run
            parameters: Parameters for the run
            
        Returns:
            run_id: ID of the started run
        """
        # Generate run ID
        run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_name = run_name or run_id
        
        # Start MLflow run if enabled
        if self.use_mlflow:
            mlflow.start_run(run_name=run_name)
            
            # Log parameters
            if parameters:
                for key, value in parameters.items():
                    mlflow.log_param(key, value)
            
            # Log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
        
        # Create run directory
        if experiment_id and experiment_id in self.experiments:
            run_dir = os.path.join(self.workspace_dir, experiment_id, run_id)
        else:
            run_dir = os.path.join(self.workspace_dir, run_id)
        
        os.makedirs(run_dir, exist_ok=True)
        
        # Create run metadata
        metadata = {
            "id": run_id,
            "name": run_name,
            "experiment_id": experiment_id,
            "tags": tags or {},
            "parameters": parameters or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "running",
            "results": []
        }
        
        # Save metadata
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Track in Langfuse if available
        if self.use_langfuse and self.langfuse:
            try:
                trace = self.langfuse.trace(
                    name=run_name,
                    metadata={
                        "experiment_id": experiment_id,
                        **metadata["parameters"]
                    },
                    tags=metadata["tags"]
                )
                metadata["langfuse_trace_id"] = trace.id
                
                # Update metadata with Langfuse trace ID
                with open(os.path.join(run_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to create Langfuse trace: {e}")
        
        # Store active run
        self.active_run = {
            "id": run_id,
            "name": run_name,
            "experiment_id": experiment_id,
            "directory": run_dir,
            "metadata": metadata
        }
        
        # Update experiment if associated
        if experiment_id and experiment_id in self.experiments:
            self.experiments[experiment_id]["runs"].append(run_id)
            self.experiments[experiment_id]["updated_at"] = datetime.now().isoformat()
            
            # Save experiment metadata
            with open(os.path.join(self.workspace_dir, experiment_id, "metadata.json"), "w") as f:
                json.dump(self.experiments[experiment_id], f, indent=2)
        
        return run_id
    
    def end_run(self):
        """End the current evaluation run."""
        if not self.active_run:
            logger.warning("No active run to end")
            return
        
        # Update run status
        self.active_run["metadata"]["status"] = "completed"
        self.active_run["metadata"]["updated_at"] = datetime.now().isoformat()
        
        # Save metadata
        with open(os.path.join(self.active_run["directory"], "metadata.json"), "w") as f:
            json.dump(self.active_run["metadata"], f, indent=2)
        
        # End MLflow run
        if self.use_mlflow:
            mlflow.end_run()
        
        # Reset active run
        self.active_run = None
        
    def setup_gemini_logger(self, 
                           api_key: Optional[str] = None,
                           project_id: Optional[str] = None,
                           location: Optional[str] = None,
                           log_inputs: bool = True,
                           log_outputs: bool = True,
                           log_metrics: bool = True,
                           log_parameters: bool = True):
        """
        Setup Gemini autologging for integration with MLflow.
        
        Args:
            api_key: Google API key for Gemini API
            project_id: Google Cloud project ID
            location: Google Cloud location
            log_inputs: Whether to log model inputs
            log_outputs: Whether to log model outputs
            log_metrics: Whether to log metrics
            log_parameters: Whether to log model parameters
            
        Returns:
            True if setup was successful, False otherwise
        """
        if not GEMINI_AVAILABLE:
            logger.warning("Google Generative AI package not found. Please install with: pip install google-generativeai")
            return False
            
        try:
            # Get MLflow tracking URI
            mlflow_uri = self.config.get("mlflow", {}).get("tracking_uri")
            
            # Initialize Gemini autologging
            self.gemini_autologging = setup_gemini_autologging(
                api_key=api_key,
                project_id=project_id,
                location=location,
                tracking_uri=mlflow_uri,
                experiment_name=self.experiment_name,
                log_inputs=log_inputs,
                log_outputs=log_outputs,
                log_metrics=log_metrics,
                log_parameters=log_parameters,
                auto_end_run=False  # Let framework handle runs
            )
            
            logger.info("Gemini autologging configured successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to set up Gemini autologging: {e}")
            return False
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number for the metrics
        """
        if not self.active_run:
            logger.warning("No active run to log metrics for")
            return
        
        # Log to MLflow if enabled
        if self.use_mlflow:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        
        # Update run metadata
        if "metrics" not in self.active_run["metadata"]:
            self.active_run["metadata"]["metrics"] = {}
        
        for key, value in metrics.items():
            if key not in self.active_run["metadata"]["metrics"]:
                self.active_run["metadata"]["metrics"][key] = []
            
            self.active_run["metadata"]["metrics"][key].append({
                "value": value,
                "step": step,
                "timestamp": datetime.now().isoformat()
            })
        
        # Save metadata
        with open(os.path.join(self.active_run["directory"], "metadata.json"), "w") as f:
            json.dump(self.active_run["metadata"], f, indent=2)
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Log an artifact for the current run.
        
        Args:
            local_path: Path to the artifact
            artifact_path: Directory within the run's artifact directory to store the artifact
        """
        if not self.active_run:
            logger.warning("No active run to log artifact for")
            return
        
        # Log to MLflow if enabled
        if self.use_mlflow:
            mlflow.log_artifact(local_path, artifact_path)
        
        # Copy artifact to run directory
        import shutil
        
        if artifact_path:
            target_dir = os.path.join(self.active_run["directory"], "artifacts", artifact_path)
        else:
            target_dir = os.path.join(self.active_run["directory"], "artifacts")
        
        os.makedirs(target_dir, exist_ok=True)
        
        if os.path.isdir(local_path):
            shutil.copytree(local_path, os.path.join(target_dir, os.path.basename(local_path)))
        else:
            shutil.copy2(local_path, target_dir)
    
    def log_evaluation_result(self, result: EvaluationResult):
        """
        Log an evaluation result for the current run.
        
        Args:
            result: Evaluation result to log
        """
        if not self.active_run:
            logger.warning("No active run to log evaluation result for")
            return
        
        # Log metrics to MLflow if enabled
        if self.use_mlflow:
            # Log basic metrics
            for key, value in result.metrics.items():
                mlflow.log_metric(key, value)
            
            # Log token usage metrics
            for key, value in result.token_usage.items():
                mlflow.log_metric(f"token_usage_{key}", value)
            
            # Log latency
            if result.latency_ms:
                mlflow.log_metric("latency_ms", result.latency_ms)
            
            # Log parameters
            if result.prompt_id:
                mlflow.log_param("prompt_id", result.prompt_id)
            if result.prompt_version:
                mlflow.log_param("prompt_version", result.prompt_version)
            if result.model_id:
                mlflow.log_param("model_id", result.model_id)
            if result.model_version:
                mlflow.log_param("model_version", result.model_version)
            
            # Log result as JSON artifact
            result_dict = {k: v for k, v in result.__dict__.items()}
            with open("evaluation_result.json", "w") as f:
                json.dump(result_dict, f, indent=2)
            
            mlflow.log_artifact("evaluation_result.json")
        
        # Log to Langfuse if enabled and we have prompt information
        if self.use_langfuse and self.langfuse and result.prompt_id:
            try:
                # Create observation in the current trace
                if self.active_run.get("metadata", {}).get("langfuse_trace_id"):
                    trace_id = self.active_run["metadata"]["langfuse_trace_id"]
                    
                    # Create generation observation
                    self.langfuse.observations.create(
                        trace_id=trace_id,
                        type="generation",
                        name=f"{result.task_id}",
                        input=str(result.request_params),
                        output=str(result.examples),
                        metadata={
                            "prompt_id": result.prompt_id,
                            "prompt_version": result.prompt_version,
                            "model_id": result.model_id,
                            "metrics": result.metrics,
                            "token_usage": result.token_usage,
                            "latency_ms": result.latency_ms
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to log evaluation result to Langfuse: {e}")
        
        # Update run metadata
        if "results" not in self.active_run["metadata"]:
            self.active_run["metadata"]["results"] = []
        
        self.active_run["metadata"]["results"].append({
            "task_id": result.task_id,
            "prompt_id": result.prompt_id,
            "model_id": result.model_id,
            "metrics": result.metrics,
            "timestamp": result.timestamp
        })
        
        # Save metadata
        with open(os.path.join(self.active_run["directory"], "metadata.json"), "w") as f:
            json.dump(self.active_run["metadata"], f, indent=2)
        
        # Save full result as JSON
        result_dict = {k: v for k, v in result.__dict__.items()}
        with open(os.path.join(self.active_run["directory"], f"result_{result.task_id}.json"), "w") as f:
            json.dump(result_dict, f, indent=2)
    
    def load_dataset(
        self, 
        dataset_path: str,
        version: str = None,
        cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load dataset from path, with optional DVC version control.
        
        Args:
            dataset_path: Path to the dataset
            version: Version to load (for DVC)
            cache: Whether to cache the dataset
            
        Returns:
            dataset: Loaded dataset
        """
        # Check if dataset is DVC-tracked and we want a specific version
        if self.use_dvc and version:
            try:
                with dvc.api.open(
                    dataset_path,
                    repo=self.dvc_repo_url,
                    rev=version
                ) as f:
                    if dataset_path.endswith(".json"):
                        dataset = json.load(f)
                    elif dataset_path.endswith(".csv"):
                        import pandas as pd
                        dataset = pd.read_csv(f).to_dict("records")
                    else:
                        raise ValueError(f"Unsupported dataset format: {dataset_path}")
            except Exception as e:
                logger.warning(f"Failed to load dataset from DVC: {e}. Falling back to local file.")
                dataset = self._load_local_dataset(dataset_path)
        else:
            dataset = self._load_local_dataset(dataset_path)
        
        return dataset
    
    def _load_local_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset from local file."""
        if dataset_path.endswith(".json"):
            with open(dataset_path, "r") as f:
                return json.load(f)
        elif dataset_path.endswith(".csv"):
            import pandas as pd
            return pd.read_csv(dataset_path).to_dict("records")
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    def evaluate_prompts(
        self,
        task_id: str,
        prompts: Dict[str, str],
        dataset: List[Dict[str, Any]], 
        inference_fn: Callable,
        metrics_fn: Callable,
        model_id: str = None,
        sample_size: int = None,
        batch_size: int = 1,
        experiment_id: str = None,
        run_name: str = None,
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate multiple prompts on a dataset.
        
        Args:
            task_id: ID of the evaluation task
            prompts: Dictionary of prompt_id -> prompt_text
            dataset: Evaluation dataset
            inference_fn: Function that takes (prompt, example, model_id) and returns prediction
            metrics_fn: Function that takes (examples, predictions) and returns metrics dictionary
            model_id: ID of the model to use
            sample_size: Number of examples to sample from the dataset (if None, use all)
            batch_size: Batch size for inference
            experiment_id: ID of the experiment to associate with the evaluation
            run_name: Name of the run
            
        Returns:
            Dictionary of prompt_id -> evaluation result
        """
        # Sample dataset if needed
        if sample_size and sample_size < len(dataset):
            import random
            dataset_sample = random.sample(dataset, sample_size)
        else:
            dataset_sample = dataset
        
        # Start run if not already started
        if not self.active_run:
            self.start_run(
                run_name=run_name or f"prompt_evaluation_{task_id}",
                experiment_id=experiment_id,
                parameters={
                    "task_id": task_id,
                    "model_id": model_id,
                    "dataset_size": len(dataset_sample),
                    "num_prompts": len(prompts)
                }
            )
        
        results = {}
        
        # Evaluate each prompt
        for prompt_id, prompt_text in prompts.items():
            logger.info(f"Evaluating prompt: {prompt_id}")
            
            # Process examples
            predictions = []
            latencies = []
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            for i in range(0, len(dataset_sample), batch_size):
                batch = dataset_sample[i:i+batch_size]
                
                for example in batch:
                    start_time = time.time()
                    try:
                        prediction = inference_fn(prompt_text, example, model_id)
                        
                        # Record latency
                        latency_ms = (time.time() - start_time) * 1000
                        latencies.append(latency_ms)
                        
                        # Handle token usage if provided
                        if isinstance(prediction, dict) and "token_usage" in prediction:
                            for k, v in prediction["token_usage"].items():
                                token_usage[k] = token_usage.get(k, 0) + v
                            
                            # Extract actual prediction if nested
                            if "prediction" in prediction:
                                prediction = prediction["prediction"]
                        
                        predictions.append({
                            "example": example,
                            "prediction": prediction,
                            "latency_ms": latency_ms
                        })
                    except Exception as e:
                        logger.error(f"Error processing example {i}: {str(e)}")
                        predictions.append({
                            "example": example,
                            "prediction": None,
                            "error": str(e),
                            "latency_ms": (time.time() - start_time) * 1000
                        })
            
            # Calculate metrics
            metrics = metrics_fn([p["example"] for p in predictions], [p["prediction"] for p in predictions])
            
            # Create evaluation result
            result = EvaluationResult(
                task_id=task_id,
                prompt_id=prompt_id,
                model_id=model_id,
                metrics=metrics,
                latency_ms=sum(latencies) / len(latencies) if latencies else None,
                token_usage=token_usage,
                examples=predictions,
                request_params={"prompt": prompt_text}
            )
            
            # Log evaluation result
            self.log_evaluation_result(result)
            
            # Store result
            results[prompt_id] = result
        
        # End run if we started it
        if not self.active_run or self.active_run.get("name") == run_name or self.active_run.get("name") == f"prompt_evaluation_{task_id}":
            self.end_run()
        
        return results
    
    def evaluate_models(
        self,
        task_id: str,
        prompt_id: str,
        models: Dict[str, Any],
        dataset: List[Dict[str, Any]],
        inference_fn: Callable,
        metrics_fn: Callable,
        sample_size: int = None,
        batch_size: int = 1,
        experiment_id: str = None,
        run_name: str = None,
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate multiple models on a dataset with the same prompt.
        
        Args:
            task_id: ID of the evaluation task
            prompt_id: ID of the prompt to use
            models: Dictionary of model_id -> model_config
            dataset: Evaluation dataset
            inference_fn: Function that takes (prompt, example, model_id) and returns prediction
            metrics_fn: Function that takes (examples, predictions) and returns metrics dictionary
            sample_size: Number of examples to sample from the dataset (if None, use all)
            batch_size: Batch size for inference
            experiment_id: ID of the experiment to associate with the evaluation
            run_name: Name of the run
            
        Returns:
            Dictionary of model_id -> evaluation result
        """
        # Get prompt text from prompt manager or directly
        if self.prompt_manager:
            try:
                prompt_data = self.prompt_manager.get_prompt(prompt_id)
                prompt_text = prompt_data["content"]
                prompt_version = prompt_data.get("metadata", {}).get("version", "unknown")
            except Exception as e:
                logger.warning(f"Failed to get prompt from prompt manager: {e}")
                prompt_text = prompt_id
                prompt_version = "unknown"
        else:
            prompt_text = prompt_id
            prompt_version = "unknown"
        
        # Sample dataset if needed
        if sample_size and sample_size < len(dataset):
            import random
            dataset_sample = random.sample(dataset, sample_size)
        else:
            dataset_sample = dataset
        
        # Start run if not already started
        if not self.active_run:
            self.start_run(
                run_name=run_name or f"model_evaluation_{task_id}",
                experiment_id=experiment_id,
                parameters={
                    "task_id": task_id,
                    "prompt_id": prompt_id,
                    "prompt_version": prompt_version,
                    "dataset_size": len(dataset_sample),
                    "num_models": len(models)
                }
            )
        
        results = {}
        
        # Evaluate each model
        for model_id, model_config in models.items():
            logger.info(f"Evaluating model: {model_id}")
            
            # Process examples
            predictions = []
            latencies = []
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            for i in range(0, len(dataset_sample), batch_size):
                batch = dataset_sample[i:i+batch_size]
                
                for example in batch:
                    start_time = time.time()
                    try:
                        prediction = inference_fn(prompt_text, example, model_id)
                        
                        # Record latency
                        latency_ms = (time.time() - start_time) * 1000
                        latencies.append(latency_ms)
                        
                        # Handle token usage if provided
                        if isinstance(prediction, dict) and "token_usage" in prediction:
                            for k, v in prediction["token_usage"].items():
                                token_usage[k] = token_usage.get(k, 0) + v
                            
                            # Extract actual prediction if nested
                            if "prediction" in prediction:
                                prediction = prediction["prediction"]
                        
                        predictions.append({
                            "example": example,
                            "prediction": prediction,
                            "latency_ms": latency_ms
                        })
                    except Exception as e:
                        logger.error(f"Error processing example {i}: {str(e)}")
                        predictions.append({
                            "example": example,
                            "prediction": None,
                            "error": str(e),
                            "latency_ms": (time.time() - start_time) * 1000
                        })
            
            # Calculate metrics
            metrics = metrics_fn([p["example"] for p in predictions], [p["prediction"] for p in predictions])
            
            # Create evaluation result
            result = EvaluationResult(
                task_id=task_id,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                model_id=model_id,
                metrics=metrics,
                latency_ms=sum(latencies) / len(latencies) if latencies else None,
                token_usage=token_usage,
                examples=predictions,
                request_params={"prompt": prompt_text, "model_config": model_config}
            )
            
            # Log evaluation result
            self.log_evaluation_result(result)
            
            # Store result
            results[model_id] = result
        
        # End run if we started it
        if not self.active_run or self.active_run.get("name") == run_name or self.active_run.get("name") == f"model_evaluation_{task_id}":
            self.end_run()
        
        return results
    
    def evaluate_pipeline(
        self,
        task_id: str,
        pipeline_fn: Callable,
        dataset: List[Dict[str, Any]],
        metrics_fn: Callable,
        pipeline_config: Dict[str, Any] = None,
        sample_size: int = None,
        batch_size: int = 1,
        experiment_id: str = None,
        run_name: str = None,
    ) -> EvaluationResult:
        """
        Evaluate an entire prediction pipeline on a dataset.
        
        Args:
            task_id: ID of the evaluation task
            pipeline_fn: Function that takes (example, config) and returns prediction
            dataset: Evaluation dataset
            metrics_fn: Function that takes (examples, predictions) and returns metrics dictionary
            pipeline_config: Configuration for the pipeline
            sample_size: Number of examples to sample from the dataset (if None, use all)
            batch_size: Batch size for inference
            experiment_id: ID of the experiment to associate with the evaluation
            run_name: Name of the run
            
        Returns:
            Evaluation result
        """
        # Sample dataset if needed
        if sample_size and sample_size < len(dataset):
            import random
            dataset_sample = random.sample(dataset, sample_size)
        else:
            dataset_sample = dataset
        
        # Start run if not already started
        if not self.active_run:
            self.start_run(
                run_name=run_name or f"pipeline_evaluation_{task_id}",
                experiment_id=experiment_id,
                parameters={
                    "task_id": task_id,
                    "dataset_size": len(dataset_sample),
                    "pipeline_config": pipeline_config or {}
                }
            )
        
        # Process examples
        predictions = []
        latencies = []
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        for i in range(0, len(dataset_sample), batch_size):
            batch = dataset_sample[i:i+batch_size]
            
            for example in batch:
                start_time = time.time()
                try:
                    prediction = pipeline_fn(example, pipeline_config)
                    
                    # Record latency
                    latency_ms = (time.time() - start_time) * 1000
                    latencies.append(latency_ms)
                    
                    # Handle token usage if provided
                    if isinstance(prediction, dict) and "token_usage" in prediction:
                        for k, v in prediction["token_usage"].items():
                            token_usage[k] = token_usage.get(k, 0) + v
                        
                        # Extract actual prediction if nested
                        if "prediction" in prediction:
                            prediction = prediction["prediction"]
                    
                    predictions.append({
                        "example": example,
                        "prediction": prediction,
                        "latency_ms": latency_ms
                    })
                except Exception as e:
                    logger.error(f"Error processing example {i}: {str(e)}")
                    predictions.append({
                        "example": example,
                        "prediction": None,
                        "error": str(e),
                        "latency_ms": (time.time() - start_time) * 1000
                    })
        
        # Calculate metrics
        metrics = metrics_fn([p["example"] for p in predictions], [p["prediction"] for p in predictions])
        
        # Create evaluation result
        result = EvaluationResult(
            task_id=task_id,
            metrics=metrics,
            latency_ms=sum(latencies) / len(latencies) if latencies else None,
            token_usage=token_usage,
            examples=predictions,
            request_params={"pipeline_config": pipeline_config}
        )
        
        # Log evaluation result
        self.log_evaluation_result(result)
        
        # End run if we started it
        if not self.active_run or self.active_run.get("name") == run_name or self.active_run.get("name") == f"pipeline_evaluation_{task_id}":
            self.end_run()
        
        return result
    
    def generate_metrics_report(
        self,
        evaluation_results: Union[EvaluationResult, Dict[str, EvaluationResult], List[EvaluationResult]],
        output_dir: str = None,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics report from evaluation results.
        
        Args:
            evaluation_results: Evaluation result(s) to report on
            output_dir: Directory to save the report and plots
            include_plots: Whether to include plots in the report
            
        Returns:
            Report data
        """
        # Handle different input types
        if isinstance(evaluation_results, EvaluationResult):
            results = {"default": evaluation_results}
        elif isinstance(evaluation_results, dict):
            results = evaluation_results
        elif isinstance(evaluation_results, list):
            results = {f"result_{i}": result for i, result in enumerate(evaluation_results)}
        else:
            raise ValueError(f"Unsupported evaluation_results type: {type(evaluation_results)}")
        
        # Create report directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.join(self.workspace_dir, "reports", f"report_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize report
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": {},
            "latency_summary": {},
            "token_usage_summary": {},
            "result_comparison": {}
        }
        
        # Process each result
        for result_id, result in results.items():
            # Add to metrics summary
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in report["metrics_summary"]:
                    report["metrics_summary"][metric_name] = {}
                report["metrics_summary"][metric_name][result_id] = metric_value
            
            # Add to latency summary
            if result.latency_ms:
                report["latency_summary"][result_id] = result.latency_ms
            
            # Add to token usage summary
            for usage_type, usage_value in result.token_usage.items():
                if usage_type not in report["token_usage_summary"]:
                    report["token_usage_summary"][usage_type] = {}
                report["token_usage_summary"][usage_type][result_id] = usage_value
            
            # Add to result comparison
            report["result_comparison"][result_id] = {
                "metrics": result.metrics,
                "latency_ms": result.latency_ms,
                "token_usage": result.token_usage,
                "prompt_id": result.prompt_id,
                "model_id": result.model_id
            }
        
        # Generate plots if requested
        if include_plots:
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Metrics comparison plot
            for metric_name, metric_values in report["metrics_summary"].items():
                plt.figure(figsize=(10, 6))
                plt.bar(metric_values.keys(), metric_values.values())
                plt.title(f"{metric_name} Comparison")
                plt.xlabel("Result ID")
                plt.ylabel(metric_name)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{metric_name}_comparison.png"))
                plt.close()
            
            # Latency comparison plot
            if report["latency_summary"]:
                plt.figure(figsize=(10, 6))
                plt.bar(report["latency_summary"].keys(), report["latency_summary"].values())
                plt.title("Latency Comparison")
                plt.xlabel("Result ID")
                plt.ylabel("Latency (ms)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "latency_comparison.png"))
                plt.close()
            
            # Token usage comparison plot
            for usage_type, usage_values in report["token_usage_summary"].items():
                plt.figure(figsize=(10, 6))
                plt.bar(usage_values.keys(), usage_values.values())
                plt.title(f"{usage_type} Comparison")
                plt.xlabel("Result ID")
                plt.ylabel(f"{usage_type}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{usage_type}_comparison.png"))
                plt.close()
        
        # Save report as JSON
        with open(os.path.join(output_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        
        # Save report as markdown
        with open(os.path.join(output_dir, "report.md"), "w") as f:
            f.write(f"# Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Metrics Summary\n\n")
            for metric_name, metric_values in report["metrics_summary"].items():
                f.write(f"### {metric_name}\n\n")
                f.write("| Result ID | Value |\n")
                f.write("| --- | --- |\n")
                for result_id, value in metric_values.items():
                    f.write(f"| {result_id} | {value:.4f} |\n")
                f.write("\n")
            
            f.write(f"## Latency Summary\n\n")
            f.write("| Result ID | Latency (ms) |\n")
            f.write("| --- | --- |\n")
            for result_id, latency in report["latency_summary"].items():
                f.write(f"| {result_id} | {latency:.2f} |\n")
            f.write("\n")
            
            f.write(f"## Token Usage Summary\n\n")
            for usage_type, usage_values in report["token_usage_summary"].items():
                f.write(f"### {usage_type}\n\n")
                f.write("| Result ID | Token Count |\n")
                f.write("| --- | --- |\n")
                for result_id, value in usage_values.items():
                    f.write(f"| {result_id} | {value} |\n")
                f.write("\n")
            
            if include_plots:
                f.write(f"## Plots\n\n")
                for metric_name in report["metrics_summary"].keys():
                    f.write(f"### {metric_name} Comparison\n\n")
                    f.write(f"![{metric_name} Comparison](plots/{metric_name}_comparison.png)\n\n")
                
                if report["latency_summary"]:
                    f.write(f"### Latency Comparison\n\n")
                    f.write(f"![Latency Comparison](plots/latency_comparison.png)\n\n")
                
                for usage_type in report["token_usage_summary"].keys():
                    f.write(f"### {usage_type} Comparison\n\n")
                    f.write(f"![{usage_type} Comparison](plots/{usage_type}_comparison.png)\n\n")
        
        return report

    def find_best_result(
        self,
        results: Dict[str, EvaluationResult],
        metric: str = "accuracy",
        higher_is_better: bool = True
    ) -> Tuple[str, EvaluationResult]:
        """
        Find the best result based on a specific metric.
        
        Args:
            results: Dictionary of ID -> evaluation result
            metric: Metric to use for comparison
            higher_is_better: Whether higher metric values are better
            
        Returns:
            Tuple of (best_id, best_result)
        """
        if not results:
            raise ValueError("No results provided")
        
        best_id = None
        best_value = None
        best_result = None
        
        for result_id, result in results.items():
            if metric in result.metrics:
                value = result.metrics[metric]
                
                if best_value is None or (higher_is_better and value > best_value) or (not higher_is_better and value < best_value):
                    best_id = result_id
                    best_value = value
                    best_result = result
        
        if best_id is None:
            raise ValueError(f"Metric '{metric}' not found in any result")
        
        return best_id, best_result
    
    def register_best_prompt(
        self,
        results: Dict[str, EvaluationResult],
        metric: str = "accuracy",
        higher_is_better: bool = True,
        environment: str = "production",
        description: str = None
    ) -> str:
        """
        Register the best prompt in the prompt management system.
        
        Args:
            results: Dictionary of prompt_id -> evaluation result
            metric: Metric to use for comparison
            higher_is_better: Whether higher metric values are better
            environment: Environment to register the prompt for
            description: Description for the prompt
            
        Returns:
            ID of the registered prompt
        """
        if not self.prompt_manager:
            raise ValueError("Prompt manager not available")
        
        # Find best result
        best_id, best_result = self.find_best_result(results, metric, higher_is_better)
        
        # Get prompt text
        if best_result.request_params and "prompt" in best_result.request_params:
            prompt_text = best_result.request_params["prompt"]
        else:
            # Try to get from prompt manager
            try:
                prompt_data = self.prompt_manager.get_prompt(best_id)
                prompt_text = prompt_data["content"]
            except:
                raise ValueError(f"Could not get prompt text for {best_id}")
        
        # Generate description if not provided
        if not description:
            description = f"Best prompt for {best_result.task_id} based on {metric} ({best_result.metrics[metric]:.4f})"
        
        # Register prompt
        production_id = f"{best_id}_production"
        
        # Update prompt with new version
        self.prompt_manager.update_prompt(
            prompt_id=production_id,
            content=prompt_text,
            description=description,
            tags=["production", f"metric:{metric}", f"value:{best_result.metrics[metric]:.4f}"]
        )
        
        logger.info(f"Registered best prompt {best_id} as {production_id} (metric: {metric}, value: {best_result.metrics[metric]:.4f})")
        
        return production_id

    def compare_results(
        self, 
        baseline_result: EvaluationResult,
        new_result: EvaluationResult,
        metrics: List[str] = None,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Compare two evaluation results and generate a comparison report.
        
        Args:
            baseline_result: Baseline evaluation result
            new_result: New evaluation result to compare
            metrics: List of metrics to compare (if None, compare all shared metrics)
            output_dir: Directory to save the comparison report
            
        Returns:
            Comparison report
        """
        # Determine metrics to compare
        if metrics:
            compare_metrics = metrics
        else:
            # Use metrics that are present in both results
            baseline_metrics = set(baseline_result.metrics.keys())
            new_metrics = set(new_result.metrics.keys())
            compare_metrics = list(baseline_metrics.intersection(new_metrics))
        
        # Create comparison report
        report = {
            "timestamp": datetime.now().isoformat(),
            "baseline_id": baseline_result.task_id,
            "new_id": new_result.task_id,
            "metrics_comparison": {},
            "latency_comparison": {
                "baseline": baseline_result.latency_ms,
                "new": new_result.latency_ms,
                "difference": new_result.latency_ms - baseline_result.latency_ms if baseline_result.latency_ms and new_result.latency_ms else None,
                "percent_change": (new_result.latency_ms - baseline_result.latency_ms) / baseline_result.latency_ms * 100 if baseline_result.latency_ms and new_result.latency_ms else None
            },
            "token_usage_comparison": {}
        }
        
        # Compare metrics
        for metric in compare_metrics:
            baseline_value = baseline_result.metrics.get(metric)
            new_value = new_result.metrics.get(metric)
            
            if baseline_value is not None and new_value is not None:
                report["metrics_comparison"][metric] = {
                    "baseline": baseline_value,
                    "new": new_value,
                    "difference": new_value - baseline_value,
                    "percent_change": (new_value - baseline_value) / baseline_value * 100 if baseline_value != 0 else None
                }
        
        # Compare token usage
        baseline_usage = baseline_result.token_usage
        new_usage = new_result.token_usage
        
        for usage_type in set(baseline_usage.keys()).union(new_usage.keys()):
            baseline_value = baseline_usage.get(usage_type, 0)
            new_value = new_usage.get(usage_type, 0)
            
            report["token_usage_comparison"][usage_type] = {
                "baseline": baseline_value,
                "new": new_value,
                "difference": new_value - baseline_value,
                "percent_change": (new_value - baseline_value) / baseline_value * 100 if baseline_value != 0 else None
            }
        
        # Save report if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as JSON
            with open(os.path.join(output_dir, "comparison_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            
            # Save as markdown
            with open(os.path.join(output_dir, "comparison_report.md"), "w") as f:
                f.write(f"# Comparison Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Baseline: {baseline_result.task_id}\n")
                f.write(f"New: {new_result.task_id}\n\n")
                
                f.write(f"## Metrics Comparison\n\n")
                f.write("| Metric | Baseline | New | Difference | % Change |\n")
                f.write("| --- | --- | --- | --- | --- |\n")
                
                for metric, values in report["metrics_comparison"].items():
                    f.write(f"| {metric} | {values['baseline']:.4f} | {values['new']:.4f} | {values['difference']:.4f} | {values['percent_change']:.2f}% |\n")
                
                f.write(f"\n## Latency Comparison\n\n")
                latency = report["latency_comparison"]
                if latency["baseline"] and latency["new"]:
                    f.write(f"- Baseline: {latency['baseline']:.2f} ms\n")
                    f.write(f"- New: {latency['new']:.2f} ms\n")
                    f.write(f"- Difference: {latency['difference']:.2f} ms\n")
                    f.write(f"- Percent Change: {latency['percent_change']:.2f}%\n")
                else:
                    f.write("Latency data incomplete\n")
                
                f.write(f"\n## Token Usage Comparison\n\n")
                f.write("| Usage Type | Baseline | New | Difference | % Change |\n")
                f.write("| --- | --- | --- | --- | --- |\n")
                
                for usage_type, values in report["token_usage_comparison"].items():
                    if values["percent_change"] is not None:
                        f.write(f"| {usage_type} | {values['baseline']} | {values['new']} | {values['difference']} | {values['percent_change']:.2f}% |\n")
                    else:
                        f.write(f"| {usage_type} | {values['baseline']} | {values['new']} | {values['difference']} | N/A |\n")
            
            # Create comparison plots
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Metrics comparison plot
            for metric, values in report["metrics_comparison"].items():
                plt.figure(figsize=(8, 6))
                plt.bar(["Baseline", "New"], [values["baseline"], values["new"]])
                plt.title(f"{metric} Comparison")
                plt.ylabel(metric)
                plt.savefig(os.path.join(plots_dir, f"{metric}_comparison.png"))
                plt.close()
            
            # Latency comparison plot
            if latency["baseline"] and latency["new"]:
                plt.figure(figsize=(8, 6))
                plt.bar(["Baseline", "New"], [latency["baseline"], latency["new"]])
                plt.title("Latency Comparison")
                plt.ylabel("Latency (ms)")
                plt.savefig(os.path.join(plots_dir, "latency_comparison.png"))
                plt.close()
            
            # Token usage comparison plot
            for usage_type, values in report["token_usage_comparison"].items():
                plt.figure(figsize=(8, 6))
                plt.bar(["Baseline", "New"], [values["baseline"], values["new"]])
                plt.title(f"{usage_type} Comparison")
                plt.ylabel("Token Count")
                plt.savefig(os.path.join(plots_dir, f"{usage_type}_comparison.png"))
                plt.close()
        
        return report

# Utility function for classification metrics
def calculate_classification_metrics(true_values, predictions, labels=None):
    """
    Calculate standard classification metrics.
    
    Args:
        true_values: List of true values
        predictions: List of predicted values
        labels: List of all possible labels
        
    Returns:
        Dictionary of metrics
    """
    # Filter out examples with None predictions
    valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
    true_values = [true_values[i] for i in valid_indices]
    predictions = [predictions[i] for i in valid_indices]
    
    # Calculate metrics
    accuracy = accuracy_score(true_values, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_values, predictions, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_values, predictions, average='micro', zero_division=0
    )
    
    # Calculate per-class metrics if labels provided
    per_class_metrics = {}
    if labels:
        precision, recall, f1, support = precision_recall_fscore_support(
            true_values, predictions, labels=labels, zero_division=0
        )
        
        for i, label in enumerate(labels):
            per_class_metrics[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
    
    # Create confusion matrix if labels provided
    cm = None
    if labels:
        cm = confusion_matrix(true_values, predictions, labels=labels).tolist()
    
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "valid_examples": len(valid_indices),
        "invalid_examples": len(predictions) - len(valid_indices)
    }
    
    if per_class_metrics:
        metrics["per_class"] = per_class_metrics
    
    if cm:
        metrics["confusion_matrix"] = cm
    
    return metrics

# Helper function for generation metrics (BLEU, ROUGE, etc.)
def calculate_generation_metrics(references, hypotheses, use_bleu=True, use_rouge=True, use_bert_score=False):
    """
    Calculate metrics for text generation tasks.
    
    Args:
        references: List of reference texts (ground truth)
        hypotheses: List of generated texts (predictions)
        use_bleu: Whether to calculate BLEU score
        use_rouge: Whether to calculate ROUGE score
        use_bert_score: Whether to calculate BERTScore
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Filter out examples with None predictions
    valid_indices = [i for i, pred in enumerate(hypotheses) if pred is not None]
    references = [references[i] for i in valid_indices]
    hypotheses = [hypotheses[i] for i in valid_indices]
    
    # Calculate BLEU score
    if use_bleu:
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            import nltk
            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            # Tokenize references and hypotheses
            tokenized_refs = [[ref.split()] for ref in references]
            tokenized_hyps = [hyp.split() for hyp in hypotheses]
            
            # Calculate BLEU score with smoothing
            smoothing = SmoothingFunction().method1
            bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothing)
            
            metrics["bleu"] = float(bleu_score)
        except ImportError:
            logger.warning("NLTK not installed. BLEU score calculation skipped.")
        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {e}")
    
    # Calculate ROUGE score
    if use_rouge:
        try:
            from rouge import Rouge
            
            rouge = Rouge()
            scores = rouge.get_scores(hypotheses, references, avg=True)
            
            metrics["rouge1_f"] = float(scores["rouge-1"]["f"])
            metrics["rouge1_p"] = float(scores["rouge-1"]["p"])
            metrics["rouge1_r"] = float(scores["rouge-1"]["r"])
            
            metrics["rouge2_f"] = float(scores["rouge-2"]["f"])
            metrics["rouge2_p"] = float(scores["rouge-2"]["p"])
            metrics["rouge2_r"] = float(scores["rouge-2"]["r"])
            
            metrics["rougeL_f"] = float(scores["rouge-l"]["f"])
            metrics["rougeL_p"] = float(scores["rouge-l"]["p"])
            metrics["rougeL_r"] = float(scores["rouge-l"]["r"])
        except ImportError:
            logger.warning("Rouge not installed. ROUGE score calculation skipped.")
        except Exception as e:
            logger.warning(f"Error calculating ROUGE score: {e}")
    
    # Calculate BERTScore
    if use_bert_score:
        try:
            from bert_score import score
            
            P, R, F1 = score(hypotheses, references, lang="en", verbose=False)
            
            metrics["bert_score_precision"] = float(P.mean())
            metrics["bert_score_recall"] = float(R.mean())
            metrics["bert_score_f1"] = float(F1.mean())
        except ImportError:
            logger.warning("BERTScore not installed. BERTScore calculation skipped.")
        except Exception as e:
            logger.warning(f"Error calculating BERTScore: {e}")
    
    metrics["valid_examples"] = len(valid_indices)
    metrics["invalid_examples"] = len(hypotheses) - len(valid_indices)
    
    return metrics