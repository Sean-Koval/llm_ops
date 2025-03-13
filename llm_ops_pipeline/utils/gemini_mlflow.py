"""
Integration between Google Gemini models and MLflow for autologging.
Allows automatic tracking of Gemini model usage, parameters, and metrics.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps

import mlflow
import numpy as np

try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    from google.ai import generativelanguage as glm
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from llm_ops_pipeline.utils.logging import get_logger

# Set up logger
logger = get_logger(name="gemini_mlflow")

class GeminiMLflowLogger:
    """
    Provides MLflow integration for Google Gemini models.
    Enables autologging of model parameters, inputs, outputs, and metrics.
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        log_inputs: bool = True,
        log_outputs: bool = True,
        log_metrics: bool = True,
        log_parameters: bool = True,
        max_input_length: int = 500,
        max_output_length: int = 500,
        auto_end_run: bool = False
    ):
        """
        Initialize the Gemini MLflow logger.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: MLflow experiment name
            log_inputs: Whether to log model inputs
            log_outputs: Whether to log model outputs
            log_metrics: Whether to log metrics (latency, token usage)
            log_parameters: Whether to log model parameters
            max_input_length: Maximum length of logged inputs
            max_output_length: Maximum length of logged outputs
            auto_end_run: Whether to end MLflow runs automatically after each call
        """
        self.log_inputs = log_inputs
        self.log_outputs = log_outputs
        self.log_metrics = log_metrics
        self.log_parameters = log_parameters
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.auto_end_run = auto_end_run
        
        # Setup MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        # Check if Gemini is available
        if not GEMINI_AVAILABLE:
            logger.warning("Google Generative AI package not found. Please install with: pip install google-generativeai")
    
    def initialize_gemini(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None
    ):
        """
        Initialize the Gemini API client.
        
        Args:
            api_key: Google API key for Gemini API (for direct API access)
            project_id: Google Cloud project ID (for Vertex AI access)
            location: Google Cloud location (for Vertex AI access)
        """
        if not GEMINI_AVAILABLE:
            logger.error("Google Generative AI package not found. Please install with: pip install google-generativeai")
            return
        
        if api_key:
            # Use API key for direct access
            genai.configure(api_key=api_key)
            logger.info("Initialized Gemini with API key")
        elif project_id and location:
            # Use Google Cloud project for Vertex AI access
            try:
                from google.cloud import aiplatform
                aiplatform.init(project=project_id, location=location)
                logger.info(f"Initialized Vertex AI with project {project_id} in {location}")
            except ImportError:
                logger.error("google-cloud-aiplatform package not found. Please install with: pip install google-cloud-aiplatform")
        else:
            # Try to use environment variables
            if os.environ.get("GOOGLE_API_KEY"):
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                logger.info("Initialized Gemini with API key from environment variable")
            elif os.environ.get("GOOGLE_CLOUD_PROJECT") and os.environ.get("GOOGLE_CLOUD_LOCATION"):
                try:
                    from google.cloud import aiplatform
                    aiplatform.init(
                        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
                    )
                    logger.info(f"Initialized Vertex AI with project from environment variables")
                except ImportError:
                    logger.error("google-cloud-aiplatform package not found. Please install with: pip install google-cloud-aiplatform")
            else:
                logger.warning("No API key or Google Cloud project provided. Please provide credentials to use Gemini.")
    
    def enable_autologging(self):
        """
        Enable autologging for Gemini models.
        This patches the GenerativeModel class to log all interactions with MLflow.
        """
        if not GEMINI_AVAILABLE:
            logger.error("Google Generative AI package not found. Cannot enable autologging.")
            return
        
        logger.info("Enabling MLflow autologging for Gemini models")
        
        # Store the original methods
        original_generate_content = GenerativeModel.generate_content
        
        # Define patched methods
        @wraps(original_generate_content)
        def patched_generate_content(self, *args, **kwargs):
            # Start MLflow run if none is active
            if not mlflow.active_run():
                run = mlflow.start_run()
                created_new_run = True
            else:
                run = mlflow.active_run()
                created_new_run = False
            
            # Log model parameters
            if hasattr(self, '_model_name'):
                mlflow.set_tag("model_name", self._model_name)
            
            generation_config = kwargs.get('generation_config', None)
            if generation_config and self.log_parameters:
                mlflow.log_params(self._extract_generation_config(generation_config))
            
            # Log inputs
            if self.log_inputs:
                contents = args[0] if args else kwargs.get('contents', None)
                if contents:
                    truncated_content = self._truncate_content(contents)
                    mlflow.log_param("inputs", truncated_content)
            
            # Time the API call
            start_time = time.time()
            
            # Make the original call
            try:
                result = original_generate_content(self, *args, **kwargs)
                
                # Log latency
                if self.log_metrics:
                    latency = (time.time() - start_time) * 1000  # ms
                    mlflow.log_metric("latency_ms", latency)
                
                # Log outputs
                if self.log_outputs and result:
                    # Log text output
                    if hasattr(result, 'text'):
                        output_text = self._truncate_text(result.text)
                        mlflow.log_text(output_text, "output.txt")
                    
                    # Log token usage
                    if hasattr(result, 'usage_metadata'):
                        token_usage = {
                            "prompt_tokens": getattr(result.usage_metadata, "prompt_token_count", 0),
                            "completion_tokens": getattr(result.usage_metadata, "candidates_token_count", 0),
                            "total_tokens": getattr(result.usage_metadata, "total_token_count", 0),
                        }
                        
                        for key, value in token_usage.items():
                            mlflow.log_metric(key, value)
                        
                        # Estimate cost based on token usage
                        cost_estimate = self._estimate_cost(
                            token_usage,
                            model_name=getattr(self, "_model_name", "gemini-1.0-pro")
                        )
                        mlflow.log_metric("cost_estimate_usd", cost_estimate)
                
                # Log additional metrics if present
                if hasattr(result, 'candidates') and result.candidates:
                    for i, candidate in enumerate(result.candidates):
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            for j, rating in enumerate(candidate.safety_ratings):
                                category = rating.category.name if hasattr(rating.category, 'name') else f"safety_{j}"
                                mlflow.log_metric(f"safety_{category}", rating.probability)
                
                # End run if auto_end_run is enabled
                if self.auto_end_run and created_new_run:
                    mlflow.end_run()
                
                return result
            
            except Exception as e:
                # Log the error
                mlflow.set_tag("error", str(e))
                mlflow.set_tag("error_type", type(e).__name__)
                
                # End run if auto_end_run is enabled
                if self.auto_end_run and created_new_run:
                    mlflow.end_run()
                
                # Re-raise the exception
                raise
        
        # Apply monkey patches
        GenerativeModel.generate_content = patched_generate_content
        GenerativeModel._extract_generation_config = self._extract_generation_config
        GenerativeModel._truncate_content = self._truncate_content
        GenerativeModel._truncate_text = self._truncate_text
        GenerativeModel._estimate_cost = self._estimate_cost
        GenerativeModel.log_inputs = self.log_inputs
        GenerativeModel.log_outputs = self.log_outputs
        GenerativeModel.log_metrics = self.log_metrics
        GenerativeModel.log_parameters = self.log_parameters
        GenerativeModel.max_input_length = self.max_input_length
        GenerativeModel.max_output_length = self.max_output_length
        GenerativeModel.auto_end_run = self.auto_end_run
        
        logger.info("Gemini autologging enabled")
    
    def disable_autologging(self):
        """
        Disable autologging for Gemini models.
        This restores the original methods.
        """
        if not GEMINI_AVAILABLE:
            logger.error("Google Generative AI package not found. No autologging to disable.")
            return
        
        # Restore original methods if they were patched
        if hasattr(GenerativeModel, '_extract_generation_config'):
            # Get original methods from the module
            import inspect
            import google.generativeai as genai_module
            
            # Get the original methods from the module
            original_source = inspect.getsource(genai_module.GenerativeModel)
            
            # Find the original generate_content method
            import re
            method_match = re.search(r'def generate_content\(self, .*?\):', original_source, re.DOTALL)
            
            if method_match:
                # Reset to original implementation (need to re-import the module)
                import importlib
                importlib.reload(genai)
                logger.info("Gemini autologging disabled")
            else:
                logger.warning("Could not find original methods to restore. Autologging may still be active.")
    
    @staticmethod
    def _extract_generation_config(config: Any) -> Dict[str, Any]:
        """Extract generation config parameters from the config object."""
        if isinstance(config, dict):
            return config
        
        # Extract parameters from the GenerationConfig object
        params = {}
        for attr in [
            "temperature", "top_p", "top_k", "candidate_count", 
            "max_output_tokens", "stop_sequences"
        ]:
            if hasattr(config, attr):
                value = getattr(config, attr)
                if value is not None:
                    params[attr] = value
        
        return params
    
    @staticmethod
    def _truncate_content(content: Any) -> str:
        """Truncate and format content for logging."""
        if isinstance(content, str):
            return content[:500] + "..." if len(content) > 500 else content
        
        elif isinstance(content, list):
            result = []
            total_length = 0
            
            for item in content:
                if isinstance(item, str):
                    # Text content
                    truncated = item[:200] + "..." if len(item) > 200 else item
                    result.append(truncated)
                    total_length += len(truncated)
                elif hasattr(item, "parts"):
                    # Content object with parts
                    parts_text = []
                    for part in item.parts:
                        if hasattr(part, "text"):
                            text = part.text[:100] + "..." if len(part.text) > 100 else part.text
                            parts_text.append(text)
                    
                    result.append(" | ".join(parts_text))
                    total_length += sum(len(t) for t in parts_text)
                else:
                    # Unknown type
                    result.append(str(item)[:100])
                    total_length += 100
                
                # Break if total length exceeds limit
                if total_length > 500:
                    result.append("...")
                    break
            
            return " | ".join(result)
        
        elif hasattr(content, "parts"):
            # Content object with parts
            parts_text = []
            for part in content.parts:
                if hasattr(part, "text"):
                    truncated = part.text[:200] + "..." if len(part.text) > 200 else part.text
                    parts_text.append(truncated)
            
            joined = " | ".join(parts_text)
            return joined[:500] + "..." if len(joined) > 500 else joined
        
        else:
            # Unknown type
            return str(content)[:500]
    
    @staticmethod
    def _truncate_text(text: str) -> str:
        """Truncate text to the maximum output length."""
        max_length = 1000  # Default max length
        return text[:max_length] + "..." if len(text) > max_length else text
    
    @staticmethod
    def _estimate_cost(token_usage: Dict[str, int], model_name: str) -> float:
        """Estimate cost based on token usage and model."""
        # Cost per 1000 tokens for different models
        costs = {
            "gemini-1.0-pro": {
                "input": 0.00125,  # $0.00125 per 1K input tokens
                "output": 0.00375,  # $0.00375 per 1K output tokens
            },
            "gemini-1.0-pro-vision": {
                "input": 0.00125,
                "output": 0.00375,
            },
            "gemini-1.0-ultra": {
                "input": 0.0125,   # $0.0125 per 1K input tokens
                "output": 0.0375,  # $0.0375 per 1K output tokens
            },
            "gemini-1.5-pro": {
                "input": 0.0025,   # $0.0025 per 1K input tokens
                "output": 0.0075,  # $0.0075 per 1K output tokens
            },
            "gemini-1.5-flash": {
                "input": 0.00025,  # $0.00025 per 1K input tokens
                "output": 0.0007,  # $0.0007 per 1K output tokens
            },
            # Default to pro pricing if model not recognized
            "default": {
                "input": 0.00125,
                "output": 0.00375,
            }
        }
        
        # Get cost rates for the model
        model_costs = costs.get(model_name, costs["default"])
        
        # Calculate cost
        input_cost = (token_usage.get("prompt_tokens", 0) / 1000) * model_costs["input"]
        output_cost = (token_usage.get("completion_tokens", 0) / 1000) * model_costs["output"]
        
        return input_cost + output_cost

# Convenience function to set up autologging
def setup_gemini_autologging(
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    log_inputs: bool = True,
    log_outputs: bool = True,
    log_metrics: bool = True,
    log_parameters: bool = True,
    auto_end_run: bool = False
) -> GeminiMLflowLogger:
    """
    Set up autologging for Gemini models.
    
    Args:
        api_key: Google API key for Gemini API
        project_id: Google Cloud project ID
        location: Google Cloud location
        tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
        log_inputs: Whether to log model inputs
        log_outputs: Whether to log model outputs
        log_metrics: Whether to log metrics
        log_parameters: Whether to log model parameters
        auto_end_run: Whether to end MLflow runs automatically
        
    Returns:
        GeminiMLflowLogger instance
    """
    logger = GeminiMLflowLogger(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        log_inputs=log_inputs,
        log_outputs=log_outputs,
        log_metrics=log_metrics,
        log_parameters=log_parameters,
        auto_end_run=auto_end_run
    )
    
    logger.initialize_gemini(
        api_key=api_key,
        project_id=project_id,
        location=location
    )
    
    logger.enable_autologging()
    return logger