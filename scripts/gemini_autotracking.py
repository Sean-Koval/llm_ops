#!/usr/bin/env python
"""
Utility script for enabling MLflow autologging with Gemini models.
Sets up the autologging integration and provides a simple CLI for managing it.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_ops_pipeline.utils.logging import setup_logger
from llm_ops_pipeline.utils.gemini_mlflow import setup_gemini_autologging

# Optional Gemini imports
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Set up logger
logger = setup_logger(name="gemini_autotracking")

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

def setup_autologging(args):
    """Set up Gemini autologging with MLflow."""
    if not GEMINI_AVAILABLE:
        logger.error("Google Generative AI package not found. Please install with: pip install google-generativeai")
        sys.exit(1)
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Get Gemini configuration from config file or arguments
    gemini_config = config.get("gemini", {})
    
    # Determine API key source
    api_key = args.api_key or gemini_config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
    project_id = args.project_id or gemini_config.get("project_id") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = args.location or gemini_config.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    # Get MLflow configuration
    mlflow_config = config.get("mlflow", {})
    tracking_uri = args.tracking_uri or mlflow_config.get("tracking_uri")
    experiment_name = args.experiment_name or mlflow_config.get("experiment_name", "gemini_experiments")
    
    # Set up autologging
    logger = setup_gemini_autologging(
        api_key=api_key,
        project_id=project_id,
        location=location,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        log_inputs=not args.disable_log_inputs,
        log_outputs=not args.disable_log_outputs,
        log_metrics=not args.disable_log_metrics,
        log_parameters=not args.disable_log_parameters,
        auto_end_run=args.auto_end_run
    )
    
    # Print status
    if logger:
        logger.info(f"Gemini MLflow autologging enabled for experiment '{experiment_name}'")
        if tracking_uri:
            logger.info(f"MLflow tracking server: {tracking_uri}")
        
        # Print authentication method
        if api_key:
            logger.info("Authentication: Using Gemini API key")
        elif project_id:
            logger.info(f"Authentication: Using Google Cloud project {project_id} in {location}")
        else:
            logger.info("Authentication: Using default credentials")
        
        return logger
    else:
        logger.error("Failed to set up Gemini autologging")
        return None

def run_gemini_example(args, logger):
    """Run a simple Gemini example with autologging."""
    # Initialize Gemini
    if args.api_key:
        genai.configure(api_key=args.api_key)
    
    # Set up model
    model = GenerativeModel(args.model)
    
    # Run example query
    prompt = args.prompt or "Explain the concept of MLOps in one paragraph."
    
    logger.info(f"Running example query with model {args.model}: {prompt}")
    
    try:
        response = model.generate_content(prompt)
        
        # Print response
        print("\nGemini Response:")
        print("---------------")
        print(response.text)
        print("\nThis interaction was automatically logged to MLflow.")
        
        if args.tracking_uri:
            print(f"Check the MLflow UI at {args.tracking_uri} to see the logged data.")
        else:
            print("Check your MLflow tracking server to see the logged data.")
        
        # Print token usage if available
        if hasattr(response, "usage_metadata"):
            print("\nToken Usage:")
            print(f"  Prompt tokens: {getattr(response.usage_metadata, 'prompt_token_count', 'N/A')}")
            print(f"  Completion tokens: {getattr(response.usage_metadata, 'candidates_token_count', 'N/A')}")
            print(f"  Total tokens: {getattr(response.usage_metadata, 'total_token_count', 'N/A')}")
    
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Gemini MLflow autologging utility")
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--api-key", type=str, help="Google API key for Gemini API")
    parser.add_argument("--project-id", type=str, help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="Google Cloud location")
    parser.add_argument("--tracking-uri", type=str, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, help="MLflow experiment name")
    
    # Logging options
    parser.add_argument("--disable-log-inputs", action="store_true", help="Disable logging model inputs")
    parser.add_argument("--disable-log-outputs", action="store_true", help="Disable logging model outputs")
    parser.add_argument("--disable-log-metrics", action="store_true", help="Disable logging metrics")
    parser.add_argument("--disable-log-parameters", action="store_true", help="Disable logging model parameters")
    parser.add_argument("--auto-end-run", action="store_true", help="End MLflow runs after each API call")
    
    # Example options
    parser.add_argument("--run-example", action="store_true", help="Run a simple example after setup")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Gemini model to use for example")
    parser.add_argument("--prompt", type=str, help="Prompt to use for example")
    
    args = parser.parse_args()
    
    # Check Gemini availability
    if not GEMINI_AVAILABLE:
        logger.error("Google Generative AI package not found. Please install with: pip install google-generativeai")
        sys.exit(1)
    
    # Set up autologging
    logger = setup_autologging(args)
    
    if logger and args.run_example:
        run_gemini_example(args, logger)

if __name__ == "__main__":
    main()