#!/usr/bin/env python
"""
Deploy the best performing prompt and model to production.
Registers the prompt in Langfuse, tracks deployment in MLflow,
and sets up monitoring configuration.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from llm_ops_pipeline.utils.logging import setup_logger
from llm_ops_pipeline.utils.prompt_management import PromptManager
from llm_ops_pipeline.evaluation.llm_evaluation_framework import LLMEvaluationFramework

# Set up logger
logger = setup_logger(name="deploy_to_production")

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

def deploy_to_production(args, config: Dict[str, Any]):
    """Deploy the best prompt and model to production."""
    # Initialize evaluation framework
    eval_framework = LLMEvaluationFramework(
        config=config,
        experiment_name=config.get("experiment_name", "prompt_tuning_example"),
        use_mlflow=True,
        use_langfuse=True
    )
    
    # Create a deployment run
    run_id = eval_framework.start_run(
        run_name=f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags={"stage": "production", "deployed_by": os.environ.get("USER", "unknown")},
        parameters={
            "prompt_id": args.prompt_id,
            "model_id": args.model_id,
            "environment": args.environment,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    # Log deployment info
    eval_framework.log_metrics({
        "deployment_timestamp": datetime.now().timestamp()
    })
    
    # Get prompt data
    prompt_text = ""
    if eval_framework.prompt_manager:
        try:
            prompt_data = eval_framework.prompt_manager.get_prompt(args.prompt_id)
            prompt_text = prompt_data["content"]
            prompt_version = prompt_data.get("metadata", {}).get("version", "unknown")
            
            logger.info(f"Retrieved prompt {args.prompt_id} (version {prompt_version}) from Langfuse")
            
            # Register prompt for production if not already
            if args.environment == "production" and args.prompt_id != f"{args.prompt_id}_production":
                production_id = eval_framework.prompt_manager.update_prompt(
                    prompt_id=f"{args.prompt_id}_production",
                    content=prompt_text,
                    description=f"Production prompt deployed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    tags=["production", f"model:{args.model_id}"]
                )
                
                logger.info(f"Registered prompt as {args.prompt_id}_production")
        except Exception as e:
            logger.error(f"Failed to get prompt from Langfuse: {e}")
            # Try to load from prompts file if provided
            if args.prompts_file and os.path.exists(args.prompts_file):
                try:
                    with open(args.prompts_file, "r") as f:
                        prompts = json.load(f)
                    
                    if args.prompt_id in prompts:
                        prompt_text = prompts[args.prompt_id]
                        logger.info(f"Loaded prompt from {args.prompts_file}")
                except Exception as e:
                    logger.error(f"Failed to load prompt from file: {e}")
    elif args.prompts_file and os.path.exists(args.prompts_file):
        try:
            with open(args.prompts_file, "r") as f:
                prompts = json.load(f)
            
            if args.prompt_id in prompts:
                prompt_text = prompts[args.prompt_id]
                logger.info(f"Loaded prompt from {args.prompts_file}")
        except Exception as e:
            logger.error(f"Failed to load prompt from file: {e}")
    
    if not prompt_text:
        logger.warning(f"Could not retrieve prompt text for {args.prompt_id}")
    
    # Create deployment record
    deployment_record = {
        "timestamp": datetime.now().isoformat(),
        "prompt_id": args.prompt_id,
        "model_id": args.model_id,
        "environment": args.environment,
        "run_id": run_id,
        "deployed_by": os.environ.get("USER", "unknown")
    }
    
    # Create deployment directory
    deployment_dir = Path(args.output_dir)
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save deployment record
    with open(deployment_dir / "deployment_record.json", "w") as f:
        json.dump(deployment_record, f, indent=2)
    
    # Save prompt text
    if prompt_text:
        with open(deployment_dir / "production_prompt.txt", "w") as f:
            f.write(prompt_text)
    
    # Create production configuration
    production_config = {
        "prompt_id": args.prompt_id if args.environment != "production" else f"{args.prompt_id}_production",
        "model_id": args.model_id,
        "environment": args.environment,
        "deployed_at": datetime.now().isoformat(),
        "monitoring": {
            "enabled": True,
            "metrics": [
                "latency_ms",
                "token_usage.total_tokens",
                "token_usage.prompt_tokens",
                "token_usage.completion_tokens"
            ],
            "alert_thresholds": {
                "error_rate": 0.05,
                "latency_p95_ms": 2000
            }
        }
    }
    
    # Save production configuration
    with open(deployment_dir / "production_config.yaml", "w") as f:
        yaml.dump(production_config, f, default_flow_style=False)
    
    logger.info(f"Deployment complete. Configuration saved to {deployment_dir}")
    
    # End the run
    eval_framework.end_run()
    
    return run_id

def main():
    parser = argparse.ArgumentParser(description="Deploy prompt and model to production")
    parser.add_argument("--config", type=str, default="../configs/evaluation_config.yaml",
                        help="Path to evaluation configuration file")
    parser.add_argument("--prompt-id", type=str, required=True,
                        help="ID of the prompt to deploy")
    parser.add_argument("--model-id", type=str, required=True,
                        help="ID of the model to deploy")
    parser.add_argument("--environment", type=str, default="production",
                        choices=["development", "staging", "production"],
                        help="Environment to deploy to")
    parser.add_argument("--prompts-file", type=str,
                        help="Path to prompts file (as fallback)")
    parser.add_argument("--output-dir", type=str, default="../deployment",
                        help="Directory to save deployment configuration")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Deploy to production
    run_id = deploy_to_production(args, config)
    
    logger.info(f"Deployment completed successfully. Run ID: {run_id}")

if __name__ == "__main__":
    main()