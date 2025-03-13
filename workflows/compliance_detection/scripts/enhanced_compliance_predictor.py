#!/usr/bin/env python
"""
Enhanced Compliance Detection Predictor

This script implements compliance detection using the prompt management system
with sophisticated templating, A/B testing, and MLflow tracking.
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

import mlflow
import yaml
import pandas as pd
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
from tqdm import tqdm

# Add the parent directory to the path so we can import from llm_ops_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm_ops_pipeline.utils.prompt_management import PromptManager
from llm_ops_pipeline.utils.prompt_templates import PromptTemplate, PromptTemplateLibrary
from llm_ops_pipeline.utils.prompt_experimentation import PromptExperiment, load_few_shot_examples

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Compliance categories
CATEGORIES = [
    "ETHICAL_BREACH",
    "ILLEGAL_ACTIVITY",
    "REGULATORY_VIOLATION",
    "CONFIDENTIAL_INFO",
    "HARASSMENT",
    "COMPLIANT"
]

class CompliancePredictor:
    """
    Compliance Predictor using the prompt management system.
    
    This class uses the PromptManager to access and use prompts for
    compliance detection. It supports A/B testing of different prompt variants
    and tracks performance metrics in MLflow.
    """
    
    def __init__(
        self,
        config_path: str,
        environment: str = "development",
        initialize_experiment: bool = True
    ):
        """
        Initialize the compliance predictor.
        
        Args:
            config_path: Path to the configuration file
            environment: Environment name (development, staging, production)
            initialize_experiment: Whether to initialize the A/B testing experiment
        """
        self.config = self._load_config(config_path)
        self.environment = environment
        self.model = None
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager(
            langfuse_api_key=os.environ.get("LANGFUSE_API_KEY"),
            langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            langfuse_host=os.environ.get("LANGFUSE_HOST"),
            environment=environment
        )
        
        # Initialize prompt template library
        prompts_dir = Path(__file__).resolve().parents[3] / "prompts"
        self.template_library = PromptTemplateLibrary(prompts_dir)
        
        # Create template for each prompt variant
        self._initialize_prompt_templates()
        
        # Initialize Vertex AI model
        self._initialize_model()
        
        # Initialize experiment if requested
        self.experiment = None
        if initialize_experiment:
            self._initialize_experiment()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_model(self) -> None:
        """Initialize the Vertex AI model."""
        model_config = self.config.get("model", {})
        
        try:
            # Initialize Vertex AI
            project_id = model_config.get("project_id")
            location = model_config.get("location", "us-central1")
            
            if project_id:
                aiplatform.init(project=project_id, location=location)
            
            # Load the generative model
            model_name = model_config.get("name", "gemini-flash")
            self.model = GenerativeModel(model_name)
            logger.info(f"Initialized Vertex AI model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI model: {e}")
            self.model = None
    
    def _initialize_prompt_templates(self) -> None:
        """Initialize the prompt templates for each variant."""
        # Define the base template with advanced features
        base_template = """
You are a compliance detection system for a financial institution. 
Analyze the provided message and determine if it contains any compliance breaches.

{% if detailed_instructions %}
## Detailed Instructions
The compliance categories are defined as follows:

- ETHICAL_BREACH: Conflicts of interest, misleading clients, dishonest behavior, or any communication that violates the institution's code of ethics.
- ILLEGAL_ACTIVITY: Fraud, money laundering, market manipulation, insider trading, or any explicitly illegal actions.
- REGULATORY_VIOLATION: Sharing material non-public information, violating compliance protocols, or breaking regulatory rules set by financial authorities.
- CONFIDENTIAL_INFO: Improper sharing of client data, unreleased financial results, proprietary strategies, or other confidential information.
- HARASSMENT: Workplace harassment, discrimination, inappropriate comments, or creating a hostile work environment.
- COMPLIANT: Professional and compliant communication that does not violate any rules or regulations.
{% endif %}

{% if step_by_step %}
## Analysis Approach
1. Read the message carefully
2. Identify any potential compliance issues
3. Determine the primary category of violation
4. If multiple violations exist, classify based on the most severe one
5. If no violation exists, classify as COMPLIANT
{% endif %}

## Examples
FEW_SHOT_EXAMPLES

## Message to Classify
"{{ message }}"

{% if output_format == "json" %}
Provide your classification as a JSON object with the following format:
```json
{
  "category": "CATEGORY_NAME",
  "confidence": 0.95,
  "reasoning": "Brief explanation of your classification"
}
```
{% else %}
Return only the category label with no additional text.
{% endif %}
"""
        
        # Create the basic template
        self.template_library.add_template(
            "compliance/basic",
            PromptTemplate(base_template)
        )
        
        # Create the detailed template
        self.template_library.add_template(
            "compliance/detailed",
            PromptTemplate(base_template)
        )
        
        # Create the step-by-step template
        self.template_library.add_template(
            "compliance/step_by_step",
            PromptTemplate(base_template)
        )
        
        # Load few-shot examples
        examples_path = Path(__file__).resolve().parent.parent / "prompts" / "few_shot_examples.json"
        try:
            few_shot_examples = load_few_shot_examples(str(examples_path))
            
            # Create the few-shot template
            self.template_library.add_template(
                "compliance/few_shot",
                PromptTemplate(base_template, few_shot_examples=few_shot_examples)
            )
        except Exception as e:
            logger.warning(f"Failed to load few-shot examples: {e}")
            
            # Create an empty few-shot template as fallback
            self.template_library.add_template(
                "compliance/few_shot",
                PromptTemplate(base_template)
            )
    
    def _initialize_experiment(self) -> None:
        """Initialize the A/B testing experiment."""
        try:
            experiment_name = self.config.get("experiment", {}).get("name", "compliance-detection")
            
            # Initialize the experiment
            self.experiment = PromptExperiment(
                experiment_name=experiment_name,
                prompt_manager=self.prompt_manager,
                mlflow_tracking_uri=self.config.get("mlflow", {}).get("tracking_uri"),
                default_metrics=["accuracy", "latency", "error_rate"]
            )
            
            # Add variants
            self.experiment.add_variant(
                name="basic",
                prompt_id="compliance/basic",
                weight=1.0,
                description="Basic compliance detection prompt"
            )
            
            self.experiment.add_variant(
                name="detailed",
                prompt_id="compliance/detailed",
                weight=1.0,
                description="Detailed compliance detection prompt with category explanations"
            )
            
            self.experiment.add_variant(
                name="step_by_step",
                prompt_id="compliance/step_by_step",
                weight=1.0,
                description="Step-by-step reasoning compliance detection prompt"
            )
            
            self.experiment.add_variant(
                name="few_shot",
                prompt_id="compliance/few_shot",
                weight=1.0,
                description="Few-shot learning compliance detection prompt"
            )
            
            logger.info(f"Initialized experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize experiment: {e}")
            self.experiment = None
    
    def classify_message(
        self,
        message: str,
        prompt_variant: Optional[str] = None,
        output_format: str = "category",
        track_in_mlflow: bool = True
    ) -> Dict[str, Any]:
        """
        Classify a message for compliance violations.
        
        Args:
            message: The message to classify
            prompt_variant: Specific prompt variant to use (or None for random)
            output_format: Output format ('category' or 'json')
            track_in_mlflow: Whether to track metrics in MLflow
            
        Returns:
            Classification result
        """
        if not self.model:
            raise ValueError("Model not initialized")
        
        # Select variant
        if self.experiment and not prompt_variant:
            variant_name, variant = self.experiment.get_random_variant()
            prompt_id = variant["prompt_id"]
        else:
            # Use the specified variant or fallback to few_shot
            variant_name = prompt_variant or "few_shot"
            if variant_name not in ["basic", "detailed", "step_by_step", "few_shot"]:
                variant_name = "few_shot"  # Default to few_shot if invalid
            prompt_id = f"compliance/{variant_name}"
        
        # Get prompt content
        try:
            prompt_data = self.prompt_manager.get_prompt(prompt_id)
            prompt_content = prompt_data.get("content", "")
        except ValueError:
            # If prompt not found in manager, create a new one from template
            logger.info(f"Creating new prompt: {prompt_id}")
            
            # Render the template
            template_name = prompt_id
            context = {
                "detailed_instructions": variant_name == "detailed",
                "step_by_step": variant_name == "step_by_step",
                "message": "placeholder",  # Will be replaced in inference
                "output_format": output_format,
                "use_few_shot": variant_name == "few_shot"
            }
            
            prompt_content = self.template_library.render(template_name, context)
            
            # Save to prompt manager
            prompt_data = self.prompt_manager.create_prompt(
                prompt_id=prompt_id,
                content=prompt_content,
                name=f"Compliance Detection - {variant_name.capitalize()}",
                description=f"Prompt for detecting compliance violations using {variant_name} approach",
                tags=["compliance", variant_name]
            )
        
        # Start tracking metrics
        start_time = time.time()
        
        try:
            # Render the template for this specific message
            template = PromptTemplate(prompt_content)
            
            context = {
                "message": message,
                "detailed_instructions": variant_name == "detailed",
                "step_by_step": variant_name == "step_by_step",
                "output_format": output_format
            }
            
            rendered_prompt = template.render(context)
            
            # Generate completion
            response = self.model.generate_content(rendered_prompt)
            completion = response.text.strip()
            
            # Parse the result
            if output_format == "json":
                # Extract JSON from the completion
                try:
                    json_str = completion
                    if "```json" in completion:
                        json_str = completion.split("```json")[1].split("```")[0].strip()
                    elif "```" in completion:
                        json_str = completion.split("```")[1].strip()
                    
                    result = json.loads(json_str)
                    
                    # Ensure category is valid
                    if "category" in result and result["category"] not in CATEGORIES:
                        result["category"] = "COMPLIANT"  # Default to compliant if invalid
                except Exception as e:
                    logger.warning(f"Failed to parse JSON result: {e}")
                    result = {
                        "category": "COMPLIANT",
                        "confidence": 0.0,
                        "reasoning": "Failed to parse result",
                        "error": str(e)
                    }
            else:
                # Simple category output
                category = completion.strip()
                
                # Ensure category is valid
                if category not in CATEGORIES:
                    category = "COMPLIANT"  # Default to compliant if invalid
                
                result = {"category": category}
            
            # Calculate metrics
            latency = time.time() - start_time
            
            # Add metadata
            result["metadata"] = {
                "prompt_id": prompt_id,
                "variant": variant_name,
                "latency": latency,
                "model": self.model._model_name,
                "timestamp": time.time()
            }
            
            # Log prompt usage if tracking is enabled
            if track_in_mlflow:
                self.prompt_manager.log_prompt_usage(
                    prompt_id=prompt_id,
                    inputs={"message": message},
                    completion=completion,
                    metadata={
                        "latency": latency,
                        "variant": variant_name,
                        "category": result.get("category"),
                        "metrics": {
                            "latency": latency,
                            "tokens_used": len(rendered_prompt.split()) + len(completion.split())
                        }
                    }
                )
            
            return result
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return {
                "category": "COMPLIANT",
                "error": str(e),
                "metadata": {
                    "prompt_id": prompt_id,
                    "variant": variant_name,
                    "latency": time.time() - start_time,
                    "error": True
                }
            }
    
    def run_ab_test(
        self,
        test_data_path: str,
        output_path: Optional[str] = None,
        sampling_method: str = "all"
    ) -> Dict[str, Any]:
        """
        Run an A/B test on a test dataset.
        
        Args:
            test_data_path: Path to the test data CSV or JSON
            output_path: Path to save results (optional)
            sampling_method: How to assign variants ('random', 'deterministic', or 'all')
            
        Returns:
            Dictionary with test results
        """
        if not self.experiment:
            raise ValueError("Experiment not initialized")
        
        # Load test data
        if test_data_path.endswith('.csv'):
            df = pd.read_csv(test_data_path)
        elif test_data_path.endswith('.json'):
            df = pd.read_json(test_data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # Prepare inputs batch
        inputs_batch = []
        for _, row in df.iterrows():
            inputs_batch.append({
                "message": row["text"],
                "true_label": row["label"] if "label" in row else None
            })
        
        # Define inference function
        def inference_fn(prompt, inputs, **kwargs):
            message = inputs["message"]
            true_label = inputs.get("true_label")
            
            # Render the template
            template = PromptTemplate(prompt["content"])
            
            context = {
                "message": message,
                "detailed_instructions": "detailed" in kwargs.get("variant", ""),
                "step_by_step": "step_by_step" in kwargs.get("variant", ""),
                "output_format": "json"
            }
            
            rendered_prompt = template.render(context)
            
            # Generate completion
            response = self.model.generate_content(rendered_prompt)
            completion = response.text.strip()
            
            # Parse the result
            try:
                json_str = completion
                if "```json" in completion:
                    json_str = completion.split("```json")[1].split("```")[0].strip()
                elif "```" in completion:
                    json_str = completion.split("```")[1].strip()
                
                result = json.loads(json_str)
                
                # Ensure category is valid
                if "category" in result and result["category"] not in CATEGORIES:
                    result["category"] = "COMPLIANT"
            except Exception as e:
                logger.warning(f"Failed to parse JSON result: {e}")
                result = {
                    "category": "COMPLIANT",
                    "confidence": 0.0,
                    "reasoning": "Failed to parse result",
                    "error": str(e)
                }
            
            # Add true label if available
            if true_label is not None:
                result["true_label"] = true_label
                result["correct"] = result["category"] == true_label
            
            return result
        
        # Define evaluation function
        def evaluation_fn(results):
            metrics = {
                "total": len(results),
                "error_count": sum(1 for r in results if "error" in r.get("results", {})),
                "latency": sum(r.get("metadata", {}).get("latency", 0) for r in results) / len(results)
            }
            
            # Calculate accuracy if true labels are available
            correct_count = sum(1 for r in results if r.get("results", {}).get("correct", False))
            metrics["accuracy"] = correct_count / len(results) if len(results) > 0 else 0
            metrics["error_rate"] = metrics["error_count"] / len(results) if len(results) > 0 else 0
            
            return metrics
        
        # Run the A/B test
        logger.info(f"Running A/B test on {len(inputs_batch)} examples with {sampling_method} sampling")
        
        with mlflow.start_run(experiment_id=self.experiment.experiment_id) as run:
            results = self.experiment.run_ab_test(
                inputs_batch=inputs_batch,
                inference_fn=inference_fn,
                sampling_method=sampling_method,
                evaluation_fn=evaluation_fn
            )
            
            # Log additional metrics
            for variant, variant_results in results.items():
                # Calculate per-category metrics
                categories = CATEGORIES
                for category in categories:
                    category_results = [r for r in variant_results 
                                     if r.get("results", {}).get("true_label") == category]
                    
                    if category_results:
                        correct = sum(1 for r in category_results if r.get("results", {}).get("correct", False))
                        accuracy = correct / len(category_results)
                        mlflow.log_metric(f"{variant}_accuracy_{category}", accuracy)
            
            # Save detailed results if output_path provided
            if output_path:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                
                # Log as artifact
                mlflow.log_artifact(output_path)
        
        # Create summary
        summary = {}
        for variant, variant_results in results.items():
            metrics = evaluation_fn(variant_results)
            summary[variant] = metrics
        
        return {
            "summary": summary,
            "run_id": run.info.run_id
        }
    
    def promote_best_variant(self, metric: str = "accuracy") -> Dict[str, Any]:
        """
        Promote the best performing variant to production.
        
        Args:
            metric: Metric to use for determining the best variant
            
        Returns:
            Dictionary with promotion results
        """
        if not self.experiment:
            raise ValueError("Experiment not initialized")
        
        # Analyze the experiment
        analysis = self.experiment.analyze_experiment()
        
        # Get the best variant for the metric
        if metric not in analysis.get("best_variants", {}):
            raise ValueError(f"Metric '{metric}' not found in experiment analysis")
        
        best_variant = analysis["best_variants"][metric]
        variant_name = best_variant["variant"]
        metric_value = best_variant["value"]
        
        logger.info(f"Best variant for {metric}: {variant_name} ({metric_value:.4f})")
        
        # Get the prompt ID for this variant
        prompt_id = self.experiment.get_variant_by_name(variant_name)["prompt_id"]
        
        # Create or update production prompt
        try:
            # Get existing prompt data
            prompt_data = self.prompt_manager.get_prompt(prompt_id)
            prompt_content = prompt_data.get("content", "")
            
            # Create or update production prompt
            production_prompt_id = "compliance/production"
            
            try:
                # Try to get existing production prompt
                self.prompt_manager.get_prompt(production_prompt_id)
                
                # Update it
                updated_prompt = self.prompt_manager.update_prompt(
                    prompt_id=production_prompt_id,
                    content=prompt_content,
                    description=f"Production compliance detection prompt (promoted from {variant_name})",
                    tags=["compliance", "production", variant_name]
                )
                
                logger.info(f"Updated production prompt from variant {variant_name}")
                
            except ValueError:
                # Create new production prompt
                updated_prompt = self.prompt_manager.create_prompt(
                    prompt_id=production_prompt_id,
                    content=prompt_content,
                    name="Compliance Detection - Production",
                    description=f"Production compliance detection prompt (promoted from {variant_name})",
                    tags=["compliance", "production", variant_name]
                )
                
                logger.info(f"Created new production prompt from variant {variant_name}")
            
            return {
                "status": "success",
                "promoted_variant": variant_name,
                "metric": metric,
                "value": metric_value,
                "prompt_id": production_prompt_id,
                "prompt_version": updated_prompt.get("metadata", {}).get("version")
            }
            
        except Exception as e:
            logger.error(f"Error promoting variant: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced compliance detection predictor")
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/compliance_config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="development",
        choices=["development", "staging", "production"],
        help="Environment to use"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="classify",
        choices=["classify", "ab_test", "promote"],
        help="Operation mode"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="Message to classify (for classify mode)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to file with messages (for classify mode) or test data (for ab_test mode)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["basic", "detailed", "step_by_step", "few_shot", "production"],
        help="Prompt variant to use (for classify mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output"
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="all",
        choices=["random", "deterministic", "all"],
        help="Sampling method for A/B testing"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        help="Metric to use for promotion"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CompliancePredictor(
        config_path=args.config,
        environment=args.environment,
        initialize_experiment=(args.mode in ["ab_test", "promote"])
    )
    
    # Run requested operation
    if args.mode == "classify":
        if args.message:
            # Classify a single message
            result = predictor.classify_message(
                message=args.message,
                prompt_variant=args.variant,
                output_format="json",
                track_in_mlflow=True
            )
            print(json.dumps(result, indent=2))
            
        elif args.file:
            # Classify messages from a file
            if args.file.endswith('.csv'):
                df = pd.read_csv(args.file)
                messages = df["text"].tolist()
            elif args.file.endswith('.json'):
                with open(args.file, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    messages = [item.get("text", "") for item in data]
                else:
                    messages = data.get("messages", [])
            else:
                # Assume plain text, one message per line
                with open(args.file, "r") as f:
                    messages = [line.strip() for line in f if line.strip()]
            
            results = []
            for message in tqdm(messages, desc="Classifying messages"):
                result = predictor.classify_message(
                    message=message,
                    prompt_variant=args.variant,
                    output_format="json",
                    track_in_mlflow=True
                )
                results.append(result)
            
            # Save results
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}")
            else:
                print(json.dumps(results, indent=2))
        
        else:
            print("Please provide a message or file to classify")
    
    elif args.mode == "ab_test":
        if not args.file:
            print("Please provide a test data file for A/B testing")
            return
        
        results = predictor.run_ab_test(
            test_data_path=args.file,
            output_path=args.output,
            sampling_method=args.sampling
        )
        
        print("A/B Test Results:")
        print(json.dumps(results["summary"], indent=2))
        print(f"MLflow Run ID: {results['run_id']}")
    
    elif args.mode == "promote":
        result = predictor.promote_best_variant(metric=args.metric)
        print("Promotion Result:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()