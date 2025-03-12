#!/usr/bin/env python
"""
Compliance predictor module to classify messages using Vertex AI Gemini.
Can be used as library or CLI tool.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import time

import google.cloud.aiplatform as vertexai
from google.cloud.aiplatform.prediction import PredictionServiceClient
import mlflow
import pandas as pd

from llm_ops_pipeline.utils.logging import setup_logging
from llm_ops_pipeline.config.config import load_config
from llm_ops_pipeline.inference.predictor import LLMPredictor

class CompliancePredictor:
    """Predictor for classifying text for compliance breaches."""
    
    def __init__(self, config_path: str):
        """Initialize the compliance predictor."""
        self.config = load_config(config_path)
        self.logger = setup_logging(
            name="compliance_predictor", 
            level=self.config.get("monitoring", {}).get("log_level", "INFO")
        )
        
        # Initialize Vertex AI
        vertexai.init(
            project=self.config["model"]["project_id"],
            location=self.config["model"]["location"],
        )
        
        # Load production prompt
        prompts_dir = Path(config_path).parent.parent / "prompts"
        production_prompt_path = prompts_dir / "production.json"
        
        if production_prompt_path.exists():
            with open(production_prompt_path, 'r') as f:
                prompt_data = json.load(f)
                self.system_prompt = prompt_data["system_prompt"]
                self.logger.info(f"Loaded production prompt: {prompt_data['name']}")
        else:
            # Fallback to basic prompt
            self.system_prompt = """You are a compliance detection system for a financial institution. 
Analyze the provided message and determine if it contains any compliance breaches.
Classify the message into exactly one of the following categories:
- ETHICAL_BREACH: Conflicts of interest, misleading clients, dishonest behavior
- ILLEGAL_ACTIVITY: Fraud, money laundering, market manipulation
- REGULATORY_VIOLATION: Sharing material non-public information, violating compliance protocols
- CONFIDENTIAL_INFO: Improper sharing of client data, unreleased financial results, proprietary strategies
- HARASSMENT: Workplace harassment or inappropriate comments
- COMPLIANT: Professional and compliant communication

Return only the category label with no additional text."""
            self.logger.warning("Production prompt not found. Using fallback prompt.")
        
        # Initialize model client
        self.model = vertexai.GenerativeModel(
            model_name=f"models/{self.config['model']['model_id']}",
        )
        
        # Set up MLflow for tracking
        if self.config.get("mlflow", {}).get("tracking_uri"):
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
    
    def predict(self, text: str, log_prediction: bool = True) -> Dict[str, Any]:
        """
        Classify a single message for compliance breaches.
        
        Args:
            text: The message text to classify
            log_prediction: Whether to log the prediction to MLflow
            
        Returns:
            Dictionary with classification result
        """
        start_time = time.time()
        
        # Generate prediction
        response = self.model.generate_content(
            [self.system_prompt, text],
            generation_config={
                "temperature": self.config["model"]["temperature"],
                "max_output_tokens": self.config["model"]["max_output_tokens"],
                "top_p": self.config["model"]["top_p"],
                "top_k": self.config["model"]["top_k"],
            }
        )
        
        # Extract prediction
        raw_prediction = response.text.strip()
        
        # Process the prediction to extract just the category
        for category in self.config["data"]["categories"]:
            if category in raw_prediction:
                prediction = category
                break
        else:
            # Default to COMPLIANT if no category is found
            prediction = "COMPLIANT"
        
        # Calculate response time
        latency_ms = (time.time() - start_time) * 1000
        
        # Create result
        result = {
            "text": text,
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "confidence": 1.0,  # Gemini doesn't return confidence scores directly
            "timestamp": datetime.now().isoformat(),
            "model_id": self.config["model"]["model_id"],
            "latency_ms": latency_ms
        }
        
        # Log prediction if requested
        if log_prediction and self.config.get("mlflow", {}).get("tracking_uri"):
            self._log_prediction(result)
        
        return result
    
    def predict_batch(self, texts: List[str], log_predictions: bool = True) -> List[Dict[str, Any]]:
        """
        Classify multiple messages for compliance breaches.
        
        Args:
            texts: List of message texts to classify
            log_predictions: Whether to log the predictions to MLflow
            
        Returns:
            List of dictionaries with classification results
        """
        results = []
        
        for text in texts:
            result = self.predict(text, log_prediction=False)
            results.append(result)
        
        # Log batch predictions if requested
        if log_predictions and self.config.get("mlflow", {}).get("tracking_uri"):
            self._log_batch_predictions(results)
        
        return results
    
    def _log_prediction(self, result: Dict[str, Any]):
        """Log a single prediction to MLflow."""
        with mlflow.start_run(run_name=f"prediction-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            # Log parameters
            mlflow.log_param("model_id", self.config["model"]["model_id"])
            mlflow.log_param("temperature", self.config["model"]["temperature"])
            
            # Log metrics
            mlflow.log_metric("latency_ms", result["latency_ms"])
            
            # Log prediction as artifact
            with open("prediction.json", "w") as f:
                json.dump(result, f, indent=2)
            mlflow.log_artifact("prediction.json")
            
            # Log tags
            mlflow.set_tag("prediction_type", "single")
            mlflow.set_tag("prediction_result", result["prediction"])
    
    def _log_batch_predictions(self, results: List[Dict[str, Any]]):
        """Log batch predictions to MLflow."""
        with mlflow.start_run(run_name=f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            # Log parameters
            mlflow.log_param("model_id", self.config["model"]["model_id"])
            mlflow.log_param("temperature", self.config["model"]["temperature"])
            mlflow.log_param("batch_size", len(results))
            
            # Log metrics
            avg_latency = sum(r["latency_ms"] for r in results) / len(results)
            mlflow.log_metric("avg_latency_ms", avg_latency)
            
            # Count predictions by category
            category_counts = {cat: 0 for cat in self.config["data"]["categories"]}
            for result in results:
                category_counts[result["prediction"]] += 1
            
            for cat, count in category_counts.items():
                mlflow.log_metric(f"count_{cat}", count)
            
            # Log predictions as artifact
            with open("predictions.json", "w") as f:
                json.dump(results, f, indent=2)
            mlflow.log_artifact("predictions.json")
            
            # Log tags
            mlflow.set_tag("prediction_type", "batch")
            mlflow.set_tag("timestamp", datetime.now().isoformat())

def main():
    """Command-line interface for the compliance predictor."""
    parser = argparse.ArgumentParser(description="Classify messages for compliance breaches")
    parser.add_argument("--config", type=str, default="workflows/compliance_detection/configs/compliance_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--input", type=str, required=True,
                        help="Input text or path to file with messages (one per line)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for predictions (JSON format)")
    parser.add_argument("--batch", action="store_true",
                        help="Process input as batch (file with one message per line)")
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CompliancePredictor(args.config)
    
    if args.batch:
        # Process batch from file
        with open(args.input, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = predictor.predict_batch(texts)
        
        # Print summary
        print(f"Processed {len(results)} messages:")
        for category in predictor.config["data"]["categories"]:
            count = sum(1 for r in results if r["prediction"] == category)
            print(f"  {category}: {count} ({count/len(results)*100:.1f}%)")
    else:
        # Process single text
        if os.path.isfile(args.input):
            with open(args.input, 'r') as f:
                text = f.read().strip()
        else:
            text = args.input
        
        result = predictor.predict(text)
        
        # Print result
        print(f"Text: {text[:100]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Latency: {result['latency_ms']:.2f} ms")
    
    # Save output if requested
    if args.output:
        with open(args.output, 'w') as f:
            if args.batch:
                json.dump(results, f, indent=2)
            else:
                json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()