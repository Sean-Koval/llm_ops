#!/usr/bin/env python
"""
Prompt tuning script for compliance breach detection.
Experiments with different prompt variants and tracks performance in MLflow.
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import random

import google.cloud.aiplatform as vertexai
from google.cloud.aiplatform.prediction import PredictionServiceClient
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from llm_ops_pipeline.utils.logging import setup_logging
from llm_ops_pipeline.config.config import load_config
from llm_ops_pipeline.evaluation.metrics import calculate_f1_score

# Setup argument parser
parser = argparse.ArgumentParser(description="Tune prompts for compliance detection")
parser.add_argument("--config", type=str, default="workflows/compliance_detection/configs/compliance_config.yaml", 
                    help="Path to configuration file")
parser.add_argument("--eval-data", type=str, default="data/compliance/validation/data.json", 
                    help="Path to evaluation data")
parser.add_argument("--prompts-dir", type=str, default="workflows/compliance_detection/prompts", 
                    help="Directory containing prompt templates")
parser.add_argument("--sample-size", type=int, default=100, 
                    help="Number of examples to sample for evaluation")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
logger = setup_logging(name="prompt_tuning", level=config.get("monitoring", {}).get("log_level", "INFO"))

# Initialize Vertex AI
vertexai.init(
    project=config["model"]["project_id"],
    location=config["model"]["location"],
)

# Initialize MLflow
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

# Prompt templates
SYSTEM_PROMPTS = {
    "basic": """You are a compliance detection system for a financial institution. 
Analyze the provided message and determine if it contains any compliance breaches.
Classify the message into exactly one of the following categories:
- ETHICAL_BREACH: Conflicts of interest, misleading clients, dishonest behavior
- ILLEGAL_ACTIVITY: Fraud, money laundering, market manipulation
- REGULATORY_VIOLATION: Sharing material non-public information, violating compliance protocols
- CONFIDENTIAL_INFO: Improper sharing of client data, unreleased financial results, proprietary strategies
- HARASSMENT: Workplace harassment or inappropriate comments
- COMPLIANT: Professional and compliant communication

Return only the category label with no additional text.""",

    "detailed": """You are a compliance detection system for a financial institution with expertise in regulatory requirements. 
Your task is to analyze workplace communications for potential compliance breaches.

For the given message, carefully examine it for signs of:
- ETHICAL_BREACH: Any conflicts of interest, misleading clients, dishonest behavior, or violations of company ethics policies
- ILLEGAL_ACTIVITY: Any hints of fraud, money laundering, insider trading, market manipulation, or other criminal activities
- REGULATORY_VIOLATION: Any sharing of material non-public information, violations of compliance protocols, or breaking industry regulations
- CONFIDENTIAL_INFO: Any improper sharing of client data, unreleased financial results, proprietary strategies, or other sensitive information
- HARASSMENT: Any workplace harassment, discrimination, threats, or inappropriate comments
- COMPLIANT: Communication that is professional and complies with all regulations and policies

Analyze the message carefully. If the message contains any breach, classify it as the most severe category that applies.
If the message is completely professional and raises no compliance concerns, classify it as COMPLIANT.

Return only the category label with no additional text.""",

    "step_by_step": """You are a compliance detection system for a financial institution with expertise in regulatory requirements.
Follow these steps to analyze the message:

1. Read the message carefully and identify any potential compliance issues.
2. Consider each category of compliance breach:
   - ETHICAL_BREACH: Conflicts of interest, misleading clients, dishonest behavior
   - ILLEGAL_ACTIVITY: Fraud, money laundering, market manipulation
   - REGULATORY_VIOLATION: Sharing material non-public information, violating compliance protocols
   - CONFIDENTIAL_INFO: Improper sharing of client data, unreleased financial results, proprietary strategies
   - HARASSMENT: Workplace harassment or inappropriate comments

3. If the message contains any breach, classify it as the most severe category that applies.
4. If the message is completely professional and raises no compliance concerns, classify it as COMPLIANT.

Return only the category label with no additional text.""",

    "few_shot": """You are a compliance detection system for a financial institution. 
Analyze the provided message and determine if it contains any compliance breaches.
Classify the message into exactly one of the following categories:
- ETHICAL_BREACH: Conflicts of interest, misleading clients, dishonest behavior
- ILLEGAL_ACTIVITY: Fraud, money laundering, market manipulation
- REGULATORY_VIOLATION: Sharing material non-public information, violating compliance protocols
- CONFIDENTIAL_INFO: Improper sharing of client data, unreleased financial results, proprietary strategies
- HARASSMENT: Workplace harassment or inappropriate comments
- COMPLIANT: Professional and compliant communication

Here are some examples:

Example 1:
Message: "I can guarantee you'll make at least 20% returns on this investment strategy."
Classification: ETHICAL_BREACH

Example 2:
Message: "Let's move these funds through our offshore accounts to avoid detection."
Classification: ILLEGAL_ACTIVITY

Example 3:
Message: "The earnings report looks great - tell your clients to buy before it's released tomorrow."
Classification: REGULATORY_VIOLATION

Example 4:
Message: "Here's the complete client list with their account balances and SSNs."
Classification: CONFIDENTIAL_INFO

Example 5:
Message: "That new analyst is only here because she's attractive, not because of her skills."
Classification: HARASSMENT

Example 6:
Message: "The quarterly report was published this morning, and it shows strong performance as expected."
Classification: COMPLIANT

Now, classify the following message. Return only the category label with no additional text."""
}

def load_evaluation_data(path: str, sample_size: int) -> List[Dict]:
    """Load evaluation data and sample a subset"""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Ensure balanced sampling
    categories = config["data"]["categories"]
    samples_per_category = sample_size // len(categories)
    
    sampled_data = []
    for category in categories:
        category_data = [item for item in data if item["label"] == category]
        if len(category_data) > samples_per_category:
            sampled_data.extend(random.sample(category_data, samples_per_category))
        else:
            sampled_data.extend(category_data)
    
    # If we don't have enough examples, take random samples to reach the desired count
    if len(sampled_data) < sample_size:
        remaining = sample_size - len(sampled_data)
        remaining_data = [item for item in data if item not in sampled_data]
        if remaining_data:
            sampled_data.extend(random.sample(remaining_data, min(remaining, len(remaining_data))))
    
    return sampled_data

def classify_message(message: str, system_prompt: str) -> str:
    """Classify a message using the Gemini model"""
    model = vertexai.GenerativeModel(
        model_name=f"models/{config['model']['model_id']}",
    )
    
    response = model.generate_content(
        [system_prompt, message],
        generation_config={
            "temperature": config["model"]["temperature"],
            "max_output_tokens": config["model"]["max_output_tokens"],
            "top_p": config["model"]["top_p"],
            "top_k": config["model"]["top_k"],
        }
    )
    
    # Extract just the category label
    prediction = response.text.strip()
    
    # Handle potential format issues by extracting just the category
    for category in config["data"]["categories"]:
        if category in prediction:
            return category
    
    # Default to COMPLIANT if no category is found
    return "COMPLIANT"

def evaluate_prompt(prompt_name: str, system_prompt: str, eval_data: List[Dict]) -> Dict[str, Any]:
    """Evaluate a prompt on the validation dataset"""
    logger.info(f"Evaluating prompt: {prompt_name}")
    
    predictions = []
    ground_truth = []
    texts = []
    
    # Run predictions
    for i, example in enumerate(eval_data):
        if i % 10 == 0:
            logger.info(f"Processing example {i}/{len(eval_data)}")
        
        try:
            prediction = classify_message(example["text"], system_prompt)
            predictions.append(prediction)
            ground_truth.append(example["label"])
            texts.append(example["text"])
        except Exception as e:
            logger.error(f"Error processing example {i}: {str(e)}")
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='macro'
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average=None, labels=config["data"]["categories"]
    )
    
    # Create confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=config["data"]["categories"])
    
    # Create results dictionary
    results = {
        "prompt_name": prompt_name,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_by_category": {cat: float(p) for cat, p in zip(config["data"]["categories"], precision)},
        "recall_by_category": {cat: float(r) for cat, r in zip(config["data"]["categories"], recall)},
        "f1_by_category": {cat: float(f) for cat, f in zip(config["data"]["categories"], f1)},
        "confusion_matrix": cm.tolist(),
        "predictions": [{"text": t, "true": g, "pred": p} for t, g, p in zip(texts, ground_truth, predictions)]
    }
    
    return results

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], title: str) -> plt.Figure:
    """Create a confusion matrix plot"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    return plt.gcf()

def log_results_to_mlflow(prompt_name: str, results: Dict[str, Any], system_prompt: str):
    """Log evaluation results to MLflow"""
    run_name = f"{config['mlflow']['run_name_prefix']}{prompt_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("prompt_name", prompt_name)
        mlflow.log_param("prompt_text", system_prompt)
        mlflow.log_param("model_id", config["model"]["model_id"])
        mlflow.log_param("temperature", config["model"]["temperature"])
        mlflow.log_param("sample_size", len(results["predictions"]))
        
        # Log metrics
        mlflow.log_metric("accuracy", results["accuracy"])
        mlflow.log_metric("precision_macro", results["precision_macro"])
        mlflow.log_metric("recall_macro", results["recall_macro"])
        mlflow.log_metric("f1_macro", results["f1_macro"])
        
        for cat in config["data"]["categories"]:
            mlflow.log_metric(f"precision_{cat}", results["precision_by_category"][cat])
            mlflow.log_metric(f"recall_{cat}", results["recall_by_category"][cat])
            mlflow.log_metric(f"f1_{cat}", results["f1_by_category"][cat])
        
        # Log confusion matrix as figure
        cm_fig = plot_confusion_matrix(
            np.array(results["confusion_matrix"]), 
            config["data"]["categories"],
            f"Confusion Matrix - {prompt_name}"
        )
        mlflow.log_figure(cm_fig, f"confusion_matrix_{prompt_name}.png")
        
        # Log predictions
        predictions_df = pd.DataFrame(results["predictions"])
        mlflow.log_table(data=predictions_df, artifact_file="predictions.json")
        
        # Log tags
        for tag_key, tag_value in config["mlflow"]["tags"].items():
            mlflow.set_tag(tag_key, tag_value)
        
        # Save the best prompt template as artifact
        with open("best_prompt.txt", "w") as f:
            f.write(system_prompt)
        mlflow.log_artifact("best_prompt.txt")
        
        # Close the figure to avoid memory leaks
        plt.close(cm_fig)

def save_prompt_to_file(prompt_name: str, system_prompt: str, metrics: Dict[str, float], prompt_dir: str):
    """Save the prompt and its metrics to a file"""
    prompt_path = Path(prompt_dir) / f"{prompt_name}.json"
    
    prompt_data = {
        "name": prompt_name,
        "system_prompt": system_prompt,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"]
        },
        "updated_at": datetime.now().isoformat()
    }
    
    with open(prompt_path, 'w') as f:
        json.dump(prompt_data, f, indent=2)

def main():
    # Create prompts directory if it doesn't exist
    Path(args.prompts_dir).mkdir(parents=True, exist_ok=True)
    
    # Load evaluation data
    eval_data = load_evaluation_data(args.eval_data, args.sample_size)
    logger.info(f"Loaded {len(eval_data)} examples for evaluation")
    
    best_f1 = 0
    best_prompt = None
    best_prompt_name = None
    
    # Evaluate each prompt template
    for prompt_name, system_prompt in SYSTEM_PROMPTS.items():
        results = evaluate_prompt(prompt_name, system_prompt, eval_data)
        
        # Log results to MLflow
        log_results_to_mlflow(prompt_name, results, system_prompt)
        
        # Save prompt to file
        save_prompt_to_file(
            prompt_name, 
            system_prompt, 
            {
                "accuracy": results["accuracy"],
                "precision_macro": results["precision_macro"],
                "recall_macro": results["recall_macro"],
                "f1_macro": results["f1_macro"]
            },
            args.prompts_dir
        )
        
        # Check if this is the best prompt
        if results["f1_macro"] > best_f1:
            best_f1 = results["f1_macro"]
            best_prompt = system_prompt
            best_prompt_name = prompt_name
    
    # Save the best prompt as "production" prompt
    if best_prompt:
        logger.info(f"Best prompt: {best_prompt_name} (F1: {best_f1:.4f})")
        
        production_path = Path(args.prompts_dir) / "production.json"
        with open(production_path, 'w') as f:
            json.dump({
                "name": f"{best_prompt_name}_production",
                "system_prompt": best_prompt,
                "metrics": {
                    "f1_macro": best_f1
                },
                "updated_at": datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Saved production prompt to {production_path}")

if __name__ == "__main__":
    main()