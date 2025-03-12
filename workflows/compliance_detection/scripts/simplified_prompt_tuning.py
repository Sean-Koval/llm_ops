#!/usr/bin/env python
"""
Simplified prompt tuning script for demonstration purposes.
This script simulates prompt experimentation and MLflow tracking.
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Mock MLflow for simplified demonstration
class MockMLflow:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.runs = []
        self.current_run = None
        print(f"MLflow initialized with experiment: {experiment_name}")
    
    def start_run(self, run_name=None):
        self.current_run = {
            "run_id": f"run_{len(self.runs) + 1}",
            "run_name": run_name or f"Run {len(self.runs) + 1}",
            "start_time": datetime.now(),
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": []
        }
        print(f"Started MLflow run: {self.current_run['run_name']}")
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.runs.append(self.current_run)
        print(f"Finished MLflow run: {self.current_run['run_name']}")
        self.current_run = None
    
    def log_param(self, key, value):
        self.current_run["params"][key] = value
        print(f"  Logged parameter: {key}={value}")
    
    def log_metric(self, key, value):
        self.current_run["metrics"][key] = value
        print(f"  Logged metric: {key}={value:.4f}")
    
    def set_tag(self, key, value):
        self.current_run["tags"][key] = value
    
    def log_figure(self, fig, artifact_file):
        self.current_run["artifacts"].append(artifact_file)
        fig.savefig(artifact_file)
        print(f"  Saved figure: {artifact_file}")
    
    def log_artifact(self, local_path):
        self.current_run["artifacts"].append(local_path)
        print(f"  Logged artifact: {local_path}")
    
    def get_best_run(self, metric="f1_macro"):
        if not self.runs:
            return None
        sorted_runs = sorted(self.runs, key=lambda x: x["metrics"].get(metric, 0), reverse=True)
        return sorted_runs[0]

# Prompt templates for testing
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

def load_data(file_path):
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def simulate_prediction(text, system_prompt, category):
    """Simulate a model prediction with controlled accuracy"""
    # Simulate different performance for different prompts
    prompt_accuracy = {
        "basic": 0.82,
        "detailed": 0.89,
        "step_by_step": 0.87,
        "few_shot": 0.92
    }
    
    # Get the expected accuracy for this prompt
    accuracy = prompt_accuracy.get(system_prompt, 0.8)
    
    # For demonstration purposes, we'll simulate the model's prediction
    # In a real scenario, this would call the Vertex AI API
    if random.random() < accuracy:
        # Correct prediction
        return category
    else:
        # Incorrect prediction, choose a random different category
        categories = ["ETHICAL_BREACH", "ILLEGAL_ACTIVITY", "REGULATORY_VIOLATION", 
                      "CONFIDENTIAL_INFO", "HARASSMENT", "COMPLIANT"]
        categories.remove(category)
        return random.choice(categories)

def evaluate_prompt(prompt_name, system_prompt, data, mlflow_client):
    """Evaluate a prompt on the validation dataset"""
    print(f"\nEvaluating prompt: {prompt_name}")
    
    predictions = []
    ground_truth = []
    texts = []
    latencies = []
    
    # Process each example
    for example in data:
        start_time = time.time()
        
        # Simulate prediction
        prediction = simulate_prediction(example["text"], prompt_name, example["label"])
        
        # Calculate simulated latency (add randomness)
        latency = (time.time() - start_time) * 1000 + random.uniform(100, 300)
        
        predictions.append(prediction)
        ground_truth.append(example["label"])
        texts.append(example["text"])
        latencies.append(latency)
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='macro'
    )
    
    # Create confusion matrix
    categories = ["ETHICAL_BREACH", "ILLEGAL_ACTIVITY", "REGULATORY_VIOLATION", 
                 "CONFIDENTIAL_INFO", "HARASSMENT", "COMPLIANT"]
    cm = confusion_matrix(ground_truth, predictions, labels=categories)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title(f'Confusion Matrix - {prompt_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Log results to MLflow
    with mlflow_client.start_run(run_name=f"prompt-{prompt_name}"):
        # Log parameters
        mlflow_client.log_param("prompt_name", prompt_name)
        mlflow_client.log_param("model", "gemini-1.5-flash-simulation")
        mlflow_client.log_param("examples_count", len(data))
        
        # Log metrics
        mlflow_client.log_metric("accuracy", accuracy)
        mlflow_client.log_metric("precision_macro", precision_macro)
        mlflow_client.log_metric("recall_macro", recall_macro)
        mlflow_client.log_metric("f1_macro", f1_macro)
        mlflow_client.log_metric("avg_latency_ms", np.mean(latencies))
        
        # Log confusion matrix
        cm_file = f"confusion_matrix_{prompt_name}.png"
        mlflow_client.log_figure(plt.gcf(), cm_file)
        
        # Close plot to avoid memory issues
        plt.close()
        
        # Save prompt to file
        prompt_dir = "workflows/compliance_detection/prompts"
        os.makedirs(prompt_dir, exist_ok=True)
        
        prompt_file = f"{prompt_dir}/{prompt_name}.json"
        with open(prompt_file, 'w') as f:
            json.dump({
                "name": prompt_name,
                "system_prompt": system_prompt,
                "metrics": {
                    "accuracy": accuracy,
                    "precision_macro": precision_macro,
                    "recall_macro": recall_macro,
                    "f1_macro": f1_macro
                },
                "updated_at": datetime.now().isoformat()
            }, f, indent=2)
        
        mlflow_client.log_artifact(prompt_file)
    
    return {
        "prompt_name": prompt_name,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm
    }

def main():
    print("Starting prompt tuning experiments...")
    
    # Load validation data
    validation_data = load_data("data/compliance/validation/data.json")
    print(f"Loaded {len(validation_data)} examples from validation set")
    
    # Initialize MockMLflow
    mlflow_client = MockMLflow("compliance-detection")
    
    results = []
    
    # Evaluate each prompt template
    for prompt_name, system_prompt in SYSTEM_PROMPTS.items():
        result = evaluate_prompt(prompt_name, system_prompt, validation_data, mlflow_client)
        results.append(result)
    
    # Find the best prompt
    best_result = max(results, key=lambda x: x["f1_macro"])
    print("\n" + "="*50)
    print(f"Best prompt: {best_result['prompt_name']} (F1: {best_result['f1_macro']:.4f})")
    
    # Save the best prompt as production prompt
    prompt_dir = "workflows/compliance_detection/prompts"
    best_prompt = SYSTEM_PROMPTS[best_result["prompt_name"]]
    
    with open(f"{prompt_dir}/production.json", 'w') as f:
        json.dump({
            "name": f"{best_result['prompt_name']}_production",
            "system_prompt": best_prompt,
            "metrics": {
                "accuracy": best_result["accuracy"],
                "precision_macro": best_result["precision_macro"],
                "recall_macro": best_result["recall_macro"],
                "f1_macro": best_result["f1_macro"]
            },
            "updated_at": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"Saved production prompt to {prompt_dir}/production.json")
    
    # Print comparison table
    print("\nPrompt Performance Comparison:")
    print("-" * 80)
    print(f"{'Prompt Name':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x["f1_macro"], reverse=True):
        print(f"{result['prompt_name']:<15} {result['accuracy']:<10.4f} {result['precision_macro']:<10.4f} {result['recall_macro']:<10.4f} {result['f1_macro']:<10.4f}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("workflows/compliance_detection/prompts", exist_ok=True)
    os.makedirs("confusion_matrix_plots", exist_ok=True)
    
    main()