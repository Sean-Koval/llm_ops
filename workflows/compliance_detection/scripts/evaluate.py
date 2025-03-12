#!/usr/bin/env python
"""
Evaluation script for the compliance detection system.
Evaluates the model on the test set and generates a comprehensive report.
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from llm_ops_pipeline.utils.logging import setup_logging
from llm_ops_pipeline.config.config import load_config
from workflows.compliance_detection.scripts.compliance_predictor import CompliancePredictor

# Setup argument parser
parser = argparse.ArgumentParser(description="Evaluate compliance detection system")
parser.add_argument("--config", type=str, default="workflows/compliance_detection/configs/compliance_config.yaml", 
                    help="Path to configuration file")
parser.add_argument("--test-data", type=str, default="data/compliance/test/data.json", 
                    help="Path to test data")
parser.add_argument("--output-dir", type=str, default="workflows/compliance_detection/evaluation", 
                    help="Output directory for evaluation results")
parser.add_argument("--full", action="store_true",
                    help="Run full evaluation (slower but more comprehensive)")
args = parser.parse_args()

def load_test_data(path: str) -> List[Dict[str, Any]]:
    """Load test data from JSON file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_model(predictor: CompliancePredictor, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate the model on the test data"""
    texts = [item["text"] for item in test_data]
    ground_truth = [item["label"] for item in test_data]
    
    # Get predictions
    start_time = time.time()
    results = predictor.predict_batch(texts, log_predictions=False)
    end_time = time.time()
    
    predictions = [result["prediction"] for result in results]
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='macro'
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average=None, labels=predictor.config["data"]["categories"]
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=predictor.config["data"]["categories"])
    
    # Calculate latency statistics
    latencies = [result["latency_ms"] for result in results]
    avg_latency = sum(latencies) / len(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    # Error analysis
    errors = []
    for i, (gt, pred) in enumerate(zip(ground_truth, predictions)):
        if gt != pred:
            errors.append({
                "text": texts[i],
                "true_label": gt,
                "predicted_label": pred,
                "index": i
            })
    
    # Generate evaluation results
    evaluation_results = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_by_category": {cat: float(p) for cat, p in zip(predictor.config["data"]["categories"], precision)},
        "recall_by_category": {cat: float(r) for cat, r in zip(predictor.config["data"]["categories"], recall)},
        "f1_by_category": {cat: float(f) for cat, f in zip(predictor.config["data"]["categories"], f1)},
        "confusion_matrix": cm.tolist(),
        "latency": {
            "avg_ms": avg_latency,
            "p50_ms": p50_latency,
            "p95_ms": p95_latency,
            "p99_ms": p99_latency,
            "total_time_s": end_time - start_time,
            "throughput_per_s": len(texts) / (end_time - start_time)
        },
        "errors": errors,
        "error_count": len(errors),
        "error_rate": len(errors) / len(texts),
        "total_examples": len(texts),
        "timestamp": datetime.now().isoformat(),
        "model_id": predictor.config["model"]["model_id"]
    }
    
    return evaluation_results

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], title: str) -> plt.Figure:
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()

def plot_category_metrics(metrics: Dict[str, Dict[str, float]], categories: List[str]) -> plt.Figure:
    """Plot precision, recall, and F1 scores for each category"""
    plt.figure(figsize=(12, 6))
    
    metrics_data = {
        "Precision": [metrics["precision_by_category"][cat] for cat in categories],
        "Recall": [metrics["recall_by_category"][cat] for cat in categories],
        "F1": [metrics["f1_by_category"][cat] for cat in categories]
    }
    
    df = pd.DataFrame(metrics_data, index=categories)
    ax = df.plot(kind='bar', figsize=(12, 6))
    plt.title('Metrics by Category')
    plt.ylabel('Score')
    plt.xlabel('Category')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    return plt.gcf()

def plot_latency_histogram(latencies: List[float]) -> plt.Figure:
    """Plot latency histogram"""
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=30, alpha=0.7, color='blue')
    plt.axvline(np.mean(latencies), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(latencies):.2f} ms')
    plt.axvline(np.percentile(latencies, 95), color='green', linestyle='dashed', linewidth=2, label=f'95th Percentile: {np.percentile(latencies, 95):.2f} ms')
    plt.title('Prediction Latency Distribution')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def save_evaluation_results(results: Dict[str, Any], output_dir: str):
    """Save evaluation results to files"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save full results as JSON
    with open(Path(output_dir) / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary as text
    with open(Path(output_dir) / "evaluation_summary.txt", 'w') as f:
        f.write(f"Compliance Detection Evaluation Summary\n")
        f.write(f"======================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {results['model_id']}\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"  Precision (macro): {results['precision_macro']:.4f}\n")
        f.write(f"  Recall (macro): {results['recall_macro']:.4f}\n")
        f.write(f"  F1 Score (macro): {results['f1_macro']:.4f}\n\n")
        
        f.write(f"Metrics by Category:\n")
        for category in results['precision_by_category'].keys():
            f.write(f"  {category}:\n")
            f.write(f"    Precision: {results['precision_by_category'][category]:.4f}\n")
            f.write(f"    Recall: {results['recall_by_category'][category]:.4f}\n")
            f.write(f"    F1 Score: {results['f1_by_category'][category]:.4f}\n")
        
        f.write(f"\nLatency:\n")
        f.write(f"  Average: {results['latency']['avg_ms']:.2f} ms\n")
        f.write(f"  50th Percentile: {results['latency']['p50_ms']:.2f} ms\n")
        f.write(f"  95th Percentile: {results['latency']['p95_ms']:.2f} ms\n")
        f.write(f"  99th Percentile: {results['latency']['p99_ms']:.2f} ms\n")
        f.write(f"  Throughput: {results['latency']['throughput_per_s']:.2f} requests/second\n\n")
        
        f.write(f"Error Analysis:\n")
        f.write(f"  Total Examples: {results['total_examples']}\n")
        f.write(f"  Error Count: {results['error_count']}\n")
        f.write(f"  Error Rate: {results['error_rate']:.4f}\n\n")
        
        f.write(f"Top Error Examples:\n")
        for i, error in enumerate(results['errors'][:10]):
            f.write(f"  Example {i+1}:\n")
            f.write(f"    Text: {error['text']}\n")
            f.write(f"    True Label: {error['true_label']}\n")
            f.write(f"    Predicted Label: {error['predicted_label']}\n\n")

def log_to_mlflow(results: Dict[str, Any], config: Dict[str, Any], output_dir: str):
    """Log evaluation results to MLflow"""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(f"{config['mlflow']['experiment_name']}_evaluation")
    
    with mlflow.start_run(run_name=f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # Log key metrics
        mlflow.log_metric("accuracy", results["accuracy"])
        mlflow.log_metric("precision_macro", results["precision_macro"])
        mlflow.log_metric("recall_macro", results["recall_macro"])
        mlflow.log_metric("f1_macro", results["f1_macro"])
        
        # Log category metrics
        for category in results["precision_by_category"].keys():
            mlflow.log_metric(f"precision_{category}", results["precision_by_category"][category])
            mlflow.log_metric(f"recall_{category}", results["recall_by_category"][category])
            mlflow.log_metric(f"f1_{category}", results["f1_by_category"][category])
        
        # Log latency metrics
        mlflow.log_metric("latency_avg_ms", results["latency"]["avg_ms"])
        mlflow.log_metric("latency_p50_ms", results["latency"]["p50_ms"])
        mlflow.log_metric("latency_p95_ms", results["latency"]["p95_ms"])
        mlflow.log_metric("latency_p99_ms", results["latency"]["p99_ms"])
        mlflow.log_metric("throughput_per_s", results["latency"]["throughput_per_s"])
        
        # Log error metrics
        mlflow.log_metric("error_count", results["error_count"])
        mlflow.log_metric("error_rate", results["error_rate"])
        mlflow.log_metric("total_examples", results["total_examples"])
        
        # Log parameters
        mlflow.log_param("model_id", results["model_id"])
        mlflow.log_param("evaluation_timestamp", results["timestamp"])
        
        # Log confusion matrix figure
        cm_fig = plot_confusion_matrix(
            np.array(results["confusion_matrix"]),
            list(results["precision_by_category"].keys()),
            "Confusion Matrix"
        )
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        plt.close(cm_fig)
        
        # Log category metrics figure
        cat_fig = plot_category_metrics(results, list(results["precision_by_category"].keys()))
        mlflow.log_figure(cat_fig, "category_metrics.png")
        plt.close(cat_fig)
        
        # Log artifacts
        mlflow.log_artifact(Path(output_dir) / "evaluation_results.json")
        mlflow.log_artifact(Path(output_dir) / "evaluation_summary.txt")
        
        # Log tags
        for tag_key, tag_value in config["mlflow"]["tags"].items():
            mlflow.set_tag(tag_key, tag_value)
        mlflow.set_tag("evaluation_type", "full" if args.full else "standard")

def main():
    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(name="evaluation", level=config.get("monitoring", {}).get("log_level", "INFO"))
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data = load_test_data(args.test_data)
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Initialize predictor
    logger.info("Initializing predictor")
    predictor = CompliancePredictor(args.config)
    
    # Evaluate model
    logger.info("Starting evaluation")
    results = evaluate_model(predictor, test_data)
    logger.info(f"Evaluation complete. Accuracy: {results['accuracy']:.4f}, F1 (macro): {results['f1_macro']:.4f}")
    
    # Save results
    logger.info(f"Saving evaluation results to {args.output_dir}")
    save_evaluation_results(results, args.output_dir)
    
    # Log to MLflow
    if config.get("mlflow", {}).get("tracking_uri"):
        logger.info("Logging results to MLflow")
        log_to_mlflow(results, config, args.output_dir)
    
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main()