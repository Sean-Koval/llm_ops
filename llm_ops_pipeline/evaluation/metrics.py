"""Evaluation metrics for LLM Ops Pipeline."""

from typing import Dict, List, Optional, Union, Any, Callable

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

from llm_ops_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def compute_perplexity(
    model: PreTrainedModel,
    dataset: Dataset,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    """Compute perplexity on a dataset.
    
    Args:
        model: Model to evaluate
        dataset: Dataset for evaluation
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        
    Returns:
        Perplexity score
    """
    model.eval()
    model = model.to(device)
    
    total_loss = 0.0
    total_length = 0
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Sum loss for all non-masked tokens
            total_loss += loss.item() * input_ids.size(0)
            total_length += input_ids.size(0)
    
    perplexity = torch.exp(torch.tensor(total_loss / total_length))
    
    logger.info(f"Perplexity: {perplexity.item():.4f}")
    return perplexity.item()


def evaluate_generated_text(
    references: List[str],
    predictions: List[str],
    metrics: List[str] = ["bleu", "rouge"]
) -> Dict[str, float]:
    """Evaluate generated text using NLG metrics.
    
    Args:
        references: List of reference texts
        predictions: List of predicted texts
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric scores
    """
    from datasets import load_metric
    
    results = {}
    
    if "bleu" in metrics:
        bleu = load_metric("bleu")
        bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
        results["bleu"] = bleu_score["bleu"]
    
    if "rouge" in metrics:
        rouge = load_metric("rouge")
        rouge_score = rouge.compute(predictions=predictions, references=references)
        results["rouge1"] = rouge_score["rouge1"].mid.fmeasure
        results["rouge2"] = rouge_score["rouge2"].mid.fmeasure
        results["rougeL"] = rouge_score["rougeL"].mid.fmeasure
    
    logger.info(f"Text generation metrics: {results}")
    return results


def evaluate_classification(
    labels: List[int],
    predictions: List[int],
    label_names: Optional[List[str]] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Evaluate classification predictions.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        label_names: Optional list of label names
        
    Returns:
        Dictionary of metric scores
    """
    accuracy = accuracy_score(labels, predictions)
    
    # Get precision, recall, F1 score
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    
    # Per-class metrics if label names are provided
    per_class_metrics = None
    if label_names:
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(labels, predictions, average=None)
            
        per_class_metrics = {}
        for i, label in enumerate(label_names):
            per_class_metrics[label] = {
                "precision": precision_per_class[i],
                "recall": recall_per_class[i],
                "f1": f1_per_class[i],
                "support": support_per_class[i]
            }
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    if per_class_metrics:
        results["per_class"] = per_class_metrics
    
    logger.info(f"Classification metrics: accuracy={accuracy:.4f}, f1={f1:.4f}")
    return results