#!/usr/bin/env python
"""
Synthetic data generation script for compliance breach detection.
Generates labeled examples of compliant and non-compliant messages.
"""

import os
import json
import yaml
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import google.cloud.aiplatform as vertexai
from google.cloud.aiplatform.prediction import PredictionServiceClient
from google.protobuf.json_format import MessageToDict
import numpy as np
import pandas as pd

from llm_ops_pipeline.utils.logging import setup_logging
from llm_ops_pipeline.config.config import load_config

# Setup argument parser
parser = argparse.ArgumentParser(description="Generate synthetic data for compliance detection")
parser.add_argument("--config", type=str, default="workflows/compliance_detection/configs/compliance_config.yaml", 
                    help="Path to configuration file")
parser.add_argument("--count", type=int, default=1000, 
                    help="Number of examples to generate")
parser.add_argument("--output-dir", type=str, default="data/compliance", 
                    help="Output directory for generated data")
parser.add_argument("--split", type=str, default="0.7,0.15,0.15", 
                    help="Train/validation/test split ratios")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
logger = setup_logging(name="data_generation", level=config.get("monitoring", {}).get("log_level", "INFO"))

# Initialize Vertex AI
vertexai.init(
    project=config["model"]["project_id"],
    location=config["model"]["location"],
)

# Prompt templates for data generation
SYSTEM_PROMPT = """You are a data generation assistant creating synthetic examples of short messages that may contain compliance breaches for the financial industry. 
You'll generate realistic examples of the requested category that could appear in workplace chat, email, or text messages."""

GENERATION_PROMPTS = {
    "ETHICAL_BREACH": "Generate a short message (1-3 sentences) that shows an ethical breach in a financial context. Examples include conflicts of interest, misleading clients, or dishonest behavior.",
    "ILLEGAL_ACTIVITY": "Generate a short message (1-3 sentences) that hints at illegal activity in a financial context. Examples include fraud, money laundering, or market manipulation.",
    "REGULATORY_VIOLATION": "Generate a short message (1-3 sentences) that contains a regulatory violation in finance. Examples include sharing material non-public information or violating compliance protocols.",
    "CONFIDENTIAL_INFO": "Generate a short message (1-3 sentences) that inappropriately shares confidential information. Examples include client data, unreleased financial results, or proprietary strategies.",
    "HARASSMENT": "Generate a short message (1-3 sentences) that contains workplace harassment or inappropriate comments.",
    "COMPLIANT": "Generate a short message (1-3 sentences) that is completely compliant and professional in a financial workplace context."
}

def generate_example(category: str, temperature: float = 0.7) -> Dict[str, Any]:
    """Generate a single synthetic example using Vertex AI Gemini"""
    prompt = GENERATION_PROMPTS[category]
    
    model = vertexai.GenerativeModel(
        model_name=f"models/{config['model']['model_id']}",
    )
    
    response = model.generate_content(
        [SYSTEM_PROMPT, prompt],
        generation_config={
            "temperature": temperature,
            "max_output_tokens": 100,
            "top_p": 0.95,
            "top_k": 40,
        }
    )
    
    text = response.text.strip()
    
    # Add a small chance of the model making a classification error
    if random.random() < 0.03 and category != "COMPLIANT":
        # Occasionally make the text completely benign despite the non-compliant label
        model = vertexai.GenerativeModel(model_name=f"models/{config['model']['model_id']}")
        response = model.generate_content(
            [SYSTEM_PROMPT, GENERATION_PROMPTS["COMPLIANT"]],
            generation_config={"temperature": temperature}
        )
        text = response.text.strip()
    
    return {
        "text": text,
        "label": category,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "source": "synthetic",
            "generator": config['model']['model_id'],
            "temperature": temperature
        }
    }

def generate_dataset(count: int, split: List[float]) -> Dict[str, List[Dict]]:
    """Generate a complete dataset with the specified number of examples"""
    logger.info(f"Generating {count} synthetic examples...")
    
    # Calculate category distribution (slightly fewer compliant examples)
    category_weights = {
        "ETHICAL_BREACH": 0.15,
        "ILLEGAL_ACTIVITY": 0.15,
        "REGULATORY_VIOLATION": 0.2,
        "CONFIDENTIAL_INFO": 0.15,
        "HARASSMENT": 0.15,
        "COMPLIANT": 0.2
    }
    
    category_counts = {cat: int(weight * count) for cat, weight in category_weights.items()}
    # Adjust to ensure we get the exact count
    remaining = count - sum(category_counts.values())
    category_counts["COMPLIANT"] += remaining
    
    all_examples = []
    
    # Generate examples for each category
    with ThreadPoolExecutor(max_workers=10) as executor:
        for category, cat_count in category_counts.items():
            logger.info(f"Generating {cat_count} examples for category: {category}")
            
            # Submit generation tasks
            futures = []
            for i in range(cat_count):
                # Vary temperature slightly for diversity
                temp = 0.7 + (random.random() * 0.3)
                futures.append(executor.submit(generate_example, category, temp))
            
            # Collect results
            for future in futures:
                try:
                    example = future.result()
                    all_examples.append(example)
                except Exception as e:
                    logger.error(f"Error generating example: {str(e)}")
    
    # Shuffle the dataset
    random.shuffle(all_examples)
    
    # Split into train/val/test
    train_end = int(split[0] * len(all_examples))
    val_end = train_end + int(split[1] * len(all_examples))
    
    return {
        "train": all_examples[:train_end],
        "validation": all_examples[train_end:val_end],
        "test": all_examples[val_end:]
    }

def save_dataset(dataset: Dict[str, List[Dict]], output_dir: str):
    """Save the dataset to disk in multiple formats"""
    base_dir = Path(output_dir)
    
    # Ensure directories exist
    for split in dataset.keys():
        (base_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Save each split in JSON format (for DVC tracking)
    for split_name, examples in dataset.items():
        # JSON format (raw)
        with open(base_dir / split_name / "data.json", 'w') as f:
            json.dump(examples, f, indent=2)
        
        # CSV format (for easier inspection/analysis)
        df = pd.DataFrame([{
            "text": ex["text"],
            "label": ex["label"],
            "timestamp": ex["timestamp"]
        } for ex in examples])
        df.to_csv(base_dir / split_name / "data.csv", index=False)
        
        # Calculate and save statistics
        stats = {
            "count": len(examples),
            "category_distribution": {cat: 0 for cat in config["data"]["categories"]},
            "avg_length": sum(len(ex["text"].split()) for ex in examples) / len(examples),
            "generated_at": datetime.now().isoformat()
        }
        
        for ex in examples:
            stats["category_distribution"][ex["label"]] += 1
        
        with open(base_dir / split_name / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    logger.info(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    # Parse split ratios
    split_ratios = [float(x) for x in args.split.split(",")]
    assert len(split_ratios) == 3 and sum(split_ratios) == 1.0, "Split must be three values that sum to 1.0"
    
    # Generate and save dataset
    dataset = generate_dataset(args.count, split_ratios)
    save_dataset(dataset, args.output_dir)
    
    # Initialize DVC tracking
    if os.system("which dvc >/dev/null 2>&1") == 0:
        os.chdir(Path(args.output_dir).parent)
        if not os.path.exists(".dvc"):
            os.system("dvc init")
        
        data_dir = Path(args.output_dir).name
        os.system(f"dvc add {data_dir}")
        os.system(f"dvc push")
        
        logger.info("Dataset tracked with DVC")
    else:
        logger.warning("DVC not found. Dataset not tracked with version control.")