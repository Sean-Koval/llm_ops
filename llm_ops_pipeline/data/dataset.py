"""Dataset handling for LLM Ops Pipeline."""

import os
from typing import Dict, List, Optional, Union, Any

from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from llm_ops_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def load_data(
    data_path: str,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Union[Dataset, DatasetDict]:
    """Load dataset from various file formats (csv, json, parquet, etc.)
    
    Args:
        data_path: Path to the dataset file or directory
        split: Optional dataset split to load (train, validation, test)
        cache_dir: Optional directory to cache the dataset
        
    Returns:
        Loaded dataset object
    """
    logger.info(f"Loading dataset from {data_path}")
    
    try:
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
            dataset = Dataset.from_pandas(df)
        elif data_path.endswith((".json", ".jsonl")):
            dataset = load_dataset("json", data_files=data_path, split=split, cache_dir=cache_dir)
        elif data_path.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files=data_path, split=split, cache_dir=cache_dir)
        else:
            # Try to load as a Hugging Face dataset
            dataset = load_dataset(data_path, split=split, cache_dir=cache_dir)
            
        logger.info(f"Successfully loaded dataset with {len(dataset)} examples")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise


def preprocess_for_llm(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    max_length: int = 512,
    is_train: bool = True
) -> Dataset:
    """Preprocess text dataset for language model training/inference.
    
    Args:
        dataset: Input dataset to preprocess
        tokenizer: Tokenizer to use for preprocessing
        text_column: Name of the column containing the text data
        max_length: Maximum sequence length
        is_train: Whether this is for training (enables truncation)
        
    Returns:
        Preprocessed dataset
    """
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    logger.info(f"Preprocessing dataset with {len(dataset)} examples")
    
    # Remove empty examples
    dataset = dataset.filter(lambda x: x[text_column] is not None and len(str(x[text_column])) > 0)
    
    # Tokenize all examples
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col != text_column]
    )
    
    logger.info(f"Finished preprocessing, dataset now has {len(tokenized_dataset)} examples")
    return tokenized_dataset


class DataCollator:
    """Collate examples for language model training."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, mlm: bool = False):
        """Initialize data collator.
        
        Args:
            tokenizer: Tokenizer to use
            mlm: Whether to use masked language modeling objective
        """
        self.tokenizer = tokenizer
        self.mlm = mlm
        
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate examples into batch.
        
        Args:
            examples: List of tokenized examples
            
        Returns:
            Batched examples
        """
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=8
        )
        
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
            
        return batch