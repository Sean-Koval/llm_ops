"""Model loading utilities for LLM Ops Pipeline."""

from typing import Dict, Optional, Tuple, Union

import torch
from transformers import (
    AutoModel, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

from llm_ops_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    model_type: str = "causal_lm",
    revision: Optional[str] = None,
    use_auth_token: bool = False,
    tokenizer_name: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **model_kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer from Hugging Face Hub.
    
    Args:
        model_name: Name of the model to load
        model_type: Type of model to load (base, causal_lm, seq2seq_lm)
        revision: Optional specific model revision to load
        use_auth_token: Whether to use the Hugging Face auth token
        tokenizer_name: Optional separate tokenizer name
        device: Device to load the model onto
        **model_kwargs: Additional keyword arguments for model loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model {model_name} of type {model_type}")
    
    # Load tokenizer
    tokenizer_to_load = tokenizer_name if tokenizer_name else model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_to_load, 
        revision=revision,
        use_auth_token=use_auth_token
    )
    
    # Ensure the tokenizer has pad token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model based on type
    if model_type == "base":
        model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            use_auth_token=use_auth_token,
            **model_kwargs
        )
    elif model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            use_auth_token=use_auth_token,
            **model_kwargs
        )
    elif model_type == "seq2seq_lm":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            revision=revision,
            use_auth_token=use_auth_token,
            **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = model.to(device)
    logger.info(f"Model and tokenizer loaded successfully, model moved to {device}")
    
    return model, tokenizer