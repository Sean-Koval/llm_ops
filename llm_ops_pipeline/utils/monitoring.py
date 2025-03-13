"""Monitoring utilities for LLM Ops Pipeline.

This module provides tools for monitoring and observability, focusing on LLM-specific metrics.
"""

import time
from typing import Dict, Optional, List, Callable, Any
from functools import wraps

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    CollectorRegistry, generate_latest
)

# Create a registry for our metrics
REGISTRY = CollectorRegistry(auto_describe=True)

# Request tracking metrics
REQUEST_COUNT = Counter(
    "llm_request_total",
    "Total number of LLM requests",
    ["endpoint", "model", "status"],
    registry=REGISTRY
)

# Latency metrics
REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "Time taken to process request",
    ["endpoint", "model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    registry=REGISTRY
)

# Token usage metrics
PROMPT_TOKENS = Counter(
    "llm_prompt_tokens_total",
    "Total number of prompt tokens used",
    ["model"],
    registry=REGISTRY
)

COMPLETION_TOKENS = Counter(
    "llm_completion_tokens_total",
    "Total number of completion tokens generated",
    ["model"],
    registry=REGISTRY
)

TOTAL_TOKENS = Counter(
    "llm_tokens_total",
    "Total number of tokens (prompt + completion)",
    ["model"],
    registry=REGISTRY
)

# Estimated cost metrics (useful for budgeting)
ESTIMATED_COST = Counter(
    "llm_estimated_cost_usd_total",
    "Estimated cost of LLM API calls in USD",
    ["model"],
    registry=REGISTRY
)

# Rate limiting and quotas
RATE_LIMIT_REMAINING = Gauge(
    "llm_rate_limit_remaining",
    "Remaining requests before rate limit is hit",
    ["model"],
    registry=REGISTRY
)

# Error metrics
ERROR_COUNT = Counter(
    "llm_error_total",
    "Total number of LLM errors",
    ["endpoint", "model", "error_type"],
    registry=REGISTRY
)

# Model cache metrics
CACHE_HIT_COUNT = Counter(
    "llm_cache_hit_total",
    "Total number of cache hits",
    ["model"],
    registry=REGISTRY
)

CACHE_MISS_COUNT = Counter(
    "llm_cache_miss_total",
    "Total number of cache misses",
    ["model"],
    registry=REGISTRY
)

# Model temperature and parameters tracking
MODEL_TEMPERATURE = Summary(
    "llm_temperature_distribution",
    "Distribution of temperature values used in requests",
    ["model"],
    registry=REGISTRY
)

# System resources for inference
GPU_MEMORY_USAGE = Gauge(
    "llm_gpu_memory_bytes",
    "GPU memory used for LLM inference",
    ["device"],
    registry=REGISTRY
)

# Hallucination/quality metrics (if available)
QUALITY_SCORE = Summary(
    "llm_quality_score",
    "Quality score of LLM responses (if available)",
    ["model", "metric_type"],
    registry=REGISTRY
)


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in a text string using the model's tokenizer.
    
    Args:
        text: The text to tokenize
        tokenizer: The model's tokenizer
        
    Returns:
        Number of tokens
    """
    tokens = tokenizer.encode(text)
    return len(tokens)


def track_token_usage(model_name: str, prompt: str, completion: str, tokenizer) -> Dict[str, int]:
    """Track token usage for a request.
    
    Args:
        model_name: Name of the model
        prompt: The prompt text
        completion: The generated completion
        tokenizer: The model's tokenizer
        
    Returns:
        Dict containing token counts
    """
    prompt_tokens = count_tokens(prompt, tokenizer)
    completion_tokens = count_tokens(completion, tokenizer)
    total_tokens = prompt_tokens + completion_tokens
    
    # Update metrics
    PROMPT_TOKENS.labels(model=model_name).inc(prompt_tokens)
    COMPLETION_TOKENS.labels(model=model_name).inc(completion_tokens)
    TOTAL_TOKENS.labels(model=model_name).inc(total_tokens)
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the estimated cost for a request.
    
    This is a simplified version - in production, you would use actual 
    pricing for your specific LLM provider.
    
    Args:
        model_name: Name of the model
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        
    Returns:
        Estimated cost in USD
    """
    # Example pricing (customize based on your model/provider)
    pricing = {
        "gpt2": {"prompt": 0.00001, "completion": 0.00002},  # Fictional pricing
        "default": {"prompt": 0.00001, "completion": 0.00002},
    }
    
    model_pricing = pricing.get(model_name, pricing["default"])
    cost = (prompt_tokens * model_pricing["prompt"]) + (completion_tokens * model_pricing["completion"])
    
    # Update metrics
    ESTIMATED_COST.labels(model=model_name).inc(cost)
    
    return cost


def track_latency(func):
    """Decorator to track latency of a function.
    
    Args:
        func: The function to track
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start_time
        
        # Extract model name (assumes the model is available in self.model or kwargs)
        model_name = "unknown"
        if hasattr(args[0], 'model'):
            if hasattr(args[0].model, 'name_or_path'):
                model_name = args[0].model.name_or_path
            elif hasattr(args[0].model, 'config') and hasattr(args[0].model.config, 'name_or_path'):
                model_name = args[0].model.config.name_or_path
        
        # Update metrics (endpoint derived from function name)
        endpoint = func.__name__
        REQUEST_LATENCY.labels(endpoint=endpoint, model=model_name).observe(latency)
        
        return result
    
    return wrapper


def track_request(func):
    """Decorator to track requests.
    
    Args:
        func: The function to track
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract model name
        model_name = "unknown"
        if hasattr(args[0], 'model'):
            if hasattr(args[0].model, 'name_or_path'):
                model_name = args[0].model.name_or_path
            elif hasattr(args[0].model, 'config') and hasattr(args[0].model.config, 'name_or_path'):
                model_name = args[0].model.config.name_or_path
        
        endpoint = func.__name__
        
        try:
            result = func(*args, **kwargs)
            REQUEST_COUNT.labels(endpoint=endpoint, model=model_name, status="success").inc()
            return result
        except Exception as e:
            # Track errors by type
            error_type = type(e).__name__
            ERROR_COUNT.labels(endpoint=endpoint, model=model_name, error_type=error_type).inc()
            REQUEST_COUNT.labels(endpoint=endpoint, model=model_name, status="error").inc()
            raise
    
    return wrapper


def track_gpu_memory(device: str = "cuda:0"):
    """Track GPU memory usage if available.
    
    Args:
        device: CUDA device to track
    """
    try:
        import torch
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device)
            GPU_MEMORY_USAGE.labels(device=device).set(memory_allocated)
    except (ImportError, RuntimeError):
        # Skip if torch is not available or other CUDA errors
        pass


def get_metrics():
    """Get all metrics in Prometheus format.
    
    Returns:
        Metrics in Prometheus format
    """
    return generate_latest(REGISTRY).decode("utf-8")


# Additional utility for tracking model parameters
def track_model_parameters(model_name: str, temperature: float, **kwargs):
    """Track model parameters.
    
    Args:
        model_name: Name of the model
        temperature: Temperature parameter
        **kwargs: Additional parameters to track
    """
    MODEL_TEMPERATURE.labels(model=model_name).observe(temperature)


def track_quality_score(model_name: str, score: float, metric_type: str = "general"):
    """Track quality score of generated responses.
    
    Args:
        model_name: Name of the model
        score: Quality score (0.0 to 1.0)
        metric_type: Type of quality metric
    """
    QUALITY_SCORE.labels(model=model_name, metric_type=metric_type).observe(score)