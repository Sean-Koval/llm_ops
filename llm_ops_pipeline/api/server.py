"""FastAPI server for LLM inference."""

import time
from typing import Dict, List, Optional, Any

import torch
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from llm_ops_pipeline.model.model_loader import load_model_and_tokenizer
from llm_ops_pipeline.inference.predictor import LLMPredictor
from llm_ops_pipeline.utils.logging import get_logger
from llm_ops_pipeline.utils.monitoring import (
    get_metrics, REQUEST_COUNT, ERROR_COUNT, ESTIMATED_COST,
    PROMPT_TOKENS, COMPLETION_TOKENS, TOTAL_TOKENS, REQUEST_LATENCY
)

logger = get_logger(__name__)

app = FastAPI(title="LLM Ops Pipeline API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Prometheus instrumentation for FastAPI
Instrumentator().instrument(app).expose(app)

# Global model and predictor instances
MODEL = None
PREDICTOR = None


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    
    generated_texts: List[str]
    prompt: str
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]


class TokenCountRequest(BaseModel):
    """Request model for token counting."""
    
    text: str


class TokenCountResponse(BaseModel):
    """Response model for token counting."""
    
    text: str
    tokens: int


def get_predictor() -> LLMPredictor:
    """Get or initialize the LLM predictor.
    
    Returns:
        Initialized LLM predictor
    """
    global MODEL, PREDICTOR
    
    if PREDICTOR is None:
        logger.info("Initializing model and predictor")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer (use environment variables or config for model name)
        model_name = "gpt2"  # Default model, should be configured
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            model_type="causal_lm",
            device=device
        )
        MODEL = model
        
        # Initialize predictor
        PREDICTOR = LLMPredictor(model=model, tokenizer=tokenizer, device=device)
        
    return PREDICTOR


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to track request processing time.
    
    Args:
        request: The incoming request
        call_next: The next middleware or route handler
        
    Returns:
        The response from the next middleware or route handler
    """
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Only track API endpoints, not metrics or static endpoints
        if not request.url.path.startswith("/metrics") and not request.url.path.startswith("/docs"):
            endpoint = request.url.path
            REQUEST_LATENCY.labels(endpoint=endpoint, model="api").observe(process_time)
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        ERROR_COUNT.labels(endpoint=request.url.path, model="api", error_type=type(e).__name__).inc()
        raise


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus metrics endpoint."""
    return get_metrics()


@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest, predictor: LLMPredictor = Depends(get_predictor)):
    """Generate text from a prompt.
    
    Args:
        request: Generation request
        predictor: LLM predictor
        
    Returns:
        Generation response
    """
    try:
        # Track request start time for our own metrics
        start_time = time.time()
        
        # Get the model name for metrics
        model_name = predictor.model_name
        
        # Count prompt tokens for metrics
        prompt_tokens = predictor.get_token_count(request.prompt)
        PROMPT_TOKENS.labels(model=model_name).inc(prompt_tokens)
        
        # Generate text
        generated_texts = predictor.generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            num_return_sequences=request.num_return_sequences
        )
        
        # Calculate response metrics
        process_time = time.time() - start_time
        
        # Count completion tokens
        completion_tokens = sum(predictor.get_token_count(text) for text in generated_texts)
        COMPLETION_TOKENS.labels(model=model_name).inc(completion_tokens)
        
        # Total tokens
        total_tokens = prompt_tokens + completion_tokens
        TOTAL_TOKENS.labels(model=model_name).inc(total_tokens)
        
        # Calculate cost (simplified)
        cost = (prompt_tokens * 0.00001) + (completion_tokens * 0.00002)
        ESTIMATED_COST.labels(model=model_name).inc(cost)
        
        # Increment success counter
        REQUEST_COUNT.labels(endpoint="/generate", model=model_name, status="success").inc()
        
        # Return response with metrics
        return GenerationResponse(
            generated_texts=generated_texts,
            prompt=request.prompt,
            parameters={
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "num_return_sequences": request.num_return_sequences
            },
            metrics={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": cost,
                "processing_time_seconds": process_time
            }
        )
    
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        
        # Increment error counter
        ERROR_COUNT.labels(
            endpoint="/generate", 
            model=getattr(predictor, "model_name", "unknown"), 
            error_type=type(e).__name__
        ).inc()
        REQUEST_COUNT.labels(
            endpoint="/generate", 
            model=getattr(predictor, "model_name", "unknown"), 
            status="error"
        ).inc()
        
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/tokens/count", response_model=TokenCountResponse)
def count_tokens(request: TokenCountRequest, predictor: LLMPredictor = Depends(get_predictor)):
    """Count tokens in a text string.
    
    Args:
        request: Token count request
        predictor: LLM predictor
        
    Returns:
        Token count response
    """
    try:
        token_count = predictor.get_token_count(request.text)
        
        return TokenCountResponse(
            text=request.text,
            tokens=token_count
        )
    
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Token counting error: {str(e)}")


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server.
    
    Args:
        host: Host to run the server on
        port: Port to run the server on
    """
    import uvicorn
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()