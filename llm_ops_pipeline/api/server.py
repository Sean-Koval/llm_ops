"""FastAPI server for LLM inference."""

from typing import Dict, List, Optional, Any

import torch
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from llm_ops_pipeline.model.model_loader import load_model_and_tokenizer
from llm_ops_pipeline.inference.predictor import LLMPredictor
from llm_ops_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="LLM Ops Pipeline API")

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


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


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
        generated_texts = predictor.generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            num_return_sequences=request.num_return_sequences
        )
        
        return GenerationResponse(
            generated_texts=generated_texts,
            prompt=request.prompt,
            parameters={
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "num_return_sequences": request.num_return_sequences
            }
        )
    
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


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