#!/usr/bin/env python
"""Script for starting the LLM API server."""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_ops_pipeline.model.model_loader import load_model_and_tokenizer
from llm_ops_pipeline.inference.predictor import LLMPredictor
from llm_ops_pipeline.api.server import app, start_server
from llm_ops_pipeline.utils.logging import get_logger

logger = get_logger("api_server")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the LLM API server")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the model directory"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to run the server on"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to run the server on"
    )
    
    return parser.parse_args()


def main():
    """Main function for starting the API server."""
    args = parse_args()
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_path,
        model_type="causal_lm"
    )
    
    # Initialize predictor (will be available globally in the server)
    predictor = LLMPredictor(model=model, tokenizer=tokenizer)
    
    # Set global variables in server module
    import llm_ops_pipeline.api.server as server
    server.MODEL = model
    server.PREDICTOR = predictor
    
    # Start server
    logger.info(f"Starting API server on {args.host}:{args.port}")
    start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()