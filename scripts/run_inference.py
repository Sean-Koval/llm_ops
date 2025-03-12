#!/usr/bin/env python
"""Script for running inference with a fine-tuned model."""

import argparse
import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_ops_pipeline.model.model_loader import load_model_and_tokenizer
from llm_ops_pipeline.inference.predictor import LLMPredictor
from llm_ops_pipeline.utils.logging import get_logger

logger = get_logger("inference")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned model")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the fine-tuned model directory"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        default=None,
        help="Input prompt for generation"
    )
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        default=None,
        help="Input file with prompts (one per line)"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="Output file to save generated text"
    )
    
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=128,
        help="Maximum number of new tokens to generate"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Nucleus sampling parameter"
    )
    
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50,
        help="Top-k sampling parameter"
    )
    
    parser.add_argument(
        "--num_return_sequences", 
        type=int, 
        default=1,
        help="Number of sequences to return"
    )
    
    return parser.parse_args()


def main():
    """Main function for running inference."""
    args = parse_args()
    
    # Check that either prompt or input_file is provided
    if args.prompt is None and args.input_file is None:
        logger.error("Either --prompt or --input_file must be provided")
        sys.exit(1)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_path,
        model_type="causal_lm"
    )
    
    # Initialize predictor
    predictor = LLMPredictor(model=model, tokenizer=tokenizer)
    
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_return_sequences": args.num_return_sequences
    }
    
    results = []
    
    # Process single prompt
    if args.prompt is not None:
        logger.info("Generating text for provided prompt")
        generated_texts = predictor.generate_text(args.prompt, **generation_kwargs)
        
        result = {
            "prompt": args.prompt,
            "generated_texts": generated_texts
        }
        results.append(result)
        
        # Print result to console
        print(f"Prompt: {args.prompt}")
        for i, text in enumerate(generated_texts):
            print(f"Generated text {i+1}: {text}")
            print("-" * 40)
    
    # Process prompts from input file
    elif args.input_file is not None:
        logger.info(f"Reading prompts from {args.input_file}")
        
        # Read prompts from file
        with open(args.input_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Generating text for {len(prompts)} prompts")
        
        # Generate text for each prompt
        for i, prompt in enumerate(prompts):
            generated_texts = predictor.generate_text(prompt, **generation_kwargs)
            
            result = {
                "prompt": prompt,
                "generated_texts": generated_texts
            }
            results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(prompts)} prompts")
    
    # Save results to output file if specified
    if args.output_file is not None:
        logger.info(f"Saving results to {args.output_file}")
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
    
    logger.info("Inference complete")


if __name__ == "__main__":
    main()