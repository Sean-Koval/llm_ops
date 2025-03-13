"""Inference utilities for LLM Ops Pipeline."""

import time
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_ops_pipeline.utils.logging import get_logger
from llm_ops_pipeline.utils.monitoring import (
    track_latency, track_request, track_token_usage,
    calculate_cost, track_model_parameters, track_gpu_memory
)

logger = get_logger(__name__)


class LLMPredictor:
    """Predictor class for LLM inference."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512
    ):
        """Initialize the predictor.
        
        Args:
            model: Pre-trained model for inference
            tokenizer: Tokenizer for the model
            device: Device to run inference on
            max_length: Maximum sequence length
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model_name = getattr(model.config, "name_or_path", 
                                 getattr(model, "name_or_path", "unknown"))
        
        # Set model to evaluation mode
        self.model.eval()
        logger.info(f"Initialized LLMPredictor with model {self.model_name} on {device}")
        
        # Track initial GPU memory
        if device.startswith("cuda"):
            track_gpu_memory(device)
    
    @track_request
    @track_latency
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt for text generation
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text sequences
        """
        # Track model parameters
        track_model_parameters(
            model_name=self.model_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        # Start generation timer
        start_time = time.time()
        
        # Tokenize input
        encoded_input = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Track GPU memory before inference
        if self.device.startswith("cuda"):
            track_gpu_memory(self.device)
        
        with torch.no_grad():
            output_sequences = self.model.generate(
                **encoded_input,
                **generation_kwargs
            )
        
        # Track generation time for internal metrics
        generation_time = time.time() - start_time
        
        # Decode and clean the generated text
        generated_texts = []
        token_metrics_list = []
        
        for output in output_sequences:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            prompt_length = len(self.tokenizer.decode(encoded_input["input_ids"][0], 
                                                     skip_special_tokens=True))
            completion = generated_text[prompt_length:].strip()
            
            # Track token usage for each completion
            token_metrics = track_token_usage(
                model_name=self.model_name,
                prompt=prompt,
                completion=completion,
                tokenizer=self.tokenizer
            )
            token_metrics_list.append(token_metrics)
            
            # Calculate cost
            cost = calculate_cost(
                model_name=self.model_name,
                prompt_tokens=token_metrics["prompt_tokens"],
                completion_tokens=token_metrics["completion_tokens"]
            )
            
            generated_texts.append(completion)
        
        # Track final GPU memory after inference
        if self.device.startswith("cuda"):
            track_gpu_memory(self.device)
        
        # Log metrics for this generation
        logger.debug(
            f"Generated {len(generated_texts)} sequences in {generation_time:.2f}s. "
            f"Tokens used: {sum(m['total_tokens'] for m in token_metrics_list)}"
        )
        
        return generated_texts
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 8,
        **generation_kwargs
    ) -> List[str]:
        """Generate text for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            batch_size: Batch size for generation
            **generation_kwargs: Generation parameters
            
        Returns:
            List of generated text sequences
        """
        all_generated = []
        total_batch_time = 0
        
        for i in range(0, len(prompts), batch_size):
            batch_start = time.time()
            batch = prompts[i:i+batch_size]
            
            for prompt in batch:
                generated = self.generate_text(
                    prompt=prompt,
                    num_return_sequences=1,
                    **generation_kwargs
                )
                all_generated.extend(generated)
            
            batch_time = time.time() - batch_start
            total_batch_time += batch_time
            
            logger.debug(f"Batch {i // batch_size + 1} processed in {batch_time:.2f}s")
        
        logger.info(f"Processed {len(prompts)} prompts in {total_batch_time:.2f}s")
        return all_generated
    
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        tokens = self.tokenizer.encode(text)
        return len(tokens)