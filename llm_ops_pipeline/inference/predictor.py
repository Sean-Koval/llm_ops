"""Inference utilities for LLM Ops Pipeline."""

from typing import Dict, List, Optional, Union, Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_ops_pipeline.utils.logging import get_logger

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
        
        # Set model to evaluation mode
        self.model.eval()
        logger.info(f"Initialized LLMPredictor with model on {device}")
    
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
        
        with torch.no_grad():
            output_sequences = self.model.generate(
                **encoded_input,
                **generation_kwargs
            )
        
        # Decode and clean the generated text
        generated_texts = []
        for output in output_sequences:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            prompt_length = len(self.tokenizer.decode(encoded_input["input_ids"][0], 
                                                     skip_special_tokens=True))
            generated_text = generated_text[prompt_length:]
            
            generated_texts.append(generated_text.strip())
        
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
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            for prompt in batch:
                generated = self.generate_text(
                    prompt=prompt,
                    num_return_sequences=1,
                    **generation_kwargs
                )
                all_generated.extend(generated)
        
        return all_generated