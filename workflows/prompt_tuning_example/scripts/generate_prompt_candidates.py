#!/usr/bin/env python
"""
Generate a variety of prompt candidates for evaluation.
Creates prompt templates with different strategies: basic, few-shot, chain-of-thought, and structured.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from llm_ops_pipeline.utils.logging import setup_logger

# Set up logger
logger = setup_logger(name="generate_prompt_candidates")

# Template strategies
STRATEGIES = ["basic", "detailed", "few_shot", "chain_of_thought", "structured_output"]

def load_examples(examples_path: str) -> List[Dict[str, Any]]:
    """Load few-shot examples from file."""
    if not os.path.exists(examples_path):
        logger.warning(f"Examples file not found: {examples_path}. Using empty examples.")
        return []
    
    with open(examples_path, "r") as f:
        return json.load(f)

def generate_basic_prompt(task_description: str, input_field: str, output_field: str, classes: List[str]) -> str:
    """Generate a basic instruction prompt."""
    class_descriptions = "\n".join([f"- {cls}" for cls in classes])
    
    return f"""You are an AI assistant that helps with {task_description.lower()}.

Classify the {input_field} into exactly one of the following categories:
{class_descriptions}

Return only the category name with no additional text or explanation."""

def generate_detailed_prompt(task_description: str, input_field: str, output_field: str, 
                             classes: List[str], class_descriptions: Dict[str, str]) -> str:
    """Generate a detailed instruction prompt with class descriptions."""
    class_desc_text = "\n".join([f"- {cls}: {class_descriptions.get(cls, '')}" for cls in classes])
    
    return f"""You are an AI assistant that specializes in {task_description.lower()}.

Your task is to analyze the given {input_field} and classify it into exactly one of the following categories:
{class_desc_text}

Carefully consider the {input_field} and categorize it based on the descriptions above.
Return only the category name with no additional text or explanation."""

def generate_few_shot_prompt(task_description: str, input_field: str, output_field: str, 
                           classes: List[str], examples: List[Dict[str, Any]]) -> str:
    """Generate a few-shot learning prompt with examples."""
    class_list = ", ".join(classes)
    examples_text = ""
    
    for i, example in enumerate(examples[:5]):  # Use up to 5 examples
        examples_text += f"\nExample {i+1}:\n{input_field}: {example.get(input_field, '')}\n{output_field}: {example.get(output_field, '')}\n"
    
    return f"""You are an AI assistant that helps with {task_description.lower()}.

Classify the {input_field} into one of these categories: {class_list}.

Here are some examples to help you understand the task:
{examples_text}

Now classify the following {input_field}. Return only the category name with no additional text."""

def generate_chain_of_thought_prompt(task_description: str, input_field: str, output_field: str, 
                                   classes: List[str], class_descriptions: Dict[str, str]) -> str:
    """Generate a chain-of-thought prompt encouraging step-by-step reasoning."""
    class_desc_text = "\n".join([f"- {cls}: {class_descriptions.get(cls, '')}" for cls in classes])
    
    return f"""You are an AI assistant that specializes in {task_description.lower()}.

Your task is to analyze the given {input_field} and classify it into exactly one of the following categories:
{class_desc_text}

Follow these steps to analyze the {input_field}:
1. Read the {input_field} carefully and identify key elements, themes, or features.
2. Consider each category and its description.
3. Evaluate which category best matches the {input_field} based on the descriptions.
4. Select the single most appropriate category.

Return only the category name with no additional text or explanation."""

def generate_structured_output_prompt(task_description: str, input_field: str, output_field: str, 
                                    classes: List[str], class_descriptions: Dict[str, str]) -> str:
    """Generate a prompt that emphasizes structured output."""
    class_desc_text = "\n".join([f"- {cls}: {class_descriptions.get(cls, '')}" for cls in classes])
    
    return f"""You are an AI assistant that specializes in {task_description.lower()}.

Your task is to analyze the given {input_field} and classify it into exactly one of the following categories:
{class_desc_text}

Analyze the {input_field} and determine the most appropriate category.

IMPORTANT: You must return ONLY the category name without any other text, explanation, or formatting.
Valid responses are ONLY: {", ".join(classes)}"""

def generate_prompt_candidates(config: Dict[str, Any], examples: List[Dict[str, Any]]) -> Dict[str, str]:
    """Generate a variety of prompt candidates based on different strategies."""
    task_description = config.get("task_description", "text classification")
    input_field = config.get("input_field", "text")
    output_field = config.get("output_field", "category")
    classes = config.get("classes", [])
    class_descriptions = config.get("class_descriptions", {})
    
    if not classes:
        raise ValueError("Classes must be provided in configuration")
    
    prompts = {}
    
    # Generate basic prompt
    prompts["basic"] = generate_basic_prompt(task_description, input_field, output_field, classes)
    
    # Generate detailed prompt
    prompts["detailed"] = generate_detailed_prompt(task_description, input_field, output_field, classes, class_descriptions)
    
    # Generate few-shot prompt
    if examples:
        prompts["few_shot"] = generate_few_shot_prompt(task_description, input_field, output_field, classes, examples)
    
    # Generate chain-of-thought prompt
    prompts["chain_of_thought"] = generate_chain_of_thought_prompt(task_description, input_field, output_field, classes, class_descriptions)
    
    # Generate structured output prompt
    prompts["structured_output"] = generate_structured_output_prompt(task_description, input_field, output_field, classes, class_descriptions)
    
    # Generate combination prompts
    if examples:
        # Detailed + few shot
        detailed_few_shot = detailed_prompt = generate_detailed_prompt(task_description, input_field, output_field, classes, class_descriptions)
        examples_text = "\n\nExamples:\n"
        for i, example in enumerate(examples[:3]):  # Use up to 3 examples
            examples_text += f"\nExample {i+1}:\n{input_field}: {example.get(input_field, '')}\n{output_field}: {example.get(output_field, '')}\n"
        
        prompts["detailed_few_shot"] = detailed_prompt + examples_text + "\n\nReturn only the category name with no additional text."
    
    return prompts

def main():
    parser = argparse.ArgumentParser(description="Generate prompt candidates for evaluation")
    parser.add_argument("--config", type=str, default="configs/prompt_generation_config.yaml",
                        help="Path to prompt generation configuration file")
    parser.add_argument("--examples", type=str, default="data/few_shot_examples.json",
                        help="Path to few-shot examples file")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to output prompts file")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            if args.config.endswith(".yaml") or args.config.endswith(".yml"):
                config = yaml.safe_load(f)
            elif args.config.endswith(".json"):
                config = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {args.config}")
                sys.exit(1)
    else:
        # Use default configuration if file not found
        logger.warning(f"Configuration file not found: {args.config}. Using default configuration.")
        config = {
            "task_description": "text classification",
            "input_field": "text",
            "output_field": "category",
            "classes": ["category_a", "category_b", "category_c"],
            "class_descriptions": {
                "category_a": "Description for category A",
                "category_b": "Description for category B",
                "category_c": "Description for category C"
            }
        }
    
    # Load examples if available
    examples = []
    if os.path.exists(args.examples):
        examples = load_examples(args.examples)
    
    # Generate prompt candidates
    prompts = generate_prompt_candidates(config, examples)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Save prompts to file
    with open(args.output_file, "w") as f:
        json.dump(prompts, f, indent=2)
    
    logger.info(f"Generated {len(prompts)} prompt candidates and saved to {args.output_file}")
    for strategy, prompt in prompts.items():
        # Print first 100 characters of each prompt
        logger.info(f"  - {strategy}: {prompt[:100]}...")

if __name__ == "__main__":
    main()