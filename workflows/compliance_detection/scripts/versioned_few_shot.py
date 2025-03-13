#!/usr/bin/env python
"""
Versioned Few-Shot Example Management for Compliance Detection

This script demonstrates how to use DVC to version and manage few-shot examples
for the compliance detection workflow.
"""

import os
import sys
import json
import argparse
import random
import subprocess
from pathlib import Path
import logging
import shutil

# Add the parent directory to the path so we can import from llm_ops_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm_ops_pipeline.utils.prompt_management import initialize_dvc_prompt_tracking
from llm_ops_pipeline.utils.prompt_experimentation import load_few_shot_examples

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def ensure_dvc_initialized(repo_root):
    """Ensure DVC is initialized for the repository."""
    dvc_dir = os.path.join(repo_root, ".dvc")
    if not os.path.exists(dvc_dir):
        logger.info("Initializing DVC...")
        subprocess.run(["dvc", "init"], cwd=repo_root, check=True)
        logger.info("DVC initialized successfully")
    
    # Initialize prompt tracking
    initialize_dvc_prompt_tracking(repo_root)


def update_few_shot_examples(
    repo_root, 
    workflow_examples_path, 
    dataset_path,
    sample_count=5, 
    selection_strategy="random",
    categories=None
):
    """
    Update few-shot examples using a versioned dataset.
    
    Args:
        repo_root: Repository root directory
        workflow_examples_path: Path to the workflow examples file
        dataset_path: Path to the versioned dataset
        sample_count: Number of examples to sample
        selection_strategy: Strategy for selecting examples ('random', 'balanced', 'prioritized')
        categories: List of categories to prioritize (for prioritized strategy)
    """
    # Load the full dataset
    try:
        examples = load_few_shot_examples(dataset_path)
        logger.info(f"Loaded {len(examples)} examples from dataset")
    except Exception as e:
        logger.error(f"Failed to load examples from {dataset_path}: {e}")
        return False
    
    # Select examples based on strategy
    selected_examples = []
    
    if selection_strategy == "balanced":
        # Group examples by category
        by_category = {}
        for example in examples:
            category = example.get("example_type")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(example)
        
        # Determine how many examples to take from each category
        categories = list(by_category.keys())
        examples_per_category = max(1, sample_count // len(categories))
        
        # Select examples from each category
        for category in categories:
            category_examples = by_category[category]
            
            # Take random sample from this category
            sample = random.sample(
                category_examples,
                min(examples_per_category, len(category_examples))
            )
            selected_examples.extend(sample)
        
        # If we need more examples, take from random categories
        while len(selected_examples) < sample_count and examples:
            remaining = sample_count - len(selected_examples)
            additional = random.sample(examples, min(remaining, len(examples)))
            
            # Filter out examples we already have
            additional = [ex for ex in additional if ex not in selected_examples]
            selected_examples.extend(additional[:remaining])
    
    elif selection_strategy == "prioritized" and categories:
        # Prioritize specified categories
        prioritized = []
        non_prioritized = []
        
        for example in examples:
            if example.get("example_type") in categories:
                prioritized.append(example)
            else:
                non_prioritized.append(example)
        
        # Calculate how many to take from prioritized vs non-prioritized
        priority_count = min(len(prioritized), int(sample_count * 0.7))  # 70% from priority categories
        remaining_count = sample_count - priority_count
        
        # Select examples
        selected_examples = random.sample(prioritized, priority_count)
        if remaining_count > 0 and non_prioritized:
            selected_examples.extend(random.sample(
                non_prioritized, 
                min(remaining_count, len(non_prioritized))
            ))
    
    else:  # Default to random selection
        selected_examples = random.sample(
            examples,
            min(sample_count, len(examples))
        )
    
    # Save the selected examples to the workflow file
    try:
        with open(workflow_examples_path, "w") as f:
            json.dump(selected_examples, f, indent=2)
        logger.info(f"Saved {len(selected_examples)} examples to {workflow_examples_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save examples to {workflow_examples_path}: {e}")
        return False


def add_examples_to_dataset(
    repo_root,
    dataset_path,
    new_examples_path,
    commit=True
):
    """
    Add new examples to the versioned dataset.
    
    Args:
        repo_root: Repository root directory
        dataset_path: Path to the versioned dataset
        new_examples_path: Path to the new examples file
        commit: Whether to commit the changes to DVC
    """
    # Load existing dataset
    try:
        existing_examples = load_few_shot_examples(dataset_path)
        logger.info(f"Loaded {len(existing_examples)} existing examples")
    except Exception as e:
        logger.warning(f"Failed to load existing examples: {e}")
        existing_examples = []
    
    # Load new examples
    try:
        new_examples = load_few_shot_examples(new_examples_path)
        logger.info(f"Loaded {len(new_examples)} new examples")
    except Exception as e:
        logger.error(f"Failed to load new examples: {e}")
        return False
    
    # Merge examples, avoiding duplicates
    merged_examples = existing_examples.copy()
    added_count = 0
    
    for new_example in new_examples:
        # Check if this example already exists (by message)
        if not any(
            ex.get("message") == new_example.get("message") 
            for ex in existing_examples
        ):
            merged_examples.append(new_example)
            added_count += 1
    
    logger.info(f"Added {added_count} new unique examples")
    
    # Save the merged dataset
    try:
        with open(dataset_path, "w") as f:
            json.dump(merged_examples, f, indent=2)
        logger.info(f"Saved {len(merged_examples)} examples to {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to save merged examples: {e}")
        return False
    
    # Update metadata
    metadata_path = os.path.join(os.path.dirname(dataset_path), "metadata.yaml")
    try:
        # Generate category distribution
        categories = {}
        for example in merged_examples:
            category = example.get("example_type")
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        # Calculate average message length
        avg_length = sum(len(ex.get("message", "").split()) for ex in merged_examples) / len(merged_examples)
        
        # Create metadata content
        from datetime import datetime
        metadata = {
            "name": "compliance-few-shot",
            "version": "1.0.0",  # Would normally increment this
            "description": "Few-shot examples for compliance breach detection",
            "category": "compliance",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "statistics": {
                "example_count": len(merged_examples),
                "category_distribution": categories,
                "avg_message_length": round(avg_length, 1)
            },
            "source": "synthetic",
            "usage": {
                "prompt_id": "compliance/few_shot",
                "template_variables": ["few_shot_examples"],
                "default_count": 5
            }
        }
        
        # Save metadata (simplified for demo)
        with open(metadata_path, "w") as f:
            f.write("name: " + metadata["name"] + "\n")
            f.write("version: " + metadata["version"] + "\n")
            f.write("description: " + metadata["description"] + "\n")
            f.write("category: " + metadata["category"] + "\n")
            f.write("updated_at: \"" + metadata["updated_at"] + "\"\n")
            f.write("statistics:\n")
            f.write(f"  example_count: {metadata['statistics']['example_count']}\n")
            f.write("  category_distribution:\n")
            for cat, count in metadata["statistics"]["category_distribution"].items():
                f.write(f"    {cat}: {count}\n")
            f.write(f"  avg_message_length: {metadata['statistics']['avg_message_length']}\n")
            f.write("source: " + metadata["source"] + "\n")
            f.write("usage:\n")
            f.write("  prompt_id: " + metadata["usage"]["prompt_id"] + "\n")
            f.write("  template_variables:\n")
            for var in metadata["usage"]["template_variables"]:
                f.write(f"    - {var}\n")
            f.write(f"  default_count: {metadata['usage']['default_count']}\n")
        
        logger.info(f"Updated metadata at {metadata_path}")
    except Exception as e:
        logger.warning(f"Failed to update metadata: {e}")
    
    # Commit to DVC if requested
    if commit:
        try:
            # Add to DVC
            subprocess.run(["dvc", "add", dataset_path], cwd=repo_root, check=True)
            
            # Also add metadata
            subprocess.run(["dvc", "add", metadata_path], cwd=repo_root, check=True)
            
            logger.info("Added dataset to DVC")
            
            # Commit to Git
            subprocess.run(["git", "add", dataset_path + ".dvc", metadata_path + ".dvc"], 
                          cwd=repo_root, check=True)
            
            subprocess.run(
                ["git", "commit", "-m", f"Update few-shot examples dataset with {added_count} new examples"],
                cwd=repo_root, check=True
            )
            
            logger.info("Committed changes to Git")
        except Exception as e:
            logger.error(f"Failed to commit changes: {e}")
            return False
    
    return True


def create_dataset_version(
    repo_root,
    dataset_path,
    version_name,
    commit=True
):
    """
    Create a new version of the dataset.
    
    Args:
        repo_root: Repository root directory
        dataset_path: Path to the versioned dataset
        version_name: Name of the new version (e.g., "v2")
        commit: Whether to commit the changes to DVC
    """
    try:
        # Create a DVC tag
        subprocess.run(
            ["dvc", "tag", "add", dataset_path, version_name],
            cwd=repo_root, check=True
        )
        
        logger.info(f"Created DVC tag '{version_name}' for dataset")
        
        if commit:
            # Commit the DVC tag to Git
            subprocess.run(
                ["git", "add", dataset_path + ".dvc"],
                cwd=repo_root, check=True
            )
            
            subprocess.run(
                ["git", "commit", "-m", f"Create version {version_name} of few-shot examples dataset"],
                cwd=repo_root, check=True
            )
            
            logger.info(f"Committed version {version_name} to Git")
    except Exception as e:
        logger.error(f"Failed to create dataset version: {e}")
        return False
    
    return True


def checkout_dataset_version(
    repo_root,
    dataset_path,
    version_name
):
    """
    Check out a specific version of the dataset.
    
    Args:
        repo_root: Repository root directory
        dataset_path: Path to the versioned dataset
        version_name: Name of the version to check out
    """
    try:
        # Checkout the specified version
        subprocess.run(
            ["dvc", "tag", "checkout", dataset_path, version_name],
            cwd=repo_root, check=True
        )
        
        logger.info(f"Checked out version '{version_name}' of dataset")
        return True
    except Exception as e:
        logger.error(f"Failed to checkout dataset version: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Manage versioned few-shot examples")
    parser.add_argument(
        "--action", 
        choices=["update", "add", "version", "checkout"],
        required=True,
        help="Action to perform"
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        help="Repository root directory"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the versioned dataset"
    )
    parser.add_argument(
        "--workflow-examples-path",
        type=str,
        default=None,
        help="Path to the workflow examples file"
    )
    parser.add_argument(
        "--new-examples-path",
        type=str,
        help="Path to the new examples file (for 'add' action)"
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Number of examples to sample (for 'update' action)"
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["random", "balanced", "prioritized"],
        default="balanced",
        help="Strategy for selecting examples (for 'update' action)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        help="Comma-separated list of categories to prioritize (for 'prioritized' strategy)"
    )
    parser.add_argument(
        "--version-name",
        type=str,
        help="Version name (for 'version' and 'checkout' actions)"
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Don't commit changes to DVC/Git"
    )
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if not args.dataset_path:
        args.dataset_path = os.path.join(
            args.repo_root, 
            "prompts/datasets/compliance/few_shot_examples.json"
        )
    
    if not args.workflow_examples_path:
        args.workflow_examples_path = os.path.join(
            args.repo_root,
            "workflows/compliance_detection/prompts/few_shot_examples.json"
        )
    
    # Parse categories if provided
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    
    # Ensure DVC is initialized
    ensure_dvc_initialized(args.repo_root)
    
    # Perform the requested action
    if args.action == "update":
        success = update_few_shot_examples(
            repo_root=args.repo_root,
            workflow_examples_path=args.workflow_examples_path,
            dataset_path=args.dataset_path,
            sample_count=args.sample_count,
            selection_strategy=args.selection_strategy,
            categories=categories
        )
        
        if success:
            print("✅ Successfully updated few-shot examples")
        else:
            print("❌ Failed to update few-shot examples")
            sys.exit(1)
    
    elif args.action == "add":
        if not args.new_examples_path:
            print("❌ --new-examples-path is required for 'add' action")
            sys.exit(1)
        
        success = add_examples_to_dataset(
            repo_root=args.repo_root,
            dataset_path=args.dataset_path,
            new_examples_path=args.new_examples_path,
            commit=not args.no_commit
        )
        
        if success:
            print("✅ Successfully added examples to dataset")
        else:
            print("❌ Failed to add examples to dataset")
            sys.exit(1)
    
    elif args.action == "version":
        if not args.version_name:
            print("❌ --version-name is required for 'version' action")
            sys.exit(1)
        
        success = create_dataset_version(
            repo_root=args.repo_root,
            dataset_path=args.dataset_path,
            version_name=args.version_name,
            commit=not args.no_commit
        )
        
        if success:
            print(f"✅ Successfully created version {args.version_name}")
        else:
            print(f"❌ Failed to create version {args.version_name}")
            sys.exit(1)
    
    elif args.action == "checkout":
        if not args.version_name:
            print("❌ --version-name is required for 'checkout' action")
            sys.exit(1)
        
        success = checkout_dataset_version(
            repo_root=args.repo_root,
            dataset_path=args.dataset_path,
            version_name=args.version_name
        )
        
        if success:
            print(f"✅ Successfully checked out version {args.version_name}")
            
            # Also update the workflow examples
            update_success = update_few_shot_examples(
                repo_root=args.repo_root,
                workflow_examples_path=args.workflow_examples_path,
                dataset_path=args.dataset_path,
                sample_count=args.sample_count,
                selection_strategy=args.selection_strategy,
                categories=categories
            )
            
            if update_success:
                print("✅ Updated workflow examples from checked out version")
            else:
                print("❌ Failed to update workflow examples")
        else:
            print(f"❌ Failed to checkout version {args.version_name}")
            sys.exit(1)


if __name__ == "__main__":
    main()