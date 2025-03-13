# Prompt Management System

This directory contains prompt templates and datasets for the LLM Ops Pipeline.

## Directory Structure

```
/prompts
├── registry/                # Environment-specific prompt registries
│   ├── development.json
│   ├── staging.json
│   └── production.json
├── [prompt_id]/            # One directory per prompt
│   ├── prompt.json         # The actual prompt content
│   └── metadata.json       # Metadata including version, tags, etc.
└── datasets/               # For few-shot learning examples (DVC-tracked)
    └── [dataset_name]/
        └── examples.json
```

## Usage

Use the PromptManager class from `llm_ops_pipeline.utils.prompt_management` to interact with prompts:

```python
from llm_ops_pipeline.utils.prompt_management import PromptManager

# Initialize manager
prompt_manager = PromptManager(environment="development")

# Get a prompt
prompt = prompt_manager.get_prompt("classification/sentiment")
prompt_content = prompt["content"]

# Use the prompt...
```

## Command Line Usage

The Makefile provides several commands for working with prompts:

```bash
# List all prompts
make prompt-list

# Create a new prompt
make prompt-create

# Update an existing prompt
make prompt-update

# Sync prompts from Langfuse
make prompt-sync

# Run prompt management tests
make prompt-test

# Launch the prompt monitoring dashboard
make prompt-dashboard
```

## Monitoring

To view prompt performance metrics, run:

```bash
make prompt-dashboard
```

This will launch a Streamlit dashboard showing:
- Prompt usage over time
- Performance metrics by prompt version
- Version history and comparisons

## Learn More

For more information, see:
- [Prompt Management Documentation](/docs/prompt_management.md)
- [Example Notebook](/notebooks/prompt_management_example.ipynb)