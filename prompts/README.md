# Prompt Management System

This directory contains the prompt management system for the LLM Ops Pipeline. It provides a robust infrastructure for managing, versioning, testing, and deploying prompts across different environments.

## Directory Structure

```
prompts/
├── registry/                 # Environment-specific registries mapping prompt IDs to active versions
│   ├── development.json
│   ├── staging.json
│   └── production.json
├── datasets/                 # Versioned datasets managed by DVC (e.g., few-shot examples)
│   └── compliance/
│       ├── few_shot_examples.json
│       └── metadata.yaml
├── compliance/               # Prompt templates for compliance detection
│   ├── basic.j2
│   ├── detailed.j2
│   ├── step_by_step.j2
│   └── few_shot.j2
└── example/                  # Example prompt (for reference)
    └── hello/
        ├── prompt.json
        └── metadata.json
```

## Prompt Management Architecture

The prompt management system integrates multiple components:

1. **Multi-tier Storage**: 
   - In-memory cache
   - Langfuse (collaborative UI)
   - Git-based storage (for resilience)

2. **PromptManager Class**: Core component that manages:
   - Prompt retrieval with fallback mechanisms
   - Version control
   - Environment-specific registries
   - MLflow tracking integration
   - Decorator pattern for easy usage

3. **Template System**:
   - Advanced templating with Jinja2-like syntax
   - Conditional blocks, loops, and variable substitution
   - Few-shot example integration
   - Template inheritance and composition

4. **Experimentation Framework**:
   - A/B testing of prompt variants
   - Performance tracking in MLflow
   - Statistical analysis of results
   - Best variant promotion

5. **Dataset Management**:
   - DVC integration for versioning large datasets
   - Few-shot example management
   - Versioned datasets with metadata

## Usage Examples

### Basic Prompt Access

```python
from llm_ops_pipeline.utils.prompt_management import PromptManager

# Initialize the prompt manager
prompt_manager = PromptManager()

# Get a prompt by ID
prompt = prompt_manager.get_prompt("compliance/basic")

# Use the prompt
content = prompt["content"]
version = prompt.get("metadata", {}).get("version", "unknown")
```

### Using Templates

```python
from llm_ops_pipeline.utils.prompt_templates import PromptTemplate, PromptTemplateLibrary

# Initialize template library
template_library = PromptTemplateLibrary("./prompts")

# Get a template
template = template_library.get_template("compliance/few_shot")

# Render with context
context = {
    "message": "Your message here",
    "output_format": "json"
}
rendered = template.render(context)
```

### Running A/B Tests

```python
from llm_ops_pipeline.utils.prompt_experimentation import PromptExperiment

# Initialize experiment
experiment = PromptExperiment(
    experiment_name="compliance-comparison",
    prompt_manager=prompt_manager
)

# Add variants
experiment.add_variant("basic", "compliance/basic")
experiment.add_variant("detailed", "compliance/detailed")

# Run test with batch of inputs
results = experiment.run_ab_test(
    inputs_batch=inputs,
    inference_fn=my_inference_function,
    sampling_method="all"
)

# Analyze results
analysis = experiment.analyze_experiment()
```

### Using the Decorator Pattern

```python
@prompt_manager.with_prompt("compliance/few_shot")
def classify_compliance(prompt, message):
    # The prompt is automatically loaded and passed
    template = PromptTemplate(prompt["content"])
    rendered = template.render({"message": message})
    
    # Use the rendered prompt with your LLM
    # ...
    
    return result
```

### Versioning Few-Shot Examples with DVC

```bash
# Add new examples to the dataset
python workflows/compliance_detection/scripts/versioned_few_shot.py \
    --action add \
    --new-examples-path new_examples.json

# Create a new version
python workflows/compliance_detection/scripts/versioned_few_shot.py \
    --action version \
    --version-name v2

# Check out a specific version
python workflows/compliance_detection/scripts/versioned_few_shot.py \
    --action checkout \
    --version-name v1
```

## CI/CD Pipeline

The prompt management system includes a GitHub Actions workflow that:

1. Validates prompt files format and structure
2. Runs unit tests for the prompt management system
3. Deploys prompts to the appropriate environment registry
4. Syncs with Langfuse if credentials are available
5. Runs a test to validate the deployment

To deploy prompts manually, use the workflow dispatch trigger in GitHub Actions.

## Best Practices

1. **Naming Convention**: Use namespaced IDs like `{domain}/{name}` (e.g., `compliance/basic`)
2. **Versioning**: Let the system handle versioning automatically
3. **Templates**: Use the templating system for complex prompts
4. **Testing**: Always A/B test significant prompt changes
5. **Few-Shot**: Version few-shot examples with DVC
6. **Environment Promotion**: Use the CI/CD pipeline to promote prompts
7. **Monitoring**: Track prompt performance over time

## Additional Resources

- [Prompt Management Documentation](../docs/prompt_management.md)
- [Example Notebook](../workflows/compliance_detection/notebooks/prompt_management_demo.ipynb)
- [Prompt Monitoring Dashboard](../scripts/prompt_monitoring_dashboard.py)