# LLM Evaluation Framework Guide

This guide provides comprehensive documentation for the LLM Evaluation Framework, a system for evaluating LLM applications throughout the MLOps lifecycle, from initial dataset receipt to optimized deployment.

## Overview

The LLM Evaluation Framework provides:

1. **Comprehensive Metrics Collection**: Automated collection of accuracy, latency, and cost metrics
2. **Experiment Tracking**: Integration with MLflow and Langfuse for experiment tracking
3. **Prompt Versioning**: Versioned prompt management with Langfuse
4. **Dataset Versioning**: Dataset versioning with DVC
5. **Visualization**: Automated report generation with plots and comparisons
6. **Model and Prompt Comparison**: Tools for comparing different models and prompt variants
7. **Result Registration**: Workflow for registering best-performing models and prompts

## Evaluation Workflow

### 1. Initial Project Setup

When you receive a new dataset or task from a customer, follow these steps:

```python
from llm_ops_pipeline.evaluation.llm_evaluation_framework import LLMEvaluationFramework

# Initialize the framework with your configuration
eval_framework = LLMEvaluationFramework(
    config_path="path/to/config.yaml",
    experiment_name="customer_project_name",
    use_mlflow=True,
    use_langfuse=True,
    use_dvc=True
)

# Create a new experiment
experiment_id = eval_framework.create_experiment(
    name="initial_evaluation",
    description="Initial evaluation of customer dataset",
    tags={"customer": "customer_name", "project": "project_name"}
)
```

### 2. Dataset Setup

```python
# Load and version your dataset
dataset = eval_framework.load_dataset("path/to/dataset.json")

# If using DVC for versioning
import dvc.api
import os

# Add dataset to DVC
os.system(f"dvc add path/to/dataset.json")
os.system(f"dvc push")  # If using remote storage
```

### 3. Prompt Engineering Evaluation

Test multiple prompt variants against your dataset:

```python
# Define prompts to evaluate
prompts = {
    "basic": "Basic instruction prompt...",
    "detailed": "More detailed prompt with examples...",
    "step_by_step": "Step-by-step reasoning prompt...",
    "few_shot": "Few-shot learning prompt with examples..."
}

# Define inference function
def inference_fn(prompt, example, model_id):
    # Your inference code here
    # Return prediction or dict with prediction and metadata
    pass

# Define metrics function
from llm_ops_pipeline.evaluation.llm_evaluation_framework import calculate_classification_metrics

def metrics_fn(examples, predictions):
    # For classification
    true_values = [ex["label"] for ex in examples]
    return calculate_classification_metrics(true_values, predictions, labels=["label1", "label2"])

# Run prompt evaluation
results = eval_framework.evaluate_prompts(
    task_id="prompt_evaluation",
    prompts=prompts,
    dataset=dataset,
    inference_fn=inference_fn,
    metrics_fn=metrics_fn,
    model_id="your-model-id",
    experiment_id=experiment_id
)

# Generate report
report = eval_framework.generate_metrics_report(
    results,
    output_dir="reports/prompt_evaluation"
)

# Register the best prompt
if eval_framework.use_langfuse:
    best_prompt_id = eval_framework.register_best_prompt(
        results,
        metric="f1_macro",
        higher_is_better=True,
        environment="development"
    )
```

### 4. Model Evaluation

Compare different models using the best prompt:

```python
# Define models to evaluate
models = {
    "model1": {"model_id": "model1-id", "config": {}},
    "model2": {"model_id": "model2-id", "config": {}}
}

# Run model evaluation
model_results = eval_framework.evaluate_models(
    task_id="model_evaluation",
    prompt_id=best_prompt_id,  # Use the best prompt from previous step
    models=models,
    dataset=dataset,
    inference_fn=inference_fn,
    metrics_fn=metrics_fn,
    experiment_id=experiment_id
)

# Generate report
model_report = eval_framework.generate_metrics_report(
    model_results,
    output_dir="reports/model_evaluation"
)
```

### 5. Fine-tuning and Optimization

After initial evaluation, you might want to fine-tune your model or optimize your prompts:

```python
# Run evaluation on the fine-tuned model
fine_tuned_result = eval_framework.evaluate_pipeline(
    task_id="fine_tuned_evaluation",
    pipeline_fn=your_inference_pipeline,
    dataset=dataset,
    metrics_fn=metrics_fn,
    pipeline_config={"model_id": "fine-tuned-model-id", "prompt_id": best_prompt_id},
    experiment_id=experiment_id
)

# Compare with baseline
comparison = eval_framework.compare_results(
    baseline_result=model_results["best_model"],
    new_result=fine_tuned_result,
    output_dir="reports/fine_tuning_comparison"
)
```

### 6. Production Deployment

When ready for production:

```python
# Register the best prompt for production
production_prompt_id = eval_framework.register_best_prompt(
    model_results,
    metric="f1_macro",
    higher_is_better=True,
    environment="production",
    description="Production prompt for customer X use case"
)

# Log the deployment
eval_framework.start_run(
    run_name="production_deployment",
    experiment_id=experiment_id,
    tags={"stage": "production"},
    parameters={"model_id": "best-model-id", "prompt_id": production_prompt_id}
)

# Log deployment metrics
eval_framework.log_metrics({
    "accuracy": fine_tuned_result.metrics["accuracy"],
    "latency_ms": fine_tuned_result.latency_ms
})

eval_framework.end_run()
```

## Integration with MLOps Tools

The framework is designed to integrate with the existing MLOps ecosystem:

### MLflow Integration

- Automatic logging of metrics, parameters, and artifacts to MLflow
- Experiment organization
- Model registry integration
- Gemini autologging for detailed model tracking

#### Using Gemini Autologging

Gemini models can be automatically tracked with MLflow using the integrated autologging feature:

```python
from llm_ops_pipeline.utils.gemini_mlflow import setup_gemini_autologging

# Initialize autologging
gemini_logger = setup_gemini_autologging(
    api_key="YOUR_API_KEY",  # Or use environment variable GOOGLE_API_KEY
    tracking_uri="http://localhost:5000",
    experiment_name="gemini_experiment"
)

# Use Gemini normally - all interactions will be logged
from google.generativeai import GenerativeModel
model = GenerativeModel("gemini-1.5-flash")
response = model.generate_content("What is MLflow?")

# MLflow will automatically track:
# - Input prompts
# - Output texts
# - Token usage and costs
# - Latency metrics
# - Model parameters
```

You can also use the CLI utility:

```bash
# Enable autologging and run an example
python scripts/gemini_autotracking.py \
  --api-key $GOOGLE_API_KEY \
  --tracking-uri http://localhost:5000 \
  --experiment-name "gemini_experiment" \
  --run-example \
  --model gemini-1.5-flash
```

### Langfuse Integration

- Prompt versioning and A/B testing
- Prompt performance tracking
- Trace-based evaluation

### DVC Integration

- Dataset versioning
- Reproducible experiments
- Large file handling

### Prometheus/Grafana Integration

- Runtime metrics monitoring
- Cost tracking
- Performance monitoring

## Evaluation Types

The framework supports different types of evaluations:

### Classification Tasks

For tasks like sentiment analysis, compliance detection, etc.:

```python
from llm_ops_pipeline.evaluation.llm_evaluation_framework import calculate_classification_metrics

def metrics_fn(examples, predictions):
    true_values = [ex["label"] for ex in examples]
    return calculate_classification_metrics(true_values, predictions, labels=["label1", "label2"])
```

### Generation Tasks

For tasks like summarization, translation, etc.:

```python
from llm_ops_pipeline.evaluation.llm_evaluation_framework import calculate_generation_metrics

def metrics_fn(examples, predictions):
    references = [ex["reference_text"] for ex in examples]
    return calculate_generation_metrics(references, predictions, use_bleu=True, use_rouge=True)
```

### Custom Metrics

You can implement custom metrics for specialized tasks:

```python
def custom_metrics_fn(examples, predictions):
    # Custom metrics implementation
    return {
        "custom_metric1": value1,
        "custom_metric2": value2
    }
```

## Human Evaluation Integration

For qualitative evaluation that requires human judgment:

1. **Generate Examples**: Use the framework to generate examples for human review
2. **Record Judgments**: Record human judgments in a structured format
3. **Analyze Results**: Integrate human evaluations with automated metrics

Example:

```python
# Generate examples for human evaluation
human_eval_examples = []
for i, example in enumerate(dataset[:100]):
    prediction = inference_fn(best_prompt, example, "best-model-id")
    human_eval_examples.append({
        "id": i,
        "input": example["text"],
        "output": prediction,
        "ground_truth": example.get("label"),
        "human_score": None  # To be filled by human evaluators
    })

# Save examples for human evaluation
import json
with open("human_evaluation.json", "w") as f:
    json.dump(human_eval_examples, f)

# Later, after human evaluation is complete:
with open("human_evaluation_complete.json", "r") as f:
    human_eval_results = json.load(f)

# Calculate human evaluation metrics
human_scores = [ex["human_score"] for ex in human_eval_results]
eval_framework.log_metrics({
    "human_eval_average": sum(human_scores) / len(human_scores),
    "human_eval_approval_rate": len([s for s in human_scores if s >= 4]) / len(human_scores)
})
```

## Best Practices

### Experiment Organization

- Use consistent naming conventions for experiments and runs
- Group related evaluations under the same experiment
- Tag runs with meaningful metadata

### Metric Selection

- Choose metrics appropriate for your task
- Consider both accuracy and performance metrics
- Include business-relevant metrics (cost, user satisfaction)

### Reporting

- Generate comprehensive reports for stakeholders
- Include visualizations for easier interpretation
- Compare results against baselines

### Version Control

- Version all inputs: datasets, prompts, and models
- Version evaluation scripts and configurations
- Document the evaluation process

## Troubleshooting

### Common Issues

- **Missing Dependencies**: Ensure all required packages are installed
- **API Key Issues**: Check that MLflow, Langfuse, and other API keys are properly configured
- **Path Issues**: Use absolute paths for datasets and outputs
- **Memory Issues**: For large datasets, use batch processing and sampling

### Logging and Debugging

- Enable debug logging with `logging.getLogger("llm_evaluation").setLevel(logging.DEBUG)`
- Check MLflow UI for experiment details
- Inspect the evaluation workspace directory for detailed outputs

## Example End-to-End Workflow

Here's a complete workflow example from dataset receipt to production:

```python
# 1. Initial setup
from llm_ops_pipeline.evaluation.llm_evaluation_framework import LLMEvaluationFramework
import json

# Initialize framework
eval_framework = LLMEvaluationFramework(
    config_path="configs/customer_project.yaml",
    experiment_name="customer_x_text_classification"
)

# 2. Create experiment
experiment_id = eval_framework.create_experiment(
    name="text_classification",
    description="Customer X text classification project",
    tags={"customer": "Customer X", "project": "Text Classification"}
)

# 3. Load and prepare dataset
with open("data/customer_dataset.json", "r") as f:
    dataset = json.load(f)

# Split into train/test if needed
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(dataset, test_size=0.2)

# 4. Define evaluation functions
def inference_fn(prompt, example, model_id):
    # Implementation depends on your model API
    import openai
    
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": example["text"]}
        ],
        temperature=0
    )
    
    prediction = response.choices[0].message.content.strip()
    
    return {
        "prediction": prediction,
        "token_usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }

def metrics_fn(examples, predictions):
    from llm_ops_pipeline.evaluation.llm_evaluation_framework import calculate_classification_metrics
    
    true_values = [ex["label"] for ex in examples]
    labels = ["category_a", "category_b", "category_c"]
    
    return calculate_classification_metrics(true_values, predictions, labels=labels)

# 5. Prompt evaluation
prompts = {
    "basic": "Classify the text into one of these categories: category_a, category_b, category_c. Return only the category name.",
    "detailed": "You are a text classification system. Analyze the provided text and classify it into exactly one of these categories:\n- category_a: [description]\n- category_b: [description]\n- category_c: [description]\nReturn only the category name with no additional text.",
    "few_shot": "Classify the text into one of these categories: category_a, category_b, category_c.\n\nExamples:\nText: [example1]\nCategory: category_a\n\nText: [example2]\nCategory: category_b\n\nText: [example3]\nCategory: category_c\n\nNow classify the following text. Return only the category name."
}

prompt_results = eval_framework.evaluate_prompts(
    task_id="initial_prompt_evaluation",
    prompts=prompts,
    dataset=test_data[:50],  # Start with a smaller sample
    inference_fn=inference_fn,
    metrics_fn=metrics_fn,
    model_id="gpt-3.5-turbo",
    experiment_id=experiment_id
)

# Generate report
initial_report = eval_framework.generate_metrics_report(
    prompt_results,
    output_dir="reports/initial_prompt_evaluation"
)

# Find best prompt
best_prompt_id, best_prompt_result = eval_framework.find_best_result(
    prompt_results,
    metric="f1_macro",
    higher_is_better=True
)

# 6. Model comparison
models = {
    "gpt-3.5-turbo": {"model_id": "gpt-3.5-turbo", "config": {}},
    "gpt-4": {"model_id": "gpt-4", "config": {}}
}

model_results = eval_framework.evaluate_models(
    task_id="model_comparison",
    prompt_id=best_prompt_id,
    models=models,
    dataset=test_data,
    inference_fn=inference_fn,
    metrics_fn=metrics_fn,
    experiment_id=experiment_id
)

model_report = eval_framework.generate_metrics_report(
    model_results,
    output_dir="reports/model_comparison"
)

# 7. Register production assets
production_prompt_id = eval_framework.register_best_prompt(
    prompt_results,
    metric="f1_macro",
    higher_is_better=True,
    environment="production"
)

# 8. Final evaluation with full dataset
final_eval = eval_framework.evaluate_pipeline(
    task_id="production_evaluation",
    pipeline_fn=lambda example, config: inference_fn(
        config["prompt_manager"].get_prompt(config["prompt_id"])["content"],
        example, 
        config["model_id"]
    ),
    dataset=test_data,
    metrics_fn=metrics_fn,
    pipeline_config={
        "prompt_id": production_prompt_id,
        "model_id": "gpt-4",
        "prompt_manager": eval_framework.prompt_manager
    },
    experiment_id=experiment_id
)

# 9. Final report
final_report = eval_framework.generate_metrics_report(
    final_eval,
    output_dir="reports/final_evaluation"
)

# Print summary
print(f"Best prompt: {best_prompt_id}, F1 score: {best_prompt_result.metrics['f1_macro']:.4f}")
print(f"Best model: {eval_framework.find_best_result(model_results, 'f1_macro')[0]}")
print(f"Final evaluation F1 score: {final_eval.metrics['f1_macro']:.4f}")
```

## Conclusion

The LLM Evaluation Framework provides a comprehensive system for evaluating and optimizing LLM applications. By following this guide, you can implement rigorous evaluation practices throughout your MLOps lifecycle, from initial dataset exploration to production deployment.

For more information, see:
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Langfuse Documentation](https://langfuse.com/docs)
- [DVC Documentation](https://dvc.org/doc)
- [Prometheus/Grafana Documentation](https://prometheus.io/docs/introduction/overview/)