# Prompt Tuning Workflow Example

This workflow demonstrates a complete end-to-end prompt tuning process using the LLM Evaluation Framework. It shows how to go from initial dataset receipt to optimized production deployment with comprehensive tracking and evaluation.

## Overview

This workflow covers:

1. **Dataset preparation and versioning with DVC**
2. **Prompt engineering with systematic evaluation**
3. **Model comparison across different provider models**
4. **Human evaluation integration**
5. **Deployment to production with Langfuse**
6. **Monitoring with Prometheus and Grafana**

## Setup

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export LANGFUSE_API_KEY="your-langfuse-api-key"  
export LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted instance
```

3. Initialize MLflow:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0
```

## Workflow Steps

### 1. Prepare Dataset

The workflow includes scripts to:
- Load raw data from customer sources
- Clean and normalize the data
- Split into train/validation/test sets
- Version with DVC for reproducibility

```bash
python scripts/prepare_dataset.py --input-dir data/raw --output-dir data/processed
```

### 2. Generate Initial Prompts

Create a variety of prompt candidates with different strategies:
- Basic instruction prompts
- Few-shot learning prompts
- Chain-of-thought prompts
- Structured output prompts

```bash
python scripts/generate_prompt_candidates.py --output-file prompts/candidates.json
```

### 3. Evaluate Prompt Candidates

Systematically evaluate all prompt candidates against validation data:

```bash
python ../../scripts/evaluate_llm.py \
  --mode prompt \
  --dataset data/processed/validation.json \
  --prompts prompts/candidates.json \
  --model-id gpt-3.5-turbo \
  --task-type classification \
  --label-field category \
  --use-mlflow \
  --use-langfuse \
  --output-dir results/prompt_evaluation
```

### 4. Optimize Best Prompts

Refine the top-performing prompts based on error analysis:

```bash
python scripts/refine_prompts.py \
  --input-file results/prompt_evaluation/report.json \
  --output-file prompts/refined.json
```

### 5. Compare Models

Evaluate the best prompts across different models:

```bash
python ../../scripts/evaluate_llm.py \
  --mode model \
  --dataset data/processed/validation.json \
  --prompt-id best_prompt_production \
  --models models/candidates.json \
  --task-type classification \
  --label-field category \
  --use-mlflow \
  --use-langfuse \
  --output-dir results/model_evaluation
```

### 6. Human Evaluation

Generate samples for human reviewers:

```bash
python scripts/generate_human_eval_samples.py \
  --dataset data/processed/test.json \
  --prompt-id best_prompt_production \
  --model-id gpt-4 \
  --output-file evaluation/human_eval_samples.json
```

After collecting human feedback:

```bash
python scripts/analyze_human_feedback.py \
  --input-file evaluation/human_eval_results.json \
  --output-dir results/human_evaluation
```

### 7. Deploy to Production

Register the production prompt and model:

```bash
python scripts/deploy_to_production.py \
  --prompt-id best_prompt_production \
  --model-id best_model \
  --environment production
```

### 8. Set up Monitoring

Configure Prometheus and Grafana for monitoring:

```bash
python scripts/setup_monitoring.py \
  --prompt-id best_prompt_production \
  --model-id best_model
```

## Directory Structure

```
workflows/prompt_tuning_example/
├── README.md                       # This file
├── configs/                        # Configuration files
│   ├── evaluation_config.yaml      # Evaluation framework config
│   └── monitoring_config.yaml      # Monitoring configuration
├── data/                           # Data directories
│   ├── raw/                        # Raw data from customer
│   └── processed/                  # Processed datasets
├── prompts/                        # Prompt files
│   ├── candidates.json             # Initial prompt candidates
│   └── refined.json                # Refined prompts after evaluation
├── models/                         # Model configurations
│   └── candidates.json             # Model candidates for evaluation
├── scripts/                        # Workflow scripts
│   ├── prepare_dataset.py          # Dataset preparation
│   ├── generate_prompt_candidates.py # Generate initial prompts
│   ├── refine_prompts.py           # Refine prompts based on evaluation
│   ├── generate_human_eval_samples.py # Generate samples for human eval
│   ├── analyze_human_feedback.py   # Process human evaluation results
│   └── deploy_to_production.py     # Deploy best prompt/model to prod
├── results/                        # Evaluation results
│   ├── prompt_evaluation/          # Results from prompt evaluation
│   ├── model_evaluation/           # Results from model evaluation
│   └── human_evaluation/           # Results from human evaluation
└── evaluation/                     # Evaluation materials
    ├── human_eval_samples.json     # Samples for human evaluation
    └── human_eval_results.json     # Results from human evaluation
```

## Integration Points

This workflow integrates with:

- **MLflow**: Experiment tracking and visualization
- **Langfuse**: Prompt management and versioning
- **DVC**: Dataset and artifact versioning
- **Prometheus/Grafana**: Runtime monitoring
- **GitHub**: Version control for code and configurations

## Key Metrics

The workflow tracks:

1. **Accuracy Metrics**:
   - F1 score (macro and per-class)
   - Precision and recall
   - Confusion matrix

2. **Performance Metrics**:
   - Latency (average, p50, p95, p99)
   - Token usage
   - Cost estimates

3. **Human Evaluation Metrics**:
   - Correctness score
   - Helpfulness score
   - Preference rates

## Customization

This workflow can be customized by:

1. Modifying the configuration files in `configs/`
2. Adding custom metrics in `scripts/custom_metrics.py`
3. Adjusting prompt templates in `prompts/templates/`

## Troubleshooting

Common issues and solutions:

1. **API Errors**: Check API keys and rate limits
2. **Evaluation Failures**: Ensure dataset format matches expectations
3. **MLflow Connection Issues**: Verify MLflow server is running

For more help, see the [LLM Evaluation Framework Guide](../../docs/evaluation_framework_guide.md).

## Example Results

The workflow will produce comprehensive evaluation reports including:

- Detailed metrics tables
- Confusion matrices
- Latency and token usage charts
- Error analysis
- Cost projections

These results are saved in the `results/` directory and also tracked in MLflow for easy visualization and comparison.

## Next Steps

After completing this workflow, consider:

1. Setting up automated re-evaluation when data distribution changes
2. Implementing A/B testing with Langfuse for continuous optimization
3. Creating alerting based on performance metrics
4. Expanding to additional use cases with the same framework