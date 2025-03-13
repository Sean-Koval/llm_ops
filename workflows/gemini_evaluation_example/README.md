# Gemini Evaluation Example

This workflow demonstrates how to use the Gemini MLflow autologging integration for evaluating and comparing different Gemini models and prompts. It provides a comprehensive example of running evaluations with automatic experiment tracking.

## Features

- Automatic tracking of Gemini model parameters, inputs, outputs, and metrics
- Cost estimation based on token usage
- Comparison of different Gemini models (e.g., Gemini 1.5 Flash vs Pro)
- Prompt optimization through comparative evaluation
- Comprehensive metrics and visualizations
- Integration with MLflow for experiment tracking

## Directory Structure

```
gemini_evaluation_example/
├── README.md                  # This file
├── configs/                   # Configuration files
│   ├── evaluation_config.yaml # Evaluation framework configuration
│   └── models_config.yaml     # Gemini model configurations
├── data/                      # Test datasets
│   ├── sentiment_analysis.json # Sample sentiment analysis dataset
│   └── entity_extraction.json  # Sample entity extraction dataset
├── prompts/                    # Prompt templates
│   ├── sentiment_prompts.json  # Prompts for sentiment analysis
│   └── entity_prompts.json     # Prompts for entity extraction
├── results/                    # Evaluation results (generated)
└── scripts/                    # Evaluation scripts
    ├── evaluate_models.py      # Script to compare different Gemini models
    ├── evaluate_prompts.py     # Script to compare different prompts
    ├── generate_report.py      # Script to generate comprehensive reports
    └── utils.py                # Utility functions for the workflow
```

## Prerequisites

- Google Generative AI Python SDK: `pip install google-generativeai`
- MLflow: `pip install mlflow`
- Pandas: `pip install pandas`
- Matplotlib and Seaborn: `pip install matplotlib seaborn`

You'll also need a Google API key for accessing Gemini models. You can set it as an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key"
```

## Quick Start

1. Install dependencies and set up your Google API key
2. Run a basic prompt comparison:

```bash
cd scripts
python evaluate_prompts.py \
    --config ../configs/evaluation_config.yaml \
    --dataset ../data/sentiment_analysis.json \
    --prompts ../prompts/sentiment_prompts.json \
    --model-id gemini-1.5-flash \
    --output-dir ../results/sentiment_prompts
```

3. Run a model comparison:

```bash
cd scripts
python evaluate_models.py \
    --config ../configs/evaluation_config.yaml \
    --dataset ../data/sentiment_analysis.json \
    --prompts ../prompts/sentiment_prompts.json \
    --output-dir ../results/model_comparison
```

4. Generate a comprehensive report:

```bash
cd scripts
python generate_report.py \
    --results-dir ../results/sentiment_prompts \
    --output-dir ../reports/sentiment_report
```

## Workflow Steps

1. **Configure Evaluation**: Set up the evaluation parameters in the config files
2. **Prepare Data**: Use or adapt the sample datasets for your use case
3. **Define Prompts**: Create prompt variants in JSON format
4. **Run Evaluation**: Compare different prompts or models
5. **Analyze Results**: View metrics and visualizations in MLflow or generated reports
6. **Deploy Best Solution**: Use the best-performing prompt and model for your production use case

## MLflow Integration

All evaluations are automatically tracked in MLflow, allowing you to:

- Compare metrics across experiments
- View parameter configurations
- Track token usage and costs
- Visualize performance

To view the MLflow UI:

```bash
mlflow ui --port 5000
```

Then navigate to http://localhost:5000 in your browser.

## Customizing for Your Use Case

To adapt this workflow for your specific use case:

1. Replace the sample datasets with your own data
2. Modify the prompts to match your task requirements
3. Adjust the evaluation metrics in the evaluation scripts
4. Use different Gemini models based on your performance and cost requirements

## Performance Considerations

- **Gemini 1.5 Flash**: Lower cost, slightly lower quality, good for high-volume applications
- **Gemini 1.5 Pro**: Higher cost, better quality, recommended for complex reasoning tasks

Choose the appropriate model based on your specific requirements and budget constraints.