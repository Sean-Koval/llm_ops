# ML Engineer's Guide to LLM-Powered Compliance Detection

This guide provides a step-by-step workflow for prompt engineers and ML engineers working on LLM-based compliance detection systems. It details each phase of the ML lifecycle with specific instructions for using the tooling in this repository.

## Table of Contents
1. [Project Setup](#1-project-setup)
2. [Data Engineering](#2-data-engineering)
3. [Prompt Engineering](#3-prompt-engineering)
4. [Experiment Tracking](#4-experiment-tracking)
5. [Model Evaluation](#5-model-evaluation)
6. [Model Registry & Versioning](#6-model-registry--versioning)
7. [Deployment & Serving](#7-deployment--serving)
8. [Monitoring & Maintenance](#8-monitoring--maintenance)
9. [Collaboration & Handoff](#9-collaboration--handoff)

## 1. Project Setup

### Environment Configuration
```bash
# Clone the repository
git clone https://github.com/yourusername/llm_ops_pipeline.git
cd llm_ops_pipeline

# Install dependencies
python -m pip install -e .

# Create necessary project directories
mkdir -p workflows/compliance_detection/{data,models,prompts,configs,notebooks,scripts}

# Configure credentials for Vertex AI
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Configuration Setup
```python
# Create a base configuration file in configs/compliance_config.yaml
# Update project-specific settings like model ID, project ID, etc.
# Run validation to ensure configuration is correct

# Verify configuration loads correctly
python -c "from llm_ops_pipeline.config.config import load_config; print(load_config('workflows/compliance_detection/configs/compliance_config.yaml'))"
```

## 2. Data Engineering

### Synthetic Data Generation
For compliance detection, high-quality labeled data is essential but often scarce. Our workflow uses synthetic data generation:

```bash
# Generate synthetic data with Vertex AI
python workflows/compliance_detection/scripts/generate_data.py --config workflows/compliance_detection/configs/compliance_config.yaml --count 1000
```

**Key Practices:**
1. **Balanced Categories**: Ensure representation across all compliance categories
2. **Edge Cases**: Include ambiguous examples that test boundaries between categories
3. **Versioning**: Use DVC to track dataset versions

```bash
# Initialize DVC for data versioning
dvc init
dvc add data/compliance
git add data/.gitignore data/compliance.dvc
git commit -m "Add initial compliance dataset"

# Push to remote storage (optional)
dvc remote add -d storage gs://your-bucket/compliance-data
dvc push
```

### Data Quality Analysis
```bash
# Run exploratory data analysis
jupyter notebook workflows/compliance_detection/notebooks/compliance_analysis.ipynb
```

**ML Engineer Tasks:**
- Review category distributions
- Analyze text length and complexity
- Check for data leakage or biased patterns
- Document data characteristics in your MLflow experiment

## 3. Prompt Engineering

### Initial Zero-Shot Prompting
The first step is to test a simple zero-shot prompt:

```python
# In Python or interactive notebook
from llm_ops_pipeline.utils.logging import setup_logging
import google.cloud.aiplatform as vertexai
from google.cloud.aiplatform.prediction import PredictionServiceClient
from datetime import datetime
import json

# Initialize logging
logger = setup_logging(name="prompt_engineering", level="INFO")
logger.info("Starting zero-shot prompt testing")

# Initialize Vertex AI
vertexai.init(project="your-project-id", location="us-central1")
model = vertexai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Define simple zero-shot prompt
ZERO_SHOT_PROMPT = """You are a compliance detection system for a financial institution. 
Analyze the provided message and determine if it contains any compliance breaches.
Classify the message into exactly one of the following categories:
- ETHICAL_BREACH: Conflicts of interest, misleading clients, dishonest behavior
- ILLEGAL_ACTIVITY: Fraud, money laundering, market manipulation
- REGULATORY_VIOLATION: Sharing material non-public information, violating compliance protocols
- CONFIDENTIAL_INFO: Improper sharing of client data, unreleased financial results, proprietary strategies
- HARASSMENT: Workplace harassment or inappropriate comments
- COMPLIANT: Professional and compliant communication

Return only the category label with no additional text."""

# Test on sample messages
test_messages = [
    "Let's adjust these numbers before the audit.",
    "The quarterly report was published this morning."
]

results = []
for message in test_messages:
    response = model.generate_content([ZERO_SHOT_PROMPT, message])
    prediction = response.text.strip()
    
    results.append({
        "text": message,
        "prediction": prediction,
        "timestamp": datetime.now().isoformat()
    })
    
    logger.info(f"Message: {message}")
    logger.info(f"Prediction: {prediction}")

# Save results for later analysis
with open("workflows/compliance_detection/prompts/zero_shot_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Prompt Variant Development
Create multiple prompt variants to test different approaches:

```bash
# Run prompt tuning script to test variants
python workflows/compliance_detection/scripts/prompt_tuning.py --config workflows/compliance_detection/configs/compliance_config.yaml --eval-data data/compliance/validation/data.json --sample-size 100
```

**Prompt Engineer Tasks:**
- Develop at least 3-4 distinct prompt variants (basic, detailed, step-by-step, few-shot)
- Document reasoning behind each prompt structure
- Version each prompt in JSONs with timestamp and author

### Manual Labeling and Validation
For edge cases or when building the initial dataset:

```python
# In a Jupyter notebook
import pandas as pd
from IPython.display import display

# Load samples that need human validation
samples = pd.read_csv("data/compliance/uncertain_samples.csv")

# Create labeling interface
for idx, row in samples.iterrows():
    print(f"Sample {idx}: {row['text']}")
    print("Current prediction:", row["predicted_label"])
    
    # Get human input
    human_label = input("Enter correct label: ")
    samples.at[idx, "human_label"] = human_label
    samples.at[idx, "needs_review"] = False if human_label == row["predicted_label"] else True
    
# Save updated labels
samples.to_csv("data/compliance/validated_samples.csv", index=False)
```

## 4. Experiment Tracking

### Configuring MLflow
```bash
# Start MLflow server (if running locally)
mlflow server --host 0.0.0.0 --port 5000

# In production, you'd use the MLflow instance configured in docker-compose.yml
```

### Tracking Prompt Engineering Experiments
```python
# In your prompt_tuning.py script
import mlflow
from llm_ops_pipeline.config.config import load_config

# Load configuration
config = load_config("workflows/compliance_detection/configs/compliance_config.yaml")

# Initialize MLflow
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

# For each prompt variant
with mlflow.start_run(run_name=f"prompt-{prompt_name}"):
    # Log parameters
    mlflow.log_param("prompt_name", prompt_name)
    mlflow.log_param("prompt_text", system_prompt)
    mlflow.log_param("model_id", config["model"]["model_id"])
    mlflow.log_param("temperature", config["model"]["temperature"])
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_macro", precision_macro)
    mlflow.log_metric("recall_macro", recall_macro)
    mlflow.log_metric("f1_macro", f1_macro)
    
    # Log confusion matrix as figure
    cm_fig = plot_confusion_matrix(...)
    mlflow.log_figure(cm_fig, f"confusion_matrix_{prompt_name}.png")
    
    # Log the prompt as artifact
    with open("prompt.txt", "w") as f:
        f.write(system_prompt)
    mlflow.log_artifact("prompt.txt")
```

### Using AdaFlow for Advanced Tracking (Optional)
If using AdaFlow integration with MLflow:

```python
# Install AdaFlow
# pip install adaflow-python

from adaflow.mlflow import AdaFlowCallback

# Initialize AdaFlow callback
adaflow_callback = AdaFlowCallback(
    mlflow_run_id=mlflow.active_run().info.run_id,
    model_id=config["model"]["model_id"]
)

# Log prompt template
adaflow_callback.log_prompt_template(system_prompt)

# For each prediction
for text, true_label in validation_data:
    # Make prediction
    prediction = predict(text, system_prompt)
    
    # Log example with AdaFlow
    adaflow_callback.log_example(
        input_text=text,
        output_text=prediction,
        expected_output=true_label,
        metadata={"category": true_label}
    )

# Generate AdaFlow report
adaflow_callback.generate_report()
```

## 5. Model Evaluation

### Comprehensive Evaluation
```bash
# Run full evaluation on test set
python workflows/compliance_detection/scripts/evaluate.py --config workflows/compliance_detection/configs/compliance_config.yaml --test-data data/compliance/test/data.json --output-dir workflows/compliance_detection/evaluation
```

**ML Engineer Tasks:**
- Analyze overall metrics (accuracy, precision, recall, F1)
- Review per-category performance
- Examine confusion matrix for systematic errors
- Conduct error analysis on misclassified examples

### Error Analysis
```python
# In evaluation.py or a Jupyter notebook
# Load evaluation results
with open("workflows/compliance_detection/evaluation/evaluation_results.json", "r") as f:
    results = json.load(f)

# Analyze errors
errors = results["errors"]

# Group errors by true/predicted label
error_matrix = {}
for error in errors:
    true_label = error["true_label"]
    pred_label = error["predicted_label"]
    
    if true_label not in error_matrix:
        error_matrix[true_label] = {}
    
    if pred_label not in error_matrix[true_label]:
        error_matrix[true_label][pred_label] = []
    
    error_matrix[true_label][pred_label].append(error)

# Print error patterns
for true_label, predictions in error_matrix.items():
    for pred_label, examples in predictions.items():
        print(f"{true_label} → {pred_label}: {len(examples)} examples")
        
        # Print sample errors
        for i, example in enumerate(examples[:3]):
            print(f"  Example {i+1}: {example['text']}")
```

## 6. Model Registry & Versioning

### Registering the Best Model
```python
# In evaluate.py or a separate script
import mlflow
from datetime import datetime

# Define model metadata
model_metadata = {
    "name": f"compliance-detector-{datetime.now().strftime('%Y%m%d')}",
    "version": "1.0.0",
    "created_at": datetime.now().isoformat(),
    "metrics": {
        "accuracy": results["accuracy"],
        "precision_macro": results["precision_macro"],
        "recall_macro": results["recall_macro"],
        "f1_macro": results["f1_macro"]
    },
    "prompt": best_prompt_data,
    "registered_by": "data_scientist@example.com",
    "tags": {
        "task": "text-classification",
        "model_type": "vertex-gemini-flash",
        "prompt_type": best_prompt_data["name"]
    }
}

# Register model in MLflow
with mlflow.start_run(run_name="model-registration"):
    # Log model metadata
    for key, value in model_metadata.items():
        if key != "metrics" and not isinstance(value, dict):
            mlflow.log_param(key, value)
    
    # Log metrics
    for metric_name, metric_value in model_metadata["metrics"].items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Log prompt as artifact
    with open("production_prompt.json", "w") as f:
        json.dump(best_prompt_data, f, indent=2)
    mlflow.log_artifact("production_prompt.json")
    
    # Register model
    model_info = mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/production_prompt",
        name="compliance-detector"
    )
    
    print(f"Model registered with version: {model_info.version}")
```

### Creating Model Cards
```python
# Generate model card for documentation
model_card = f"""# Model Card: Compliance Detector

## Model Details
- **Name:** {model_metadata["name"]}
- **Version:** {model_metadata["version"]}
- **Type:** Prompt-based classification using Vertex AI Gemini Flash
- **Date:** {datetime.now().strftime('%Y-%m-%d')}
- **Developer:** Example Organization Data Science Team

## Model Description
This model classifies text messages for compliance breaches in financial contexts.

## Intended Use
- **Primary Use Case:** Monitoring workplace communications for potential compliance breaches
- **Intended Users:** Compliance teams, risk management personnel

## Performance Metrics
- **Accuracy:** {results["accuracy"]:.4f}
- **Precision (macro):** {results["precision_macro"]:.4f}
- **Recall (macro):** {results["recall_macro"]:.4f}
- **F1 Score (macro):** {results["f1_macro"]:.4f}

## Limitations
- The model performs less effectively on ambiguous cases
- The model was trained on synthetic data

## Ethical Considerations
- This model is intended as a first-pass screening tool
- All flagged messages should be reviewed by a human reviewer
"""

with open("workflows/compliance_detection/model_registry/model_card.md", "w") as f:
    f.write(model_card)
```

## 7. Deployment & Serving

### API Deployment
```bash
# Deploy the model API
python workflows/compliance_detection/scripts/deploy_api.py --config workflows/compliance_detection/configs/compliance_config.yaml --host 0.0.0.0 --port 8000 --metrics-port 8001
```

### Docker Deployment
```bash
# Build and deploy with Docker Compose
docker-compose -f docker-compose.yml up -d

# Check services
docker-compose ps
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes (with proper namespace)
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/monitoring-deployment.yaml

# Check deployment status
kubectl get pods
```

## 8. Monitoring & Maintenance

### Setting Up Monitoring
```yaml
# In Prometheus configuration (monitoring/prometheus.yml)
scrape_configs:
  - job_name: 'compliance-api'
    scrape_interval: 15s
    static_configs:
      - targets: ['compliance-api:8001']
```

### Creating Dashboards
In Grafana, create dashboards for:
1. Request rate and latency
2. Classification distribution
3. Error rates
4. Model drift metrics

### Automated Testing
```bash
# Regular evaluation with new data
python workflows/compliance_detection/scripts/evaluate.py --config workflows/compliance_detection/configs/compliance_config.yaml --test-data data/compliance/new_test_data.json --output-dir workflows/compliance_detection/evaluation/periodic
```

### Drift Detection
```python
# In a monitoring script
import pandas as pd
from scipy.stats import ks_2samp

# Load baseline predictions distribution
baseline = pd.read_csv("workflows/compliance_detection/evaluation/baseline_distribution.csv")

# Load recent predictions
recent = pd.read_csv("logs/recent_predictions.csv")

# Compare distributions
drift_score, p_value = ks_2samp(baseline["prediction"], recent["prediction"])

if p_value < 0.05:
    # Log drift alert
    logger.warning(f"Potential distribution drift detected: score={drift_score}, p={p_value}")
    
    # Send alert
    send_alert("Model drift detected in compliance classifier")
```

## 9. Collaboration & Handoff

### Documentation
- Maintain up-to-date model cards
- Document all experiment results
- Keep track of dataset versions
- Record deployment configurations

### Handoff Process
1. **Team Walkthrough**: Present end-to-end workflow to stakeholders
2. **Documentation Review**: Ensure all documentation is complete
3. **Access Transfer**: Provide access to MLflow, monitoring, and source control
4. **Support Period**: Maintain availability for questions during transition
5. **Training**: Train operations team on monitoring alerts and incident response

### Collaboration Tools
- Use MLflow for experiment tracking
- Store prompts in version control
- Document key decisions in model registry
- Set up shared dashboards for cross-team visibility

---

## Appendix: Complete Workflow Example

The following represents the complete workflow from initial development to production:

1. **Setup Project**: Configure environment and dependencies
2. **Generate Data**: Create synthetic dataset for compliance categories
3. **Version Data**: Track dataset with DVC
4. **Develop Prompts**: Create multiple prompt variants
5. **Track Experiments**: Log all prompt tests in MLflow
6. **Evaluate Models**: Run comprehensive evaluation on test set
7. **Register Model**: Register best-performing model and prompt
8. **Deploy API**: Deploy FastAPI service with selected prompt
9. **Monitor Performance**: Track metrics in Prometheus/Grafana
10. **Maintain System**: Regular evaluation, drift detection, and updates

For specific command details, refer to the relevant sections above.