# Compliance Breach Detection Workflow

This workflow demonstrates using the LLM Ops Pipeline for detecting compliance breaches in short text messages using Google's Vertex AI Gemini Flash model.

## Use Case

Financial institutions and regulated industries need to monitor communications for potential compliance breaches, including:
- Unethical behavior
- Illegal activities
- Regulatory violations
- Confidential information sharing
- Harassment and inappropriate content

## Workflow Overview

1. **Data Generation & Management**
   - Generate synthetic training/test data with LLMs
   - Version control datasets with DVC
   - Create evaluation benchmarks

2. **Model & Prompt Engineering**
   - Design classification prompts for Gemini Flash
   - Optimize prompts through experimentation
   - Track prompt variants in MLflow

3. **Deployment & Serving**
   - Deploy optimized prompts to production
   - Set up monitoring and logging
   - Implement feedback loops

4. **Evaluation & Monitoring**
   - Track classification metrics (precision, recall, F1)
   - Monitor for concept drift
   - Implement human feedback mechanisms

## Key Components

- **Google Vertex AI Gemini Flash**: Primary LLM for classification
- **DVC**: Data version control
- **MLflow**: Experiment tracking and model registry
- **Prometheus/Grafana**: Monitoring
- **FastAPI**: Serving infrastructure

## Getting Started

1. Setup environment: `make setup`
2. Generate synthetic data: `python workflows/compliance_detection/scripts/generate_data.py`
3. Experiment with prompts: `python workflows/compliance_detection/scripts/prompt_tuning.py`
4. Run evaluation: `python workflows/compliance_detection/scripts/evaluate.py`
5. Deploy API: `make deploy-compliance-api`