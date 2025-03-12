# Model Card: Compliance Detector

## Model Details
- **Name:** compliance-detector-20250312
- **Version:** 1.0.0
- **Type:** Prompt-based classification using Vertex AI Gemini Flash
- **Date:** 2025-03-12
- **Developer:** Example Organization Data Science Team

## Model Description
This model classifies text messages for compliance breaches in financial contexts. It uses a few-shot prompting approach with Google Vertex AI Gemini Flash to detect various categories of compliance issues.

## Intended Use
- **Primary Use Case:** Monitoring workplace communications for potential compliance breaches
- **Intended Users:** Compliance teams, risk management personnel

## Training Data
The model was tuned using a synthetic dataset of 1,000 examples covering the following categories:
- Ethical breaches
- Illegal activities
- Regulatory violations
- Confidential information sharing
- Harassment
- Compliant messages

## Performance Metrics
- **Accuracy:** 0.9667
- **Precision (macro):** 0.9696
- **Recall (macro):** 0.9673
- **F1 Score (macro):** 0.9679

## Limitations
- The model performs less effectively on ambiguous cases that could fall into multiple categories
- The model was trained on synthetic data and may not fully capture real-world diversity
- The model may have implicit biases based on its training data

## Ethical Considerations
- This model is intended as a first-pass screening tool and should not replace human judgment
- All flagged messages should be reviewed by a human reviewer before any action is taken
- User privacy considerations must be addressed before deployment

## Maintenance
- Regular retraining is recommended as new compliance issues emerge
- Performance monitoring should track concept drift and accuracy over time
