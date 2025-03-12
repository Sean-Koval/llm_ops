# LLM Ops Pipeline

A comprehensive MLOps pipeline for Large Language Models, supporting the entire lifecycle from data preparation to model deployment and monitoring.

## Features

- Data preparation and versioning
- Model training with experiment tracking through MLflow and Weights & Biases
- Model evaluation and testing with standardized metrics
- Model deployment and serving via FastAPI
- Inference monitoring with Prometheus and Grafana
- CI/CD integration

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm_ops_pipeline.git
cd llm_ops_pipeline

# Install the package
python -m pip install -e .
```

### Deployment Options

#### Option 1: Local Development with Docker Compose

For local development, you can use the provided Docker Compose setup:

```bash
# Start all services (API, MLflow, Prometheus, Grafana)
docker compose up -d

# To view logs
docker compose logs -f

# To stop all services
docker compose down
```

This will start:
- The model serving API on port 8000
- MLflow tracking server on port 5000
- Prometheus on port 9090
- Grafana on port 3000 (default credentials: admin/admin)

You can also use the Makefile for convenience:

```bash
# Start services
make compose-up

# View logs
make compose-logs

# Stop services
make compose-down
```

#### Option 2: Local Deployment with Docker Compose

For the simplest local deployment without Kubernetes:

```bash
# Make scripts executable
make setup-scripts

# Deploy using Docker Compose
make docker-deploy

# Check logs
make compose-logs

# Clean up when done
make docker-cleanup
```

This approach skips Kubernetes entirely and uses plain Docker Compose, which is more reliable in WSL2/Windows environments.

#### Option 3: Using Kubernetes with Minikube in WSL2/Windows

For testing Kubernetes deployments with Docker Desktop/WSL2 setup:

1. **Prerequisites**:
   - Docker Desktop installed on Windows with Kubernetes enabled
   - WSL2 configured with kubectl installed
   - Kubectl configured to use the Kubernetes in Docker Desktop

2. **Setup**:
   ```bash
   # Make scripts executable
   make setup-scripts

   # Deploy services to Kubernetes
   make minikube-setup

   # Forward ports to access services
   make port-forward

   # Test the deployment
   make test-deployment

   # Clean up resources
   make minikube-cleanup
   ```

3. **Accessing services**:
   - API: http://localhost:8000
   - MLflow: http://localhost:5000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (default credentials: admin/admin)

#### Option 3: Deploying to Google Kubernetes Engine (GKE)

For production deployments, follow these steps to deploy to GKE:

1. **Set up a GKE cluster**

```bash
# Create a GKE cluster with GPU support
gcloud container clusters create llm-ops-cluster \
    --machine-type=n1-standard-4 \
    --num-nodes=1 \
    --zone=us-central1-a \
    --cluster-version=latest \
    --accelerator="type=nvidia-tesla-t4,count=1"

# Install NVIDIA drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

2. **Create Kubernetes manifests**

Create Kubernetes deployment files in a `k8s/` directory:
- `k8s/api-deployment.yaml` - For the model API service
- `k8s/mlflow-deployment.yaml` - For MLflow tracking server
- `k8s/monitoring-deployment.yaml` - For Prometheus and Grafana

3. **Deploy to GKE**

```bash
# Set up persistent volumes
kubectl apply -f k8s/volumes.yaml

# Deploy components
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/monitoring-deployment.yaml

# Get the external IP for the API service
kubectl get service llm-api-service
```

4. **Scale as needed**

```bash
# Scale the API deployment
kubectl scale deployment llm-api --replicas=3
```

#### Option 4: Automatic Deployment with GitHub Actions

This repository includes a GitHub Actions workflow for automatically deploying to GKE:

1. **Set up GitHub Secrets**

Add the following secrets to your GitHub repository:
- `GCP_PROJECT_ID`: Your Google Cloud project ID (e.g., `sk-ml-inference`)
- `GCP_SA_KEY`: Your Google Cloud service account key (the entire JSON content of credentials.json)

2. **Enable GitHub Actions**

The workflow will automatically deploy to GKE when pushing to the main branch.

```bash
# View deployment status
https://github.com/yourusername/llm_ops_pipeline/actions
```

3. **Manual Deployment**

You can also trigger a manual deployment from the GitHub Actions tab by using the "workflow_dispatch" event.

4. **Troubleshooting**

If you encounter issues with the GitHub Actions workflow:
- Check that your service account has the necessary permissions
- Verify that the GKE cluster name matches the one in the workflow file
- Look at the detailed logs in the GitHub Actions tab

## End-to-End Workflow Examples

### 1. Fine-tuning an LLM for Classification

This example shows how to fine-tune a pre-trained model for a classification task:

```bash
# Configure your experiment in a YAML file
# See config_samples/base_config.yaml for an example

# Run the fine-tuning script
python scripts/fine_tune.py --config path/to/your_config.yaml

# Track the experiment in MLflow
# Visit http://localhost:5000 after starting Docker Compose
```

Example configuration for classification:

```yaml
experiment:
  name: "sentiment-classification"
  tracking_uri: "http://localhost:5000"
  tags:
    task_type: "classification"
    model_type: "bert"

model:
  name: "bert-base-uncased"
  revision: "main"

data:
  train_path: "data/sentiment/train.csv"
  validation_path: "data/sentiment/validation.csv"
  text_column: "text"
  label_column: "sentiment"

training:
  batch_size: 16
  learning_rate: 5e-5
  num_epochs: 3
  warmup_steps: 500
  max_length: 128
  gradient_accumulation_steps: 1
```

### 2. Deploying a Model as an API Endpoint

After training, you can deploy your model:

```bash
# Start the API server
python scripts/start_server.py --model-path ./models/sentiment-classification

# Or, use Docker Compose to deploy the full stack
docker compose up -d
```

### 3. Running Inference

```bash
# Using the CLI
python scripts/run_inference.py --model-path ./models/sentiment-classification --input "I really enjoyed this movie"

# Using the API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed this movie", "options": {"return_probabilities": true}}'
```

### 4. Monitoring the Deployment

- View API metrics in Prometheus: http://localhost:9090
- View dashboards in Grafana: http://localhost:3000
  - Import the sample dashboard from `monitoring/dashboards/api_dashboard.json`

### 5. Experiment Tracking and Versioning

- Track experiments in MLflow: http://localhost:5000
- Compare runs, view metrics, and manage model versions
- Export models for deployment

```bash
# Register model in MLflow
mlflow models register -m "runs:/<run_id>/model" --name "sentiment-classifier"

# Promote model to production
mlflow models transition-stage --name "sentiment-classifier" --version 1 --stage "Production"
```

## Project Structure

```
llm_ops_pipeline/
├── data/            # Data processing and preparation
├── model/           # Model architecture and configuration
├── training/        # Training pipelines and utilities
├── inference/       # Inference and serving code
├── evaluation/      # Evaluation metrics and tests
├── utils/           # Shared utilities
├── config/          # Configuration management
└── api/             # API for model serving

tests/
├── unit/            # Unit tests
└── integration/     # Integration tests
```

## Setting Up CI/CD with Google Cloud and GitHub

To set up continuous deployment to Google Kubernetes Engine:

1. **Create a Google Cloud Service Account**

```bash
# Create a service account
gcloud iam service-accounts create github-actions

# Assign necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/container.developer"
    
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# Create and download a JSON key
gcloud iam service-accounts keys create credentials.json \
    --iam-account=github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

2. **Add Secrets to GitHub**

- Go to your GitHub repository > Settings > Secrets and variables > Actions
- Create a new repository secret `GCP_PROJECT_ID` with your Google Cloud project ID
- Create a new repository secret `GCP_SA_KEY` with the entire content of your `credentials.json` file

3. **Secure Your Service Account Key**

Once you've added the key to GitHub Secrets, make sure to:
- Delete the local copy of `credentials.json`
- Add `credentials.json` to your `.gitignore` file (already included)

## License

This project is licensed under the MIT License - see the LICENSE file for details.