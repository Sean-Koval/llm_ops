# Kubernetes Deployment for LLM Ops Pipeline

This directory contains Kubernetes manifests for deploying the LLM Ops Pipeline to Google Kubernetes Engine (GKE).

## Files

- `volumes.yaml`: Persistent Volume Claims for storing models, configuration, and monitoring data
- `api-deployment.yaml`: Deployment and Service for the LLM API
- `mlflow-deployment.yaml`: Deployment and Service for MLflow tracking server
- `monitoring-deployment.yaml`: Deployments and Services for Prometheus and Grafana

## Deployment Steps

1. **Build and push the Docker image**

```bash
# Build the Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/llm-ops-pipeline:latest .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/llm-ops-pipeline:latest
```

2. **Update the image name in api-deployment.yaml**

Replace `YOUR_PROJECT_ID` with your actual Google Cloud Project ID.

3. **Deploy to GKE**

```bash
# Set up persistent volumes
kubectl apply -f volumes.yaml

# Deploy components
kubectl apply -f api-deployment.yaml
kubectl apply -f mlflow-deployment.yaml
kubectl apply -f monitoring-deployment.yaml
```

4. **Get the external IPs for services**

```bash
kubectl get service llm-api-service
kubectl get service mlflow-service
kubectl get service grafana-service
```

## Scaling

To scale the API deployment:

```bash
kubectl scale deployment llm-api --replicas=3
```

## Cleanup

To delete all resources:

```bash
kubectl delete -f api-deployment.yaml
kubectl delete -f mlflow-deployment.yaml
kubectl delete -f monitoring-deployment.yaml
kubectl delete -f volumes.yaml
```