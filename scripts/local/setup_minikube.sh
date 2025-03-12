#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Minikube setup for LLM Ops Pipeline${NC}"

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Please install docker first.${NC}"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi

echo -e "${YELLOW}Running in WSL2 with Docker Desktop integration${NC}"
echo -e "${GREEN}Using kubectl to interact with Windows Minikube${NC}"

# Create persistent volumes
echo -e "${GREEN}Creating persistent volumes...${NC}"
kubectl apply -f k8s/minikube/volumes.yaml

# Build and tag the Docker image
echo -e "${GREEN}Building LLM API Docker image...${NC}"
docker build -t llm-ops-pipeline:latest .

# Verify Docker image was built
echo -e "${GREEN}Verifying Docker image...${NC}"
docker images | grep llm-ops-pipeline

# Create ConfigMap for Prometheus
echo -e "${GREEN}Deploying services...${NC}"
kubectl apply -f k8s/minikube/api-deployment.yaml
kubectl apply -f k8s/minikube/mlflow-deployment.yaml
kubectl apply -f k8s/minikube/monitoring-deployment.yaml

# Wait for deployments to be ready
echo -e "${GREEN}Waiting for deployments to be ready...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/llm-api || echo "Waiting for llm-api deployment..."
kubectl wait --for=condition=available --timeout=300s deployment/mlflow || echo "Waiting for mlflow deployment..."
kubectl wait --for=condition=available --timeout=300s deployment/prometheus || echo "Waiting for prometheus deployment..."
kubectl wait --for=condition=available --timeout=300s deployment/grafana || echo "Waiting for grafana deployment..."

# Show pods status
echo -e "${GREEN}Current pod status:${NC}"
kubectl get pods

# Display service information
echo -e "${GREEN}Service information:${NC}"
kubectl get services

echo -e "${GREEN}Setup complete!${NC}"
echo -e "\n${YELLOW}To get service URLs in Windows, run:${NC}"
echo -e "minikube service llm-api-service --url"
echo -e "minikube service mlflow-service --url"
echo -e "minikube service prometheus-service --url"
echo -e "minikube service grafana-service --url"

echo -e "\n${YELLOW}For port forwarding in WSL2, run:${NC}"
echo -e "make port-forward"