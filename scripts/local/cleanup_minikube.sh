#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Cleaning up LLM Ops Pipeline from Minikube${NC}"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi

echo -e "${YELLOW}Running in WSL2 with Docker Desktop integration${NC}"

# Delete all deployments
echo -e "${GREEN}Deleting deployments...${NC}"
kubectl delete -f k8s/minikube/api-deployment.yaml --ignore-not-found=true
kubectl delete -f k8s/minikube/mlflow-deployment.yaml --ignore-not-found=true
kubectl delete -f k8s/minikube/monitoring-deployment.yaml --ignore-not-found=true

# Delete persistent volumes
echo -e "${GREEN}Deleting persistent volumes...${NC}"
kubectl delete -f k8s/minikube/volumes.yaml --ignore-not-found=true

# Show remaining resources
echo -e "${GREEN}Checking remaining resources...${NC}"
kubectl get pods
kubectl get services
kubectl get pvc

echo -e "${GREEN}Cleanup complete!${NC}"
echo -e "${YELLOW}Note: Minikube is managed by Docker Desktop on Windows.${NC}"
echo -e "${YELLOW}To manage Minikube, use the Windows command prompt or PowerShell.${NC}"