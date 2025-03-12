#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up port forwarding for LLM Ops Pipeline services${NC}"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi

echo -e "${YELLOW}Running in WSL2 with Docker Desktop integration${NC}"

# Kill any existing port-forward processes
echo -e "${GREEN}Cleaning up any existing port forwards...${NC}"
pkill -f "kubectl port-forward" || true

# Check if services exist
echo -e "${GREEN}Checking services...${NC}"
kubectl get svc llm-api-service || { echo -e "${RED}llm-api-service not found. Have you run 'make minikube-setup'?${NC}"; exit 1; }

# Start port forwarding in the background
echo -e "${GREEN}Starting port forwarding...${NC}"
kubectl port-forward svc/llm-api-service 8000:8000 &
kubectl port-forward svc/mlflow-service 5000:5000 &
kubectl port-forward svc/prometheus-service 9090:9090 &
kubectl port-forward svc/grafana-service 3000:3000 &

echo -e "${GREEN}Port forwarding setup complete!${NC}"
echo -e "Services are available at:"
echo -e "LLM API: ${YELLOW}http://localhost:8000${NC}"
echo -e "MLflow: ${YELLOW}http://localhost:5000${NC}"
echo -e "Prometheus: ${YELLOW}http://localhost:9090${NC}"
echo -e "Grafana: ${YELLOW}http://localhost:3000${NC} (default credentials: admin/admin)"
echo -e "\n${YELLOW}Press Ctrl+C to stop port forwarding${NC}"

# Wait for user to press Ctrl+C
wait