#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Testing LLM Ops Pipeline deployment in Minikube${NC}"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi

echo -e "${YELLOW}Running in WSL2 with Docker Desktop integration${NC}"

# Check pod status
echo -e "${GREEN}Checking pod status...${NC}"
kubectl get pods

# Test services by port forwarding temporarily
API_PORT=8001
MLFLOW_PORT=5001
PROMETHEUS_PORT=9091
GRAFANA_PORT=3001

echo -e "${GREEN}Testing API service...${NC}"
kubectl port-forward svc/llm-api-service ${API_PORT}:8000 > /dev/null 2>&1 &
API_PID=$!
sleep 3
if curl -s --head --request GET http://localhost:${API_PORT}/health | grep "200" > /dev/null; then 
    echo -e "${GREEN}✓ API service is running${NC}"
else
    echo -e "${RED}✗ API service is not responding${NC}"
fi
kill $API_PID 2> /dev/null || true

echo -e "${GREEN}Testing MLflow service...${NC}"
kubectl port-forward svc/mlflow-service ${MLFLOW_PORT}:5000 > /dev/null 2>&1 &
MLFLOW_PID=$!
sleep 3
if curl -s --head --request GET http://localhost:${MLFLOW_PORT} | grep "200" > /dev/null; then 
    echo -e "${GREEN}✓ MLflow service is running${NC}"
else
    echo -e "${RED}✗ MLflow service is not responding${NC}"
fi
kill $MLFLOW_PID 2> /dev/null || true

echo -e "${GREEN}Testing Prometheus service...${NC}"
kubectl port-forward svc/prometheus-service ${PROMETHEUS_PORT}:9090 > /dev/null 2>&1 &
PROMETHEUS_PID=$!
sleep 3
if curl -s --head --request GET http://localhost:${PROMETHEUS_PORT}/-/healthy | grep "200" > /dev/null; then 
    echo -e "${GREEN}✓ Prometheus service is running${NC}"
else
    echo -e "${RED}✗ Prometheus service is not responding${NC}"
fi
kill $PROMETHEUS_PID 2> /dev/null || true

echo -e "${GREEN}Testing Grafana service...${NC}"
kubectl port-forward svc/grafana-service ${GRAFANA_PORT}:3000 > /dev/null 2>&1 &
GRAFANA_PID=$!
sleep 3
if curl -s --head --request GET http://localhost:${GRAFANA_PORT}/api/health | grep "200" > /dev/null; then 
    echo -e "${GREEN}✓ Grafana service is running${NC}"
else
    echo -e "${RED}✗ Grafana service is not responding${NC}"
fi
kill $GRAFANA_PID 2> /dev/null || true

# Clean up any remaining port-forward processes
pkill -f "kubectl port-forward" || true

echo -e "\n${GREEN}Testing complete!${NC}"
echo -e "To access services, run: ${YELLOW}make port-forward${NC}"