## Makefile for LLM Ops Pipeline

.PHONY: help setup install-deps test lint typecheck clean build docs docker-build docker-run docker-compose-up \
	minikube-setup minikube-deploy port-forward-all test-deployment compliance-generate-data \
	compliance-prompt-tuning compliance-evaluate compliance-deploy

# Default target
help:
	@echo "Available targets:"
	@echo "  setup              Install development dependencies"
	@echo "  install-deps       Install project dependencies"
	@echo "  test               Run all tests"
	@echo "  lint               Run linters"
	@echo "  typecheck          Run type checker"
	@echo "  clean              Clean build artifacts"
	@echo "  build              Build the project"
	@echo "  docs               Generate documentation"
	@echo "  docker-build       Build Docker image"
	@echo "  docker-run         Run Docker container"
	@echo "  docker-compose-up  Run all services with docker-compose"
	@echo "  minikube-setup     Setup Minikube cluster"
	@echo "  minikube-deploy    Deploy to Minikube"
	@echo "  port-forward-all   Port forward all services"
	@echo "  test-deployment    Test the deployment"
	@echo ""
	@echo "Compliance Detection Workflow:"
	@echo "  compliance-generate-data    Generate synthetic data for compliance detection"
	@echo "  compliance-prompt-tuning    Run prompt tuning experiments"
	@echo "  compliance-evaluate         Evaluate the compliance detection model"
	@echo "  compliance-deploy           Deploy the compliance detection API"

# Setup
setup: install-deps

install-deps:
	pip install -e .
	pip install -r requirements.txt

# Testing
test:
	pytest

# Code quality
lint:
	flake8
	black --check .

typecheck:
	mypy .

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	python -m pip install -e .

# Documentation
docs:
	cd docs && make html

# Docker
docker-build:
	docker build -t llm-ops-pipeline:latest .

docker-run:
	docker run -p 8000:8000 llm-ops-pipeline:latest

docker-compose-up:
	docker-compose up -d

# Minikube
minikube-setup:
	./scripts/local/setup_minikube.sh

minikube-deploy:
	kubectl apply -f k8s/minikube/volumes.yaml
	kubectl apply -f k8s/minikube/mlflow-deployment.yaml
	kubectl apply -f k8s/minikube/monitoring-deployment.yaml
	kubectl apply -f k8s/minikube/api-deployment.yaml

port-forward-all:
	./scripts/local/port_forward.sh

test-deployment:
	./scripts/local/test_deployment.sh

# Compliance Detection Workflow
compliance-generate-data:
	python workflows/compliance_detection/scripts/generate_data.py --count 1000

compliance-prompt-tuning:
	python workflows/compliance_detection/scripts/prompt_tuning.py --sample-size 100

compliance-evaluate:
	python workflows/compliance_detection/scripts/evaluate.py

compliance-deploy:
	python workflows/compliance_detection/scripts/deploy_api.py