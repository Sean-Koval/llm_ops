## Makefile for LLM Ops Pipeline

.PHONY: help setup install-uv install-uv-wsl install-deps test lint typecheck clean build docs docker-build docker-run docker-compose-up \
	docker-deploy docker-cleanup minikube-setup minikube-deploy minikube-cleanup port-forward-all test-deployment \
	compliance-generate-data compliance-prompt-tuning compliance-evaluate compliance-deploy run-inference start-server \
	minimal-server fine-tune create-venv activate-venv

# Default target
help:
	@echo "Available targets:"
	@echo "  setup              Install development dependencies with UV"
	@echo "  install-uv         Install UV package manager"
	@echo "  install-uv-wsl     Install UV package manager for Ubuntu WSL2"
	@echo "  install-deps       Install project dependencies with UV"
	@echo "  create-venv        Create a UV virtual environment"
	@echo "  activate-venv      Show how to activate the virtual environment"
	@echo "  test               Run all tests"
	@echo "  lint               Run linters"
	@echo "  typecheck          Run type checker"
	@echo "  clean              Clean build artifacts"
	@echo "  build              Build the project"
	@echo "  docs               Generate documentation"
	@echo ""
	@echo "Deployment and Infrastructure:"
	@echo "  docker-build       Build Docker image"
	@echo "  docker-run         Run Docker container"
	@echo "  docker-compose-up  Run all services with docker-compose"
	@echo "  docker-deploy      Deploy using Docker Compose"
	@echo "  docker-cleanup     Clean up Docker Compose deployment"
	@echo "  minikube-setup     Setup Minikube cluster"
	@echo "  minikube-deploy    Deploy to Minikube"
	@echo "  minikube-cleanup   Clean up Minikube deployment"
	@echo "  port-forward-all   Port forward all services"
	@echo "  test-deployment    Test the deployment"
	@echo ""
	@echo "Scripts:"
	@echo "  run-inference      Run inference using model"
	@echo "  start-server       Start the API server"
	@echo "  minimal-server     Start a minimal test server"
	@echo "  fine-tune          Run model fine-tuning"
	@echo ""
	@echo "Compliance Detection Workflow:"
	@echo "  compliance-generate-data    Generate synthetic data for compliance detection"
	@echo "  compliance-prompt-tuning    Run prompt tuning experiments"
	@echo "  compliance-evaluate         Evaluate the compliance detection model"
	@echo "  compliance-deploy           Deploy the compliance detection API"

# Setup
setup: install-uv install-deps

install-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing UV package manager..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}

install-uv-wsl:
	@echo "Setting up UV for Ubuntu WSL2..."
	@chmod +x scripts/local/setup_uv.sh
	@./scripts/local/setup_uv.sh

install-deps:
	uv pip install -e .
	uv pip install -r requirements.txt

# Testing
test:
	uv pip run pytest

# Code quality
lint:
	uv pip run flake8
	uv pip run black --check .

typecheck:
	uv pip run mypy .

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	uv pip install -e .

# Documentation
docs:
	cd docs && make html

# Scripts
run-inference:
	uv pip run python scripts/run_inference.py

start-server:
	uv pip run python scripts/start_server.py

minimal-server:
	uv pip run python scripts/minimal_server.py

fine-tune:
	uv pip run python scripts/fine_tune.py

# Docker
docker-build:
	docker build -t llm-ops-pipeline:latest .

docker-run:
	docker run -p 8000:8000 llm-ops-pipeline:latest

docker-compose-up:
	docker-compose up -d

docker-deploy:
	./scripts/local/docker_deploy.sh

docker-cleanup:
	./scripts/local/docker_cleanup.sh

# Minikube
minikube-setup:
	./scripts/local/setup_minikube.sh

minikube-deploy:
	kubectl apply -f k8s/minikube/volumes.yaml
	kubectl apply -f k8s/minikube/mlflow-deployment.yaml
	kubectl apply -f k8s/minikube/monitoring-deployment.yaml
	kubectl apply -f k8s/minikube/api-deployment.yaml

minikube-cleanup:
	./scripts/local/cleanup_minikube.sh

port-forward-all:
	./scripts/local/port_forward.sh

test-deployment:
	./scripts/local/test_deployment.sh

# Compliance Detection Workflow
compliance-generate-data:
	uv pip run python workflows/compliance_detection/scripts/generate_data.py --count 1000

compliance-prompt-tuning:
	uv pip run python workflows/compliance_detection/scripts/prompt_tuning.py --sample-size 100

compliance-evaluate:
	uv pip run python workflows/compliance_detection/scripts/evaluate.py

compliance-deploy:
	uv pip run python workflows/compliance_detection/scripts/deploy_api.py

# Virtual environment management
create-venv:
	uv venv

activate-venv:
	@echo "To activate the virtual environment, run:"
	@echo "source .venv/bin/activate"