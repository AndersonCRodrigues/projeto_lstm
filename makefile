.PHONY: help install lint test train run-api run-mlflow docker-build docker-run clean

VENV_NAME = .venv
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip
DOCKER_IMAGE_NAME = lstm-api-service

help:
	@echo "Production LSTM Project Makefile"
	@echo "--------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Create virtual environment and install dependencies
	@echo "Setting up virtual environment..."
	@if [ ! -d "$(VENV_NAME)" ]; then python3 -m venv $(VENV_NAME); fi
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@$(PIP) install -e .

lint: ## Run linter and formatter (Ruff) on code
	@$(PYTHON) -m ruff check src/ tests/ --fix
	@$(PYTHON) -m ruff format src/ tests/

test: ## Execute tests with pytest
	@$(PYTHON) -m pytest

run-mlflow: ## Start MLflow UI
	@echo "Starting MLflow UI at http://127.0.0.1:5000..."
	@mlflow ui

train: ## Execute complete training pipeline
	@$(PYTHON) src/production_lstm/training/pipeline.py

run-api: ## Start API locally in development mode
	@echo "Starting API at http://localhost:8000..."
	@$(PYTHON) -m uvicorn production_lstm.api.main:app --reload --port 8000

docker-build: ## Build Docker image for application
	@docker build -t $(DOCKER_IMAGE_NAME):latest .

docker-run: ## Run API inside Docker container
	@docker run --rm -p 8000:8000 --env-file .env $(DOCKER_IMAGE_NAME)

clean: ## Remove temporary files, cache and virtual environment
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .pytest_cache .coverage mlruns/ artifacts/ logs/ build/ dist/ *.egg-info
	@if [ -d "$(VENV_NAME)" ]; then rm -rf $(VENV_NAME); fi