#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ml-pipeline-automation
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python3
APP_HOST ?= 127.0.0.1
APP_PORT ?= 8000
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000
MLFLOW_BACKEND_STORE_URI ?= sqlite:///artifacts/mlflow/mlflow.db
MLFLOW_ARTIFACTS_DESTINATION ?= $(CURDIR)/artifacts/mlflow/mlartifacts
MLFLOW_TRACKING_URI ?= http://127.0.0.1:$(MLFLOW_PORT)
MODELS ?= log_reg random_forest catboost
N_JOBS ?= -1
DATA_PATH ?= data/processed/UCI_Credit_Card.csv

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Download, validate and clean the dataset into data/processed
.PHONY: data
data:
	uv run python -m src.data.make_dataset

## Validate and clean the existing raw dataset
.PHONY: prepare-data
prepare-data:
	uv run python -m src.data.prepare_dataset

## Start the credit scoring API (localhost:8000 by default)
.PHONY: run
run:
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" uv run uvicorn src.api.app:app \
		--host "$(APP_HOST)" --port "$(APP_PORT)"

## Start the credit scoring API with automatic reload
.PHONY: dev
dev:
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" uv run uvicorn src.api.app:app \
		--host "$(APP_HOST)" --port "$(APP_PORT)" --reload

## Start MLflow UI and tracking API (localhost:5000 by default)
.PHONY: mlflow
mlflow:
	mkdir -p artifacts/mlflow
	uv run mlflow server --host "$(MLFLOW_HOST)" --port "$(MLFLOW_PORT)" \
		--backend-store-uri "$(MLFLOW_BACKEND_STORE_URI)" \
		--artifacts-destination "$(MLFLOW_ARTIFACTS_DESTINATION)"

## Start MLflow UI (alias for make mlflow)
.PHONY: mlflow-ui
mlflow-ui: mlflow

## Start MLflow tracking server (alias for make mlflow)
.PHONY: mlflow-server
mlflow-server: mlflow

## Train credit default model and log to local MLflow (override DATA_PATH if needed)
.PHONY: train
train: data
	uv run python -m src.models.train --data-path "$(DATA_PATH)" --models $(MODELS) --n-jobs $(N_JOBS)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
