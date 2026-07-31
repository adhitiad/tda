.PHONY: help install backtest live cockpit test lint format check docker-up docker-down

help: ## Show this help message
	@echo "Quantuis - Trading Data Analysis Framework"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -e .

backtest: ## Run backtest
	python -m crypto_trading_framework.cli backtest

live: ## Start live trading bot
	python -m crypto_trading_framework.cli live

cockpit: ## Launch Streamlit observability cockpit
	python -m crypto_trading_framework.cli cockpit

signals: ## Generate trading signals
	python -m crypto_trading_framework.cli signals

train: ## Train models
	python -m crypto_trading_framework.cli train

test: ## Run pytest
	pytest tests/ -v

lint: ## Run ruff linter
	ruff check .

format: ## Run ruff formatter and black
	ruff format .
	black .

check: ## Run all checks (lint + format + mypy)
	ruff check .
	ruff format --check .
	black --check .
	mypy crypto_trading_framework/

docker-up: ## Start Docker Compose stack
	docker-compose up -d

docker-down: ## Stop Docker Compose stack
	docker-compose down

clean: ## Remove cache and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name build -exec rm -rf {} + 2>/dev/null || true