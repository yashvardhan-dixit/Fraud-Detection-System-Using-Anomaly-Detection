.PHONY: help install test train api dashboard docker-build docker-up docker-down clean

help:
	@echo "Fraud Detection System - Available Commands"
	@echo "============================================="
	@echo "install         - Install dependencies"
	@echo "test            - Run tests"
	@echo "train           - Train models"
	@echo "api             - Run FastAPI server"
	@echo "dashboard       - Run Streamlit dashboard"
	@echo "docker-build    - Build Docker images"
	@echo "docker-up       - Start Docker services"
	@echo "docker-down     - Stop Docker services"
	@echo "clean           - Clean temporary files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

train:
	python train.py

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/visualization/dashboard.py --server.port 8501

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d api dashboard

docker-train:
	docker-compose --profile training up training

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf build/ dist/

lint:
	flake8 src/ tests/ --max-line-length=120

format:
	black src/ tests/ --line-length=120
