# Resume Parser Makefile

.PHONY: help install dev test lint format clean run health

# Default target
help:
	@echo "Available targets:"
	@echo "  install    Install dependencies"
	@echo "  dev        Install development dependencies"
	@echo "  test       Run tests"
	@echo "  lint       Run linting"
	@echo "  format     Format code"
	@echo "  clean      Clean up temporary files"
	@echo "  run        Run the application"
	@echo "  health     Check application health"
	@echo "  intent-test Run 20+ prompts through /analyze-query-intent"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
dev: install
	pip install pytest pytest-asyncio black flake8 mypy

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-coverage:
	pytest tests/ --cov=src/resume_parser --cov-report=html

# Run linting
lint:
	flake8 src/ tests/ app.py --max-line-length=100
	mypy src/ --ignore-missing-imports

# Format code
format:
	black src/ tests/ app.py config/

# Clean up temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf .coverage htmlcov/
	rm -rf logs/*.log

# Run the application
run:
	uvicorn app:app --reload --host 0.0.0.0 --port 8001

# Check application health
health:
	curl -s http://localhost:8001/health | python -m json.tool

# Run intent analyzer test suite (set INTENT_BASE_URL if not 8000)
intent-test:
	INTENT_BASE_URL?=http://localhost:8000 \
	python scripts/intent_test_runner.py

# Setup development environment
setup-dev: dev
	cp .env.example .env
	mkdir -p logs uploads
	@echo "Development environment setup complete!"
	@echo "Edit .env file with your configuration before running the app."

# Docker targets
docker-build:
	docker build -t resume-parser .

docker-run:
	docker run -p 8000:8000 --env-file .env resume-parser
