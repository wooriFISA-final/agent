.PHONY: setup install test lint format clean run

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

install:
	pip install -e .

test:
	pytest tests/ -v --cov=agents --cov=graph

lint:
	flake8 agents/ graph/ core/ main.py

format:
	black agents/ graph/ core/ main.py
	isort agents/ graph/ core/ main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/

run:
	python main.py
