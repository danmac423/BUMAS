.PHONY: install fmt lint typecheck test check run clean

install:
	uv sync
	uv run pre-commit install

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run ty check

test:
	uv run pytest --cov

check: lint typecheck test

run:
	uv run lab_1

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .coverage coverage.xml htmlcov dist build *.egg-info
