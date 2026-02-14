.PHONY: format test test-all

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

test:
	uv run pytest tests/test_wright.py tests/test_gradients.py -v

test-all:
	uv run pytest tests/ -v
