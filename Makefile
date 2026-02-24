.PHONY: format test test-all

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

test:
	uv run --group test pytest tests/test_wright.py tests/test_gradients.py -v

test-all:
	uv run --group test pytest tests/ -v
