# -*- mode: makefile -*-

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn

.PHONY: setup data-download data-extract data-split train evaluate serve lint test ci ship clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

data-download:
	$(PYTHON) -m scripts.download_data

data-extract:
	$(PYTHON) -m scripts.extract_features

data-split:
	$(PYTHON) -m scripts.make_splits

train:
	$(PYTHON) -m scripts.train

evaluate:
	$(PYTHON) -m scripts.evaluate

serve:
	$(UVICORN) timbre.api.main:app --reload --port 8000

lint:
	$(PYTHON) -m ruff check src scripts tests

test:
	$(PYTHON) -m pytest

ci: lint test

# Run checks, create one meaningful commit, and push the current branch.
# Usage: make ship MESSAGE="Improve voice scanner animation"
ship:
	@bash scripts/ship.sh "$(MESSAGE)"

clean:
	rm -rf data/processed/* models/* experiments/* .venv
