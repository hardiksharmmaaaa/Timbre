# -*- mode: makefile -*-

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn
WORKERS ?= 4

.PHONY: setup data-download data-prepare data-extract data-augment data-split data-ready data-augment-ready train-baselines train train-all train-augmented calibrate calibrate-augmented evaluate evaluate-augmented test-models augmentation-ablation model-search-v2 model-search-dynamic-v2 tune-dynamic-v2 validate-dynamic-finalist-v2 serve lint test ci tracker-update ship clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

data-download:
	$(PYTHON) -m scripts.download_data

data-prepare:
	$(PYTHON) -m scripts.prepare_data --workers $(WORKERS)

data-extract:
	$(PYTHON) -m scripts.extract_features --workers $(WORKERS)

data-augment:
	$(PYTHON) -m scripts.extract_augmented_features --workers $(WORKERS)

# Compatibility alias; canonical preparation now freezes splits before extraction.
data-split:
	$(PYTHON) -m scripts.prepare_data --workers $(WORKERS)

data-ready:
	$(PYTHON) -m scripts.verify_data

data-augment-ready:
	$(PYTHON) -m scripts.verify_data --augmented

train-baselines:
	$(PYTHON) -m scripts.train_baselines

train:
	$(PYTHON) -m scripts.train

train-augmented:
	$(PYTHON) -m scripts.train --include-augmented --model-dir models/augmented

calibrate:
	$(PYTHON) -m scripts.calibrate

calibrate-augmented:
	$(PYTHON) -m scripts.calibrate --model-dir models/augmented

train-all: data-ready
	$(MAKE) train-baselines
	$(MAKE) train
	$(MAKE) calibrate

evaluate:
	$(PYTHON) -m scripts.evaluate

evaluate-augmented:
	$(PYTHON) -m scripts.evaluate --model-dir models/augmented --run-name six_class_v1_augmented

test-models: evaluate

augmentation-ablation: data-ready data-augment data-augment-ready
	$(MAKE) train-augmented
	$(MAKE) calibrate-augmented
	$(MAKE) evaluate-augmented

# Compare predeclared model families using speaker-grouped folds from training data only.
model-search-v2: data-ready
	$(PYTHON) -m scripts.search_models

model-search-dynamic-v2: data-ready
	$(PYTHON) -m scripts.search_models --representation dynamic_v2 \
		--output experiments/model_search_dynamic_v2/summary.json

tune-dynamic-v2: data-ready
	$(PYTHON) -m scripts.tune_dynamic_model

validate-dynamic-finalist-v2: data-ready
	$(PYTHON) -m scripts.tune_dynamic_model --folds 5 \
		--candidate hgb_leaf7_agreement \
		--output experiments/dynamic_finalist_v2/summary.json

serve:
	$(UVICORN) timbre.api.main:app --reload --port 8000

lint:
	$(PYTHON) -m ruff check src scripts tests

test:
	$(PYTHON) -m pytest

ci: lint test

# Refresh tracker status and select the next engineering task after checks pass.
tracker-update: ci
	$(PYTHON) -m scripts.update_tracker --checks-passed

# Run checks, refresh the tracker, create one meaningful commit, and push the current branch.
# Usage: make ship MESSAGE="Improve voice scanner animation"
ship:
	@bash scripts/ship.sh "$(MESSAGE)"

clean:
	rm -rf data/manifests/* data/processed/* data/processed_augmented/* data/splits/* models/* experiments/* .venv

## to run the app: make serve
