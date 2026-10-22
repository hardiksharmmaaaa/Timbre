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
	$(PYTHON) -m ml.data.download

data-prepare:
	$(PYTHON) -m ml.data.prepare --workers $(WORKERS)

data-extract:
	$(PYTHON) -m ml.training.extract_features --workers $(WORKERS)

data-augment:
	$(PYTHON) -m ml.training.extract_augmented --workers $(WORKERS)

# Compatibility alias; canonical preparation now freezes splits before extraction.
data-split:
	$(PYTHON) -m ml.data.prepare --workers $(WORKERS)

data-ready:
	$(PYTHON) -m ml.data.verify

data-augment-ready:
	$(PYTHON) -m ml.data.verify --augmented

train-baselines:
	$(PYTHON) -m ml.training.train_baselines

train:
	$(PYTHON) -m ml.training.train

train-augmented:
	$(PYTHON) -m ml.training.train --include-augmented --model-dir models/augmented

calibrate:
	$(PYTHON) -m ml.training.calibrate

calibrate-augmented:
	$(PYTHON) -m ml.training.calibrate --model-dir models/augmented

train-all: data-ready
	$(MAKE) train-baselines
	$(MAKE) train
	$(MAKE) calibrate

evaluate:
	$(PYTHON) -m ml.training.evaluate

evaluate-augmented:
	$(PYTHON) -m ml.training.evaluate --model-dir models/augmented --run-name six_class_v1_augmented

test-models: evaluate

augmentation-ablation: data-ready data-augment data-augment-ready
	$(MAKE) train-augmented
	$(MAKE) calibrate-augmented
	$(MAKE) evaluate-augmented

# Compare predeclared model families using speaker-grouped folds from training data only.
model-search-v2: data-ready
	$(PYTHON) -m ml.training.search_models

model-search-dynamic-v2: data-ready
	$(PYTHON) -m ml.training.search_models --representation dynamic_v2 \
		--output experiments/model_search_dynamic_v2/summary.json

tune-dynamic-v2: data-ready
	$(PYTHON) -m ml.training.tune_dynamic_model

validate-dynamic-finalist-v2: data-ready
	$(PYTHON) -m ml.training.tune_dynamic_model --folds 5 \
		--candidate hgb_leaf7_agreement \
		--output experiments/dynamic_finalist_v2/summary.json

serve:
	$(UVICORN) timbre.api.main:app --reload --port 8000

lint:
	$(PYTHON) -m ruff check src ml scripts tests

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
