PYTHON ?= .venv/bin/python
CONFIG ?= configs/mt-small-deen.yaml
CHECKPOINT ?= artifacts/run/best.pt

.PHONY: install data tokenizer prepare train eval test lint format clean

install:
	$(PYTHON) -m pip install -e ".[data,eval,dev]"

data:
	$(PYTHON) scripts/download_data.py --corpora wmt14,opus100

tokenizer:
	$(PYTHON) scripts/train_tokenizer.py

prepare:
	$(PYTHON) scripts/prepare_data.py

train:
	$(PYTHON) -m mt.train --config $(CONFIG)

eval:
	$(PYTHON) -m mt.evaluate --model $(CHECKPOINT) \
		--test-sets wmt14,wmt19 --output results/mt-small.json

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff format --check src tests scripts
	$(PYTHON) -m ruff check src tests scripts

format:
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

clean:
	rm -rf artifacts/run data/processed
