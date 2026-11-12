# Common tasks. Every target is a thin wrapper around a documented command, so nothing
# here is required: see README.md for the same commands without make.
#
# EVAL_DATA is a local checkout of the private eval-data repository used by CI.

PY ?= python
EVAL_DATA ?= ../localrag-eval-data
CONFIG ?= configs/baseline.yaml

.PHONY: help install check test corpus eval matrix smoke gate baseline record-full export-data

help:
	@echo "install      install dependencies and the rag-eval command"
	@echo "check        lint, format check, type check, tests (what CI runs)"
	@echo "corpus       download + verify + prepare the FastAPI docs corpus"
	@echo "eval         full evaluation of CONFIG on the verified golden set"
	@echo "matrix       the 12-configuration matrix (updates the README table)"
	@echo "smoke        build the 40-item stratified smoke subset"
	@echo "gate         run the PR gate locally and record its model calls for CI"
	@echo "baseline     record the gate baseline (run on main after merging)"
	@echo "record-full  record the full evaluation's model calls for the nightly job"
	@echo "export-data  copy dataset, smoke subset and manifest into EVAL_DATA"

install:
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install --no-deps -e .

check:
	ruff check . && ruff format --check . && mypy && actionlint && pytest

corpus:
	$(PY) -m rag.corpus fetch
	$(PY) -m rag.corpus prepare

eval:
	$(PY) -m rag.cli run --config $(CONFIG) --dataset data/golden/v1.jsonl

matrix:
	$(PY) -m rag.cli matrix --base $(CONFIG) --dataset data/golden/v1.jsonl

smoke:
	$(PY) -m rag.golden smoke --n 40 --seed 11

gate:
	$(PY) -m rag.cli gate --config $(CONFIG) --dataset data/golden/smoke.jsonl \
		--baseline results/baseline.json --record-to $(EVAL_DATA)/cache/smoke.sqlite

baseline:
	$(PY) -m rag.cli gate --config $(CONFIG) --dataset data/golden/smoke.jsonl \
		--write-baseline --record-to $(EVAL_DATA)/cache/smoke.sqlite

record-full:
	$(PY) -m rag.cli run --config $(CONFIG) --dataset data/golden/v1.jsonl \
		--record-to $(EVAL_DATA)/cache/full.sqlite

export-data:
	$(PY) -c "import shutil, pathlib; d = pathlib.Path('$(EVAL_DATA)'); \
	[ (d / p).parent.mkdir(parents=True, exist_ok=True) or shutil.copy2(s, d / p) for s, p in [ \
	('data/golden/v1.jsonl', 'golden/v1.jsonl'), ('data/golden/smoke.jsonl', 'golden/smoke.jsonl'), \
	('data/manifests/fastapi.json', 'manifests/fastapi.json')] ]; print('copied to', d)"
