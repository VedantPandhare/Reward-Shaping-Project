PY ?= python

setup:
	$(PY) -m venv .venv
	.venv/Scripts/python -m pip install -e ".[dev]" || .venv/bin/python -m pip install -e ".[dev]"

smoke:
	$(PY) scripts/run_all.py --smoke --steps 20000 --seeds 0

train-all:
	$(PY) scripts/run_all.py

eval:
	$(PY) scripts/make_gifs.py

plots:
	$(PY) -m rewardlab.analysis

demo:
	$(PY) -m http.server -d web 8000

web-export:
	$(PY) scripts/export_policies.py

web-test:
	cd web && npm test

test:
	$(PY) -m pytest -q

lint:
	ruff check .

.PHONY: setup smoke train-all eval plots demo web-export web-test test lint
