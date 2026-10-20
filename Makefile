PY ?= python

setup:
	$(PY) -m venv .venv
	.venv/Scripts/python -m pip install -e ".[dev,demo]" || .venv/bin/python -m pip install -e ".[dev,demo]"

smoke:
	$(PY) scripts/run_all.py --smoke --steps 20000 --seeds 0

train-all:
	$(PY) scripts/run_all.py

eval:
	$(PY) scripts/make_gifs.py

plots:
	$(PY) -m rewardlab.analysis

demo:
	$(PY) app/app.py

test:
	$(PY) -m pytest -q

lint:
	ruff check .

.PHONY: setup smoke train-all eval plots demo test lint
