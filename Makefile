.PHONY: help install install-dev test run-fast run-full prepare-data cam segment evaluate visualize push-kaggle watch-kaggle package-artifacts fetch-results clean

help:
	@echo "Available commands:"
	@echo "  make install            Install package and dependencies"
	@echo "  make install-dev        Install with dev extras (pytest, ruff)"
	@echo "  make test               Run unit + smoke test suite"
	@echo "  make run-fast           Full pipeline on synthetic mini-VOC (seconds)"
	@echo "  make run-full           Full pipeline on real VOC2012"
	@echo "  make prepare-data       Download / verify Pascal VOC2012"
	@echo "  make cam                Generate CAM pseudo masks (naive + SEAM [+CRF])"
	@echo "  make evaluate           Evaluate all models on VOC val"
	@echo "  make package-artifacts  Package outputs for Kaggle (no upload)"
	@echo "  make push-kaggle        Push kernel to Kaggle and trigger execution"
	@echo "  make watch-kaggle       Follow live Kaggle kernel output stream"
	@echo "  make fetch-results      Pull latest Kaggle results and restore locally"
	@echo "  make clean              Remove caches and temp build files"

install:
	pip install -e .

install-dev:
	pip install -e '.[dev]'

test:
	pytest tests/ -v

run-fast:
	python scripts/run_full_pipeline.py --force fast_dev_run=true

run-full:
	python scripts/run_full_pipeline.py --force device=auto

prepare-data:
	python scripts/prepare_data.py

cam:
	python scripts/generate_cam.py

evaluate:
	python scripts/evaluate.py

package-artifacts:
	python scripts/publish_kaggle_artifacts.py --no-upload

push-kaggle:
	kaggle kernels push -p .

watch-kaggle:
	python scripts/watch_kaggle_kernel.py

fetch-results:
	python scripts/fetch_kaggle_results.py

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
