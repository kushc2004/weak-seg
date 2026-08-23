#!/usr/bin/env python3
"""Publish WeakSeg training artifacts as a Kaggle Dataset (create or new version).

This is the durable cross-session cache: checkpoints, pseudo-masks, metrics,
reports, and the pipeline state land in a private dataset laid out as
``<repo>/outputs/...``, which the notebook's restore cell picks up on the next
run so completed stages are skipped instead of retrained.

Works both locally (your own kaggle.json) and INSIDE a Kaggle notebook, where
the session carries your API identity automatically.

Usage:
    # From a finished run (notebook does this automatically):
    python scripts/publish_kaggle_dataset.py

    # Seed the very first dataset from manually downloaded checkpoints:
    python scripts/publish_kaggle_dataset.py \
        --seed-dir ~/Downloads/kaggle-output/weak-seg/outputs
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "kushchaudhari/weakseg-classifier-cache"
STAGING = ROOT / "artifacts" / "kaggle-dataset"

# Subpaths of outputs/ worth persisting between sessions. Deliberately excludes
# multi-GB raw data (VOC2012 lives under data/, never here).
INCLUDE_DIRS = (
    "checkpoints",       # trained models (~350 MB total)
    "pseudo_masks",      # CAM / CRF pngs - regenerating costs ~40 min GPU
    "metrics",
    "reports",
    "visualizations",
)
INCLUDE_FILES = ("pipeline_state.json", "pipeline.log")


def _run_kaggle(*args: str) -> tuple[int, str]:
    result = subprocess.run(["kaggle", *args], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.returncode, result.stdout.strip()


def _stage(source_dir: Path) -> Path:
    """Copy selected artifacts into the clean staging tree ``weak-seg/outputs/...``."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    target_root = STAGING / "weak-seg" / "outputs"
    copied = []
    for name in INCLUDE_DIRS:
        src = source_dir / name
        if src.is_dir() and any(src.iterdir()):
            shutil.copytree(src, target_root / name)
            copied.append(name)
    for name in INCLUDE_FILES:
        src = source_dir / name
        if src.is_file():
            target_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target_root / name)
            copied.append(name)
    if not any(target_root.rglob("*")):
        raise FileNotFoundError(f"No includable artifacts found under {source_dir}")
    print(f"[publish] staged: {', '.join(sorted(copied))}")

    size_mb = sum(f.stat().st_size for f in STAGING.rglob("*") if f.is_file()) >> 20
    print(f"[publish] staging size: {size_mb} MB")
    return STAGING


def _write_metadata(slug: str) -> None:
    owner, title = slug.split("/")
    metadata = {
        "title": title,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (STAGING / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="target dataset id")
    parser.add_argument("--seed-dir", type=Path, default=None,
                        help="external outputs/ dir to seed from (e.g. manually "
                             "downloaded checkpoints); defaults to <repo>/outputs")
    parser.add_argument("--message", default="WeakSeg artifact sync",
                        help="commit message when creating a new dataset version")
    args = parser.parse_args()

    source = args.seed_dir.resolve() if args.seed_dir else ROOT / "outputs"
    if args.seed_dir and not source.exists():
        raise FileNotFoundError(f"--seed-dir does not exist: {source}")

    staging = _stage(source)
    _write_metadata(args.slug)

    code, out = _run_kaggle("datasets", "version", "-p", str(staging),
                            "-m", args.message, "--dir-mode", "zip")
    if code != 0:
        print(f"[publish] version failed ({out}) - trying first-time create ...")
        code, out = _run_kaggle("datasets", "create", "-p", str(staging),
                                "--dir-mode", "zip")
    if code != 0:
        print(f"[publish] FAILED to publish dataset: {out}\n"
              f"[publish] staging kept at {staging} for a manual retry.")
        return 1
    print(f"[publish] dataset synced: https://www.kaggle.com/datasets/{args.slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
