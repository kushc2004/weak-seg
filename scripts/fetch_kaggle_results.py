#!/usr/bin/env python3
"""Fetch the latest WeakSeg kernel output from Kaggle and restore it locally."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "kushchaudhari/weakseg"
OUTPUT_DIR = ROOT / ".kaggle-outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", args.slug, "-p", str(OUTPUT_DIR)], check=True)

    archives = sorted(OUTPUT_DIR.glob("weakseg_artifacts.tar.gz"))
    if archives:
        subprocess.run(
            ["python", str(ROOT / "scripts/restore_kaggle_artifacts.py"), str(archives[0])],
            check=True,
        )
    else:
        print("No weakseg_artifacts.tar.gz found in kernel output; raw files kept in .kaggle-outputs/")
    summary = OUTPUT_DIR / "RESULTS.md"
    local_results = ROOT / "RESULTS.md"
    if summary.is_file():
        shutil.copy2(summary, local_results)
        print(f"RESULTS.md refreshed at {local_results}")


if __name__ == "__main__":
    main()
