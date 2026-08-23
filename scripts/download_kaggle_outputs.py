#!/usr/bin/env python3
"""Fetch a kernel run's output and keep only the useful artifacts.

The Kaggle API cannot partial-download a kernel's output, so this script pulls
the full snapshot into a temp dir and then copies ONLY the whitelisted paths
(everything under ``outputs/`` plus RESULTS.md) into the destination - leaving
behind the multi-GB VOC tarball and repo clones that make raw snapshots so slow.

Usage:
    python scripts/download_kaggle_outputs.py                       # latest version
    python scripts/download_kaggle_outputs.py --version 5           # specific version
    python scripts/download_kaggle_outputs.py -o ~/weakseg-results  # custom dest
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_SLUG = "kushchaudhari/weakseg"
KEEP_PREFIXES = ("outputs/",)
KEEP_FILES = ("RESULTS.md",)


def _download(slug: str, version: str | None, tmp: Path) -> None:
    command = ["kaggle", "kernels", "output", slug, "-p", str(tmp)]
    if version:
        command += ["-v", version]
    print(f"[fetch] downloading {slug} output ... (API downloads everything; we filter after)")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError("kaggle kernels output failed")


def _filter(tmp: Path, dest: Path) -> list[str]:
    kept = []
    for prefix in KEEP_PREFIXES:
        for src in sorted((tmp / prefix).rglob("*")) if (tmp / prefix).is_dir() else []:
            if src.is_file():
                rel = src.relative_to(tmp)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                kept.append(str(rel))
    for name in KEEP_FILES:
        src = tmp / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            kept.append(name)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--version", default=None, help="kernel version number")
    parser.add_argument("-o", "--out", type=Path, default=Path(".kaggle-outputs"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="weakseg-fetch-") as tmp_name:
        _download(args.slug, args.version, Path(tmp_name))
        kept = _filter(Path(tmp_name), args.out)

    if not kept:
        print("[fetch] nothing matched the keep-list - was there any output?")
        return 1
    total_mb = sum((args.out / k).stat().st_size for k in kept) >> 20
    print(f"[fetch] kept {len(kept)} files ({total_mb} MB) -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
