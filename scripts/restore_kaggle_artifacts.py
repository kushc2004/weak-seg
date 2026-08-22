#!/usr/bin/env python3
"""Restore packaged WeakSeg artifacts from a tar.gz archive or uncompressed folder."""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PREFIXES = ("outputs",)


def restore_from_archive(archive_path: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            norm_name = str(Path(member.name))
            if not any(norm_name.startswith(p) for p in ALLOWED_PREFIXES):
                raise ValueError(f"Refusing to restore unsafe path from archive: {member.name}")
        print(f"Extracting {len(members)} items from {archive_path} to {ROOT}...")
        archive.extractall(ROOT)


def restore_from_directory(directory: Path) -> None:
    count = 0
    for prefix in ALLOWED_PREFIXES:
        src = directory / prefix
        dst = ROOT / prefix
        if src.is_dir():
            print(f"Copying {src} -> {dst}...")
            shutil.copytree(src, dst, dirs_exist_ok=True)
            count += 1
    if count == 0:
        raise ValueError(f"No allowed artifact directories found in {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to weakseg_artifacts.tar.gz or directory")
    args = parser.parse_args()

    src = args.source.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if src.is_file() and (src.name.endswith(".tar.gz") or src.name.endswith(".tgz")):
        restore_from_archive(src)
    elif src.is_dir():
        restore_from_directory(src)
    else:
        raise ValueError(f"Unsupported source format: {src}")
    print("Artifact restoration complete.")


if __name__ == "__main__":
    main()
