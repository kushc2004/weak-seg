#!/usr/bin/env python3
"""Package WeakSeg outputs into a Kaggle-friendly artifact bundle (optionally publish)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ("outputs",)
DEFAULT_DATASET = "kushchaudhari/weakseg-artifacts"
STAGING = ROOT / "artifacts" / ".kaggle-staging"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_files() -> list[Path]:
    files: list[Path] = []
    for include in INCLUDE_DIRS:
        base = ROOT / include
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and ".tmp" not in path.suffix:
                files.append(path)
    return sorted(files)


def package() -> tuple[Path, Path]:
    files = _collect_files()
    if not files:
        raise FileNotFoundError("No output artifacts found to package; run the pipeline first.")

    STAGING.mkdir(parents=True, exist_ok=True)
    archive_path = ROOT / "artifacts" / "kaggle" / "weakseg_artifacts.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive_path.name,
        "archive_sha256": _sha256(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "num_files": len(files),
        "files": [str(p.relative_to(ROOT)) for p in files],
    }
    manifest_path = ROOT / "outputs" / "kaggle_artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return archive_path, manifest_path


def publish(archive_path: Path, manifest_path: Path, dataset_slug: str) -> None:
    version_dir = STAGING / "dataset"
    if version_dir.exists():
        shutil.rmtree(version_dir)
    version_dir.mkdir(parents=True)
    shutil.copy2(archive_path, version_dir / archive_path.name)
    shutil.copy2(manifest_path, version_dir / manifest_path.name)
    (version_dir / "dataset-metadata.json").write_text(
        json.dumps({
            "title": "weakseg-artifacts",
            "id": dataset_slug,
            "licenses": [{"name": "MIT"}],
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(["kaggle", "datasets", "version", "-p", str(version_dir),
                    "-m", "weakseg artifact bundle update", "--dir-mode", "zip"],
                   check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="Also push to Kaggle datasets")
    parser.add_argument("--no-upload", dest="upload", action="store_false")
    parser.set_defaults(upload=False)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    archive_path, manifest_path = package()
    print(f"Packaged {archive_path}")
    print(f"Manifest: {manifest_path}")
    if args.upload:
        publish(archive_path, manifest_path, args.dataset)
        print("Uploaded to Kaggle datasets.")


if __name__ == "__main__":
    main()
