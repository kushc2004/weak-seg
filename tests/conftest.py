"""Shared fixtures: a tiny synthetic VOC dataset for fast unit tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from weakseg.data.download import create_synthetic_voc


@pytest.fixture(scope="session")
def synthetic_voc(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("weakseg-data")
    return create_synthetic_voc(root, num_images=12, image_size=64, seed=3)


@pytest.fixture(scope="session")
def synthetic_vocab(synthetic_voc: Path) -> list[str]:
    """Foreground class names only (classes.txt includes background at index 0)."""
    lines = [line for line in (synthetic_voc / "classes.txt").read_text().splitlines() if line]
    return lines[1:]
