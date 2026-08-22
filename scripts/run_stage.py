#!/usr/bin/env python3
"""Run a single pipeline stage by name (used by the thin per-step CLI wrappers)."""
from __future__ import annotations

from pathlib import Path

from weakseg.pipeline import STAGES, FullPipeline

from run_full_pipeline import ROOT, load_config


def run_single_stage(stage: str, overrides: list[str] | None = None,
                     root: Path | None = None, force: bool = True) -> None:
    assert stage in STAGES, f"unknown stage {stage!r}; choose from {STAGES}"
    root = root or ROOT
    pipeline = FullPipeline(root, load_config(overrides or []), force=force)
    pipeline.run(stage, stage)
