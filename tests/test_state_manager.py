"""PipelineStateManager resume semantics."""
from __future__ import annotations

import json

import pytest

from weakseg.utils.checkpoint import PipelineStateManager


@pytest.fixture()
def state_path(tmp_path):
    return tmp_path / "outputs" / "pipeline_state.json"


def test_start_complete_roundtrip(state_path, tmp_path):
    manager = PipelineStateManager(state_path)
    manager.start_stage("train_seg_cam")
    assert not manager.is_stage_complete("train_seg_cam")

    artifact = tmp_path / "model.pth"
    artifact.write_bytes(b"x")
    manager.complete_stage("train_seg_cam", artifacts=[artifact], metrics={"miou": 55.0},
                           duration=1.5)

    fresh = PipelineStateManager(state_path)  # simulates process restart
    assert fresh.is_stage_complete("train_seg_cam")
    stage = fresh.state["stages"]["train_seg_cam"]
    assert stage["duration_seconds"] == 1.5
    assert stage["metrics"] == {"miou": 55.0}


def test_missing_artifact_invalidates_stage(state_path, tmp_path):
    manager = PipelineStateManager(state_path)
    manager.start_stage("evaluate")
    ghost = tmp_path / "ghost.pth"
    manager.complete_stage("evaluate", artifacts=[ghost])
    assert not manager.is_stage_complete("evaluate")  # recorded artifact does not exist


def test_fail_stage_records_error(state_path):
    manager = PipelineStateManager(state_path)
    manager.start_stage("generate_report")
    manager.fail_stage("generate_report", "boom")
    assert json.loads(state_path.read_text())["stages"]["generate_report"]["error"] == "boom"
