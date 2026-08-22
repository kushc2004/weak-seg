"""End-to-end fast_dev_run of every pipeline stage on the synthetic mini-VOC (CPU).

This is THE integration test: it exercises data generation, both classifier
trainers, CAM extraction, pseudo-mask export, all segmentation runs, evaluation,
visualization grids, and RESULTS.md generation - in seconds, without network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from weakseg.pipeline import STAGES, FullPipeline


SMOKE_OVERRIDES = {
    "device": "cpu",
    "seed": 0,
    "fast_dev_run": True,
    "pretrained": False,
    "cls_crop_size": 48,
    "cls_resize_long": [48, 64],
    "cls_batch_size": 4,
    "cls_num_workers": 0,
    "seg_crop_size": 48,
    "seg_resize_long": [48, 64],
    "seg_batch_size": 2,
    "seg_num_workers": 0,
    "cam_seam_scales": [1.0],
    "cam_seam_flips": False,
    "cam_max_long_side": 128,
    "eval_max_images": 4,
    "eval_num_workers": 0,
    "viz_n_examples": 2,
}


@pytest.fixture(scope="module")
def smoke_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("weakseg-smoke")
    pipeline = FullPipeline(root, dict(SMOKE_OVERRIDES), force=True)
    summary = pipeline.run()
    return root, summary


def test_all_stages_complete(smoke_results):
    from weakseg.weak.crf import CRF_AVAILABLE

    _, summary = smoke_results
    assert set(summary) == set(STAGES)
    expected = [s for s in STAGES if s != "train_seg_cam_crf" or CRF_AVAILABLE]
    for stage in expected:
        assert summary[stage]["status"] == "complete", stage


def test_checkpoints_written(smoke_results):
    from weakseg.weak.crf import CRF_AVAILABLE

    root, _ = smoke_results
    checkpoints = root / "outputs/checkpoints"
    names = ["classifier_plain", "classifier_seam", "seg_fully_sup", "seg_cam", "seg_seam"]
    if CRF_AVAILABLE:
        names.append("seg_cam_crf")
    for name in names:
        assert (checkpoints / f"{name}.pth").is_file(), name


def test_pseudo_masks_and_quality(smoke_results):
    root, _ = smoke_results
    masks = root / "outputs/pseudo_masks"
    naive_masks = list((masks / "cam_naive").glob("*.png"))
    seam_masks = list((masks / "cam_seam").glob("*.png"))
    assert naive_masks and seam_masks

    quality = json.loads((root / "outputs/metrics/pseudo_quality.json").read_text())
    assert "cam_naive" in quality and "miou" in quality["cam_naive"]


def test_evaluation_metrics_and_report(smoke_results):
    root, summary = smoke_results
    metrics = json.loads((root / "outputs/metrics/val_metrics.json").read_text())
    for method in ("fully_sup", "cam", "seam"):
        assert method in metrics, method
        assert 0.0 <= metrics[method]["miou"] <= 1.0

    report = root / "outputs/reports/experiment_summary.md"
    assert report.is_file() and "WeakSeg — Results" in report.read_text(encoding="utf-8")
    assert "| Method | Supervision | mIoU" in report.read_text(encoding="utf-8")
    assert summary["evaluate"]["metrics"], "evaluation stage must record mIoU metrics"


def test_visualization_grids(smoke_results):
    root, _ = smoke_results
    grids = list((root / "outputs/visualizations").glob("*.jpg"))
    assert len(grids) >= 1
