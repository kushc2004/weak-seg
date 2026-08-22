"""RESULTS.md generation and qualitative comparison grids."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from weakseg import VOC_CLASS_NAMES
from weakseg.data.datasets import load_image
from weakseg.utils.voc_palette import colorize_mask

METHOD_LABELS = {
    "fully_sup": "DeepLab fully supervised",
    "cam": "CAM baseline",
    "cam_crf": "CAM + DenseCRF",
    "seam": "SEAM (ours pipeline)",
}
SUPERVISION = {
    "fully_sup": "Pixel labels",
    "cam": "Image labels",
    "cam_crf": "Image labels",
    "seam": "Image labels",
}


def build_comparison_rows(val_metrics: dict) -> list[dict]:
    """One row per method with the annotation-efficiency gap vs the supervised bound."""
    reference = val_metrics.get("fully_sup", {}).get("miou")
    rows = []
    for method in ("fully_sup", "cam", "cam_crf", "seam"):
        metrics = val_metrics.get(method)
        if not metrics:
            continue
        gap = (round(reference - metrics["miou"], 2)
               if isinstance(reference, (int, float)) else None)
        retained = (round(100 * metrics["miou"] / reference, 1)
                    if isinstance(reference, (int, float)) and reference else None)
        rows.append({
            "method": method,
            "label": METHOD_LABELS[method],
            "supervision": SUPERVISION[method],
            "miou": round(metrics["miou"], 2),
            "dice": round(metrics["dice"], 2),
            "pixel_accuracy": round(metrics["pixel_accuracy"], 2),
            "gap_vs_fully_sup": gap,
            "retained_pct": retained,
        })
    return rows


def _fmt(value) -> str:
    return "-" if value is None else f"{value}"


def write_results_markdown(path: Path | str, val_metrics: dict,
                           pseudo_quality: dict | None, state_summary: dict,
                           config_echo: dict,
                           class_names_fg: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fg_names = class_names_fg or VOC_CLASS_NAMES[1:]
    rows = build_comparison_rows(val_metrics)
    lines = [
        "# WeakSeg — Results",
        "",
        "Annotation-efficient semantic segmentation on PASCAL VOC2012.",
        "Weak methods see **image-level labels only** (`ImageSets/Main`); pixel masks are used exclusively for evaluation.",
        "",
        "## Main comparison (val split)",
        "",
        "| Method | Supervision | mIoU (%) | Dice (%) | Pixel Acc (%) | Δ mIoU vs fully-sup | Retained |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['supervision']} | {row['miou']:.2f} | "
            f"{row['dice']:.2f} | {row['pixel_accuracy']:.2f} | "
            f"{_fmt(row['gap_vs_fully_sup'])} | {_fmt(row['retained_pct'])}% |"
        )

    if pseudo_quality:
        lines += [
            "",
            "## Pseudo-label quality (train split, diagnostic only)",
            "",
            "| Source | mIoU vs GT | Dice vs GT |",
            "| --- | ---: | ---: |",
        ]
        for name in sorted(pseudo_quality):
            entry = pseudo_quality[name]
            lines.append(f"| {name} | {entry['miou']:.2f} | {entry['dice']:.2f} |")
        lines += ["", "_Ground-truth masks are read here only to *measure* pseudo-label quality; they never enter weak training._"]

    lines += [
        "",
        "## Per-class IoU (val split)",
        "",
        "| Class | " + " | ".join(METHOD_LABELS[r["method"]] for r in rows) + " |",
        "| --- | " + " | ".join("---:" for _ in rows) + " |",
    ]
    for cls_idx, name in enumerate(fg_names, start=1):
        values = []
        for row in rows:
            iou_list = val_metrics[row["method"]].get("per_class_iou", [])
            values.append(f"{iou_list[cls_idx]:.2f}" if cls_idx < len(iou_list) else "-")
        lines.append(f"| {name} | " + " | ".join(values) + " |")

    lines += ["", "## Stage runtimes", "", "| Stage | Status | Seconds |", "| --- | --- | ---: |"]
    for stage, info in state_summary.items():
        if info.get("status") == "running":
            continue  # the report stage itself is still open while this renders
        duration = info.get("duration_seconds")
        lines.append(f"| {stage} | {info.get('status')} | {duration if duration is not None else '-'} |")

    lines += ["", "## Configuration echo", "", "```yaml"]
    for key, value in config_echo.items():
        lines.append(f"{key}: {value}")
    lines += ["```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _label_strip(width: int, headers: list[str]) -> Image.Image:
    strip = Image.new("RGB", (width, 22), color=(250, 250, 250))
    draw = ImageDraw.Draw(strip)
    cell = width // max(1, len(headers))
    for index, text in enumerate(headers):
        draw.text((index * cell + 4, 5), text, fill=(20, 20, 20))
    return strip


def _blend(image: Image.Image, mask: np.ndarray, alpha: float = 0.55) -> Image.Image:
    return Image.blend(image.copy(), colorize_mask(mask), alpha=alpha)


def write_qualitative_grid(out_path: Path | str, voc_root: Path | str, image_id: str,
                           gt_mask: np.ndarray | None,
                           pseudo_dirs: dict[str, Path],
                           prediction_masks: dict[str, np.ndarray],
                           crf_available: bool = True) -> Path:
    """Compose one row: input | GT | pseudo masks | final segmentation predictions."""
    base_image = load_image(voc_root, image_id)
    panels: list[tuple[str, Image.Image]] = [("input", base_image)]
    if gt_mask is not None:
        panels.append(("ground truth", _blend(base_image, gt_mask)))

    pseudo_specs = [
        ("CAM mask", "cam_naive"),
        ("CAM+CRF", "cam_naive_crf"),
        ("SEAM mask", "cam_seam"),
    ]
    for header, key in pseudo_specs:
        directory = pseudo_dirs.get(key)
        candidate = Path(directory) / f"{image_id}.png" if directory else None
        if candidate and candidate.is_file():
            mask = np.asarray(Image.open(candidate), dtype=np.uint8)
            panels.append((header, _blend(base_image, mask)))
        elif header == "CAM+CRF" and not crf_available:
            panels.append(("CAM+CRF (n/a)", Image.new("RGB", base_image.size, (235, 235, 235))))

    for header, key in (("pred full-sup", "fully_sup"), ("pred CAM", "cam"), ("pred SEAM", "seam")):
        if key in prediction_masks:
            panels.append((header, _blend(base_image, prediction_masks[key])))

    height = min(panel.size[1] for _, panel in panels)
    resized = [panel.resize((int(panel.size[0] * height / panel.size[1]), height)) for _, panel in panels]
    width = sum(panel.size[0] for panel in resized)
    grid = Image.new("RGB", (width, height + 22), color=(255, 255, 255))
    x_offset = 0
    for panel in resized:
        grid.paste(panel, (x_offset, 22))
        x_offset += panel.size[0]
    grid.paste(_label_strip(width, [header for header, _ in panels]), (0, 0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=90)
    return out_path
