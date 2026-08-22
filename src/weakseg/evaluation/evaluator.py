"""Full-image segmentation inference and VOC evaluation loop."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from weakseg.data.datasets import VOCSegInferenceDataset, pad_to_multiple
from weakseg.evaluation.metrics import ConfusionMatrix
from weakseg.utils.sampling import worker_init_fn


@torch.no_grad()
def predict_mask(model, image_tensor: torch.Tensor, device) -> np.ndarray:
    """Full-image prediction with /32 padding; returns HxW class-id array."""
    model.eval()
    padded, (h, w) = pad_to_multiple(image_tensor.unsqueeze(0), multiple=32)
    logits = model(padded.to(device))["out"]
    pred = torch.argmax(logits, dim=1)[0, :h, :w]
    return pred.cpu().numpy().astype(np.uint8)


@torch.no_grad()
def evaluate_segmentation(model, voc_root: Path | str, image_ids: list[str],
                          gt_dir: Path | str, device, num_classes: int,
                          max_images: int | None = None,
                          num_workers: int = 2) -> dict:
    """Evaluate a segmentation model against GT masks; returns ConfusionMatrix.summary()."""
    from PIL import Image

    ids = image_ids if max_images is None else image_ids[:max_images]
    dataset = VOCSegInferenceDataset(voc_root, ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                        worker_init_fn=worker_init_fn)
    matrix = ConfusionMatrix(num_classes)
    gt_dir = Path(gt_dir)

    for index, (image_id, image_tensor) in enumerate(loader):
        pred = predict_mask(model, image_tensor[0], device)
        gt = np.asarray(Image.open(gt_dir / f"{image_id[0]}.png"), dtype=np.uint8)
        matrix.update(pred, gt)
        if index % 200 == 0:
            print(f"  eval {index + 1}/{len(dataset)}", flush=True)
    return matrix.summary()
