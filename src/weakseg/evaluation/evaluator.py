"""Full-image segmentation inference and VOC evaluation loop."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from weakseg.data.datasets import VOCSegInferenceDataset, pad_to_multiple
from weakseg.evaluation.metrics import ConfusionMatrix
from weakseg.utils.sampling import worker_init_fn


def _padded_batch_collate(batch):
    """Stack variable-size images by zero-padding to the batch max (/32 aligned).

    Predictions are sliced back to each original size, so results are identical
    to batch-size-1 evaluation - just several times faster.
    """
    ids = [item[0] for item in batch]
    tensors = [item[1] for item in batch]
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    max_h += (-max_h % 32)
    max_w += (-max_w % 32)
    stacked = torch.zeros(len(tensors), 3, max_h, max_w)
    sizes = torch.empty(len(tensors), 2, dtype=torch.int64)
    for index, tensor in enumerate(tensors):
        stacked[index, :, :tensor.shape[1], :tensor.shape[2]] = tensor
        sizes[index] = torch.tensor(tensor.shape[1:])
    return ids, stacked, sizes


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
                          num_workers: int = 2, batch_size: int = 1) -> dict:
    """Evaluate a segmentation model against GT masks; returns ConfusionMatrix.summary()."""
    from PIL import Image

    ids = image_ids if max_images is None else image_ids[:max_images]
    dataset = VOCSegInferenceDataset(voc_root, ids)
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False,
                        num_workers=num_workers, collate_fn=_padded_batch_collate,
                        worker_init_fn=worker_init_fn)
    matrix = ConfusionMatrix(num_classes)
    gt_dir = Path(gt_dir)

    seen = 0
    for image_id_list, images, sizes in loader:
        model.eval()
        logits = model(images.to(device))["out"]
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        for b, image_id in enumerate(image_id_list):
            h, w = int(sizes[b][0]), int(sizes[b][1])
            pred = preds[b][:h, :w].astype(np.uint8)
            gt = np.asarray(Image.open(gt_dir / f"{image_id}.png"), dtype=np.uint8)
            matrix.update(pred, gt)
        seen += len(image_id_list)
        if seen % (batch_size * 50) < batch_size:
            print(f"  eval {seen}/{len(dataset)}", flush=True)
    return matrix.summary()
