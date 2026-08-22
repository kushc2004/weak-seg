"""Datasets and joint transforms for classification, CAM inference, and segmentation."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from weakseg import IMAGENET_MEAN, IMAGENET_STD
from weakseg.data.labels import read_split_list

NORMALIZE = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
TO_TENSOR = transforms.Compose([transforms.ToTensor(), NORMALIZE])


def load_image(voc_root: Path | str, image_id: str) -> Image.Image:
    path = Path(voc_root) / "JPEGImages" / f"{image_id}.jpg"
    if not path.is_file():
        path = Path(voc_root) / "JPEGImages" / f"{image_id}.png"
    return Image.open(path).convert("RGB")


class RandomResizeLong:
    """Resize so the long side lands in [min_long, max_long] (AffinityNet-style)."""

    def __init__(self, min_long: int, max_long: int):
        self.min_long, self.max_long = min_long, max_long

    def __call__(self, img: Image.Image) -> Image.Image:
        long = max(img.size)
        target = random.randint(self.min_long, self.max_long)
        scale = target / long
        size = (max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale)))
        return img.resize(size, Image.BILINEAR)


class RandomCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if h < self.size or w < self.size:
            pad_h, pad_w = max(0, self.size - h), max(0, self.size - w)
            padded = Image.new("RGB", (max(w, self.size), max(h, self.size)))
            padded.paste(img, (pad_w // 2, pad_h // 2))
            img = padded
            w, h = img.size
        x = random.randint(0, w - self.size)
        y = random.randint(0, h - self.size)
        return img.crop((x, y, x + self.size, y + self.size))


class VOCClassificationDataset(Dataset):
    """Images + multi-label vectors for classifier training (no masks involved)."""

    def __init__(self, voc_root: Path | str, image_ids: list[str], labels: np.ndarray,
                 crop_size: int = 448, resize_range: tuple[int, int] = (448, 768)):
        self.voc_root = Path(voc_root)
        self.image_ids = image_ids
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        self.transform = transforms.Compose([
            RandomResizeLong(*resize_range),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            RandomCrop(crop_size),
            TO_TENSOR,
        ])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        return self.transform(load_image(self.voc_root, image_id)), self.labels[index]


class VOCMultiScaleDataset(Dataset):
    """Full-image multi-scale (+flip) tensors for CAM extraction; one item per image."""

    def __init__(self, voc_root: Path | str, image_ids: list[str],
                 labels: np.ndarray | None = None, scales: tuple[float, ...] = (1.0,),
                 flips: bool = False, max_long_side: int = 960):
        self.voc_root = Path(voc_root)
        self.image_ids = image_ids
        self.labels = None if labels is None else torch.from_numpy(np.asarray(labels, dtype=np.float32))
        self.scales = scales
        self.flips = flips
        self.max_long_side = max_long_side

    def __len__(self) -> int:
        return len(self.image_ids)

    def _scaled_views(self, img: Image.Image) -> list[tuple[torch.Tensor, bool]]:
        views = []
        for scale in self.scales:
            scaled = img.resize(
                (max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale))),
                Image.BILINEAR,
            )
            long = max(scaled.size)
            if long > self.max_long_side:
                factor = self.max_long_side / long
                scaled = scaled.resize(
                    (max(1, round(scaled.size[0] * factor)), max(1, round(scaled.size[1] * factor))),
                    Image.BILINEAR,
                )
            for flip in ((False, True) if self.flips else (False,)):
                views.append((TO_TENSOR(scaled.transpose(Image.FLIP_LEFT_RIGHT) if flip else scaled), flip))
        return views

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        original = load_image(self.voc_root, image_id)
        views = self._scaled_views(original)
        label = self.labels[index] if self.labels is not None else torch.empty(0)
        return image_id, views, label, original.size


def _joint_resize_crop_flip(image: Image.Image, mask: Image.Image,
                            crop_size: int, resize_range: tuple[int, int]):
    long = max(image.size)
    target = random.randint(*resize_range)
    scale = target / long
    new_size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
    image = image.resize(new_size, Image.BILINEAR)
    mask = mask.resize(new_size, Image.NEAREST)

    w, h = image.size
    x = random.randint(0, max(0, w - crop_size))
    y = random.randint(0, max(0, h - crop_size))
    box = (x, y, min(x + crop_size, w), min(y + crop_size, h))
    image, mask = image.crop(box), mask.crop(box)
    if image.size != (crop_size, crop_size):  # pad when the crop ran off the edge
        padded_img = Image.new("RGB", (crop_size, crop_size))
        padded_img.paste(image, (0, 0))
        padded_mask = Image.new("L", (crop_size, crop_size), color=255)
        padded_mask.paste(mask, (0, 0))
        image, mask = padded_img, padded_mask
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return image, mask


class VOCSegDataset(Dataset):
    """Paired image+mask dataset for supervised DeepLab training.

    ``mask_dir`` points either at ground-truth ``SegmentationClass`` (fully
    supervised baseline) or at a generated pseudo-mask directory (weak runs);
    the trainer is identical in both cases.
    """

    def __init__(self, voc_root: Path | str, image_ids: list[str], mask_dir: Path | str,
                 crop_size: int = 512, resize_range: tuple[int, int] = (512, 768)):
        self.voc_root = Path(voc_root)
        self.image_ids = image_ids
        self.mask_dir = Path(mask_dir)
        self.crop_size = crop_size
        self.resize_range = resize_range

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = load_image(self.voc_root, image_id)
        raw_mask = np.asarray(Image.open(self.mask_dir / f"{image_id}.png"), dtype=np.uint8)
        mask = Image.fromarray(raw_mask, mode="L")
        image, mask = _joint_resize_crop_flip(image, mask, self.crop_size, self.resize_range)
        return TO_TENSOR(image), torch.from_numpy(np.asarray(mask, dtype=np.int64))


class VOCSegInferenceDataset(Dataset):
    """Full-resolution images for segmentation evaluation / prediction export."""

    def __init__(self, voc_root: Path | str, image_ids: list[str]):
        self.voc_root = Path(voc_root)
        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        return image_id, TO_TENSOR(load_image(self.voc_root, image_id))


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, tuple[int, int]]:
    """Right/bottom-pad an image tensor to a size divisible by ``multiple``.

    Returns the padded tensor plus the ORIGINAL (h, w) for unpadding predictions.
    """
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
    return tensor, (h, w)
