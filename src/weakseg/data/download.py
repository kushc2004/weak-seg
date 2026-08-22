"""PASCAL VOC 2012 acquisition plus a synthetic mini-VOC for smoke runs.

The official tarball (~1.9 GB) is fetched from a fast mirror with the official
host as fallback. Ground-truth ``SegmentationClass`` masks are used ONLY for
evaluation and reporting - never during weak-supervision training.
"""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from weakseg.utils.logging import get_logger

VOC_TAR_NAME = "VOCtrainval_11-May-2012.tar"
VOC_URLS = (
    "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
)

EXPECTED_COUNTS = {"JPEGImages": 17_125, "SegmentationClass": 2_913}


def _download(url: str, dest: Path, logger) -> None:
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        done = 0
        with dest.open("wb") as sink:
            for chunk in response.iter_content(chunk_size=1 << 20):
                sink.write(chunk)
                done += len(chunk)
                if total and (done >> 20) % 100 == 0:
                    logger.info("  %d / %d MB", done >> 20, total >> 20)


def _verify(voc_root: Path) -> None:
    jpeg = voc_root / "JPEGImages"
    seg = voc_root / "SegmentationClass"
    splits = voc_root / "ImageSets" / "Segmentation"
    missing = [
        str(p)
        for p in (jpeg, seg, splits / "train.txt", splits / "val.txt")
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(f"VOC2012 layout incomplete, missing: {missing}")
    n_jpeg = len(list(jpeg.glob("*.jpg")))
    n_seg = len(list(seg.glob("*.png")))
    for name, expected in EXPECTED_COUNTS.items():
        actual = n_jpeg if name == "JPEGImages" else n_seg
        if actual < expected:
            raise RuntimeError(f"{name}: found {actual} files, expected >= {expected}")


def prepare_voc2012(data_dir: Path | str, logger=None) -> Path:
    """Download + extract VOC2012 under ``data_dir`` and return the VOC2012 root."""
    logger = logger or get_logger("weakseg.data")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    voc_root = data_dir / "VOC2012"

    try:
        _verify(voc_root)
        logger.info("VOC2012 already present at %s", voc_root)
        return voc_root
    except (FileNotFoundError, RuntimeError):
        pass

    tar_path = data_dir / VOC_TAR_NAME
    if not tar_path.is_file():
        last_error: Exception | None = None
        for url in VOC_URLS:
            try:
                _download(url, tar_path, logger)
                break
            except Exception as error:  # noqa: BLE001 - fall through to next mirror
                last_error = error
                logger.warning("Mirror failed (%s): %s", url, error)
                tar_path.unlink(missing_ok=True)
        if not tar_path.is_file():
            raise RuntimeError(f"All VOC2012 mirrors failed; last error: {last_error}")

    logger.info("Extracting %s ...", tar_path)
    with tarfile.open(tar_path) as archive:
        archive.extractall(data_dir)
    extracted = data_dir / "VOCdevkit" / "VOC2012"
    if extracted.is_dir() and not voc_root.is_dir():
        extracted.rename(voc_root)
    shutil.rmtree(data_dir / "VOCdevkit", ignore_errors=True)
    _verify(voc_root)
    logger.info("VOC2012 ready at %s", voc_root)
    return voc_root


# ---------------------------------------------------------------------------
# Synthetic mini-VOC (fast_dev_run smoke testing without any download)
# ---------------------------------------------------------------------------

SYN_COLORS = [
    (180, 130, 70), (90, 180, 90), (200, 80, 80), (80, 120, 200),
    (220, 200, 60), (140, 60, 160), (60, 190, 190),
]


def create_synthetic_voc(root: Path | str, num_images: int = 24, image_size: int = 128,
                         val_fraction: float = 0.25, seed: int = 0) -> Path:
    """Generate a tiny VOC-shaped dataset of colored shapes with clean masks.

    Layout matches real VOC2012 exactly (JPEGImages, SegmentationClass,
    ImageSets/Segmentation/{train,val}.txt, ImageSets/Main/<cls>_<split>.txt)
    so every downstream code path is exercised.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    voc = root / "synthetic_voc"
    for sub in ("JPEGImages", "SegmentationClass",
                "ImageSets/Segmentation", "ImageSets/Main"):
        (voc / sub).mkdir(parents=True, exist_ok=True)

    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle"]
    ids = []
    for i in range(num_images):
        image_id = f"syn_{i:04d}"
        ids.append(image_id)
        size = image_size + int(rng.integers(-16, 17))
        img = np.full((size, size, 3), rng.integers(30, 90), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        for cls_idx in range(len(class_names)):
            if rng.random() < 0.55:
                x0, y0 = rng.integers(4, size - 40, size=2)
                w = h = int(rng.integers(24, max(25, size // 2)))
                color = SYN_COLORS[cls_idx % len(SYN_COLORS)]
                img[y0:y0 + h, x0:x0 + w] = color
                mask[y0:y0 + h, x0:x0 + w] = cls_idx + 1
        Image.fromarray(img).save(voc / "JPEGImages" / f"{image_id}.jpg", quality=90)

        from weakseg.utils.voc_palette import save_class_mask

        save_class_mask(mask, voc / "SegmentationClass" / f"{image_id}.png")

    rng.shuffle(ids)
    n_val = max(2, int(len(ids) * val_fraction))
    val_ids, train_ids = sorted(ids[:n_val]), sorted(ids[n_val:])
    (voc / "ImageSets/Segmentation/train.txt").write_text("\n".join(train_ids) + "\n")
    (voc / "ImageSets/Segmentation/val.txt").write_text("\n".join(val_ids) + "\n")

    # Image-level labels in the official Main format: "<id>  -1|0|1".
    labels = {image_id: set() for image_id in ids}
    for image_id in ids:
        png = np.asarray(Image.open(voc / "SegmentationClass" / f"{image_id}.png"))
        for c in np.unique(png):
            if c > 0:
                labels[image_id].add(int(c))
    for split, split_ids in (("train", train_ids), ("val", val_ids)):
        for cls_idx, name in enumerate(class_names, start=1):
            lines = []
            for image_id in split_ids:
                flag = 1 if cls_idx in labels[image_id] else -1
                lines.append(f"{image_id}  {flag}")
            (voc / "ImageSets/Main" / f"{name}_{split}.txt").write_text("\n".join(lines) + "\n")

    # The synthetic set defines its own 5-class vocabulary.
    (voc / "classes.txt").write_text("\n".join(["background"] + class_names) + "\n")
    return voc
