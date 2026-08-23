"""WeakSeg end-to-end pipeline with resumable stage checkpoints.

Stages (in order):
    data_prep -> train_classifier_plain -> train_classifier_seam ->
    generate_pseudo_masks -> train_seg_fully_sup -> train_seg_cam ->
    train_seg_cam_crf -> train_seg_seam -> evaluate -> visualize -> generate_report

Protocol guarantee: ground-truth ``SegmentationClass`` masks are opened ONLY by
``data_prep`` verification, pseudo-label quality *diagnostics*, evaluation, and
visualization. Weak-supervision training reads images plus ``ImageSets/Main``
classification labels exclusively.
"""
from __future__ import annotations

import json
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from weakseg import VOC_NUM_CLASSES, VOC_NUM_FG_CLASSES
from weakseg.data.datasets import (
    VOCClassificationDataset,
    VOCSegDataset,
    load_image,
)
from weakseg.data.download import create_synthetic_voc, prepare_voc2012
from weakseg.data.labels import filter_labelled, load_cls_labels, read_split_list
from weakseg.evaluation.evaluator import evaluate_segmentation, predict_mask
from weakseg.evaluation.metrics import ConfusionMatrix
from weakseg.models.cam_classifier import CamClassifier
from weakseg.models.deeplab import build_deeplab
from weakseg.models.seam import SeamNet
from weakseg.reporting.export import build_comparison_rows, write_qualitative_grid, write_results_markdown
from weakseg.utils.checkpoint import PipelineStateManager
from weakseg.utils.device import get_device, get_device_name
from weakseg.utils.logging import get_logger
from weakseg.utils.sampling import worker_init_fn
from weakseg.utils.seed import seed_everything
from weakseg.weak.cam import generate_cam_scores
from weakseg.weak.crf import CRF_AVAILABLE
from weakseg.weak.losses import (
    adaptive_min_pooling_loss,
    cross_refined_consistency,
    equivariant_loss,
    max_norm,
    max_onehot,
)
from weakseg.weak.pseudo import cams_to_argmax_mask, crf_argmax_mask, save_pseudo_mask

STAGES = [
    "data_prep",
    "train_classifier_plain",
    "train_classifier_seam",
    "generate_pseudo_masks",
    "train_seg_fully_sup",
    "train_seg_cam",
    "train_seg_cam_crf",
    "train_seg_seam",
    "evaluate",
    "visualize",
    "generate_report",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "device": "auto",
    "seed": 42,
    "fast_dev_run": False,
    "amp": True,
    "pretrained": True,
    # data
    "data_root": "data",
    "train_list": "train",           # 'train' | 'val' | 'train_aug' (if masks available)
    "label_source_split": "auto",    # Main split derived from train_list unless overridden
    # classifier training
    "cls_crop_size": 448,
    "cls_resize_long": [448, 768],
    "cls_batch_size": 8,
    "cls_lr": 0.01,
    "cls_momentum": 0.9,
    "cls_weight_decay": 5e-4,
    "cls_epochs_plain": 15,
    "cls_epochs_seam": 30,
    "cls_head_lr_mult": 10.0,
    "cls_num_workers": 4,
    "cls_scale_factor": 0.3,         # SEAM equivariant downscale factor
    "max_steps_per_epoch": None,
    # CAM extraction / pseudo masks
    "cam_naive_scales": [1.0],
    "cam_naive_flips": False,
    "cam_seam_scales": [0.5, 1.0, 1.5, 2.0],
    "cam_seam_flips": True,
    "cam_bg_alpha": 0.26,
    "cam_use_crf": True,
    "cam_crf_alpha": 4.0,
    "cam_max_long_side": 960,
    "cam_include_val_for_viz": True,
    # segmentation training
    "seg_init": "coco",              # 'coco' | 'random'
    "seg_crop_size": 512,
    "seg_resize_long": [512, 768],
    "seg_batch_size": 8,
    "seg_lr": 0.01,
    "seg_momentum": 0.9,
    "seg_weight_decay": 1e-4,
    "seg_epochs": 30,
    "seg_num_workers": 4,
    # evaluation / reporting
    "eval_max_images": None,
    "eval_num_workers": 2,
    "eval_batch_size": 8,
    "crf_workers": 4,                # CPU processes refining CRF while the GPU proceeds
    "viz_n_examples": 10,
}


def _crf_worker(payload: tuple[str, dict[int, np.ndarray], float, str]) -> str:
    """Refine one image's CAM scores with DenseCRF and save the mask (runs in a CPU process)."""
    image_path, cam_dict, alpha, out_path = payload
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    mask = crf_argmax_mask(rgb, cam_dict, alpha=alpha)
    save_pseudo_mask(mask, Path(out_path).parent, Path(out_path).stem)
    return out_path


class FullPipeline:
    """Coordinates execution of all WeakSeg experiment stages."""

    def __init__(self, root_dir: Path | str, config: dict[str, Any], force: bool = False):
        self.root = Path(root_dir)
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(config or {})
        self.force = force
        self.logger = get_logger("weakseg.pipeline", self.root / "outputs" / "pipeline.log")

        self.state_mgr = PipelineStateManager(self.root / "outputs/pipeline_state.json")
        self.device = get_device(str(self.config["device"]))
        seed_everything(int(self.config["seed"]))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.fast_dev_run = bool(self.config["fast_dev_run"])
        self.use_amp = bool(self.config["amp"]) and self.device.type == "cuda"
        self.num_classes = VOC_NUM_CLASSES          # resolved for synthetic data in data_prep
        self.num_fg_classes = VOC_NUM_FG_CLASSES
        self.class_names_fg = None                  # resolved for synthetic data in data_prep

        self.data_dir = self.root / str(self.config["data_root"])
        self.outputs_dir = self.root / "outputs"
        self.checkpoints_dir = self.outputs_dir / "checkpoints"
        self.masks_dir = self.outputs_dir / "pseudo_masks"
        self.metrics_dir = self.outputs_dir / "metrics"
        self.viz_dir = self.outputs_dir / "visualizations"
        self.reports_dir = self.outputs_dir / "reports"
        for directory in (self.checkpoints_dir, self.masks_dir, self.metrics_dir,
                          self.viz_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("Device: %s | amp=%s | fast_dev_run=%s",
                         get_device_name(self.device), self.use_amp, self.fast_dev_run)

    # ------------------------------------------------------------------ utils

    @property
    def voc_root(self) -> Path:
        if self.fast_dev_run:
            return self.data_dir / "synthetic_voc"
        return self.data_dir / "VOC2012"

    def _autocast(self):
        return torch.autocast("cuda") if self.use_amp else nullcontext()

    def _scaler(self):
        return torch.amp.GradScaler("cuda") if self.use_amp else None

    def _poly_optimizer(self, param_groups: list[dict], max_steps: int):
        optimizer = torch.optim.SGD(param_groups, momentum=float(self.config["cls_momentum"]))
        return optimizer, PolyLr(optimizer, power=0.9, max_steps=max_steps)

    def _steps_this_epoch(self, loader: DataLoader) -> int:
        if self.config["max_steps_per_epoch"]:
            return min(len(loader), int(self.config["max_steps_per_epoch"]))
        if self.fast_dev_run:
            return min(len(loader), 2)
        return len(loader)

    def _resolve_splits(self) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
        """Train ids/labels from the configured Main split; optional val ids for viz."""
        label_split = str(self.config["label_source_split"])
        if label_split == "auto":
            label_split = {"train": "train", "val": "val", "train_aug": "train"}.get(
                str(self.config["train_list"]), "train")
        train_ids, train_labels = load_cls_labels(
            self.voc_root, label_split, self.num_fg_classes, self.class_names_fg)
        train_ids, train_labels = filter_labelled(train_ids, train_labels)

        val_ids, val_labels = [], np.zeros((0, self.num_fg_classes))
        if self.config["cam_include_val_for_viz"] and not self.fast_dev_run:
            seg_val = self.voc_root / "ImageSets/Segmentation/val.txt"
            if seg_val.is_file():
                val_ids, val_labels = load_cls_labels(
                    self.voc_root, "val", self.num_fg_classes, self.class_names_fg)
        return train_ids, train_labels, val_ids, val_labels

    def _seg_train_ids(self) -> list[str]:
        ids = read_split_list(
            self.voc_root / "ImageSets/Segmentation" / f"{self.config['train_list']}.txt")
        gt_dir = self.gt_mask_dir()
        missing = [i for i in ids[:50] if not (gt_dir / f"{i}.png").is_file()]
        if missing:
            raise FileNotFoundError(
                f"Segmentation masks missing for e.g. {missing[:3]} under {gt_dir}. "
                "The 'train_aug' list requires SegmentationClassAug masks."
            )
        return ids

    def _weak_training_ids(self) -> tuple[list[str], int]:
        """Segmentation-train ids that ALSO carry an official image-level label.

        ~300 VOC2012 segmentation-train images (inherited from the VOC2007 pool,
        e.g. ``2007_000032``) were never classified under VOC2012's Main task, so
        no honest image-level label exists for them. They are excluded from ALL
        experiment rows alike so every method trains on the identical image set.
        """
        labelled = set(self._cls_label_union())
        seg_ids = self._seg_train_ids()
        covered = [image_id for image_id in seg_ids if image_id in labelled]
        return covered, len(seg_ids) - len(covered)

    def _viz_val_ids(self) -> list[str]:
        """Evenly spaced val ids used for qualitative grids (kept small on purpose)."""
        val_ids = read_split_list(self.voc_root / "ImageSets/Segmentation/val.txt")
        n = max(1, int(self.config["viz_n_examples"]))
        stride = max(1, len(val_ids) // n)
        return val_ids[::stride][:n]

    def _labels_for(self, image_ids: list[str], lookup: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([lookup[image_id] for image_id in image_ids])

    def _cls_label_union(self) -> dict[str, np.ndarray]:
        """Image-level labels merged over Main train+val (= trainval coverage).

        VOC2012's segmentation splits are sampled from the classification
        trainval pool and are deliberately NOT aligned with either Main split
        alone - a segmentation-train image may sit in Main/val.txt and vice
        versa - so pseudo-mask generation must look up labels in the union.
        """
        lookup: dict[str, np.ndarray] = {}
        for split in ("train", "val"):
            ids, labels = load_cls_labels(
                self.voc_root, split, self.num_fg_classes, self.class_names_fg)
            lookup.update(zip(ids, labels))
        return lookup

    def gt_mask_dir(self) -> Path:
        if self.class_names_fg is not None:  # synthetic dataset
            return self.voc_root / "SegmentationClass"
        aug = self.voc_root / "SegmentationClassAug"
        if str(self.config["train_list"]) == "train_aug" and aug.is_dir():
            return aug
        return self.voc_root / "SegmentationClass"

    def _load_classifier_checkpoint(self, arch: str):
        model = CamClassifier(self.num_classes, pretrained=False) if arch == "plain" \
            else SeamNet(self.num_classes, pretrained=False)
        path = self.checkpoints_dir / f"classifier_{arch}.pth"
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        return model.to(self.device)

    def _load_segmentation_checkpoint(self, run_name: str):
        model = build_deeplab(num_classes=self.num_classes, init="random")
        path = self.checkpoints_dir / f"seg_{run_name}.pth"
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.to(self.device)

    # ------------------------------------------------------------------- run

    def run(self, from_stage: str | None = None, to_stage: str | None = None) -> dict:
        start = STAGES.index(from_stage) if from_stage else 0
        end = STAGES.index(to_stage) if to_stage else len(STAGES) - 1
        if start > end:
            raise ValueError(f"from_stage {from_stage} is after to_stage {to_stage}")

        for stage in STAGES[start:end + 1]:
            if not self.force and self.state_mgr.is_stage_complete(stage):
                self.logger.info("[skip] %s already complete", stage)
                continue
            self.state_mgr.start_stage(stage)
            self.logger.info("=== stage: %s ===", stage)
            started = time.perf_counter()
            try:
                artifacts, metrics = getattr(self, f"_stage_{stage}")()
            except Exception as error:
                self.state_mgr.fail_stage(stage, str(error))
                self.logger.exception("stage %s failed", stage)
                raise
            duration = time.perf_counter() - started
            self.state_mgr.complete_stage(stage, artifacts=artifacts or [],
                                          metrics=metrics or {}, duration=duration)
            self.logger.info("=== %s complete in %.1fs ===", stage, duration)
            if stage == "generate_report":
                # Re-render once the report stage's own runtime is final.
                self._stage_generate_report()

        summary = self.state_mgr.summary()
        (self.outputs_dir / "pipeline_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    # ----------------------------------------------------------------- stages

    def _stage_data_prep(self):
        if self.fast_dev_run:
            voc_root = create_synthetic_voc(self.data_dir, num_images=16, image_size=96)
            names = (voc_root / "classes.txt").read_text().splitlines()
            self.class_names_fg = names[1:]
            self.num_fg_classes = len(self.class_names_fg)
            self.num_classes = self.num_fg_classes + 1
            self.logger.info("Synthetic VOC ready: %d classes", self.num_classes)
            return {"artifacts": [str(voc_root)]}, {}
        voc_root = prepare_voc2012(self.data_dir, self.logger)
        return {"artifacts": [str(voc_root)]}, {}

    # -- classification ------------------------------------------------------

    def _train_classifier(self, arch: str) -> tuple[list[str], dict]:
        pretrained = bool(self.config["pretrained"]) and not self.fast_dev_run
        model = CamClassifier(self.num_classes, pretrained=pretrained) if arch == "plain" \
            else SeamNet(self.num_classes, pretrained=pretrained)
        model.to(self.device)

        train_ids, train_labels, _, _ = self._resolve_splits()
        self.logger.info("[%s] training on %d image-level labelled images", arch, len(train_ids))
        dataset = VOCClassificationDataset(
            self.voc_root, train_ids, train_labels,
            crop_size=int(self.config["cls_crop_size"]),
            resize_range=tuple(self.config["cls_resize_long"]))
        loader = DataLoader(dataset, batch_size=int(self.config["cls_batch_size"]), shuffle=True,
                            num_workers=int(self.config["cls_num_workers"]), drop_last=True,
                            pin_memory=self.device.type == "cuda", worker_init_fn=worker_init_fn)
        epochs = int(self.config[f"cls_epochs_{arch}"])

        steps_per_epoch = self._steps_this_epoch(loader)
        max_steps = steps_per_epoch * epochs
        optimizer, scheduler = self._poly_optimizer(
            model.parameter_groups(float(self.config["cls_lr"]),
                                   float(self.config["cls_head_lr_mult"]),
                                   float(self.config["cls_weight_decay"])),
            max_steps)
        scaler = self._scaler()

        model.train()
        running = 0.0
        global_step = 0
        for epoch in range(epochs):
            for batch_index, (images, label) in enumerate(loader):
                if batch_index >= steps_per_epoch:
                    break
                images = images.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                with self._autocast():
                    loss = self._classifier_loss(model, arch, images, label)
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                running += float(loss.item())
                global_step += 1
                if global_step % 50 == 0:
                    self.logger.info("[%s] epoch %d it %d/%d loss %.4f lr %.5f", arch,
                                     epoch + 1, batch_index + 1, steps_per_epoch,
                                     running / 50, scheduler.current_lr())
                    running = 0.0

        out_path = self.checkpoints_dir / f"classifier_{arch}.pth"
        torch.save(model.state_dict(), out_path)
        return [out_path], {"final_train_loss_approx": round(running or 0.0, 5)}

    def _classifier_loss(self, model, arch: str, images, label):
        label21 = torch.cat([torch.ones(label.size(0), 1, device=label.device), label], dim=1)
        label_map = label21[:, :, None, None]

        if arch == "plain":
            cam = model(images)
            scores = cam.mean(dim=(2, 3))
            return F.multilabel_soft_margin_loss(scores[:, 1:], label)

        # SEAM: scale-equivariant consistency + PCM cross-consistency (SEAM recipe).
        n, _, h, w = images.shape
        small = F.interpolate(images, scale_factor=float(self.config["cls_scale_factor"]),
                              mode="bilinear", align_corners=True)
        cam_full, cam_rv_full = model(images)
        cam_small, cam_rv_small = model(small)

        loss_cls = 0.5 * (
            F.multilabel_soft_margin_loss(cam_full.mean(dim=(2, 3))[:, 1:], label)
            + F.multilabel_soft_margin_loss(cam_small.mean(dim=(2, 3))[:, 1:], label))

        rvmin = 0.5 * (
            adaptive_min_pooling_loss((cam_rv_full * label_map)[:, 1:, :, :])
            + adaptive_min_pooling_loss((cam_rv_small * label_map)[:, 1:, :, :]))

        norm_full = max_norm(cam_full) * label_map
        norm_small = max_norm(cam_small) * label_map
        norm_rv_full = max_norm(cam_rv_full) * label_map
        norm_rv_small = max_norm(cam_rv_small) * label_map
        size = small.shape[-2:]
        norm_full_ds = F.interpolate(norm_full, size=size, mode="bilinear", align_corners=True)
        norm_rv_full_ds = F.interpolate(norm_rv_full, size=size, mode="bilinear", align_corners=True)

        loss_er = torch.mean(torch.abs(norm_full_ds[:, 1:, :, :] - norm_small[:, 1:, :, :]))
        loss_ecr = cross_refined_consistency(norm_small, norm_rv_full_ds, self.num_classes) \
            + cross_refined_consistency(norm_full_ds, norm_rv_small, self.num_classes)
        return loss_cls + rvmin + loss_er + loss_ecr

    def _stage_train_classifier_plain(self):
        return self._train_classifier("plain")

    def _stage_train_classifier_seam(self):
        return self._train_classifier("seam")

    # -- pseudo masks ---------------------------------------------------------

    def _stage_generate_pseudo_masks(self):
        # Classifiers may train on all Main-labelled images (5,717 for VOC train),
        # but pseudo masks are only needed where they are consumed: the segmentation
        # training ids plus the handful of val ids used in qualitative grids.
        seg_train_ids, dropped = self._weak_training_ids()
        viz_ids = self._viz_val_ids() if self.config["cam_include_val_for_viz"] else []
        label_lookup = self._cls_label_union()

        if dropped:
            self.logger.warning(
                "%d/%d segmentation-train ids carry no Main image-level label "
                "(VOC2007-inherited images) - excluded from ALL rows for a controlled comparison",
                dropped, dropped + len(seg_train_ids))
        viz_ids = [i for i in viz_ids if i in label_lookup]

        target_ids = list(dict.fromkeys(seg_train_ids + viz_ids))
        target_labels = self._labels_for(target_ids, label_lookup)
        self.logger.info("pseudo masks: %d segmentation-train ids + %d viz ids",
                         len(seg_train_ids), len(viz_ids))

        use_crf = bool(self.config["cam_use_crf"]) and CRF_AVAILABLE
        if bool(self.config["cam_use_crf"]) and not CRF_AVAILABLE:
            self.logger.warning("pydensecrf unavailable -> skipping CAM+CRF variant "
                                "(install pydensecrf2 to enable the ownership experiment)")
        if use_crf and not self._crf_smoke_ok():
            use_crf = False

        jobs = [
            ("cam_naive", "plain", tuple(self.config["cam_naive_scales"]),
             bool(self.config["cam_naive_flips"])),
            ("cam_seam", "seam", tuple(self.config["cam_seam_scales"]),
             bool(self.config["cam_seam_flips"])),
        ]
        artifacts: list[str] = []
        crf_futures: deque[Future] = deque()
        crf_workers = max(0, int(self.config["crf_workers"]))
        executor = ProcessPoolExecutor(max_workers=crf_workers) if (use_crf and crf_workers) else None

        def _drain_crf(limit: int) -> None:
            while len(crf_futures) > limit:
                done = [f for f in crf_futures if f.done()]
                if not done:
                    time.sleep(0.2)
                    continue
                for future in done:
                    future.result()
                    crf_futures.remove(future)

        try:
            for out_name, arch, scales, flips in jobs:
                # The ownership experiment refines the NAIVE CAMs only: Raw CAM vs
                # CAM+DenseCRF vs SEAM keeps one refinement variable per row.
                use_crf_here = use_crf and out_name == "cam_naive"
                model = self._load_classifier_checkpoint(arch)
                out_dir = self.masks_dir / out_name
                count = 0
                for image_id, cam_dict in generate_cam_scores(
                        model, self.voc_root, target_ids, target_labels, self.device,
                        scales=scales, flips=flips,
                        max_long_side=int(self.config["cam_max_long_side"])):
                    mask = cams_to_argmax_mask(cam_dict, bg_alpha=float(self.config["cam_bg_alpha"]))
                    save_pseudo_mask(mask, out_dir, image_id)
                    if use_crf_here:
                        # Owned writable copies: read-only / view arrays break both
                        # multiprocessing pickling and pydensecrf's Cython buffers.
                        cam_payload = {
                            c: np.array(scores, dtype=np.float32, copy=True)
                            for c, scores in cam_dict.items()
                        }
                        out_path = self.masks_dir / f"{out_name}_crf" / f"{image_id}.png"
                        if executor is not None:
                            payload = (
                                str(self.voc_root / "JPEGImages" / f"{image_id}.jpg"),
                                cam_payload,
                                float(self.config["cam_crf_alpha"]),
                                str(out_path),
                            )
                            crf_futures.append(executor.submit(_crf_worker, payload))
                            _drain_crf(32)  # bound in-flight RAM; GPU keeps streaming
                        else:
                            rgb = np.asarray(load_image(self.voc_root, image_id))
                            save_pseudo_mask(
                                crf_argmax_mask(rgb, cam_payload,
                                                alpha=float(self.config["cam_crf_alpha"])),
                                out_path.parent, image_id)
                    count += 1
                self.logger.info("generated %d %s pseudo-masks%s", count, out_name,
                                 " (+CRF)" if use_crf_here else "")
                del model
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            if executor is not None:
                for future in crf_futures:   # finish any remaining refinements
                    future.result()
                executor.shutdown(wait=True)

        missing_crf = (self.masks_dir / "cam_naive_crf")
        if use_crf and not any(missing_crf.glob("*.png")):
            raise RuntimeError("DenseCRF was enabled but produced no masks - check worker logs")

        artifacts.append(str(self.masks_dir))
        quality = self._pseudo_label_quality(seg_train_ids)
        (self.metrics_dir / "pseudo_quality.json").write_text(
            json.dumps(quality, indent=2) + "\n", encoding="utf-8")
        return artifacts, {"crf_available": CRF_AVAILABLE}

    def _crf_smoke_ok(self) -> bool:
        """One tiny end-to-end DenseCRF call before committing 1,151 refinements to it.

        Converts any residual environment incompatibility (writable-buffer rules,
        ABI quirks) into graceful degradation of the CAM+CRF row instead of a
        dead multi-hour session.
        """
        try:
            from weakseg.weak.crf import dense_crf_inference

            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, :, 0] = 128
            scores = np.stack([np.full((64, 64), 0.4), np.full((64, 64), 0.6)]).astype(np.float32)
            result = dense_crf_inference(image, scores, iterations=2)
            return result.shape == (2, 64, 64)
        except Exception as exc:  # noqa: BLE001 - any failure disables the feature
            self.logger.warning("DenseCRF smoke test failed (%s) -> disabling CAM+CRF row", exc)
            return False

    def _pseudo_label_quality(self, seg_train_ids: list[str]) -> dict:
        """Diagnostic-only comparison of pseudo masks against GT train masks."""
        gt_dir = self.gt_mask_dir()
        report: dict[str, Any] = {}
        variants = ["cam_naive", "cam_seam"] + (["cam_naive_crf"] if CRF_AVAILABLE else [])
        for variant in variants:
            directory = self.masks_dir / variant
            if not directory.is_dir():
                continue
            matrix = ConfusionMatrix(self.num_classes)
            for image_id in seg_train_ids:
                pred_path = directory / f"{image_id}.png"
                gt_path = gt_dir / f"{image_id}.png"
                if not pred_path.is_file() or not gt_path.is_file():
                    continue
                matrix.update(np.asarray(Image.open(pred_path), dtype=np.uint8),
                              np.asarray(Image.open(gt_path), dtype=np.uint8))
            if matrix.matrix.sum():
                report[variant] = matrix.summary()
        return report

    # -- segmentation ----------------------------------------------------------

    def _train_segmentation(self, run_name: str, mask_dir: Path) -> tuple[list[str], dict]:
        # Every row - including fully supervised - trains on the SAME id set
        # (segmentation-train ids that carry an official image-level label),
        # so differences in the results table come purely from supervision.
        train_ids, _ = self._weak_training_ids()
        init = "random" if self.fast_dev_run else str(self.config["seg_init"])
        model = build_deeplab(num_classes=self.num_classes, init=init)
        model.to(self.device)
        self.logger.info("[seg:%s] %d images | masks=%s | init=%s",
                         run_name, len(train_ids), mask_dir, init)

        dataset = VOCSegDataset(self.voc_root, train_ids, mask_dir,
                                crop_size=int(self.config["seg_crop_size"]),
                                resize_range=tuple(self.config["seg_resize_long"]))
        loader = DataLoader(dataset, batch_size=int(self.config["seg_batch_size"]), shuffle=True,
                            num_workers=int(self.config["seg_num_workers"]), drop_last=True,
                            pin_memory=self.device.type == "cuda", worker_init_fn=worker_init_fn)
        epochs = int(self.config["seg_epochs"])
        steps_per_epoch = self._steps_this_epoch(loader)
        max_steps = steps_per_epoch * epochs
        optimizer = torch.optim.SGD(model.parameters(), lr=float(self.config["seg_lr"]),
                                    momentum=float(self.config["seg_momentum"]),
                                    weight_decay=float(self.config["seg_weight_decay"]))
        scheduler = PolyLr(optimizer, power=0.9, max_steps=max_steps)
        scaler = self._scaler()
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        model.train()
        running, global_step = 0.0, 0
        for epoch in range(epochs):
            for batch_index, (images, masks) in enumerate(loader):
                if batch_index >= steps_per_epoch:
                    break
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                with self._autocast():
                    logits = model(images)["out"]
                    loss = criterion(logits, masks)
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                running += float(loss.item())
                global_step += 1
                if global_step % 100 == 0:
                    self.logger.info("[seg:%s] ep %d it %d/%d loss %.4f", run_name,
                                     epoch + 1, batch_index + 1, steps_per_epoch,
                                     running / 100)
                    running = 0.0

        out_path = self.checkpoints_dir / f"seg_{run_name}.pth"
        torch.save(model.state_dict(), out_path)
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return [out_path], {}

    def _stage_train_seg_fully_sup(self):
        return self._train_segmentation("fully_sup", self.gt_mask_dir())

    def _stage_train_seg_cam(self):
        return self._train_segmentation("cam", self.masks_dir / "cam_naive")

    def _stage_train_seg_cam_crf(self):
        crf_dir = self.masks_dir / "cam_naive_crf"
        if not any(crf_dir.glob("*.png")):
            self.logger.warning("No CRF pseudo masks present - marking stage skipped")
            return [], {"skipped": True}
        return self._train_segmentation("cam_crf", crf_dir)

    def _stage_train_seg_seam(self):
        return self._train_segmentation("seam", self.masks_dir / "cam_seam")

    # -- evaluation ------------------------------------------------------------

    def _stage_evaluate(self):
        # Always the official segmentation val split (1,449 ids) - NOT the larger
        # ImageSets/Main classification val list, most of which has no GT masks.
        val_ids = read_split_list(self.voc_root / "ImageSets/Segmentation/val.txt")
        runs = {
            "fully_sup": "seg_fully_sup.pth",
            "cam": "seg_cam.pth",
            "cam_crf": "seg_cam_crf.pth",
            "seam": "seg_seam.pth",
        }
        results: dict[str, Any] = {}
        for method, filename in runs.items():
            path = self.checkpoints_dir / filename
            if not path.is_file():
                self.logger.warning("checkpoint missing for %s (%s)", method, path)
                continue
            model = self._load_segmentation_checkpoint(method)
            metrics = evaluate_segmentation(
                model, self.voc_root, val_ids, self.gt_mask_dir(), self.device,
                self.num_classes, max_images=self.config["eval_max_images"],
                num_workers=int(self.config["eval_num_workers"]),
                batch_size=int(self.config["eval_batch_size"]))
            results[method] = metrics
            self.logger.info("[eval] %-10s mIoU %.2f dice %.2f pixacc %.2f", method,
                             metrics["miou"], metrics["dice"], metrics["pixel_accuracy"])
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        out_path = self.metrics_dir / "val_metrics.json"
        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return [out_path], {m: v["miou"] for m, v in results.items()}

    # -- visualization -----------------------------------------------------------

    def _stage_visualize(self):
        chosen = self._viz_val_ids()

        models = {}
        for method in ("fully_sup", "cam", "seam"):
            if (self.checkpoints_dir / f"seg_{method}.pth").is_file():
                models[method] = self._load_segmentation_checkpoint(method)

        pseudo_dirs = {
            "cam_naive": self.masks_dir / "cam_naive",
            "cam_naive_crf": self.masks_dir / "cam_naive_crf",
            "cam_seam": self.masks_dir / "cam_seam",
        }
        artifacts = []
        for image_id in chosen:
            image_tensor = self._tensor_for(image_id)
            predictions = {
                method: predict_mask(model, image_tensor, self.device)
                for method, model in models.items()
            }
            gt_path = self.gt_mask_dir() / f"{image_id}.png"
            gt = np.asarray(Image.open(gt_path), dtype=np.uint8) if gt_path.is_file() else None
            out_path = write_qualitative_grid(
                self.viz_dir / f"{image_id}.jpg", self.voc_root, image_id, gt,
                pseudo_dirs, predictions, crf_available=CRF_AVAILABLE)
            artifacts.append(out_path)
        return artifacts, {}

    def _tensor_for(self, image_id: str):
        from weakseg.data.datasets import TO_TENSOR, load_image as _li

        return TO_TENSOR(_li(self.voc_root, image_id))

    # -- report ---------------------------------------------------------------

    def _stage_generate_report(self):
        val_metrics = {}
        metrics_path = self.metrics_dir / "val_metrics.json"
        if metrics_path.is_file():
            val_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        pseudo_quality = {}
        pq_path = self.metrics_dir / "pseudo_quality.json"
        if pq_path.is_file():
            pseudo_quality = json.loads(pq_path.read_text(encoding="utf-8"))

        echo = {k: v for k, v in self.config.items() if k != "data_root"}
        echo["device_used"] = str(self.device)
        report = write_results_markdown(
            self.reports_dir / "experiment_summary.md", val_metrics,
            pseudo_quality, self.state_mgr.summary(), echo,
            class_names_fg=self.class_names_fg)
        root_copy = self.root / "RESULTS.md"
        root_copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")

        rows = build_comparison_rows(val_metrics)
        return [report, root_copy], {row["method"]: row["miou"] for row in rows}


class PolyLr:
    """Poly learning-rate decay: lr_t = lr_0 * (1 - t/T)^power, applied per group."""

    def __init__(self, optimizer: torch.optim.Optimizer, power: float, max_steps: int):
        self.optimizer = optimizer
        self.power = power
        self.max_steps = max(1, max_steps)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_count = 0

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def step(self) -> None:
        self.step_count += 1
        factor = (1 - self.step_count / self.max_steps) ** self.power
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * max(0.0, factor)
