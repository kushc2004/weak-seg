# WeakSeg: Annotation-Efficient Semantic Segmentation Using Image-Level Supervision

WeakSeg compares how much segmentation performance survives when expensive **pixel-level
annotations** are replaced with cheap **image-level labels** on PASCAL VOC2012:

| # | Method | Supervision | Pipeline |
| - | ------ | ----------- | -------- |
| 1 | DeepLabV3-ResNet50 (baseline) | Pixel masks | image + GT mask → DeepLabV3 |
| 2 | CAM baseline (naive) | Image labels only | ResNet-50 classifier → CAM → threshold → pseudo-mask → DeepLabV3 |
| 3 | CAM + DenseCRF (ownership experiment) | Image labels only | CAM scores → DenseCRF → pseudo-mask → DeepLabV3 |
| 4 | SEAM refinement (Wang et al., CVPR 2020) | Image labels only | equivariant-attention training + PCM affinity propagation → pseudo-mask → DeepLabV3 |

**Research question:** *how much segmentation performance can be recovered when pixel
annotations are replaced by image-level supervision?*

**Protocol guarantee** — weak-supervision training never reads `SegmentationClass`.
Image-level labels come from the official `ImageSets/Main` classification annotations;
ground-truth masks are opened exclusively by evaluation, visualization, and a clearly-marked
pseudo-label-quality diagnostic.

Results land in [`RESULTS.md`](RESULTS.md) (generated) plus per-image qualitative grids
under `outputs/visualizations/`.

---

## Method notes

### Naive CAM baseline
A ResNet-50 multi-label classifier whose head is a 1×1 convolution over OS16 features,
so global-average-pooled scores and class activation maps share weights exactly. CAMs are
label-masked, min-max normalized, then binarized by an argmax against a constant background
score α=0.26 (SEAM's protocol). Single-scale by design: this row intentionally shows what
raw localization gives.

### SEAM components (ported onto torchvision ResNet-50)
The original [SEAM repo](https://github.com/YudeWang/SEAM) targets PyTorch 0.4 + MXNet
ResNet-38 weights; WeakSeg re-implements its method faithfully on modern torchvision:

- **Equivariant regularization** — the same image at two scales must produce consistent
  CAMs (`loss_er`, L1 between rescaled CAM pairs).
- **Cross-view consistency** — the PCM-refined map must agree with the other view's raw
  argmax on the hardest 20% of pixels (`loss_ecr`), plus SEAM's adaptive-min-pooling loss
  that lifts weak foreground activations.
- **Pixel Correlation Module (PCM)** — learned projections of mid-level features plus the
  image build a dense affinity matrix that propagates the normalized CAM across similar
  pixels, expanding activations from discriminative parts toward whole objects.
- Inference aggregates multi-scale (0.5/1/1.5/2) horizontal-flip CAMs exactly like
  `infer_SEAM.py`; the refined map is consumed at prediction time.

Channel mapping ResNet-38 → ResNet-50 is exact: conv4(512)/conv5(1024)/conv6 ↔ layer2/layer3/layer4.

### Ownership experiment: pseudo-label refinement comparison
Pseudo-mask quality is measured for Raw CAM vs CAM+DenseCRF vs SEAM against train-split GT
(diagnostic only), and each variant trains its own DeepLabV3 to show how pseudo-label quality
propagates to final mIoU. DenseCRF uses SEAM's background rule `bg = (1 − max_fg)^α` with α=4.

### Controlled-comparison choices
- All four segmentation runs share identical architecture and initialization
  (torchvision COCO-pretrained DeepLabV3-RN50); only the label source differs.
- Classifier backbones start from ImageNet weights.
- Default training split: VOC2012 `train` (1,464 images). Evaluation: `val` (1,449),
  ignore label 255, mIoU/Dice averaged over classes present in ground truth.
- Metrics: mIoU, Dice, pixel accuracy + per-class IoU.

---

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

Smoke-test every stage end-to-end on a synthetic mini-VOC (no downloads, CPU-friendly):

```bash
make run-fast          # python scripts/run_full_pipeline.py --force fast_dev_run=true
```

Full pipeline on real VOC2012 (~4–5 h on one T4/P100):

```bash
make prepare-data      # downloads VOCtrainval_11-May-2012.tar (mirror + fallback)
make run-full          # all stages, resumable via outputs/pipeline_state.json
```

Stages execute in order and are individually resumable:

```text
data_prep -> train_classifier_plain -> train_classifier_seam -> generate_pseudo_masks
         -> train_seg_fully_sup -> train_seg_cam -> train_seg_cam_crf -> train_seg_seam
         -> evaluate -> visualize -> generate_report
```

Resume or rerun selectively:

```bash
python scripts/run_full_pipeline.py                       # skips completed stages
python scripts/run_full_pipeline.py --from-stage evaluate # re-evaluate only
python scripts/run_full_pipeline.py seg_epochs=40         # key=value overrides
python scripts/train_classifier.py --arch seam            # single-step CLIs
python scripts/train_segmentation.py --labels cam_crf
```

DenseCRF is optional: if `pydensecrf` cannot install, the CRF stage degrades gracefully
and the table simply omits the row. Try `pip install pydensecrf2` to enable it.

## Running on Kaggle GPU

Ready-to-run notebook: `notebooks/kaggle_weakseg.ipynb` (+ `kernel-metadata.json`).

```bash
# 1. Push kernel to Kaggle and trigger execution
kaggle kernels push -p .

# 2. Watch live logs
python scripts/watch_kaggle_kernel.py

# 3. Pull results back and restore locally
python scripts/fetch_kaggle_results.py
```

The notebook clones this repo, installs DenseCRF, verifies the GPU, restores any attached
artifact cache (`weakseg_artifacts.tar.gz`) to skip finished stages, runs the full pipeline,
renders RESULTS.md + qualitative grids, and repackages artifacts for download/reuse.

## Project layout

```text
weak-seg/
├── configs/                 # YAML defaults merged at runtime (key=value CLI overrides)
├── src/weakseg/
│   ├── pipeline.py          # FullPipeline + STAGES + state checkpointing
│   ├── data/                # VOC download, synthetic mini-VOC, labels, datasets
│   ├── models/              # RN50-OS16 backbone, CAM classifier, SeamNet, DeepLab wrapper
│   ├── weak/                # losses, CAM extraction, pseudo-masks, optional DenseCRF
│   ├── evaluation/          # confusion matrix metrics (mIoU/Dice/pixel acc), evaluator
│   ├── reporting/           # RESULTS.md generation, qualitative grids
│   └── utils/               # device/seed/checkpoint/palette/logging
├── scripts/                 # run_full_pipeline.py + per-step CLIs + Kaggle helpers
├── tests/                   # unit tests + full synthetic end-to-end smoke test
├── notebooks/kaggle_weakseg.ipynb
└── RESULTS.md               # generated comparison report
```

## Interview talking points

1. **Classification vs segmentation** — classification answers *what*, segmentation answers
   *where* per pixel; CAM bridges them because GAP classifiers retain spatial structure
   before pooling.
2. **Why CAM works / why it's incomplete** — class scores are dot products of feature maps
   with classifier weights, so high-response regions localize objects; but only the most
   discriminative parts activate (dog face, not paws), and backgrounds leak.
3. **Why SEAM helps** — equivariance turns scale consistency into free supervision; the PCM
   spreads evidence across pixel affinities so masks cover whole objects.
4. **How pseudo labels inject noise** — systematic errors (missing regions, background bleed)
   become training targets; CE memorizes them unless refined (CRF/affinity) or regularized.
5. **mIoU** — mean over classes of TP/(TP+FP+FN), void ignored; Dice ≡ F1 per class.
