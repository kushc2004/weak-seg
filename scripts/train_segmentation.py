#!/usr/bin/env python3
"""Train DeepLabV3-ResNet50 on a chosen supervision source.

--labels gt        fully supervised baseline (SegmentationClass masks)
--labels cam       naive CAM pseudo masks
--labels cam_crf   DenseCRF-refined CAM pseudo masks
--labels seam      SEAM pseudo masks
"""
from __future__ import annotations

import argparse

from run_stage import run_single_stage

STAGE_FOR_LABELS = {
    "gt": "train_seg_fully_sup",
    "cam": "train_seg_cam",
    "cam_crf": "train_seg_cam_crf",
    "seam": "train_seg_seam",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", choices=sorted(STAGE_FOR_LABELS), default="gt")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    run_single_stage(STAGE_FOR_LABELS[args.labels], args.overrides)


if __name__ == "__main__":
    main()
