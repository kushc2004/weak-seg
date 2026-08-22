#!/usr/bin/env python3
"""Evaluate every trained segmentation model on the VOC val set (mIoU / Dice / pixel acc)."""
from __future__ import annotations

import argparse

from run_stage import run_single_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    run_single_stage("evaluate", args.overrides)


if __name__ == "__main__":
    main()
