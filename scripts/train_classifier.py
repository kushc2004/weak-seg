#!/usr/bin/env python3
"""Train the multi-label classifier (CAM head) - naive (--arch plain) or SEAM (--arch seam)."""
from __future__ import annotations

import argparse

from run_stage import run_single_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("plain", "seam"), default="plain")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    overrides = list(args.overrides)
    if args.epochs is not None:
        overrides.append(f"cls_epochs_{args.arch}={args.epochs}")
    stage = "train_classifier_plain" if args.arch == "plain" else "train_classifier_seam"
    run_single_stage(stage, overrides)


if __name__ == "__main__":
    main()
