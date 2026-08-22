#!/usr/bin/env python3
"""Generate CAMs and pseudo segmentation masks (naive / SEAM, plus optional DenseCRF)."""
from __future__ import annotations

import argparse

from run_stage import run_single_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-crf", action="store_true", help="Skip DenseCRF refinement")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    overrides = ["cam_use_crf=false"] if args.no_crf else []
    run_single_stage("generate_pseudo_masks", args.overrides + overrides)


if __name__ == "__main__":
    main()
