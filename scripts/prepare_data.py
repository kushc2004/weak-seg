#!/usr/bin/env python3
"""Download / verify Pascal VOC2012 (or build the synthetic mini set with fast_dev_run=true)."""
from __future__ import annotations

import argparse

from run_stage import run_single_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    run_single_stage("data_prep", args.overrides)


if __name__ == "__main__":
    main()
