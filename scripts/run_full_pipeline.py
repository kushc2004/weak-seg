#!/usr/bin/env python3
"""Run the complete WeakSeg pipeline with durable stage checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from weakseg.pipeline import STAGES, FullPipeline


ROOT = Path(__file__).resolve().parents[1]


def load_config(overrides: list[str]) -> dict:
    config: dict = {}
    for cfg_path in sorted((ROOT / "configs").glob("**/*.yaml")):
        with cfg_path.open(encoding="utf-8") as source:
            config.update(yaml.safe_load(source) or {})
    for value in overrides:
        if "=" not in value:
            raise ValueError(f"override must be key=value, received {value!r}")
        key, raw = value.split("=", 1)
        config[key] = yaml.safe_load(raw)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", choices=STAGES, help="Starting stage name")
    parser.add_argument("--to-stage", choices=STAGES, help="Ending stage name")
    parser.add_argument("--force", action="store_true", help="Rerun stages even if complete")
    parser.add_argument("overrides", nargs="*", help="Configuration overrides as key=value")
    args = parser.parse_args()

    config = load_config(args.overrides)
    pipeline = FullPipeline(ROOT, config, force=args.force)
    pipeline.run(args.from_stage, args.to_stage)
    print(f"\nPipeline state written to: {(ROOT / 'outputs/pipeline_state.json').resolve()}")


if __name__ == "__main__":
    main()
