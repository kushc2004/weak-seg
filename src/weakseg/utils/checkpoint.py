"""Durable stage-checkpointing for resumable pipeline runs.

Mirrors the atomic ``outputs/pipeline_state.json`` protocol used by the
sibling next-gen-rec project so both repos behave identically on Kaggle:
every completed stage records its artifacts, and re-runs skip stages whose
recorded artifacts still exist.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PipelineStateManager:
    def __init__(self, state_file_path: Path | str):
        self.path = Path(state_file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.is_file():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stages": {},
        }

    def save(self) -> None:
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(self.path)

    def is_stage_complete(self, stage_name: str) -> bool:
        stage = self.state.get("stages", {}).get(stage_name)
        if not stage or stage.get("status") != "complete":
            return False
        for rel_path in stage.get("artifacts", []):
            if not Path(rel_path).exists():
                return False
        return True

    def start_stage(self, stage_name: str) -> None:
        self.state.setdefault("stages", {})[stage_name] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": None,
            "artifacts": [],
            "metrics": {},
        }
        self.save()

    def complete_stage(
        self,
        stage_name: str,
        artifacts: list[str | Path] | None = None,
        metrics: dict[str, Any] | None = None,
        duration: float | None = None,
    ) -> None:
        stage = self.state.setdefault("stages", {}).setdefault(stage_name, {})
        stage["status"] = "complete"
        stage["completed_at"] = datetime.now(timezone.utc).isoformat()
        if duration is not None:
            stage["duration_seconds"] = round(duration, 3)
        if artifacts:
            stage["artifacts"] = [str(p) for p in artifacts]
        if metrics:
            stage["metrics"] = metrics
        self.save()

    def fail_stage(self, stage_name: str, error: str) -> None:
        stage = self.state.setdefault("stages", {}).setdefault(stage_name, {})
        stage["status"] = "failed"
        stage["error"] = error
        stage["failed_at"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "status": stage.get("status"),
                "duration_seconds": stage.get("duration_seconds"),
                "metrics": stage.get("metrics", {}),
            }
            for name, stage in self.state.get("stages", {}).items()
        }
