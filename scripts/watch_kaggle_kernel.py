#!/usr/bin/env python3
"""Stream a Kaggle notebook run's logs to stdout and a local append-only log file."""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SLUG = "kushchaudhari/weakseg"
DEFAULT_LOG = Path(__file__).resolve().parents[1] / ".kaggle-run.log"


def _run_kaggle(*args: str) -> tuple[int, str]:
    result = subprocess.run(
        ["kaggle", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    return result.returncode, result.stdout.rstrip()


def _status(slug: str) -> str:
    code, output = _run_kaggle("kernels", "status", slug)
    if code:
        return f"STATUS_COMMAND_FAILED: {output}"
    marker = 'status "'
    if marker in output:
        return output.split(marker, 1)[1].split('"', 1)[0]
    return output.splitlines()[-1] if output else "UNKNOWN"


def _write_line(output, line: str) -> None:
    print(line, flush=True)
    output.write(line + "\n")
    output.flush()


def watch(slug: str, log_path: Path, once: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        started = datetime.now(timezone.utc).isoformat()
        _write_line(log, f"===== Kaggle log watcher started {started} ({slug}) =====")

        if once:
            timestamp = datetime.now(timezone.utc).isoformat()
            current_status = _status(slug)
            _write_line(log, f"[{timestamp}] status: {current_status}")
            code, logs = _run_kaggle("kernels", "logs", slug)
            if code:
                _write_line(log, f"[{timestamp}] log command failed ({code}): {logs}")
                return code
            for line in logs.splitlines():
                _write_line(log, line)
            _write_line(log, f"[watch] stopped with status {current_status}")
            return 1 if "ERROR" in current_status.upper() else 0

        _write_line(log, "[watch] following live log stream; Ctrl-C stops this watcher only")
        process = subprocess.Popen(
            ["kaggle", "kernels", "logs", "--follow", slug],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            _write_line(log, line.rstrip())
        code = process.wait()
        current_status = _status(slug)
        timestamp = datetime.now(timezone.utc).isoformat()
        _write_line(log, f"[{timestamp}] stream ended (code={code}) status={current_status}")
        return 1 if "ERROR" in current_status.upper() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--once", action="store_true",
                        help="Print the latest buffered logs and exit instead of following")
    args = parser.parse_args()
    raise SystemExit(watch(args.slug, args.log, args.once))


if __name__ == "__main__":
    main()
