"""Preserve the original worker exception before torchrun's shutdown summary."""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

from . import config


def save_worker_error(error: Exception) -> None:
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown"))
    record = {
        "run_id": config.RUN_ID,
        "rank": rank,
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    if type(error).__name__ == "OutOfMemoryError" or "Insufficient free memory before GPU model loading" in str(error):
        from .prepare import gpu_process_inventory
        record["gpu_processes"] = gpu_process_inventory()
    path = config.RUNTIME / "worker-errors" / config.RUN_ID / f"rank-{rank}.json"
    try:
        config.atomic_json(path, record)
        print(f"WORKER_EXCEPTION: rank={rank} {type(error).__name__}: {error}\nSaved: {path}", file=sys.stderr, flush=True)
    except OSError as write_error:
        # Failure to write diagnostics must not replace the original exception.
        print(f"Could not save worker exception: {write_error}", file=sys.stderr, flush=True)


def log_excerpt(path: Path, offset: int = 0) -> str:
    if not path.is_file():
        return "Log file does not exist yet."
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        # Bound RAM usage even after long-running deployments. Prefer the current
        # run's first traceback over the final multi-rank termination summary.
        if offset > size:
            offset = 0
        stream.seek(max(offset, size - 1024 * 1024))
        lines = stream.read().decode(errors="replace").splitlines()
    for index, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            return "\n".join(lines[max(0, index - 8):index + 160])
    return "\n".join(lines[-60:])


def failure_report() -> str:
    run = config.read_json(config.RUNTIME / "last-run.json")
    run_id = run.get("run_id")
    records = []
    if run_id:
        for path in (config.RUNTIME / "worker-errors" / run_id).glob("rank-*.json"):
            record = config.read_json(path)
            if record.get("run_id") == run_id and record.get("traceback"):
                records.append(record)
    sections = []
    if records:
        records.sort(key=lambda record: record.get("time_ns", 0))
        first = records[0]
        sections.append(f"--- First recorded worker exception: rank {first['rank']} ---\n{first['traceback']}")
        if first.get("gpu_processes"):
            sections.append(first["gpu_processes"])
    offsets = run.get("log_offsets", {})
    for name in ("worker.log", "api.log", "service.log"):
        if name == "worker.log" and records:
            continue
        sections.append(f"--- {name}: exception context / recent output ---\n"
                        + log_excerpt(config.RUNTIME / name, offsets.get(name, 0)))
    return "\n".join(sections)
