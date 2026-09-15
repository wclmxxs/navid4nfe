"""Choose a model-loading group size from the current host/container RAM budget."""
from __future__ import annotations

import os
from pathlib import Path

GIB = 1024**3
LOAD_RANK_GIB = 192  # ~140 GiB weights plus temporary CPU tensors
HOST_RESERVE_GIB = 64


def available_memory_bytes(proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")) -> int:
    memory = dict(line.split(":", 1) for line in (proc_root / "meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    try:
        memberships = (proc_root / "self/cgroup").read_text().splitlines()
    except FileNotFoundError:
        memberships = []
    for membership in memberships:
        _, controllers, group = membership.split(":", 2)
        if controllers == "":
            root, limit_name, usage_name = cgroup_root, "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            root, limit_name, usage_name = cgroup_root / "memory", "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        # Check parent limits too. A cgroup namespace may expose only its root.
        root = root.resolve()
        directory = (root / group.lstrip("/")).resolve()
        if not directory.is_relative_to(root):
            directory = root
        while True:
            try:
                limit = (directory / limit_name).read_text().strip()
                if limit != "max":
                    usage = int((directory / usage_name).read_text())
                    available = min(available, max(0, int(limit) - usage))
            except FileNotFoundError:
                pass
            if directory == root:
                break
            directory = directory.parent
    return available


def loading_parallelism(available_gib: float, requested="auto", world_size=8) -> int:
    if available_gib < 160:
        raise RuntimeError(f"At least 160 GiB available host/container RAM required; got {available_gib:.1f} GiB")
    affordable = max(1, int((available_gib - HOST_RESERVE_GIB) // LOAD_RANK_GIB))
    choices = [n for n in (1, 2, 4, 8) if n <= world_size]
    if requested == "auto":
        return max(n for n in choices if n <= affordable)
    if str(requested) not in {str(n) for n in choices}:
        raise ValueError("MODEL_LOAD_PARALLELISM must be auto, 1, 2, 4 or 8 (no larger than world size)")
    value = int(requested)
    if value > affordable:
        raise RuntimeError(f"MODEL_LOAD_PARALLELISM={value} needs a loading budget of "
                           f"{value * LOAD_RANK_GIB + HOST_RESERVE_GIB} GiB; available={available_gib:.1f} GiB")
    return value


def load_plan(world_size=8) -> dict:
    available_gib = available_memory_bytes() / GIB
    requested = os.environ.get("MODEL_LOAD_PARALLELISM", "auto")
    count = loading_parallelism(available_gib, requested, world_size)
    return {"parallelism": count, "requested": requested, "available_gib": round(available_gib, 1),
            "per_rank_budget_gib": LOAD_RANK_GIB, "reserve_gib": HOST_RESERVE_GIB}


def dit_compile_enabled() -> bool:
    return _compile_enabled("DIT_COMPILE")


def vae_compile_enabled() -> bool:
    return _compile_enabled("VAE_COMPILE")


def _compile_enabled(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"
