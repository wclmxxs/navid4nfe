"""Bound the number of ranks materializing CPU weights at the same time."""
from __future__ import annotations


def load_in_groups(load, *, rank: int, world_size: int, parallelism: int, barrier) -> None:
    if not 1 <= parallelism <= world_size or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed model-loading group")
    for first in range(0, world_size, parallelism):
        barrier()
        if first <= rank < min(first + parallelism, world_size):
            load()
        barrier()
