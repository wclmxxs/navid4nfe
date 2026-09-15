"""DBCache: one probe block, cached remaining-block residual.

The relative change of the first block's residual controls reuse. This is a
local integration of the Cache-DiT DBCache algorithm (Fn=1, Bn=0), not AdaLN
precomputation. Global sums give every Ulysses rank the same skip decision.
"""
from __future__ import annotations

import math


class ResidualCache:
    def __init__(self, reduce_sums=None):
        self.reduce_sums = reduce_sums
        self.reset({"enabled": False, "warmup": 1, "rdt": 0.08, "max_continuous_cached_steps": 1})

    def reset(self, options, total_steps: int = 4):
        if total_steps not in (4, 8):
            raise ValueError("DBCache expects 4 or 8 total steps")
        self.total_steps = total_steps
        self.options = dict(options)
        self.previous_probe = self.tail_residual = self.tail_input = None
        self.skip = False
        self.accumulated = 0.0
        self.consecutive = 0
        self.decisions = []
        self.blocks_computed = self.blocks_reused = 0

    def probe(self, before, after, step: int, valid_rows: int):
        import torch

        # clone before executing the probe in the wrapper: fused kernels may
        # otherwise overwrite a snapshot retained by reference.
        residual = (after[:, :valid_rows].float() - before[:, :valid_rows].float())
        change = None
        if self.previous_probe is not None:
            sums = torch.stack(((residual - self.previous_probe).abs().sum(),
                                self.previous_probe.abs().sum()))
            if self.reduce_sums is not None:
                sums = self.reduce_sums(sums)
            num, den = sums.tolist()
            change = num / den if den > 1e-12 else (0.0 if num <= 1e-12 else math.inf)
        self.previous_probe = residual.detach().clone()
        if change is not None:
            self.accumulated += change
        eligible = (step >= self.options["warmup"] and step < self.total_steps - 1 and
                    self.tail_residual is not None and change is not None and
                    math.isfinite(self.accumulated) and
                    self.consecutive < self.options["max_continuous_cached_steps"])
        self.skip = eligible and self.accumulated < self.options["rdt"]
        self.decisions.append({"step": step, "cached": self.skip,
                               "relative_change": change if change is not None and math.isfinite(change) else None})
        if self.skip:
            self.consecutive += 1
        else:
            self.consecutive = 0
            self.accumulated = 0.0
            self.tail_input = after.detach().clone()

    def finish_tail(self, output):
        if self.skip:
            return output + self.tail_residual
        self.tail_residual = (output - self.tail_input).detach().clone()
        self.tail_input = None
        return output

    def stats(self):
        return {"algorithm": "DBCache", "Fn": 1, "Bn": 0, "nfe": self.total_steps, **self.options,
                "steps": list(self.decisions), "cached_steps": sum(d["cached"] for d in self.decisions),
                "blocks_computed": self.blocks_computed, "blocks_reused": self.blocks_reused}

    def release(self):
        self.previous_probe = self.tail_residual = self.tail_input = None
