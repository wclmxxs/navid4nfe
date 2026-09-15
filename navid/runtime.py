"""Request-scoped H200 optimization and coarse sequence compilation."""
from __future__ import annotations

import math
import os
import time

from .dit_cache import ResidualCache
from .options import BUCKET, RequestRejected, resolve_optimization
from .startup import dit_compile_enabled


def dense_attention(q, k, v):
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
        dropout_p=0.0, is_causal=False).squeeze(0).transpose(0, 1)


def target_video_start(indices):
    breaks = (indices[1:] - indices[:-1] != 1).nonzero()
    start = int(breaks[-1].item()) + 1 if breaks.numel() else 0
    return int(indices[start].item())


class RequestRuntime:
    native_int8_qkv = False

    def __init__(self, engine):
        import torch
        import torch.distributed as dist
        import triton
        from h3_runtime import ulysses
        from h3_runtime.parallel_hooks import with_cp_reapplied

        # Triton host TensorDescriptors need device scratch storage on Hopper.
        triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.uint8))
        self.engine = engine
        self.compile_enabled = dit_compile_enabled()
        self.compiles = 0
        self.compile_s = 0.0
        self.seen_shapes = set()
        self.gated_shapes = set()

        def reduce_sums(sums):
            if dist.is_initialized():
                dist.all_reduce(sums)
            return sums

        self.cache = ResidualCache(reduce_sums)
        # One stable attention callable; request overrides never repatch modules.
        ulysses.install(engine.transformer, attention_fn=self)

        def compile_backend(graph, inputs):
            from torch._dynamo.backends.registry import lookup_backend
            started = time.perf_counter()
            compiled = lookup_backend("inductor")(graph, inputs)
            self.compiles += 1
            self.compile_s += time.perf_counter() - started
            print(f"DIT_COMPILE: rank={engine.rank} graph={self.compiles} elapsed={time.perf_counter()-started:.2f}s", flush=True)
            return compiled

        def install_blocks():
            blocks = engine.transformer.transformer_blocks
            for index, block in enumerate(blocks):
                # Collectives and runtime lengths stay eager. The surrounding
                # numerical work compiles at padded local rows, never prompt IDs.
                if self.compile_enabled:
                    block.attn.forward = torch.compiler.disable(block.attn.forward)
                    numerical = torch.compile(block.forward, backend=compile_backend, dynamic=False)
                else:
                    numerical = block.forward

                def forward(hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None,
                            _index=index, _run=numerical):
                    self.layer = _index
                    enabled = self.cache.options["enabled"]
                    if enabled and _index > 0 and self.cache.skip:
                        result = hidden_states
                        self.cache.blocks_reused += 1
                    else:
                        snapshot = hidden_states.clone() if enabled and _index == 0 else None
                        result = _run(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
                        self.cache.blocks_computed += 1
                        if enabled and _index == 0:
                            local_rows = result.shape[1]
                            valid = max(0, min(local_rows, self.logical_tokens - engine.rank * local_rows))
                            self.cache.probe(snapshot, result, self.step, valid)
                    if enabled and _index == len(blocks) - 1:
                        result = self.cache.finish_tail(result)
                    return result

                block.forward = forward

        with_cp_reapplied(engine.transformer, install_blocks)
        engine.transformer.register_forward_pre_hook(self.before_transformer, with_kwargs=True)
        engine.transformer.register_forward_hook(self.after_transformer)
        engine.request_runtime = self

    def begin(self, options: dict | None, duration: int):
        import torch

        options = resolve_optimization(options)
        self.sol = options["sol_attn"]
        self.cache.reset(options["cache_dit"])
        self.duration = duration
        self.step = -1
        self.layer = 0
        self.logical_tokens = self.capacity = 0
        self.sparse_calls = self.dense_calls = 0
        self.density = None
        self.shape_seen = False
        self.compiles_before, self.compile_s_before = self.compiles, self.compile_s
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        self.events[0].record()

    def before_transformer(self, _module, args, kwargs):
        import torch
        import torch.nn.functional as F

        if args:
            raise RuntimeError("Expected the pinned Diffusers transformer keyword call")
        self.step += 1
        logical = kwargs["position_ids"].shape[0]
        capacity = ((logical + BUCKET - 1) // BUCKET) * BUCKET
        if capacity > int(os.environ.get("MAX_PACKED_TOKENS", "262144")):
            raise RequestRejected(f"Actual packed sequence {capacity} exceeds MAX_PACKED_TOKENS")
        if self.step == 0:
            self.events[1].record()
            self.logical_tokens, self.capacity = logical, capacity
            self.video_start = target_video_start(kwargs["video_indices"])
            self.sink_tokens = self.video_start if self.sol["sink_conditioning"] != "off" else 0
            # Target video positions are monotonic in latent time. Convert the
            # requested wall-clock prefix to a count of target latent frames.
            positions = kwargs["position_ids"][self.video_start:, 0]
            times = torch.unique_consecutive(positions)
            dense_frames = min(times.numel(), math.ceil(
                times.numel() * self.sol["dense_prefix_seconds"] / self.duration))
            self.dense_video_rows = int((positions <= times[dense_frames - 1]).sum().item()) if dense_frames else 0
            shape = (capacity, kwargs["hidden_states"].dtype, self.sol["enabled"])
            self.shape_key = shape
            self.shape_seen = shape in self.seen_shapes
            print(f"EXECUTION: rank={self.engine.rank} tokens={logical} bucket={capacity} "
                  f"shape_seen={self.shape_seen} sol={self.sol} cache={self.cache.options}", flush=True)
        elif logical != self.logical_tokens:
            raise RuntimeError("Packed layout changed within a request")
        if bool((kwargs["token_tags"] < 0).any()):
            raise RuntimeError("Unexpected upstream padding; bucket attention expects a contiguous live sequence")
        if capacity != logical:
            kwargs = dict(kwargs)
            # Placeholder tag 0 avoids allocating an O(S^2) upstream mask.
            # The attention callable physically excludes these rows from Q/K/V.
            for name in ("token_tags", "timestep_indices"):
                kwargs[name] = F.pad(kwargs[name], (0, capacity - logical), value=0)
            kwargs["position_ids"] = F.pad(kwargs["position_ids"], (0, 0, 0, capacity - logical))
        return args, kwargs

    def after_transformer(self, _module, _args, _output):
        self.events[2].record()

    def __call__(self, q, k, v):
        import torch

        tokens = self.logical_tokens
        ql, kl, vl = q[:tokens], k[:tokens], v[:tokens]
        use_sol = (self.sol["enabled"] and self.step >= self.sol["dense_steps"]
                   and self.dense_video_rows < tokens - self.video_start)
        if not use_sol:
            result = dense_attention(ql, kl, vl)
            self.dense_calls += 1
        else:
            from h3_runtime.sparse_attention import (
                _agree,
                _estimate_density,
                _import_kernel,
            )

            _import_kernel()  # establishes the vendored sol_attn module path
            from sol_attn.triton_ref import sol_attn

            qb, kb, vb = (x.unsqueeze(0) for x in (ql, kl, vl))
            gate_key = (self.capacity, q.shape[1])
            if gate_key not in self.gated_shapes:
                got = sol_attn(qb, kb, vb, tau=-1000.0, compile_bucket_size=BUCKET)
                want = dense_attention(ql, kl, vl).unsqueeze(0)
                diff = got.float() - want.float()
                rel = float((diff.norm() / want.float().norm().clamp_min(1e-12)).item())
                passed = _agree(bool(torch.isfinite(got).all()) and rel < 0.01)
                if not passed:
                    raise RuntimeError(f"H200 Sol dense-limit correctness gate failed: rel_l2={rel}")
                self.gated_shapes.add(gate_key)
                print(f"SOL_GATE: rank={self.engine.rank} backend=triton_tma_sm90 rel_l2={rel:.6f}", flush=True)
            result = sol_attn(qb, kb, vb, tau=self.sol["tau"], sink_start=0,
                              sink_tokens=self.sink_tokens, compile_bucket_size=BUCKET)[0]
            if self.sol["sink_conditioning"] == "exact_kv_and_rows":
                result[:self.video_start] = dense_attention(ql[:self.video_start], kl, vl)
            if self.dense_video_rows:
                lo, hi = self.video_start, self.video_start + self.dense_video_rows
                result[lo:hi] = dense_attention(ql[lo:hi], kl, vl)
            self.sparse_calls += 1
            if self.density is None:
                # Sample one head once per request; never materialize all heads'
                # block routing matrices just for diagnostics.
                self.density = _estimate_density(qb[:, :, :1], kb[:, :, :1], vb[:, :, :1],
                    tau=self.sol["tau"], thresh_type="diag", sink_start=0,
                    sink_tokens=self.sink_tokens, compile_bucket_size=BUCKET)
        if tokens == q.shape[0]:
            return result
        output = torch.zeros_like(q)
        output[:tokens] = result
        return output

    def finish(self):
        import torch

        self.events[3].record()
        torch.cuda.synchronize(self.engine.device)
        self.seen_shapes.add(self.shape_key)
        new_compiles = self.compiles - self.compiles_before
        result = {
            "sol_attn": {**self.sol, "backend": "triton_tma_sm90" if self.sparse_calls else "dense",
                         "sparse_calls": self.sparse_calls, "dense_calls": self.dense_calls,
                         "route_density_first_call_head0": self.density},
            "cache_dit": self.cache.stats(),
            "compile": {"enabled": self.compile_enabled, "bucket_alignment": BUCKET,
                        "logical_tokens": self.logical_tokens, "padded_tokens": self.capacity,
                        "shape_seen": self.shape_seen, "inductor_invocations": new_compiles,
                        "inductor_compile_s": round(self.compile_s - self.compile_s_before, 3),
                        "graph_reused": self.compile_enabled and self.shape_seen and new_compiles == 0},
            "timing": {"conditioning_gpu_s": self.events[0].elapsed_time(self.events[1]) / 1000,
                       "denoising_gpu_s": self.events[1].elapsed_time(self.events[2]) / 1000,
                       "decode_gpu_s": self.events[2].elapsed_time(self.events[3]) / 1000},
        }
        self.cache.release()
        return result
