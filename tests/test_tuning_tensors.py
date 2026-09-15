import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from navid.dit_cache import ResidualCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/sol_h3"))
from h3_runtime.output import prepare_output


def test_exact_frames_canvas_and_stereo_audio():
    video = torch.rand(107, 3, 32, 64)
    audio = torch.rand(2, 150000)
    result, sound = prepare_output(video, audio, 32000, 4, (64, 30))
    assert result.shape == (96, 3, 30, 64)
    assert sound.shape == (2, 128000)
    torch.testing.assert_close(sound, audio[:, :128000])
    assert video.shape[0] == 107
    _, transposed = prepare_output(video, audio.T, 32000, 4, (64, 32))
    torch.testing.assert_close(transposed, audio.T[:128000])


def test_cache_probe_reuse_limit_and_last_step():
    cache = ResidualCache()
    cache.reset({"enabled": True, "warmup": 1, "rdt": 0.1, "max_continuous_cached_steps": 1})
    for step in range(4):
        before = torch.full((1, 8, 4), step * 1.0)
        after = before + 1
        cache.probe(before, after, step, 8)
        result = cache.finish_tail(after if cache.skip else after + 3)
        torch.testing.assert_close(result, after + 3)
    assert [d["cached"] for d in cache.decisions] == [False, True, False, False]
    cache.release()
    assert cache.tail_residual is None
    cache.reset({"enabled": True, "warmup": 4, "rdt": 1, "max_continuous_cached_steps": 2})
    assert not cache.decisions and cache.previous_probe is None


def test_cache_padding_excluded_and_zero_denominator_is_finite():
    cache = ResidualCache()
    cache.reset({"enabled": True, "warmup": 1, "rdt": 0.1, "max_continuous_cached_steps": 2})
    x = torch.zeros(1, 8, 4)
    cache.probe(x, x, 0, 3)
    cache.finish_tail(x + 3)
    y = x.clone()
    y[:, 3:] = float("nan")
    cache.probe(x, y, 1, 3)
    assert cache.skip
    cache.probe(x, x + 1, 2, 3)
    assert not cache.skip


def test_cache_all_ranks_use_global_sums():
    # One shard changes strongly while the other barely changes. A global
    # reduction must force both to recompute, independent of their local RDT.
    caches = [ResidualCache(lambda sums: torch.tensor([10.0, 20.0])) for _ in range(2)]
    for cache in caches:
        cache.reset({"enabled": True, "warmup": 1, "rdt": 0.1, "max_continuous_cached_steps": 1})
        x = torch.zeros(1, 4, 4)
        cache.probe(x, x + 1, 0, 4)
        cache.finish_tail(x + 3)
        cache.probe(x, x + 1.001, 1, 4)
        assert not cache.skip
        assert cache.decisions[-1]["relative_change"] == 0.5


@pytest.mark.parametrize("nfe", (4, 8))
def test_cache_protects_final_step_for_each_profile_and_resets(nfe):
    cache = ResidualCache()
    cache.reset({"enabled": True, "warmup": 1, "rdt": 1,
                 "max_continuous_cached_steps": nfe - 2}, total_steps=nfe)
    for step in range(nfe):
        before = torch.full((1, 4, 2), float(step))
        after = before + 1
        cache.probe(before, after, step, 4)
        output = cache.finish_tail(after if cache.skip else after + 3)
        torch.testing.assert_close(output, after + 3)
    assert [d["cached"] for d in cache.decisions] == [False] + [True] * (nfe - 2) + [False]
    assert cache.stats()["nfe"] == nfe
    cache.reset({"enabled": True, "warmup": nfe, "rdt": 1,
                 "max_continuous_cached_steps": 1}, total_steps=nfe)
    for step in range(nfe):
        x = torch.zeros(1, 4, 2)
        cache.probe(x, x + 1, step, 4)
        cache.finish_tail(x + 4)
    assert not any(d["cached"] for d in cache.decisions)


def test_disabled_compile_and_cache_execute_every_block(monkeypatch):
    from types import SimpleNamespace

    import h3_runtime
    from navid.runtime import RequestRuntime

    monkeypatch.setenv("DIT_COMPILE", "0")
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(set_allocator=lambda allocator: None))
    installed = []
    monkeypatch.setattr(h3_runtime, "ulysses", SimpleNamespace(
        install=lambda transformer, attention_fn: installed.append(attention_fn)), raising=False)
    monkeypatch.setitem(sys.modules, "h3_runtime.parallel_hooks", SimpleNamespace(
        with_cp_reapplied=lambda transformer, install: install()))

    def unexpected_compile(*args, **kwargs):
        pytest.fail("DIT_COMPILE=0 must not invoke torch.compile")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    calls = []

    def numerical(hidden_states, *args):
        calls.append(1)
        return hidden_states + 1

    blocks = [SimpleNamespace(forward=numerical, attn=SimpleNamespace(forward=numerical)) for _ in range(3)]
    transformer = SimpleNamespace(transformer_blocks=blocks,
        register_forward_pre_hook=lambda *args, **kwargs: None,
        register_forward_hook=lambda *args, **kwargs: None)
    runtime = RequestRuntime(SimpleNamespace(transformer=transformer, rank=0))
    assert installed == [runtime]  # Attention remains installed without DiT compilation.
    runtime.cache.skip = True  # A stale skip flag cannot bypass blocks when caching is off.
    for step in range(4):
        result = torch.zeros(1, 2, 3)
        for block in blocks:
            result = block.forward(result, None, None, None)
        torch.testing.assert_close(result, torch.full_like(result, 3))
    assert len(calls) == runtime.cache.blocks_computed == 12
    assert runtime.cache.blocks_reused == 0
    assert runtime.cache.previous_probe is runtime.cache.tail_residual is None
    assert not runtime.compile_enabled and runtime.compiles == 0


def test_bucket_padding_never_participates_in_dense_attention():
    from types import SimpleNamespace

    from navid.options import SolOptions
    from navid.runtime import RequestRuntime, dense_attention

    runtime = RequestRuntime.__new__(RequestRuntime)
    runtime.engine = SimpleNamespace(rank=0)
    runtime.step = -1
    runtime.events = [SimpleNamespace(record=lambda: None)] * 4
    runtime.sol = SolOptions(enabled=False).model_dump()
    runtime.cache = ResidualCache()
    runtime.seen_shapes = set()
    runtime.duration = 8
    runtime.dense_calls = 0
    for logical in (79, 131):
        runtime.step = -1
        kwargs = {"position_ids": torch.zeros(logical, 3), "token_tags": torch.zeros(logical, dtype=torch.long),
                  "timestep_indices": torch.zeros(logical, dtype=torch.long),
                  "video_indices": torch.cat((torch.arange(10, 15), torch.arange(20, logical))),
                  "hidden_states": torch.zeros(1, 3, 8)}
        _, padded = runtime.before_transformer(None, (), kwargs)
        assert padded["position_ids"].shape[0] == 4096
        assert kwargs["position_ids"].shape[0] == logical  # no mutation of pipeline state
        packed = torch.randn(4096, 2, 24)
        # Deliberately poison the extra keys: masking must be semantic, not
        # merely assuming all padded K/V activations remain zero after blocks.
        packed[logical:] = float("nan")
        q, k, v = packed.split(8, dim=-1)
        output = runtime(q, k, v)
        torch.testing.assert_close(output[:logical], dense_attention(q[:logical], k[:logical], v[:logical]))
        assert torch.isfinite(output).all()
        assert not output[logical:].any()


@pytest.mark.parametrize("parallelism", [1, 2, 4, 8])
def test_distributed_load_groups_bound_concurrency_and_load_every_rank(parallelism):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from h3_runtime.loading import load_in_groups

    barrier = threading.Barrier(8, timeout=5)
    group_barriers = [threading.Barrier(parallelism, timeout=5) for _ in range(8 // parallelism)]
    lock = threading.Lock()
    active = set()
    seen = []
    peaks = []

    def rank_main(rank):
        def load():
            with lock:
                active.add(rank)
                peaks.append(len(active))
                assert len({r // parallelism for r in active}) == 1
                seen.append(rank)
            group_barriers[rank // parallelism].wait()
            with lock:
                active.remove(rank)

        load_in_groups(load, rank=rank, world_size=8, parallelism=parallelism, barrier=barrier.wait)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(rank_main, range(8)))
    assert sorted(seen) == list(range(8))
    assert max(peaks) == parallelism
