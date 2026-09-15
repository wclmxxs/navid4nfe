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


def test_bucket_padding_never_participates_in_dense_attention():
    from types import SimpleNamespace

    from navid.options import SolOptions
    from navid.runtime import RequestRuntime, dense_attention

    runtime = RequestRuntime.__new__(RequestRuntime)
    runtime.engine = SimpleNamespace(rank=0)
    runtime.step = -1
    runtime.events = [SimpleNamespace(record=lambda: None)] * 4
    runtime.sol = SolOptions().model_dump()
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
