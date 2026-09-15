import pytest
from pydantic import ValidationError

from navid.options import VideoRequest, native_frames, optimization_defaults, reference_size, resolve_optimization


def request(**kwargs):
    return VideoRequest(prompt="test", references=["ref"], **kwargs)


def test_all_integer_durations_and_native_alignment():
    for seconds in range(4, 16):
        execution = request(duration=seconds).execution()
        assert execution["output_frames"] == seconds * 24
        assert execution["native_frames"] % 17 == 5
        assert 0 <= execution["native_frames"] - seconds * 24 < 17
    assert native_frames(8) == 192
    for value in (3, 16, 4.5, 5.0, "8", True):
        with pytest.raises(ValidationError):
            request(duration=value)


def test_canvases_and_ambiguous_geometry():
    assert request().output_size() == (1344, 768)
    for size, canvas in [((1280, 720), (1280, 736)), ((1080, 1920), (1088, 1920)),
                         ((512, 512), (512, 512))]:
        item = request(width=size[0], height=size[1])
        assert item.output_size() == size
        assert item.inference_size() == canvas
    assert request(resolution=720, ratio="9:16").output_size() == (720, 1280)
    for kwargs in ({"width": 720}, {"width": 512, "height": 512, "resolution": 512},
                   {"width": 129, "height": 256}, {"width": 4096, "height": 2048},
                   {"reference_allow_upscale": True}):
        with pytest.raises(ValidationError):
            request(**kwargs)


def test_reference_resize_does_not_expand_small_sources():
    assert reference_size(1080, 1900) == (768, 1344)
    assert reference_size(1080, 1900, 512) == (512, 896)
    assert reference_size(256, 512, 512) == (256, 512)
    assert reference_size(256, 512, 512, True) == (512, 1024)


def test_partial_optimization_defaults_and_request_isolation(monkeypatch):
    monkeypatch.setenv("SOL_ATTN_ENABLED", "1")
    monkeypatch.setenv("SOL_ATTN_TAU", "1.5")
    monkeypatch.setenv("CACHE_DIT_RDT", "0.12")
    override = request(optimization={"sol_attn": {"enabled": False}, "cache_dit": {"enabled": True}})
    actual = override.resolved_optimization()
    assert actual["sol_attn"]["enabled"] is False
    assert actual["sol_attn"]["tau"] == 1.5
    assert actual["cache_dit"]["rdt"] == 0.12
    assert request().resolved_optimization()["sol_attn"]["enabled"] is True
    assert request().resolved_optimization()["cache_dit"]["enabled"] is False
    for options in ({"sol_attn": {"dense_steps": 5}}, {"cache_dit": {"rdt": float("nan")}},
                    {"cache_dit": {"warmup": 0}}, {"sol_attn": {"tau": -1}}):
        with pytest.raises(ValidationError):
            request(optimization=options)


def test_sol_only_defaults_match_api_warmup_and_runtime(monkeypatch):
    for name in ("SOL_ATTN_ENABLED", "SOL_ATTN_TAU", "SOL_ATTN_DENSE_STEPS", "CACHE_DIT_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    expected = optimization_defaults()
    assert expected["sol_attn"]["enabled"] is True
    assert expected["sol_attn"]["tau"] == 1.5
    assert expected["sol_attn"]["dense_steps"] == 1
    assert expected["sol_attn"]["sink_conditioning"] == "exact_kv_and_rows"
    assert expected["cache_dit"]["enabled"] is False
    assert request().execution()["optimization"] == resolve_optimization(None) == expected
    monkeypatch.setenv("SOL_ATTN_TAU", "1.8")
    partial = {"sol_attn": {"enabled": False}}
    assert resolve_optimization(partial) == request(optimization=partial).resolved_optimization()
    assert resolve_optimization(partial)["sol_attn"]["tau"] == 1.8
    assert resolve_optimization(None)["sol_attn"]["enabled"] is True
