import pytest

from navid.startup import (
    GIB,
    available_memory_bytes,
    dit_compile_enabled,
    loading_parallelism,
)


def test_parallelism_scales_with_free_ram_and_respects_overrides():
    assert loading_parallelism(1946) == 8
    assert loading_parallelism(1000) == 4
    assert loading_parallelism(500) == 2
    assert loading_parallelism(256) == 1
    assert loading_parallelism(160) == 1
    assert loading_parallelism(1946, "1") == 1
    assert loading_parallelism(1946, "4") == 4
    with pytest.raises(RuntimeError, match="loading budget"):
        loading_parallelism(500, "8")
    with pytest.raises(RuntimeError, match="160 GiB"):
        loading_parallelism(159)
    with pytest.raises(ValueError, match="MODEL_LOAD_PARALLELISM"):
        loading_parallelism(2000, "3")


def test_container_limit_caps_large_host_ram_including_parent(tmp_path):
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (cgroup / "parent/child").mkdir(parents=True)
    (proc / "meminfo").write_text(f"MemAvailable: {2000 * 1024**2} kB\n")
    (proc / "self/cgroup").write_text("0::/parent/child\n")
    (cgroup / "parent/child/memory.max").write_text("max")
    (cgroup / "parent/memory.max").write_text(str(550 * GIB))
    (cgroup / "parent/memory.current").write_text(str(50 * GIB))
    (cgroup / "memory.max").write_text("max")
    available = available_memory_bytes(proc, cgroup) / GIB
    assert available == 500
    assert loading_parallelism(available) == 2


def test_compile_defaults_on_but_keeps_explicit_disable(monkeypatch):
    monkeypatch.delenv("DIT_COMPILE", raising=False)
    assert dit_compile_enabled()
    monkeypatch.setenv("DIT_COMPILE", "0")
    assert not dit_compile_enabled()
    monkeypatch.setenv("DIT_COMPILE", "unexpected")
    with pytest.raises(ValueError, match="DIT_COMPILE"):
        dit_compile_enabled()
