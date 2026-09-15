import pytest

from navid import config, errors, prepare, worker


def test_original_error_survives_shutdown_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME", tmp_path)
    (tmp_path / "worker.log").write_text(
        "Loading GPU rank 0\nTraceback (most recent call last):\n"
        '  File "engine.py", line 215, in load\nRuntimeError: original load failure\n'
        + "rank received SIGTERM\n" * 200
    )
    report = errors.failure_report()
    assert "RuntimeError: original load failure" in report


def test_current_run_does_not_report_old_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME", tmp_path)
    old = "Traceback (most recent call last):\nRuntimeError: OLD ERROR\n"
    (tmp_path / "worker.log").write_text(old + "Traceback (most recent call last):\nValueError: CURRENT ERROR\n")
    config.atomic_json(tmp_path / "last-run.json", {"run_id": "current", "log_offsets": {"worker.log": len(old)}})
    report = errors.failure_report()
    assert "CURRENT ERROR" in report
    assert "OLD ERROR" not in report


def test_worker_records_and_reraises_original_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME", tmp_path)
    monkeypatch.setattr(config, "RUN_ID", "current")
    monkeypatch.setenv("RANK", "3")
    config.atomic_json(tmp_path / "last-run.json", {"run_id": "current"})

    def fail():
        raise ValueError("specific model error")

    monkeypatch.setattr(worker, "main", fail)
    with pytest.raises(ValueError, match="specific model error"):
        worker.run_worker()
    record = config.read_json(tmp_path / "worker-errors/current/rank-3.json")
    assert record["type"] == "ValueError"
    assert "ValueError: specific model error" in record["traceback"]
    assert "rank 3" in errors.failure_report()


def test_diagnostic_write_failure_keeps_original_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME", tmp_path)

    def unavailable(*args, **kwargs):
        raise OSError("disk full")

    def fail():
        raise RuntimeError("original CUDA failure")

    monkeypatch.setattr(config, "atomic_json", unavailable)
    monkeypatch.setattr(worker, "main", fail)
    with pytest.raises(RuntimeError, match="original CUDA failure"):
        worker.run_worker()


def test_oom_report_preserves_gpu_process_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME", tmp_path)
    monkeypatch.setattr(config, "RUN_ID", "oom-run")
    monkeypatch.setenv("RANK", "0")
    config.atomic_json(tmp_path / "last-run.json", {"run_id": "oom-run"})
    monkeypatch.setattr(prepare, "gpu_process_inventory", lambda: "GPU-abc, 179841, python, 44144 MiB")

    class OutOfMemoryError(RuntimeError):
        pass

    def fail():
        raise OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(worker, "main", fail)
    with pytest.raises(OutOfMemoryError):
        worker.run_worker()
    report = errors.failure_report()
    assert "CUDA out of memory" in report
    assert "179841, python, 44144 MiB" in report
