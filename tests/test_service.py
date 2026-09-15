import os

from navid import config, service
from navid.store import Store


def test_dead_worker_stops_api_and_fails_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    config.initialize_dirs()
    monkeypatch.setattr(service, "process_stamp", lambda _pid: "test-stamp")
    monkeypatch.setattr(service.signal, "signal", lambda *_: None)
    children = []
    cleaned = []
    jobs = []

    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(children)
            self.worker = "torch.distributed.run" in command
            children.append(self)
            if self.worker:
                store = Store(config.DATA)
                jobs.append(store.enqueue({"duration": 5, "seed": 1}, 1))

        def poll(self):
            return 7 if self.worker else None

    monkeypatch.setattr(service.subprocess, "Popen", Child)
    monkeypatch.setattr(service, "terminate", lambda processes: cleaned.extend(processes))
    assert service.serve() == 1
    assert cleaned == children and len(children) == 2
    assert Store(config.DATA).get(jobs[0])["status"] == "failed"
    assert config.read_json(config.RUNTIME / "supervisor.json")["phase"] == "stopped"
    assert not config.read_json(config.RUNTIME / "pid.json")


def test_pid_reuse_is_not_a_running_service(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME", tmp_path)
    config.atomic_json(tmp_path / "pid.json", {"pid": os.getpid(), "stamp": "old"})
    monkeypatch.setattr(service, "process_stamp", lambda _pid: "new")
    assert service.running() is None


def test_probe_uses_configured_bind_host():
    assert service.endpoint({"host": "0.0.0.0", "port": 8000}) == "http://127.0.0.1:8000"
    assert service.endpoint({"host": "10.0.0.1", "port": 8000}) == "http://10.0.0.1:8000"
    assert service.endpoint({"host": "::", "port": 8000}) == "http://[::1]:8000"
