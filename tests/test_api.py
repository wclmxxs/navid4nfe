from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from navid import config
from navid.store import Store


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    monkeypatch.setattr(config, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(config, "RUN_ID", "test-run")
    config.initialize_dirs()
    (config.RUNTIME / "api.key").write_text("test-key\n")
    from navid import api
    monkeypatch.setattr(api, "store", Store(config.DATA))
    config.atomic_json(config.RUNTIME / "supervisor.json",
                       {"run_id": "test-run", "phase": "running", "heartbeat": time.time()})
    config.atomic_json(config.RUNTIME / "worker.json", {"run_id": "test-run", "phase": "ready"})
    with TestClient(api.app, headers={"X-API-Key": "test-key"}) as client:
        yield api, client


def upload_image(client):
    data = io.BytesIO()
    Image.new("RGB", (128, 128), "red").save(data, format="PNG")
    response = client.post("/v1/references?kind=image", content=data.getvalue())
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_submit_query_and_download(api):
    module, client = api
    ref = upload_image(client)
    response = client.post("/v1/videos", json={"prompt": "An orange ball rolls", "duration": 5, "seed": 7, "references": [ref]})
    assert response.status_code == 202
    job_id = response.json()["id"]
    assert client.get(f"/v1/videos/{job_id}/content").status_code == 409
    job = module.store.claim()
    assert job["seed"] == 7
    assert job["references"][0]["kind"] == "image"
    # Test the API handoff, not GPU synthesis; output bytes are an explicit fixture.
    output = config.DATA / "outputs" / f"{job_id}.mp4"
    output.write_bytes(b"fixture-video")
    module.store.finish(job_id, output=str(output), inference_s=12.5)
    result = client.get(f"/v1/videos/{job_id}").json()
    assert result["status"] == "succeeded"
    assert result["nfe"] == 4
    assert "output" not in result
    assert client.get(result["content_url"]).content == b"fixture-video"


def test_auth_and_readiness(api):
    _, client = api
    assert client.get("/readyz").status_code == 200
    assert client.post("/v1/videos", headers={"X-API-Key": "wrong"}, json={}).status_code == 401
    config.atomic_json(config.RUNTIME / "worker.json", {"run_id": "old-run", "phase": "ready"})
    assert client.get("/readyz").status_code == 503
    assert client.post("/v1/videos", json={"prompt": "p", "references": ["r"]}).status_code == 503
    config.atomic_json(config.RUNTIME / "worker.json", {"run_id": "test-run", "phase": "ready"})
    config.atomic_json(config.RUNTIME / "supervisor.json",
                       {"run_id": "test-run", "phase": "running", "heartbeat": time.time() - 20})
    assert client.get("/readyz").status_code == 503


def test_reject_bad_input_before_gpu(api, monkeypatch):
    module, client = api
    assert client.post("/v1/references?kind=image", content=b"not an image").status_code == 422
    assert not list((config.DATA / "references").iterdir())
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    assert client.post("/v1/references?kind=image", content=b"x" * (1024**2 + 1)).status_code == 413
    assert not list((config.DATA / "references").iterdir())
    ref = upload_image(client)
    for payload in [
        {"prompt": " ", "references": [ref]},
        {"prompt": "p", "references": ["missing"]},
        {"prompt": "p", "references": [ref], "duration": 6},
        {"prompt": "p", "references": [ref], "steps": 50},
        {"prompt": "p", "references": [ref], "seed": -1},
        {"prompt": "p", "references": [ref] * 10},
    ]:
        assert client.post("/v1/videos", json=payload).status_code == 422
    audio = config.DATA / "references" / "audio"
    audio.write_bytes(b"fixture")
    module.store.add_reference("audio", "audio", audio)
    assert client.post("/v1/videos", json={"prompt": "p", "references": ["audio"]}).status_code == 422
    assert module.store.claim() is None


def test_full_queue_and_failed_task(api, monkeypatch):
    module, client = api
    monkeypatch.setenv("MAX_QUEUE", "1")
    ref = upload_image(client)
    payload = {"prompt": "p", "references": [ref]}
    job_id = client.post("/v1/videos", json=payload).json()["id"]
    assert client.post("/v1/videos", json=payload).status_code == 429
    module.store.fail_pending("GPU worker exited")
    result = client.get(f"/v1/videos/{job_id}").json()
    assert result["status"] == "failed"
    assert result["error"] == "GPU worker exited"
    assert client.get(f"/v1/videos/{job_id}/content").status_code == 409
    assert client.post("/v1/videos", json=payload).status_code == 202
