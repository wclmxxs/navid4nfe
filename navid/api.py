from __future__ import annotations

import asyncio
import hmac
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from . import config
from .media import validate_reference
from .store import QueueFull, Store

config.initialize_dirs()
store = Store(config.DATA)
app = FastAPI(title="Navid 4 NFE · H200 Ref2VA", version="1.0.0")
upload_slots = asyncio.Semaphore(2)


def health() -> tuple[bool, dict]:
    supervisor = config.read_json(config.RUNTIME / "supervisor.json")
    worker = config.read_json(config.RUNTIME / "worker.json")
    alive = (bool(config.RUN_ID) and supervisor.get("run_id") == config.RUN_ID
             and supervisor.get("phase") == "running"
             and time.time() - supervisor.get("heartbeat", 0) < 10)
    ready = alive and worker.get("run_id") == config.RUN_ID and worker.get("phase") in {"ready", "busy"}
    return ready, {"ready": ready, "phase": worker.get("phase", "starting") if alive else "unavailable",
                   "task": "ref2va", "nfe": 4, "gpus": 8, "attention": "dense", "compute": "bf16"}


def authorize(x_api_key: str | None = Header(default=None),
              authorization: str | None = Header(default=None)) -> None:
    expected = (config.RUNTIME / "api.key").read_text().strip()
    supplied = x_api_key or (authorization[7:] if authorization and authorization.startswith("Bearer ") else "")
    if not supplied or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(401, "Provide X-API-Key or Authorization: Bearer <key>")


@app.get("/healthz")
@app.get("/readyz")
def readiness():
    ready, value = health()
    return JSONResponse(value, status_code=200 if ready else 503)


@app.post("/v1/references", dependencies=[Depends(authorize)], status_code=201)
async def upload_reference(request: Request, kind: Literal["image", "video", "audio"]):
    """Upload raw file bytes (curl --data-binary @file), not multipart."""
    ref_id = uuid.uuid4().hex
    path = config.DATA / "references" / ref_id
    limit = int(os.environ.get("MAX_UPLOAD_MB", "64")) * 1024 * 1024
    async with upload_slots:
        try:
            size = 0
            with path.open("xb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"Reference exceeds {limit // 1024 // 1024} MiB")
                    stream.write(chunk)
            if not size:
                raise HTTPException(422, "Empty reference file")
            try:
                await run_in_threadpool(validate_reference, path, kind)
            except Exception as error:
                raise HTTPException(422, f"Invalid {kind}: {error}") from error
            store.add_reference(ref_id, kind, path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return {"id": ref_id, "kind": kind, "bytes": size}


class VideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=32000)
    duration: Literal[5, 10, 15] = 5
    seed: int = Field(default_factory=lambda: secrets.randbits(32), ge=0, le=2**63 - 1, strict=True)
    references: list[str] = Field(min_length=1, max_length=12)


@app.post("/v1/videos", dependencies=[Depends(authorize)], status_code=202)
def submit(payload: VideoRequest):
    if not health()[0]:
        raise HTTPException(503, "Worker is not ready; check /readyz")
    if not payload.prompt.strip():
        raise HTTPException(422, "Prompt must not be blank")
    references = []
    for ref_id in payload.references:
        reference = store.reference(ref_id)
        if reference is None or not Path(reference["path"]).is_file():
            raise HTTPException(422, f"Unknown reference: {ref_id}")
        references.append({"kind": reference["kind"], "path": reference["path"]})
    kinds = [r["kind"] for r in references]
    if not {"image", "video"}.intersection(kinds):
        raise HTTPException(422, "Audio must be paired with an image or video")
    if kinds.count("image") > 9 or kinds.count("video") > 3 or kinds.count("audio") > 3:
        raise HTTPException(422, "At most 9 images, 3 videos and 3 audio references")
    try:
        job_id = store.enqueue({**payload.model_dump(), "references": references},
                               int(os.environ.get("MAX_QUEUE", "32")))
    except QueueFull as error:
        raise HTTPException(429, "Queue is full", headers={"Retry-After": "10"}) from error
    return {"id": job_id, "status": "queued", "status_url": f"/v1/videos/{job_id}"}


@app.get("/v1/videos/{job_id}", dependencies=[Depends(authorize)])
def get_job(job_id: str):
    result = store.get(job_id)
    if result is None:
        raise HTTPException(404, "Unknown task")
    result.pop("output")
    if result["status"] == "succeeded":
        result["content_url"] = f"/v1/videos/{job_id}/content"
    return result


@app.get("/v1/videos/{job_id}/content", dependencies=[Depends(authorize)])
def download(job_id: str):
    result = store.get(job_id)
    if result is None:
        raise HTTPException(404, "Unknown task")
    if result["status"] != "succeeded":
        raise HTTPException(409, f"Task is {result['status']}")
    if not result["output"] or not Path(result["output"]).is_file():
        raise HTTPException(410, "Output is no longer available")
    return FileResponse(result["output"], media_type="video/mp4", filename=f"{job_id}.mp4")
