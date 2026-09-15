from __future__ import annotations

import os
import time
import traceback
from datetime import timedelta

from . import config
from .media import verify_output
from .store import Store


def state(phase: str, **extra) -> None:
    config.atomic_json(config.RUNTIME / "worker.json",
                       {"run_id": config.RUN_ID, "phase": phase, "since": time.time(), **extra})


def main() -> None:
    import torch
    import torch.distributed as dist
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
    from h3_runtime import MiniMaxH3Inference
    from h3_runtime.engine import DURATION_FRAMES, INFERENCE_STEPS
    from PIL import Image, ImageDraw

    from .h200 import configure

    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 8:
        raise RuntimeError("This deployment requires exactly eight GPU ranks")
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability(rank) != (9, 0) or "H200" not in torch.cuda.get_device_name(rank):
        raise RuntimeError("This deployment profile requires NVIDIA H200 (SM90)")
    if INFERENCE_STEPS != 5:
        raise RuntimeError("Expected five scheduler points / four DiT forwards")
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank),
                            timeout=timedelta(seconds=int(os.environ.get("TASK_TIMEOUT", "1800"))))
    configure()
    store = Store(config.DATA) if rank == 0 else None
    if rank == 0:
        state("loading")

    engine = MiniMaxH3Inference(str(config.MODEL), config.ADAPTER, attention_backend="dense",
                               task="ref2va", compute_quant="none", reference_image_resize_mode="match")

    # Smoke test the same Ref2VA path as production, including reference encoding,
    # four actual DiT calls, eight-rank communication, VAE and MP4/audio encoding.
    smoke_image = config.RUNTIME / "warmup-reference.png"
    if rank == 0:
        image = Image.new("RGB", (768, 768), "#d7e4ec")
        draw = ImageDraw.Draw(image)
        draw.ellipse((200, 180, 570, 550), fill="#d07b37")
        draw.rectangle((110, 555, 660, 620), fill="#604531")
        image.save(smoke_image)
        state("warming")
    dist.barrier()
    calls = [0]

    def count_forward(_module, _args):
        calls[0] += 1

    hook = engine.transformer.register_forward_pre_hook(count_forward)
    media = engine.generate(
        "A continuous shot of the round orange object on the table. The camera slowly moves closer. Quiet room ambience.",
        duration=5, seed=42, references=[MiniMaxH3Reference(image=str(smoke_image))])
    hook.remove()
    counts = [None] * 8
    dist.all_gather_object(counts, calls[0])
    if counts != [4] * 8:
        raise RuntimeError(f"Four-step verification failed: per-rank DiT calls={counts}")
    if rank == 0:
        smoke_output = config.RUNTIME / "warmup.mp4"
        media.save(smoke_output)
        verify_output(smoke_output, expected_frames=DURATION_FRAMES[5])
        print(f"WARMUP_OK: 8 x H200; Ref2VA; DiT calls={counts}; MP4 video+audio", flush=True)
    del media
    dist.barrier()
    if rank == 0:
        state("ready", verified_nfe=counts)

    while True:
        # Only rank zero accesses the queue. Broadcast only trusted, locally
        # validated jobs; all ranks execute exactly the same collective sequence.
        if rank == 0:
            job = store.claim()
            if job is None:
                time.sleep(0.25)
            else:
                state("busy", job_id=job["id"])
        else:
            job = None
        message = [job]
        dist.broadcast_object_list(message, src=0)
        job = message[0]
        if job is None:
            continue
        output = config.DATA / "outputs" / f"{job['id']}.mp4"
        temporary = output.with_suffix(".partial.mp4")
        try:
            references = [MiniMaxH3Reference(**{r["kind"]: r["path"]}) for r in job["references"]]
            media = engine.generate(job["prompt"], duration=job["duration"], seed=job["seed"], references=references)
            if rank == 0:
                media.save(temporary)
                verify_output(temporary, expected_frames=DURATION_FRAMES[job["duration"]])
                temporary.replace(output)
                store.finish(job["id"], output=str(output), inference_s=round(media.elapsed_s, 3))
                print(f"TASK_OK: {job['id']} inference_s={media.elapsed_s:.3f}", flush=True)
            del media
            dist.barrier()
            if rank == 0:
                state("ready")
        except Exception as error:
            temporary.unlink(missing_ok=True)
            if rank == 0:
                store.finish(job["id"], error=str(error)[:2000])
                state("failed", error=str(error)[:2000])
            # A failed collective/CUDA context is not safe to reuse. Torchrun and
            # the supervisor terminate every rank and mark pending jobs failed.
            raise


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
