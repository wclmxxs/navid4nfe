from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from . import config

COMPONENTS = ("audio_scheduler", "audio_vae", "processor", "scheduler", "text_encoder", "tokenizer", "vae", "transformer_ref")
MIN_FREE_GPU_GIB = 125


def gpu_process_inventory() -> str:
    lines = ["GPU process inventory (read-only; no processes will be stopped):"]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        lines.append(result.stdout.strip() or result.stderr.strip() or "No compute processes reported.")
    except (OSError, subprocess.TimeoutExpired) as error:
        lines.append(f"Could not query GPU processes: {error}")
    lines.append("If memory is occupied but no owner is listed, run nvidia-smi on the host outside the container.")
    return "\n".join(lines)


def print_gpu_processes() -> None:
    print("\n" + gpu_process_inventory(), flush=True)


def check_gpus(cuda) -> None:
    if cuda.device_count() != 8:
        print_gpu_processes()
        raise RuntimeError(f"Expected 8 visible GPUs, got {cuda.device_count()}")
    check_gpu_memory(cuda, range(8))


def check_gpu_memory(cuda, indices) -> None:
    # During serial loading, earlier ranks already own model weights. Inspect
    # only the rank about to move its CPU model to CUDA in that case.
    issues = []
    print("GPU memory before model loading (GiB; indices follow CUDA_VISIBLE_DEVICES):", flush=True)
    for i in indices:
        props = cuda.get_device_properties(i)
        free, total = cuda.mem_get_info(i)
        print(f"GPU {i}: {props.name}; total={total / 1024**3:.1f}, "
              f"used={(total - free) / 1024**3:.1f}, free={free / 1024**3:.1f}; "
              f"required free>={MIN_FREE_GPU_GIB} GiB", flush=True)
        if "H200" not in props.name or (props.major, props.minor) != (9, 0):
            issues.append(f"GPU {i} must be H200 SM90, got {props.name}")
        if free < MIN_FREE_GPU_GIB * 1024**3:
            issues.append(f"GPU {i} has only {free / 1024**3:.1f} GiB free")
    if issues:
        print_gpu_processes()
        raise RuntimeError(
            "; ".join(issues) + ". Insufficient free memory before GPU model loading. "
            "This BF16/Ulysses profile keeps full model weights on each GPU; eight GPUs do not divide "
            "the weight memory by eight. Identify the listed workloads and stop the ones you intend "
            "to replace, then rerun ./deploy.sh. Installed dependencies will be reused."
        )


def gpucheck() -> None:
    import torch
    check_gpus(torch.cuda)


def validate_model(root: Path) -> None:
    for name in ("modular_model_index.json", "model_index.json"):
        json.loads((root / name).read_text())
    for name in COMPONENTS:
        directory = root / name
        if not directory.is_dir() or not list(directory.glob("*.json")):
            raise RuntimeError(f"Missing model component: {directory}")
        if name in {"audio_vae", "text_encoder", "vae", "transformer_ref"}:
            if not list(directory.glob("*.safetensors")):
                raise RuntimeError(f"No safetensors weights in {directory}")
            for index in directory.glob("*.safetensors.index.json"):
                for shard in set(json.loads(index.read_text())["weight_map"].values()):
                    path = (directory / shard).resolve()
                    if not path.is_relative_to(directory.resolve()) or not path.is_file() or not path.stat().st_size:
                        raise RuntimeError(f"Missing or invalid model shard: {path}")


def validate_adapter(path: Path) -> None:
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        if not keys or not any("lora_A" in k for k in keys) or any(k.endswith(".set_weight") for k in keys):
            raise RuntimeError("Expected the LightX2V Ref2VA four-step BF16 LoRA")


def download() -> None:
    from huggingface_hub import snapshot_download
    config.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("MODEL_DIR"):
        print("Downloading pinned Ref2VA checkpoint; interrupted transfers resume on the next run.", flush=True)
        snapshot_download("MiniMaxAI/MiniMax-H3", revision=config.MODEL_REVISION,
                          local_dir=config.MODEL, max_workers=4,
                          allow_patterns=["model_index.json", "modular_model_index.json", "LICENSE"]
                          + [f"{name}/*" for name in COMPONENTS])
    if not os.environ.get("ADAPTER_PATH"):
        snapshot_download("lightx2v/Minimax-h3-Turbo", revision=config.ADAPTER_REVISION,
                          local_dir=config.ADAPTER.parent, allow_patterns=[config.ADAPTER_NAME])
    validate_model(config.MODEL)
    validate_adapter(config.ADAPTER)
    config.atomic_json(config.RUNTIME / "checkpoints.json", {
        "model": str(config.MODEL), "model_revision": config.MODEL_REVISION if not os.environ.get("MODEL_DIR") else "local",
        "adapter": str(config.ADAPTER), "adapter_revision": config.ADAPTER_REVISION if not os.environ.get("ADAPTER_PATH") else "local"})
    print("Checkpoint layout and LoRA validation passed.", flush=True)


def preflight() -> None:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if torch.__version__ != "2.10.0+cu128":
        raise RuntimeError(f"Unexpected PyTorch build: {torch.__version__}")
    check_gpus(torch.cuda)
    # Require native CUDA 12.8 driver support: Triton compiles kernels at runtime.
    version = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0]
    if tuple(int(p) for p in version.split(".")[:2]) < (570, 26):
        raise RuntimeError(f"Driver {version} is too old; install NVIDIA driver >=570.26 for this CUDA 12.8 profile")
    torch.cuda.set_device(0)
    q = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.bfloat16)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        y = F.scaled_dot_product_attention(q, q, q)
    if not torch.isfinite(y).all():
        raise RuntimeError("Hopper flash SDPA smoke test failed")
    from h3_runtime.fusions import fused_swiglu
    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    expected = x[:, :128] * F.silu(x[:, 128:])
    torch.testing.assert_close(fused_swiglu(x), expected, rtol=0.02, atol=0.02)
    torch.cuda.synchronize()
    print("CUDA, native Flash SDPA and Triton kernel checks passed.", flush=True)
    available = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    memory_gib = int(available["MemAvailable"].split()[0]) / 1024**2
    if memory_gib < 160:
        raise RuntimeError(f"At least 160 GiB available host RAM required; got {memory_gib:.1f} GiB")
    print(f"Available host RAM: {memory_gib:.1f} GiB; ranks load sequentially.", flush=True)
    config.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    print(f"Free checkpoint disk: {shutil.disk_usage(config.CHECKPOINTS).free / 1024**3:.1f} GiB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["download", "preflight", "gpucheck"])
    args = parser.parse_args()
    config.initialize_dirs()
    globals()[args.action]()
