"""Resident 768p MiniMax-H3 T2V/I2V/Ref2VA inference engine."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from PIL import Image, ImageOps

from .compute_quant import COMPUTE_QUANT_MODES


WIDTH = 1344
HEIGHT = 768
FPS = 24
INFERENCE_STEPS = 5  # Five scheduler points execute four DiT forwards.
DURATION_FRAMES = {seconds: seconds * FPS + (5 - seconds * FPS) % 17 for seconds in range(4, 16)}
ATTENTION_BACKENDS = {"dense", "sol", "sol_bsa"}
TASKS = {"t2v", "i2v", "ref2va"}
REFERENCE_IMAGE_RESIZE_MODES = {"match", "diffusers"}


def _resolve_reference_image_size(
    width: int,
    height: int,
    *,
    mode: str,
    short_edge: int | None = None,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """Resolve Ref2VA image geometry for the selected deployment profile."""
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")

    if short_edge is not None:
        scale = short_edge / min(width, height)
        if not allow_upscale:
            scale = min(1.0, scale)
    elif mode == "match":
        # Match the fastest validated Ref2VA profile: never upscale a source,
        # and cap larger inputs to the generated 768p canvas area.
        scale = min(1.0, math.sqrt((WIDTH * HEIGHT) / (width * height)))
    elif mode == "diffusers":
        # Official Diffusers parity mode. This intentionally upscales every
        # reference image to a 2048-pixel short edge.
        scale = 2048 / min(width, height)
    else:
        raise ValueError(
            f"reference_image_resize_mode must be one of {sorted(REFERENCE_IMAGE_RESIZE_MODES)}"
        )

    multiple = 32
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def _state_value(state: Any, name: str) -> Any:
    if hasattr(state, "get"):
        value = state.get(name)
        if value is not None:
            return value
    return getattr(state, name, None)


@dataclass
class GeneratedMedia:
    """Rank-zero output. Other ranks return ``None`` from ``generate``."""

    video: Any
    audio: Any
    audio_sample_rate: int | None
    elapsed_s: float
    duration: int
    seed: int
    metadata: dict = field(default_factory=dict)

    def save_async(self, output_path: str | Path):
        """Stage output and encode it in the background."""
        from .encoding import start_fast_video_encode

        if self.video is None:
            raise RuntimeError("this output has already been handed to an encoder")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        job = start_fast_video_encode(
            self.video,
            fps=FPS,
            output_path=str(output_path),
            audio=self.audio,
            audio_sample_rate=self.audio_sample_rate,
            chunk_frames=8,
            fragmented=True,
            encoder_threads=36,
            pyav_zero_copy=True,
            overlap_audio=True,
        )
        job.wait_staged()
        self.video = None
        self.audio = None
        return job

    def save(self, output_path: str | Path) -> Path:
        job = self.save_async(output_path)
        job.wait()
        output_path = Path(output_path)
        return output_path


class MiniMaxH3Inference:
    """Load once and generate fixed-profile 768p T2V, I2V, or Ref2VA."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | Path,
        attention_backend: str = "sol_bsa",
        task: str = "t2v",
        reference_image_resize_mode: str = "match",
        compute_quant: str = "none",
        before_gpu_load: Callable[[int], None] | None = None,
        load_parallelism: int = 1,
        inference_nfe: int = 4,
    ) -> None:
        if type(inference_nfe) is not int or inference_nfe not in (4, 8):
            raise ValueError("inference_nfe must be 4 or 8 with a matching adapter")
        self.inference_nfe = inference_nfe
        self._owns_process_group = False
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        if dist.is_available() and not dist.is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1:
                dist.init_process_group(backend="nccl", device_id=self.device)
                self._owns_process_group = True

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.world_size not in {1, 2, 4, 8}:
            raise ValueError("GPU process count must be 1, 2, 4, or 8")
        task = task.lower()
        if task not in TASKS:
            raise ValueError(f"task must be one of {sorted(TASKS)}")
        attention_backend = attention_backend.lower()
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"attention_backend must be one of {sorted(ATTENTION_BACKENDS)}"
            )
        if attention_backend != "dense" and self.world_size == 1:
            raise ValueError("SOL attention requires 2, 4, or 8 GPU processes")
        if task == "ref2va" and attention_backend == "sol":
            raise ValueError("Ref2VA supports dense or sol_bsa attention")
        reference_image_resize_mode = reference_image_resize_mode.lower()
        if reference_image_resize_mode not in REFERENCE_IMAGE_RESIZE_MODES:
            raise ValueError(
                "reference_image_resize_mode must be one of "
                f"{sorted(REFERENCE_IMAGE_RESIZE_MODES)}"
            )
        compute_quant = compute_quant.lower()
        if compute_quant not in COMPUTE_QUANT_MODES:
            raise ValueError(f"compute_quant must be one of {sorted(COMPUTE_QUANT_MODES)}")
        if compute_quant == "mxfp8":
            from .compute_quant import validate_platform

            validate_platform()
        self.task = task
        self.attention_backend = attention_backend
        self.reference_image_resize_mode = reference_image_resize_mode
        self.compute_quant = compute_quant
        self.reference_short_edge = None
        self.reference_allow_upscale = False
        self.reference_geometry = []

        from diffusers import ComponentsManager, ModularPipeline

        # The released pipeline originally enforced a shorter public limit. The
        # 362-frame shape is a native 17*n+5 H3 sequence and is used directly.
        from diffusers.modular_pipelines.minimax_h3 import before_encoder

        before_encoder.MINIMAX_H3_MIN_DURATION = 4
        before_encoder.MINIMAX_H3_MAX_DURATION = 364 / FPS
        if task == "ref2va":
            # before_encoder imports the resolver by name, so override that
            # binding without modifying the installed Diffusers package.
            def resolve(width, height):
                h, w = _resolve_reference_image_size(
                    width, height, mode=self.reference_image_resize_mode,
                    short_edge=self.reference_short_edge, allow_upscale=self.reference_allow_upscale)
                self.reference_geometry.append({"original_size": [width, height], "encoded_size": [w, h]})
                return h, w

            before_encoder.resolve_reference_image_size = resolve

        manager = ComponentsManager()
        if task == "ref2va":
            from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks

            self.pipe = MiniMaxH3Ref2VABlocks().init_pipeline(
                model_path, components_manager=manager
            )
        else:
            self.pipe = ModularPipeline.from_pretrained(
                model_path, components_manager=manager
            )
        load_kwargs = {"dtype": torch.bfloat16}
        if task == "ref2va":
            # Resolve every Ref2VA component below the selected model root. This
            # also avoids stale absolute paths in locally converted indexes.
            load_kwargs["pretrained_model_name_or_path"] = model_path
        from .loading import load_in_groups

        def load_copy():
            loading_started = time.monotonic()
            print(f"Loading model on GPU rank {self.rank}/{self.world_size}; pid={os.getpid()}", flush=True)
            self.pipe.load_components(**load_kwargs)
            cpu_seconds = time.monotonic() - loading_started
            print(f"CPU weights ready on GPU rank {self.rank}/{self.world_size}; "
                  f"cpu_load_s={cpu_seconds:.1f}; transferring to CUDA", flush=True)
            if before_gpu_load is not None:
                before_gpu_load(local_rank)
            transfer_started = time.monotonic()
            self.pipe.to(self.device)
            torch.cuda.synchronize(self.device)
            print(f"Loaded model on GPU rank {self.rank}/{self.world_size}; "
                  f"cpu_load_s={cpu_seconds:.1f}; cuda_transfer_s={time.monotonic() - transfer_started:.1f}; "
                  f"elapsed={time.monotonic() - loading_started:.1f}s", flush=True)

        load_in_groups(load_copy, rank=self.rank, world_size=self.world_size,
                       parallelism=load_parallelism,
                       barrier=dist.barrier if dist.is_initialized() else lambda: None)
        if self.rank == 0:
            print("All GPU model copies loaded; fusing LoRA and configuring inference kernels", flush=True)
        self.transformer = (
            self.pipe.transformer_ref if task == "ref2va" else self.pipe.transformer
        )
        if self.transformer is None:
            component = "transformer_ref" if task == "ref2va" else "transformer"
            raise RuntimeError(
                f"failed to load {component} from {model_path!r}; "
                f"make sure {component}/config.json and all checkpoint shards exist"
            )
        self.pipe.scheduler.set_shift(12.0)
        self.pipe.audio_scheduler.set_shift(3.0)

        from .lora import fuse_lora

        fuse_lora(
            self.transformer,
            adapter_path,
            alpha=8 if task == "ref2va" else 64,
            scale=1.0,
        )

        if self.world_size > 1:
            from diffusers.models._modeling_parallel import ContextParallelConfig

            from .cp_plan import MINIMAX_H3_CP_PLAN, assert_no_attention_mask

            assert_no_attention_mask(self.transformer)
            self.transformer.enable_parallelism(
                config=ContextParallelConfig(
                    ulysses_degree=self.world_size,
                    ulysses_anything=True,
                ),
                cp_plan=MINIMAX_H3_CP_PLAN,
            )

        from . import adaln, fusion_install, sparse_attention, ulysses, vae_parallel

        self.compute_quant_report = None
        if compute_quant == "mxfp8":
            from .compute_quant import install as install_compute_quant

            self.compute_quant_report = install_compute_quant(self.transformer)
            if self.rank == 0:
                report = self.compute_quant_report
                ratio = report.original_bytes / report.quantized_bytes
                print(
                    "MXFP8 compute: "
                    f"blocks={report.first_block}..{report.last_block}, "
                    f"linears={report.quantized_linears}, "
                    f"storage={report.original_bytes / 2**30:.2f}->"
                    f"{report.quantized_bytes / 2**30:.2f} GiB ({ratio:.2f}x)"
                )
        fusion_install.install(self.transformer)
        adaln.enable_adaln_precompute(
            self.transformer,
            verbose=self.rank == 0,
            component_name="transformer_ref" if task == "ref2va" else "transformer",
        )
        self.sparse_attention = None
        if attention_backend != "dense":
            self.sparse_attention = sparse_attention.install(
                self.transformer,
                backend=attention_backend,
                tau=1.0,
                dense_steps=0 if task == "ref2va" else 1,
                dense_layers=0 if task == "ref2va" else 2,
                sink_mode="text_audio" if task == "ref2va" else "prefix",
            )
        if self.world_size > 1:
            ulysses.install(
                self.transformer,
                attention_fn=self.sparse_attention,
            )
        vae_parallel.install(
            self.pipe.vae,
            batched=True,
            compile_mode="default",
            encode_parallel=task == "ref2va",
        )
        self._adaln_checked = False

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def _check_adaln(self) -> None:
        if self._adaln_checked:
            return
        for index, block in enumerate(self.transformer.transformer_blocks):
            table = getattr(block.adaln_proj, "table", None)
            schedules = 3 if self.task == "ref2va" else 2
            if (table is None or table.ndim != 4 or table.shape[0] != schedules
                    or table.shape[1] != self.inference_nfe):
                raise RuntimeError(f"invalid AdaLN cache at transformer block {index}")
        self._adaln_checked = True

    @staticmethod
    def _load_image(image: str | Path | Image.Image | None) -> Image.Image | None:
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")
        with Image.open(image) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        duration: int,
        seed: int = 0,
        image: str | Path | Image.Image | None = None,
        references: list[Any] | None = None,
        width: int = WIDTH,
        height: int = HEIGHT,
        output_size: tuple[int, int] | list[int] | None = None,
        reference_short_edge: int | None = None,
        reference_allow_upscale: bool = False,
        optimization: dict | None = None,
    ) -> GeneratedMedia | None:
        """Generate one video. ``references`` is required only for Ref2VA."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if duration not in DURATION_FRAMES:
            raise ValueError("duration must be an integer between 4 and 15 seconds")
        if self.task == "ref2va":
            if image is not None:
                raise ValueError("image is a first-frame I2V input; use references for Ref2VA")
            if not references:
                raise ValueError("Ref2VA requires at least one reference")
        elif references is not None:
            raise ValueError("references are only valid when task='ref2va'")

        if width % 32 or height % 32:
            raise ValueError("Inference dimensions must be multiples of 32")
        self.reference_short_edge = reference_short_edge
        self.reference_allow_upscale = reference_allow_upscale
        self.reference_geometry = []
        first_frame = self._load_image(image)
        if self.world_size > 1:
            from .ulysses import reset_row_counts

            reset_row_counts()
            dist.barrier()

        request = {
            "prompt": prompt.strip(),
            "height": height,
            "width": width,
            "num_frames": DURATION_FRAMES[duration],
            "num_inference_steps": self.inference_nfe + 1,
            "generator": torch.Generator().manual_seed(int(seed)),
            "output_type": "pt",
        }
        if first_frame is not None:
            request["image"] = first_frame
        if self.task == "ref2va":
            request["references"] = list(references)

        torch.cuda.synchronize(self.device)
        runtime = getattr(self, "request_runtime", None)
        if runtime is not None:
            runtime.begin(optimization, duration)
        started = time.perf_counter()
        state = self.pipe(**request)
        torch.cuda.synchronize(self.device)
        if self.world_size > 1:
            dist.barrier()
        elapsed = time.perf_counter() - started
        self._check_adaln()
        metadata = runtime.finish() if runtime is not None else {}
        metadata.update(reference_images=self.reference_geometry, native_frames=DURATION_FRAMES[duration],
                        output_frames=duration * FPS, inference_size=[width, height],
                        output_size=list(output_size or (width, height)))

        if not self.is_rank_zero:
            return None
        videos = _state_value(state, "videos")
        if videos is None:
            raise RuntimeError("pipeline returned no video")
        audio = _state_value(state, "audio")
        from .output import prepare_output

        video, waveform = prepare_output(
            videos[0], None if audio is None else audio[0],
            _state_value(state, "sampling_rate"), duration, output_size or (width, height))
        return GeneratedMedia(
            video=video,
            audio=waveform,
            audio_sample_rate=_state_value(state, "sampling_rate"),
            elapsed_s=elapsed,
            duration=duration,
            seed=int(seed),
            metadata=metadata,
        )

    def generate_to_file(
        self,
        output_path: str | Path,
        prompt: str,
        *,
        duration: int,
        seed: int = 0,
        image: str | Path | Image.Image | None = None,
        references: list[Any] | None = None,
    ):
        """Generate and start background MP4 encoding on rank zero."""
        media = self.generate(
            prompt,
            duration=duration,
            seed=seed,
            image=image,
            references=references,
        )
        return None if media is None else media.save_async(output_path)

    def warmup(
        self,
        *,
        duration: int = 5,
        prompt: str | None = None,
        image: str | Path | Image.Image | None = None,
        references: list[Any] | None = None,
    ) -> None:
        self.generate(
            prompt
            or "A continuous cinematic shot of morning light moving across a quiet room.",
            duration=duration,
            seed=0,
            image=image,
            references=references,
        )

    def close(self) -> None:
        if self._owns_process_group and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            self._owns_process_group = False

    def __enter__(self) -> "MiniMaxH3Inference":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
