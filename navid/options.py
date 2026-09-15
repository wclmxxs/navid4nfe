"""Request validation and geometry, shared by API and worker (no CUDA imports)."""
from __future__ import annotations

import math
import os
import secrets
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FPS = 24
BUCKET = 4096


class RequestRejected(ValueError):
    """A deterministic, pre-DiT admission failure that does not poison CUDA."""


def native_frames(seconds: int) -> int:
    frames = seconds * FPS
    return frames + (5 - frames) % 17


def reference_size(width: int, height: int, short_edge: int | None = None,
                   allow_upscale: bool = False) -> tuple[int, int]:
    scale = (short_edge / min(width, height) if short_edge is not None
             else math.sqrt(1344 * 768 / (width * height)))
    if not allow_upscale:
        scale = min(1.0, scale)
    return tuple(max(32, round(side * scale / 32) * 32) for side in (width, height))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SolOptions(StrictModel):
    enabled: bool = True
    tau: float = Field(1.5, gt=0, le=10)
    dense_steps: int = Field(1, ge=0, le=4, strict=True)
    sink_conditioning: Literal["exact_kv", "exact_kv_and_rows", "off"] = "exact_kv_and_rows"
    dense_prefix_seconds: float = Field(0.0, ge=0, le=15)


class CacheOptions(StrictModel):
    enabled: bool = False
    warmup: int = Field(1, ge=1, le=4, strict=True)
    rdt: float = Field(0.08, ge=0, le=1)
    max_continuous_cached_steps: int = Field(1, ge=1, le=2, strict=True)


class Optimization(StrictModel):
    sol_attn: SolOptions = Field(default_factory=SolOptions)
    cache_dit: CacheOptions = Field(default_factory=CacheOptions)


def optimization_defaults() -> dict:
    return Optimization(
        sol_attn=SolOptions(enabled=os.environ.get("SOL_ATTN_ENABLED", "1") == "1",
                            tau=float(os.environ.get("SOL_ATTN_TAU", "1.5")),
                            dense_steps=int(os.environ.get("SOL_ATTN_DENSE_STEPS", "1"))),
        cache_dit=CacheOptions(enabled=os.environ.get("CACHE_DIT_ENABLED", "0") == "1",
                               warmup=int(os.environ.get("CACHE_DIT_WARMUP", "1")),
                               rdt=float(os.environ.get("CACHE_DIT_RDT", "0.08")),
                               max_continuous_cached_steps=int(os.environ.get("CACHE_DIT_MAX_CONTINUOUS", "1"))),
    ).model_dump()


def resolve_optimization(overrides: dict | None = None) -> dict:
    result = optimization_defaults()
    for name, values in Optimization.model_validate(overrides or {}).model_dump(exclude_unset=True).items():
        result[name].update(values)
    return Optimization.model_validate(result).model_dump()


class VideoRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=32000)
    duration: int = Field(5, ge=4, le=15, strict=True)
    seed: int = Field(default_factory=lambda: secrets.randbits(32), ge=0, le=2**63 - 1, strict=True)
    references: list[str] = Field(min_length=1, max_length=12)
    width: int | None = Field(None, ge=128, le=4096, multiple_of=2, strict=True)
    height: int | None = Field(None, ge=128, le=4096, multiple_of=2, strict=True)
    resolution: int | None = Field(None, ge=128, le=2048, multiple_of=2, strict=True)
    ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"] | None = None
    reference_short_edge: int | None = Field(None, ge=128, le=2048, strict=True)
    reference_allow_upscale: bool = False
    optimization: Optimization = Field(default_factory=Optimization)

    @model_validator(mode="after")
    def check_geometry(self):
        if self.reference_allow_upscale and self.reference_short_edge is None:
            raise ValueError("reference_allow_upscale requires reference_short_edge")
        if (self.width is None) != (self.height is None):
            raise ValueError("Provide both width and height")
        if self.width is not None and (self.resolution is not None or self.ratio is not None):
            raise ValueError("Use width/height OR resolution/ratio")
        width, height = self.output_size()
        if max(width, height) > min(width, height) * 4 or max(width, height) > 4096:
            raise ValueError("Output aspect ratio must be 1:4..4:1, longest edge <=4096")
        iw, ih = self.inference_size()
        if iw * ih > int(os.environ.get("MAX_OUTPUT_PIXELS", str(1920 * 1088))):
            raise ValueError("Inference canvas exceeds MAX_OUTPUT_PIXELS")
        return self

    def output_size(self) -> tuple[int, int]:
        if self.width is not None:
            return self.width, self.height
        if self.resolution is None and self.ratio is None:
            return 1344, 768
        short = self.resolution or 768
        a, b = map(int, (self.ratio or "16:9").split(":"))
        return tuple(round(short * side / min(a, b) / 2) * 2 for side in (a, b))

    def inference_size(self) -> tuple[int, int]:
        return tuple(math.ceil(side / 32) * 32 for side in self.output_size())

    def resolved_optimization(self) -> dict:
        return resolve_optimization(self.optimization.model_dump(exclude_unset=True))

    def execution(self) -> dict:
        return {"output_size": list(self.output_size()), "inference_size": list(self.inference_size()),
                "output_frames": self.duration * FPS, "native_frames": native_frames(self.duration),
                "optimization": self.resolved_optimization(), "compile_bucket": BUCKET}


def check_workload(payload: VideoRequest, references: list[dict]) -> None:
    """Conservative pre-queue token budget; exact layout is checked in the worker too."""
    from PIL import Image, ImageOps

    width, height = payload.inference_size()
    frames = native_frames(payload.duration)
    latent_frames = (frames - 5) // 17 * 5 + 2
    # VAE spatial compression is 16, DiT patches are 2x2.
    rows = latent_frames * (width // 32) * (height // 32)
    for ref in references:
        if ref["kind"] == "image":
            with Image.open(ref["path"]) as im:
                w, h = reference_size(*ImageOps.exif_transpose(im).size,
                                      payload.reference_short_edge, payload.reference_allow_upscale)
            rows += (w // 32) * (h // 32)
        elif ref["kind"] == "video":
            # Native reference videos use the fixed ~1MP canvas and no more
            # frames than the target. Include two streams when audio is present.
            rows += latent_frames * 1100
    rows += 16384  # reserve text/vision-language conditioning and audio rows
    if rows > int(os.environ.get("MAX_PACKED_TOKENS", "262144")):
        raise ValueError("Resolution, duration and references exceed MAX_PACKED_TOKENS; reduce one of them")
