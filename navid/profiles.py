"""Pinned Ref2VA adapters paired with their trained sampling schedules."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

ADAPTER_REVISION = "3ec17a324ced54151364f24f8b5fb6bf7e26414f"


@dataclass(frozen=True)
class Ref2VAProfile:
    nfe: int
    adapter_name: str
    adapter_sha256: str
    video_shift: float = 12.0
    audio_shift: float = 3.0
    lora_alpha: int = 8

    def metadata(self) -> dict:
        return {**asdict(self), "adapter_revision": ADAPTER_REVISION,
                "scheduler_points": self.nfe + 1, "sampler": "euler"}


# Eight-step settings: https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/51
# Four-step settings: https://github.com/ModelTC/Minimax-H3-Turbo#1-model-specs
PROFILES = {
    4: Ref2VAProfile(4, "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
                    "9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99"),
    8: Ref2VAProfile(8, "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors",
                    "9bac880b1a5d7ac052171cf6cce769f0cceaaa42ffa51de4b8e41143a2bdd2d2"),
}


def configured_profile() -> Ref2VAProfile:
    value = os.environ.get("REF2VA_NFE", "8")
    if value not in {"4", "8"}:
        raise ValueError("REF2VA_NFE must be 4 or 8; each selects its own pinned LoRA")
    return PROFILES[int(value)]


def require_profile(nfe: int | None, profile: Ref2VAProfile) -> None:
    if nfe is not None and nfe != profile.nfe:
        raise ValueError(f"This worker is loaded for {profile.nfe} NFE. Set REF2VA_NFE={nfe} "
                         "and run ./deploy.sh restart to load the matching LoRA; "
                         "a request cannot switch resident weights.")
