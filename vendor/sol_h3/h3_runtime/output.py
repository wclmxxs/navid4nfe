"""Deliver exact frame count and requested canvas without changing playback speed."""
from __future__ import annotations


def prepare_output(video, audio, sample_rate, duration, output_size):
    import torch
    import torch.nn.functional as F

    frames = duration * 24
    if video.shape[0] < frames:
        raise RuntimeError(f"Decoder produced only {video.shape[0]} frames, expected >= {frames}")
    video = video[:frames]
    width, height = output_size
    if tuple(video.shape[-2:]) != (height, width):
        # Resize a few frames at a time to bound temporary FP32 storage at 1080p.
        output = torch.empty((frames, video.shape[1], height, width), device=video.device, dtype=video.dtype)
        for start in range(0, frames, 8):
            output[start:start + 8] = F.interpolate(video[start:start + 8].float(),
                size=(height, width), mode="bilinear", align_corners=False, antialias=True).to(video.dtype)
        video = output
    if audio is not None:
        count = round(duration * sample_rate)
        # Diffusers uses channels x samples; the encoder also accepts samples x channels.
        axis = -1 if audio.ndim == 1 or audio.shape[0] <= 2 else 0
        if audio.shape[axis] < count:
            raise RuntimeError("Decoded audio is shorter than the requested duration")
        audio = audio[..., :count] if axis == -1 else audio[:count]
    return video, audio
