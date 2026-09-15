"""Decode references on CPU before admitting them to the distributed worker."""
from __future__ import annotations

from pathlib import Path


def validate_reference(path: Path, kind: str) -> None:
    if kind == "image":
        from PIL import Image, ImageOps
        with Image.open(path) as image:
            image.load()
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            if min(width, height) < 32 or max(width, height) > 8192:
                raise ValueError("Image dimensions must be 32..8192 pixels")
            if width > height * 4 or height > width * 4:
                raise ValueError("Image aspect ratio must be between 1:4 and 4:1")
        return
    import av
    with av.open(str(path)) as container:
        streams = [s for s in container.streams if s.type == kind]
        if not streams:
            raise ValueError(f"File has no {kind} stream")
        stream = streams[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else (
            container.duration / av.time_base if container.duration else None)
        if duration is None or not 0 < duration <= 15:
            raise ValueError("Audio/video references must have a known duration of at most 15 seconds")
        if kind == "video":
            width, height = stream.width, stream.height
            if not min(width, height) >= 32 or max(width, height) > 2048:
                raise ValueError("Reference video dimensions must be 32..2048 pixels")
            if width > 4 * height or height > 4 * width:
                raise ValueError("Video aspect ratio must be between 1:4 and 4:1")
        # Decode the complete file; malformed tails should not crash an NCCL worker.
        count = sum(1 for _ in container.decode(stream))
        if count == 0:
            raise ValueError(f"File contains no decodable {kind} frames")


def verify_output(path: Path, expected_frames: int | None = None,
                  expected_size=(1344, 768), expected_duration: int | None = None) -> dict:
    import av
    with av.open(str(path)) as container:
        video = next((s for s in container.streams if s.type == "video"), None)
        audio = next((s for s in container.streams if s.type == "audio"), None)
        if video is None or audio is None:
            raise RuntimeError("Generated MP4 must contain both video and audio")
        if (video.width, video.height) != tuple(expected_size):
            raise RuntimeError("Generated MP4 has unexpected dimensions")
        width, height = video.width, video.height
        if video.average_rate != 24:
            raise RuntimeError("Generated MP4 must be 24 FPS")
        frames = sum(1 for _ in container.decode(video))
        if not frames or (expected_frames is not None and frames != expected_frames):
            raise RuntimeError(f"Generated MP4 has {frames} video frames; expected {expected_frames or 'at least one'}")
    with av.open(str(path)) as container:
        samples = sum(frame.samples for frame in container.decode(audio=0))
        if not samples:
            raise RuntimeError("Generated MP4 has no decodable audio frame")
        audio_seconds = samples / container.streams.audio[0].rate
        # AAC may add a partial final codec frame; permit at most 0.1 s padding.
        if expected_duration is not None and abs(audio_seconds - expected_duration) > 0.1:
            raise RuntimeError(f"Generated audio duration {audio_seconds:.3f}s differs from {expected_duration}s")
    return {"width": width, "height": height, "frames": frames, "audio": True, "audio_seconds": audio_seconds}
