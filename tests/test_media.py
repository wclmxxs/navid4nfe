import wave

import av
import numpy as np
import pytest

from navid.media import validate_reference, verify_output


def test_real_audio_upload_and_duration_limit(tmp_path):
    for seconds in (1, 16):
        path = tmp_path / f"audio-{seconds}.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 16000 * seconds)
        if seconds == 1:
            validate_reference(path, "audio")
        else:
            with pytest.raises(ValueError, match="15 seconds"):
                validate_reference(path, "audio")


def test_video_without_audio_fails_output_gate(tmp_path):
    path = tmp_path / "silent.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = 1344
        stream.height = 768
        stream.pix_fmt = "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(np.zeros((768, 1344, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    validate_reference(path, "video")
    with pytest.raises(RuntimeError, match="both video and audio"):
        verify_output(path)
