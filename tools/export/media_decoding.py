"""Audio and video decoding shared by the export and reference scripts.

The commands must stay identical to the Rust runtime's (src/av.rs), so the reference embeddings
and the runtime see the same samples and pixels.
"""
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def decode_audio(path: Path) -> np.ndarray:
    """16 kHz mono float32 PCM."""
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(pcm, dtype=np.float32).copy()


def decode_video_frames(path: Path, seconds_per_frame: int, width: int, height: int) -> np.ndarray:
    """RGB frames [N, height, width, 3], one every `seconds_per_frame` seconds."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps=1/{seconds_per_frame},scale={width}:{height}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3).copy()
