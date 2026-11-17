from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import wave

from .corpus import local_path


def ffmpeg_binary(explicit: str | None = None) -> str:
    path = explicit or os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install the audio extra or put ffmpeg on PATH") from exc


def extract_row(row: dict, destination: Path, ffmpeg: str | None = None) -> float:
    import numpy as np
    if not row["candidate"]:
        raise ValueError("Only clean candidates may be extracted automatically")
    pieces = []
    for span in row["spans"]:
        if span["rate"] != 1:
            raise ValueError("Speed-changed samples need explicit alignment review")
        command = [ffmpeg_binary(ffmpeg), "-v", "error", "-ss", str(span["source_start"]),
                   "-i", str(local_path(span["path"])), "-t", str(span["source_end"]-span["source_start"]),
                   "-map", f"0:a:{span['stream']}", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
        pieces.append(np.frombuffer(subprocess.check_output(command), dtype="<f4"))
    samples = np.concatenate(pieces)
    if len(samples) < 1600:
        raise ValueError("Audio is empty or shorter than 100 ms")
    rms = float(np.sqrt(np.mean(samples**2)))
    if rms < .00001:
        raise ValueError("Audio is silent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes((np.clip(samples, -1, 1)*32767).astype("<i2").tobytes())
    return rms


def read_wav(path: Path):
    import numpy as np
    # Avoid PyAV decoder compatibility differences. Faster Whisper accepts PCM.
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise ValueError("Expected mono 16-bit PCM at 16 kHz")
        return np.frombuffer(source.readframes(source.getnframes()), "<i2").astype("float32")/32768
