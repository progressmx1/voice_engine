"""Audio input and output: any file to mono float32 at a chosen sample rate, and WAV out."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

#: Speech-to-text models expect 16 kHz mono.
STT_RATE = 16_000


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average channels: (frames, channels) to (frames,)."""
    return samples.mean(axis=1) if samples.ndim == 2 else samples


def resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Polyphase resampling (band-limited, no aliasing) from `rate` to `target` Hz."""
    if rate == target:
        return samples.astype(np.float32, copy=False)
    g = gcd(rate, target)
    return resample_poly(samples, target // g, rate // g).astype(np.float32)


def load_audio(path: str | Path, rate: int = STT_RATE) -> np.ndarray:
    """Read WAV, FLAC, OGG or MP3 into mono float32 in [-1, 1] at `rate` Hz."""
    data, file_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return resample(to_mono(data), int(file_rate), rate)


def duration(path: str | Path) -> float:
    """Length of an audio file in seconds.

    WAV and FLAC headers are exact. MP3 headers are only estimates (variable bit rate), so
    compressed files are decoded and counted.
    """
    info = sf.info(str(path))
    if info.format in {"WAV", "FLAC", "AIFF"}:
        return info.frames / info.samplerate
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return len(data) / rate


def save_wav(path: str | Path, samples: np.ndarray, rate: int) -> Path:
    """Write mono float samples as 16-bit PCM WAV, creating folders as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), np.clip(samples, -1.0, 1.0), rate, subtype="PCM_16")
    return out
