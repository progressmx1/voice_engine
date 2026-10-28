"""Edith's text-to-speech engine: text in, smooth speech out.

    text -> normalize (edith.text) -> clause chunks -> backend (acoustic model + vocoder)
         -> pauses -> Smoother (edith.tts.smooth) -> audio

The backend is pluggable. `KokoroBackend` (Kokoro-82M ONNX) is the baseline voice until Edith's
own fine-tuned model exists (PLAN.md M5). The engine owns everything else: what gets said, where
it is split, how long the pauses are and how the chunks are joined.

To keep the timbre steady across chunks, Kokoro's per-length style vector is locked to one row
for the whole utterance (`lock_style`), instead of changing with every chunk's length.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from edith import models
from edith.audio import save_wav
from edith.text import normalize
from edith.tts.smooth import Smoother, fade_curve, ms_to_samples

#: Voice blends saved as {"voice_name": weight, ...}; `voices/edith.json` is Edith's voice.
VOICES_DIR = Path(__file__).resolve().parents[3] / "voices"
#: Used for "edith" until `voices/edith.json` exists.
FALLBACK_VOICE = "af_heart"
#: Fade at the end of each chunk's speech before its pause.
FADE_OUT_MS = 5.0

Voice = str | Mapping[str, float]


class Backend(Protocol):
    """An acoustic model + vocoder that turns one chunk of normalised text into audio."""

    rate: int

    def voices(self) -> list[str]:
        """Names of the voices this backend can speak with."""
        ...

    def prepare(self, voice: Voice, utterance: str) -> Any:
        """Resolve `voice` once per utterance into whatever `synthesize` needs."""
        ...

    def synthesize(self, text: str, prepared: Any, speed: float) -> np.ndarray:
        """Speak one chunk; returns mono float32 at `rate` with edge silence trimmed."""
        ...


# --- Chunking -------------------------------------------------------------------------------


def split_chunks(
    text: str, max_chars: int = 180, sentence_pause: float = 0.25, clause_pause: float = 0.1
) -> list[tuple[str, float]]:
    """Split normalised text into (chunk, pause after in seconds).

    Sentences are kept whole when they fit in `max_chars`; longer ones are split at clause
    punctuation (, ; :), and only if a clause is still too long, at spaces. The last chunk has
    no pause.
    """
    out: list[tuple[str, float]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sentence:
            continue
        pieces = _pack(re.split(r"(?<=[,;:])\s+", sentence), max_chars)
        pieces = [p for piece in pieces for p in (_pack(piece.split(), max_chars)
                                                  if len(piece) > max_chars else [piece])]
        out += [(p, clause_pause) for p in pieces[:-1]] + [(pieces[-1], sentence_pause)]
    if out:
        out[-1] = (out[-1][0], 0.0)
    return out


def _pack(parts: list[str], max_chars: int) -> list[str]:
    """Greedily join `parts` with spaces into pieces of at most `max_chars` (when possible)."""
    pieces: list[str] = []
    for part in parts:
        if pieces and len(pieces[-1]) + 1 + len(part) <= max_chars:
            pieces[-1] += " " + part
        else:
            pieces.append(part)
    return pieces


# --- Voices ---------------------------------------------------------------------------------


def load_blend(name: str, voices_dir: Path = VOICES_DIR) -> dict[str, float] | None:
    """Weights saved in `voices/<name>.json`, or None if there is no such file."""
    file = voices_dir / f"{name}.json"
    if not file.is_file():
        return None
    weights = json.loads(file.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in weights.items()}


def blend(styles: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """Weighted average of voice style arrays (weights are normalised to sum to 1)."""
    total = sum(weights.values())
    if not weights or total <= 0:
        raise ValueError("a blend needs at least one positive weight")
    missing = [n for n in weights if n not in styles]
    if missing:
        raise ValueError(f"unknown voice(s) in blend: {', '.join(missing)}")
    return sum(styles[n].astype(np.float32) * (w / total) for n, w in weights.items())


def lock_style(style: np.ndarray, length: int) -> np.ndarray:
    """Repeat the style row for `length` phonemes over every row, so chunk length can't change it."""
    row = style[min(max(length, 1), len(style)) - 1]
    return np.broadcast_to(row, style.shape).copy()


# --- Kokoro backend -------------------------------------------------------------------------

_KOKORO_CACHE: dict[tuple[str, str, tuple[str, ...]], Any] = {}
#: onnxruntime threads for Kokoro on CPU; more threads than this get slower, not faster.
CPU_THREADS = 6


def _providers(device: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        # Load CUDA/cuDNN from the NVIDIA wheels (site-packages/nvidia/*) before the session.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        warnings.warn("CUDA is not available to onnxruntime; Kokoro runs on CPU.", RuntimeWarning,
                      stacklevel=3)
    elif device not in ("auto", "cpu"):
        raise ValueError(f"device must be 'auto', 'cpu' or 'cuda', got {device!r}")
    return ["CPUExecutionProvider"]


class KokoroBackend:
    """Kokoro-82M ONNX (Apache-2.0) through kokoro-onnx; the model is loaded once and cached."""

    rate = 24_000

    def __init__(self, device: str = "auto", model: str = "kokoro-v1.0.onnx",
                 voices_file: str = "voices-v1.0.bin", locked: bool = True) -> None:
        self.providers = _providers(device)
        self.device = "cuda" if self.providers[0].startswith("CUDA") else "cpu"
        self.locked = locked
        model_path, voices_path = models.ensure(model), models.ensure(voices_file)
        key = (str(model_path), str(voices_path), tuple(self.providers))
        if key not in _KOKORO_CACHE:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro

            options = ort.SessionOptions()
            if self.device == "cpu":
                # Measured on the 22-thread laptop CPU: 6 threads RTF 0.24, default 0.31, 22 0.86.
                options.intra_op_num_threads = CPU_THREADS
            session = ort.InferenceSession(str(model_path), options, providers=self.providers)
            _KOKORO_CACHE[key] = Kokoro.from_session(session, str(voices_path))
        self.kokoro = _KOKORO_CACHE[key]
        # onnxruntime drops to CPU (with only a log line) if a CUDA library fails to load.
        actual = self.kokoro.sess.get_providers()[0]
        self.device = "cuda" if actual.startswith("CUDA") else "cpu"
        if self.providers[0].startswith("CUDA") and self.device == "cpu":
            warnings.warn("onnxruntime could not start CUDA; Kokoro runs on CPU.", RuntimeWarning,
                          stacklevel=2)

    def voices(self) -> list[str]:
        return list(self.kokoro.get_voices())

    def prepare(self, voice: Voice, utterance: str) -> np.ndarray:
        if isinstance(voice, str):
            weights = load_blend(voice) or ({FALLBACK_VOICE: 1.0} if voice == "edith" else None)
            style = (blend(self.kokoro.voices, weights) if weights
                     else self.kokoro.get_voice_style(voice))
        else:
            style = blend(self.kokoro.voices, voice)
        # Characters approximate phonemes closely enough to pick the style row.
        return lock_style(style, len(utterance)) if self.locked else style

    def synthesize(self, text: str, prepared: np.ndarray, speed: float) -> np.ndarray:
        samples, _ = self.kokoro.create(text, voice=prepared, speed=speed, lang="en-us", trim=True)
        return np.asarray(samples, dtype=np.float32)


# --- Synthesizer ----------------------------------------------------------------------------


class Synthesizer:
    """Edith's voice: `synthesize(text)`, `stream(text)`, `save(text, path)`, `voices()`."""

    def __init__(
        self,
        voice: Voice = "edith",
        speed: float = 1.0,
        device: str = "auto",
        backend: Backend | None = None,
        max_chars: int = 180,
        sentence_pause: float = 0.25,
        clause_pause: float = 0.1,
        crossfade_ms: float = 10.0,
    ) -> None:
        if not 0.5 <= speed <= 2.0:
            raise ValueError(f"speed must be between 0.5 and 2.0, got {speed}")
        self.voice = voice
        self.speed = speed
        self.backend: Backend = backend or KokoroBackend(device=device)
        self.max_chars = max_chars
        self.sentence_pause = sentence_pause
        self.clause_pause = clause_pause
        self.crossfade_ms = crossfade_ms
        #: Output sample indices of the chunk joins in the most recent utterance.
        self.last_joins: list[int] = []

    @property
    def rate(self) -> int:
        return self.backend.rate

    def voices(self) -> list[str]:
        """Backend voices plus saved blends (`voices/*.json`)."""
        blends = sorted(p.stem for p in VOICES_DIR.glob("*.json")) if VOICES_DIR.is_dir() else []
        return sorted(set(self.backend.voices()) | set(blends) | {"edith"})

    def chunks(self, text: str) -> list[tuple[str, float]]:
        """The normalised chunks and pauses `text` will be spoken as."""
        return split_chunks(normalize(text), self.max_chars, self.sentence_pause, self.clause_pause)

    def stream(self, text: str, voice: Voice | None = None) -> Iterator[np.ndarray]:
        """Yield smoothed audio as each chunk is synthesized; together they are gapless."""
        parts = self.chunks(text)
        if not parts:
            return
        prepared = self.backend.prepare(voice or self.voice, " ".join(c for c, _ in parts))
        smoother = Smoother(self.rate, crossfade_ms=self.crossfade_ms)
        self.last_joins = smoother.joins
        for chunk, pause in parts:
            audio = self.backend.synthesize(chunk, prepared, self.speed)
            if pause:
                # Speech meets silence here, inside the chunk: fade it so the cut can't click.
                n = min(ms_to_samples(FADE_OUT_MS, self.rate), len(audio))
                audio = audio.copy()
                audio[len(audio) - n :] *= fade_curve(n)[::-1]
                audio = np.concatenate([audio, np.zeros(int(pause * self.rate), np.float32)])
            out = smoother.feed(audio)
            if out.size:
                yield out
        tail = smoother.flush()
        if tail.size:
            yield tail

    def synthesize(self, text: str, voice: Voice | None = None) -> tuple[np.ndarray, int]:
        """The whole utterance as (samples, sample_rate)."""
        parts = list(self.stream(text, voice))
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return audio, self.rate

    def save(self, text: str, path: str | Path, voice: Voice | None = None) -> Path:
        """Speak `text` into a 16-bit WAV file."""
        audio, rate = self.synthesize(text, voice)
        return save_wav(path, audio, rate)
