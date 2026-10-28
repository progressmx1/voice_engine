"""Speech-to-text: offline transcription with faster-whisper (CTranslate2).

The ``Transcriber`` loads one model and reuses it for every call. On Windows,
CUDA needs the cuBLAS/cuDNN DLL folders from the NVIDIA wheels to be on the DLL
search path *before* CTranslate2 is imported; :func:`setup_cuda_dlls` does that.
"""

from __future__ import annotations

import os
import site
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Windows: the NVIDIA wheels keep cuBLAS/cuDNN in ``site-packages/nvidia/*/bin``,
# which Python does not put on the DLL search path. CTranslate2 loads those DLLs
# when faster-whisper is imported, so this must run first.


def _nvidia_bin_dirs() -> list[Path]:
    """Every ``site-packages/nvidia/*/bin`` folder, deduplicated and sorted."""
    try:
        roots = site.getsitepackages() + [site.getusersitepackages()]
    except Exception:  # pragma: no cover - unusual Python installs
        roots = [p for p in sys.path if p and os.path.isdir(p)]

    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        nvidia = Path(root) / "nvidia"
        if not nvidia.is_dir():
            continue
        for pkg in sorted(nvidia.iterdir()):
            bin_dir = pkg / "bin"
            if bin_dir.is_dir():
                key = os.path.normcase(str(bin_dir.resolve()))
                if key not in seen:
                    seen.add(key)
                    found.append(bin_dir)
    return found


def _prepend_nvidia_bin_dirs_to_path(bin_dirs: list[Path]) -> None:
    """Prepend ``bin_dirs`` to ``PATH`` once each, ahead of existing entries.

    ``os.add_dll_directory`` only affects extension-module and ``ctypes``
    loads. CTranslate2's own native library later calls ``LoadLibrary`` for
    cuBLAS/cuDNN using the standard Windows search order, which includes
    ``PATH`` — so the NVIDIA ``bin`` folders must also be on ``PATH``.
    """
    if not bin_dirs:
        return
    existing = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    seen = {os.path.normcase(os.path.normpath(p)) for p in existing}
    additions: list[str] = []
    for bin_dir in bin_dirs:
        value = str(bin_dir)
        if os.path.normcase(os.path.normpath(value)) in seen:
            continue
        additions.append(value)
        seen.add(os.path.normcase(os.path.normpath(value)))
    if additions:
        os.environ["PATH"] = os.pathsep.join(additions + existing)


def setup_cuda_dlls() -> None:
    """Make the NVIDIA wheel ``bin`` folders loadable on Windows.

    Two mechanisms are used because different loaders use different search
    orders:

    * :func:`os.add_dll_directory` helps Python extension modules and ctypes.
    * Prepending the folders to ``PATH`` helps native libraries such as
      CTranslate2 that call ``LoadLibrary`` themselves.

    This must run before ``faster_whisper``/CTranslate2 is imported. No-op on
    non-Windows machines and if the folders are missing.
    """
    if os.name != "nt":
        return
    bin_dirs = _nvidia_bin_dirs()
    for bin_dir in bin_dirs:
        try:
            os.add_dll_directory(str(bin_dir))
        except OSError:  # pragma: no cover - folder vanished between checks
            pass
    _prepend_nvidia_bin_dirs_to_path(bin_dirs)


# Must happen before ``import faster_whisper`` (which imports CTranslate2).
setup_cuda_dlls()

import faster_whisper  # noqa: E402  (needs the DLL setup above)

from edith.models import MODELS_DIR  # noqa: E402

#: faster-whisper downloads its CTranslate2 models here (git-ignored), not the user cache.
WHISPER_DIR = MODELS_DIR / "whisper"


def _load_model(model: str, device: str, compute_type: str) -> faster_whisper.WhisperModel:
    """Load (downloading on first use into ``models/whisper``) a faster-whisper model."""
    return faster_whisper.WhisperModel(
        model, device=device, compute_type=compute_type, download_root=str(WHISPER_DIR)
    )


@dataclass(frozen=True)
class Segment:
    """One recognised span of speech, in seconds from the start of the audio."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """The result of :meth:`Transcriber.transcribe`."""

    text: str
    segments: tuple[Segment, ...]
    language: str
    seconds: float
    rtf: float


def _cuda_device_count() -> int:
    """Number of CUDA devices CTranslate2 can see, or 0 if it cannot load."""
    try:
        import ctranslate2
    except Exception:
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


_CUDA_LIBRARY_HINTS = ("cublas", "cudnn")


def _is_cuda_library_error(exc: BaseException) -> bool:
    """True when ``exc`` means a CUDA compute library could not be loaded.

    CTranslate2 raises this mid-encode (not at model load) when cuBLAS/cuDNN
    are missing, e.g. ``Library cublas64_12.dll is not found or cannot be
    loaded``. These errors mean the GPU is not usable, so the caller falls
    back to CPU int8.
    """
    message = str(exc).lower()
    return any(hint in message for hint in _CUDA_LIBRARY_HINTS)


def _resolve_device(device: str, cuda_count: int) -> str:
    """Turn ``auto`` into ``cuda``/``cpu``; fall back to CPU with a warning."""
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"device must be 'auto', 'cuda' or 'cpu', not {device!r}")
    if device == "auto":
        if cuda_count > 0:
            return "cuda"
        warnings.warn(
            "CUDA is not available to CTranslate2; falling back to CPU (int8).",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    if device == "cuda" and cuda_count == 0:
        warnings.warn(
            "device='cuda' requested but CUDA is not available; using CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    return device


def _resolve_compute_type(compute_type: str, device: str) -> str:
    """Pick float16 on CUDA and int8 on CPU when ``compute_type`` is ``auto``."""
    if compute_type == "auto":
        return "float16" if device == "cuda" else "int8"
    if not compute_type:
        raise ValueError("compute_type must not be empty")
    return compute_type


class Transcriber:
    """Offline speech-to-text with faster-whisper.

    ``device="auto"`` uses CUDA (float16) when it works and CPU (int8)
    otherwise. The model is loaded once and reused across :meth:`transcribe`
    calls.
    """

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self.model = model
        self.device = _resolve_device(device, _cuda_device_count())
        self.compute_type = _resolve_compute_type(compute_type, self.device)
        try:
            self._model = _load_model(self.model, self.device, self.compute_type)
        except RuntimeError as exc:
            if not _is_cuda_library_error(exc) or self.device != "cuda":
                raise
            warnings.warn(
                f"CUDA model load failed ({exc}); falling back to CPU (int8).",
                RuntimeWarning,
                stacklevel=2,
            )
            self.device = "cpu"
            self.compute_type = "int8"
            self._model = _load_model(self.model, "cpu", "int8")

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        language: str = "en",
        *,
        beam_size: int = 5,
    ) -> Transcript:
        """Transcribe an audio path or 16 kHz mono float32 samples.

        ``audio`` may be a path to any file :func:`edith.audio.load_audio` can
        read, or an in-memory ``(frames,)`` / ``(frames, channels)`` array at
        16 kHz. Returns the joined text, per-segment timings, detected
        language, audio length and the real-time factor of this call.

        If a CUDA model fails because its compute libraries (cuBLAS/cuDNN) are
        missing, the transcriber reloads on CPU int8 and retries once.
        """
        audio_arg = self._coerce_audio(audio)
        try:
            return self._transcribe(audio_arg, language=language, beam_size=beam_size)
        except RuntimeError as exc:
            if not _is_cuda_library_error(exc) or self.device != "cuda":
                raise
            warnings.warn(
                f"CUDA transcription failed ({exc}); falling back to CPU (int8).",
                RuntimeWarning,
                stacklevel=2,
            )
            self._reload_on_cpu()
            return self._transcribe(audio_arg, language=language, beam_size=beam_size)

    def _reload_on_cpu(self) -> None:
        """Swap the model for a CPU int8 model after a CUDA failure."""
        self.device = "cpu"
        self.compute_type = "int8"
        self._model = _load_model(self.model, "cpu", "int8")

    def _transcribe(
        self, audio_arg: str | np.ndarray, language: str, beam_size: int
    ) -> Transcript:
        start = time.perf_counter()
        segments_iter, info = self._model.transcribe(
            audio_arg, language=language, beam_size=beam_size
        )
        segments = tuple(
            Segment(
                start=round(float(seg.start), 3),
                end=round(float(seg.end), 3),
                text=(seg.text or "").strip(),
            )
            for seg in segments_iter
        )
        text = " ".join(seg.text for seg in segments if seg.text).strip()
        seconds = float(info.duration)
        rtf = (time.perf_counter() - start) / seconds if seconds > 0 else 0.0
        return Transcript(
            text=text,
            segments=segments,
            language=info.language or language,
            seconds=seconds,
            rtf=rtf,
        )

    @staticmethod
    def _coerce_audio(audio: str | Path | np.ndarray) -> str | np.ndarray:
        if isinstance(audio, (str, Path)):
            return str(audio)
        if isinstance(audio, np.ndarray):
            return _as_float32_mono(audio)
        raise TypeError(
            "audio must be a path or a numpy array, not "
            f"{type(audio).__name__}"
        )


def _as_float32_mono(audio: np.ndarray) -> np.ndarray:
    """Normalise an in-memory array to mono float32 (channels are averaged)."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=1).astype(np.float32)
    raise ValueError(f"audio must be 1-D or 2-D, not {arr.ndim}-D")
