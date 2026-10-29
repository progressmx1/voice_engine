"""JSON bridge between the Electron UI and the Python engine.

The UI spawns ``python -m edith.bridge`` once and talks to it over stdin/stdout:
one JSON request per line in, one JSON response per line out. Each request is
``{"id": <int>, "method": <str>, "params": {…}}``; each response is either
``{"id": …, "result": …}`` or ``{"id": …, "error": {"code": …, "message": …}}``.

The transcriber is cached per ``(model, device)`` and the TTS backend per device, so
the UI does not reload models on every request.

TTS methods: ``tts_voices`` lists voices and presets; ``tts`` speaks text into
``output/ui-tts/NNN.wav`` and returns timings, the normalised chunks and voice
measures; ``tts_check`` transcribes such a file with Whisper and scores it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .stt import Transcriber
from .samples import PROJECT_ROOT

#: Loaded transcribers, keyed by (model, device). Model weights are expensive
#: to load, so keep them for the lifetime of the bridge process.
_caches: dict[tuple[str, str], Transcriber] = {}


def _transcribe(params: dict[str, Any]) -> dict[str, Any]:
    audio = params.get("audio")
    if not audio or not isinstance(audio, str):
        raise ValueError("'audio' (a path to a WAV/MP3/FLAC/OGG file) is required")
    model = str(params.get("model") or "large-v3-turbo")
    device = str(params.get("device") or "auto")
    language = str(params.get("language") or "en")

    transcriber = _transcriber(model, device)
    transcript = transcriber.transcribe(audio, language=language)
    return {
        "text": transcript.text,
        "language": transcript.language,
        "seconds": transcript.seconds,
        "rtf": transcript.rtf,
        "model": transcriber.model,
        "device": transcriber.device,
        "compute_type": transcriber.compute_type,
        "segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in transcript.segments
        ],
    }


def _transcriber(model: str, device: str) -> Transcriber:
    key = (model, device)
    if key not in _caches:
        _caches[key] = Transcriber(model=model, device=device, compute_type="auto")
    return _caches[key]


# ---- TTS ----------------------------------------------------------------------------------

#: Where the UI's clips are written (git-ignored).
TTS_DIR = PROJECT_ROOT / "output" / "ui-tts"
MAX_TEXT = 2000
#: The candidates compared when choosing Edith's intonation (PLAN.md M4).
PRESETS = [
    {"name": "A: af_heart, intonation 1.0 (Kokoro as is)", "voice": "af_heart", "intonation": 1.0},
    {"name": "B: af_heart, intonation 1.3", "voice": "af_heart", "intonation": 1.3},
    {"name": "C: heart + lily, intonation 1.0", "voice": {"af_heart": 0.5, "bf_lily": 0.5},
     "intonation": 1.0},
    {"name": "D: heart + lily, intonation 1.3", "voice": {"af_heart": 0.5, "bf_lily": 0.5},
     "intonation": 1.3},
    {"name": "E: heart + lily, intonation 1.5", "voice": {"af_heart": 0.5, "bf_lily": 0.5},
     "intonation": 1.5},
]
_tts_backends: dict[str, Any] = {}
_clip_counter = [0]


def _tts_backend(device: str) -> Any:
    from .tts.engine import KokoroBackend

    if device not in _tts_backends:
        _tts_backends[device] = KokoroBackend(device=device)
    return _tts_backends[device]


def _voice_param(value: Any) -> str | dict[str, float]:
    """A voice name, or a blend {name: weight} with positive weights."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and value:
        weights = {str(k): float(v) for k, v in value.items() if float(v) > 0}
        if weights:
            return weights
    raise ValueError("'voice' must be a voice name or a blend like {\"af_heart\": 0.5, ...}")


def _tts_voices(params: dict[str, Any]) -> dict[str, Any]:
    from .tts.engine import Synthesizer

    synth = Synthesizer(backend=_tts_backend(str(params.get("device") or "auto")))
    return {"voices": synth.voices(), "presets": PRESETS, "default_text": DEFAULT_TEXT}


DEFAULT_TEXT = (
    "Good evening! You've got two new messages. Sarah says the meeting moved to Thursday at "
    "3 pm, and she's wondering if that still works for you. Also, your package arrived this "
    "afternoon. Oh, and it looks like it might rain tomorrow, so you'll probably want an "
    "umbrella. Should I reply to Sarah for you?"
)


def _tts(params: dict[str, Any]) -> dict[str, Any]:
    from .audio import save_wav
    from .quality import voice_measures
    from .tts.engine import Synthesizer

    text = params.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("'text' is required")
    if len(text) > MAX_TEXT:
        raise ValueError(f"'text' is limited to {MAX_TEXT} characters")
    device = str(params.get("device") or "auto")
    synth = Synthesizer(
        voice=_voice_param(params.get("voice") or "edith"),
        speed=float(params.get("speed", 1.0)),
        intonation=float(params.get("intonation", 1.0)),
        backend=_tts_backend(device),
    )
    synth.reference_hz()  # calibrate outside the timing, as a running engine would have
    start = time.perf_counter()
    first = None
    parts = []
    for part in synth.stream(text):
        if first is None:
            first = time.perf_counter() - start
        parts.append(part)
    total = time.perf_counter() - start
    audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    seconds = len(audio) / synth.rate
    _clip_counter[0] += 1
    path = save_wav(TTS_DIR / f"{_clip_counter[0]:03d}.wav", audio, synth.rate)
    m = voice_measures(audio, synth.rate) if seconds > 0.3 else None
    return {
        "path": str(path),
        "spoken": " ".join(chunk for chunk, _ in synth.chunks(text)),
        "chunks": [chunk for chunk, _ in synth.chunks(text)],
        "seconds": seconds,
        "first_audio_ms": (first or 0.0) * 1000,
        "synth_seconds": total,
        "rtf": total / seconds if seconds else 0.0,
        "device": getattr(synth.backend, "device", ""),
        "rate": synth.rate,
        "measures": None if m is None else {
            "jitter": m.jitter, "shimmer": m.shimmer,
            "f0_median": m.f0_median, "f0_spread": m.f0_spread,
        },
    }


def _tts_check(params: dict[str, Any]) -> dict[str, Any]:
    from .audio import STT_RATE, load_audio
    from .roundtrip import score

    path = Path(str(params.get("path") or ""))
    if not path.is_file() or path.resolve().parent != TTS_DIR.resolve():
        raise ValueError("'path' must be a clip the UI made in output/ui-tts")
    spoken = str(params.get("spoken") or "")
    transcriber = _transcriber(str(params.get("model") or "large-v3-turbo"),
                               str(params.get("device") or "auto"))
    transcript = transcriber.transcribe(load_audio(path, STT_RATE))
    return {"heard": transcript.text, "wer": score(spoken, transcript.text),
            "model": transcriber.model, "device": transcriber.device}


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    """Handle one parsed request and return the response dict."""
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") if isinstance(request.get("params"), dict) else {}

    try:
        if method == "ping":
            result: dict[str, Any] = {"ok": True, "version": __version__}
        elif method == "transcribe":
            result = _transcribe(params)
        elif method == "tts_voices":
            result = _tts_voices(params)
        elif method == "tts":
            result = _tts(params)
        elif method == "tts_check":
            result = _tts_check(params)
        else:
            return {
                "id": request_id,
                "error": {"code": -32601, "message": f"unknown method: {method!r}"},
            }
        return {"id": request_id, "result": result}
    except Exception as exc:  # pragma: no cover - defensive boundary for the UI
        return {"id": request_id, "error": {"code": -32000, "message": str(exc)}}


def process_line(line: str) -> str:
    """Parse and dispatch one input line, returning the JSON response line."""
    line = line.strip()
    if not line:
        return ""
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        return json.dumps(
            {"id": None, "error": {"code": -32700, "message": f"parse error: {exc}"}}
        )
    if not isinstance(request, dict):
        return json.dumps(
            {"id": None, "error": {"code": -32600, "message": "request must be an object"}}
        )
    return json.dumps(dispatch(request))


def main(argv: list[str] | None = None) -> int:
    """Run the bridge on stdin/stdout until stdin closes."""
    for line in sys.stdin:
        out = process_line(line)
        if out:
            sys.stdout.write(out + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
