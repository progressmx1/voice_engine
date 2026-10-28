"""JSON bridge between the Electron UI and the Python engine.

The UI spawns ``python -m edith.bridge`` once and talks to it over stdin/stdout:
one JSON request per line in, one JSON response per line out. Each request is
``{"id": <int>, "method": <str>, "params": {…}}``; each response is either
``{"id": …, "result": …}`` or ``{"id": …, "error": {"code": …, "message": …}}``.

The transcriber is cached per ``(model, device)`` so the UI does not reload the
model on every request.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from . import __version__
from .stt import Transcriber

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

    key = (model, device)
    transcriber = _caches.get(key)
    if transcriber is None:
        transcriber = Transcriber(model=model, device=device, compute_type="auto")
        _caches[key] = transcriber

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
        elif method == "tts":
            result = {"ok": False, "error": "TTS is not implemented yet (M4 milestone)"}
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
