"""JSON bridge dispatch and line protocol (no model download needed)."""

from __future__ import annotations

import json

from edith import bridge
from edith.stt import Segment, Transcript


class _FakeTranscriber:
    def __init__(self, model: str = "large-v3-turbo", device: str = "auto", compute_type: str = "auto") -> None:
        self.model = model
        self.device = "cpu" if device == "cpu" else "cuda"
        self.compute_type = "int8" if self.device == "cpu" else "float16"

    def transcribe(self, audio, language: str = "en", **kwargs):
        return Transcript(
            text="hello world",
            segments=(Segment(0.0, 1.0, "hello"), Segment(1.0, 2.0, "world")),
            language=language,
            seconds=2.0,
            rtf=0.25,
        )


def test_ping_returns_version() -> None:
    response = bridge.dispatch({"id": 1, "method": "ping", "params": {}})
    assert response["id"] == 1
    assert response["result"]["ok"] is True
    assert response["result"]["version"]


def test_transcribe_returns_json_result(monkeypatch) -> None:
    monkeypatch.setattr(bridge, "_caches", {})
    monkeypatch.setattr(bridge, "Transcriber", _FakeTranscriber)

    response = bridge.dispatch(
        {
            "id": 2,
            "method": "transcribe",
            "params": {"audio": "clip.wav", "model": "small.en", "device": "cpu"},
        }
    )

    assert response["result"]["text"] == "hello world"
    assert response["result"]["rtf"] == 0.25
    assert response["result"]["device"] == "cpu"
    assert response["result"]["compute_type"] == "int8"
    assert response["result"]["segments"] == [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "world"},
    ]


def test_transcribe_requires_audio_path(monkeypatch) -> None:
    monkeypatch.setattr(bridge, "_caches", {})
    response = bridge.dispatch({"id": 3, "method": "transcribe", "params": {}})
    assert response["error"]["code"] == -32000
    assert "audio" in response["error"]["message"]


def test_transcribe_caches_per_model_and_device(monkeypatch) -> None:
    created: list[tuple] = []

    class TrackingFake(_FakeTranscriber):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            created.append((self.model, self.device))

    monkeypatch.setattr(bridge, "_caches", {})
    monkeypatch.setattr(bridge, "Transcriber", TrackingFake)

    params = {"audio": "clip.wav", "model": "small.en", "device": "cpu"}
    bridge.dispatch({"id": 4, "method": "transcribe", "params": params})
    bridge.dispatch({"id": 5, "method": "transcribe", "params": params})

    assert created == [("small.en", "cpu")]


def test_tts_is_a_placeholder() -> None:
    response = bridge.dispatch({"id": 6, "method": "tts", "params": {}})
    assert response["result"]["ok"] is False


def test_unknown_method_returns_jsonrpc_style_error() -> None:
    response = bridge.dispatch({"id": 7, "method": "nope"})
    assert response["error"]["code"] == -32601


def test_process_line_handles_bad_json() -> None:
    out = bridge.process_line("not json")
    doc = json.loads(out)
    assert doc["error"]["code"] == -32700


def test_process_line_returns_empty_for_blank_lines() -> None:
    assert bridge.process_line("   ") == ""
