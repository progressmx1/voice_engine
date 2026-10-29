"""JSON bridge dispatch and line protocol (no model download needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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


class _FakeTtsBackend:
    """Speaks every chunk as half a second of a buzzy 200 Hz vowel."""

    rate = 24_000
    device = "cpu"

    def voices(self) -> list[str]:
        return ["af_heart", "bf_lily"]

    def prepare(self, voice, utterance: str):
        self.voice = voice

    def synthesize(self, text: str, prepared, speed: float) -> np.ndarray:
        phase = 2 * np.pi * 200 * np.arange(self.rate // 2) / self.rate
        return (0.2 * sum(np.sin(k * phase) / k for k in range(1, 10))).astype(np.float32)


@pytest.fixture
def fake_tts(monkeypatch, tmp_path):
    backend = _FakeTtsBackend()
    monkeypatch.setattr(bridge, "_tts_backends", {"auto": backend, "cpu": backend})
    monkeypatch.setattr(bridge, "TTS_DIR", tmp_path / "ui-tts")
    return backend


def test_tts_voices_lists_voices_and_presets(fake_tts) -> None:
    result = bridge.dispatch({"id": 6, "method": "tts_voices", "params": {}})["result"]
    assert {"edith", "af_heart", "bf_lily"} <= set(result["voices"])
    assert len(result["presets"]) == 5 and result["default_text"]


def test_tts_speaks_and_reports_timings_text_and_measures(fake_tts) -> None:
    pytest.importorskip("parselmouth")
    params = {"text": "It costs $5. Bye!", "voice": {"af_heart": 0.5, "bf_lily": 0.5},
              "intonation": 1.0, "speed": 1.0}
    result = bridge.dispatch({"id": 7, "method": "tts", "params": params})["result"]
    assert Path(result["path"]).is_file()
    assert result["spoken"] == "It costs five dollars. Bye!"
    assert result["chunks"] == ["It costs five dollars.", "Bye!"]
    assert result["seconds"] > 0.9 and result["rtf"] > 0 and result["first_audio_ms"] > 0
    assert result["measures"]["f0_median"] == pytest.approx(200, rel=0.05)
    assert fake_tts.voice == {"af_heart": 0.5, "bf_lily": 0.5}


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({}, "text"),
        ({"text": "x" * 2001}, "limited"),
        ({"text": "hi", "voice": {"af_heart": 0}}, "voice"),
        ({"text": "hi", "intonation": 5}, "intonation"),
    ],
)
def test_tts_rejects_bad_params(fake_tts, params, message) -> None:
    response = bridge.dispatch({"id": 8, "method": "tts", "params": params})
    assert message in response["error"]["message"]


def test_tts_check_scores_whisper_against_spoken_text(fake_tts, monkeypatch) -> None:
    pytest.importorskip("parselmouth")

    class Heard(_FakeTranscriber):
        def transcribe(self, audio, language: str = "en", **kwargs):
            return Transcript("It costs $5. Hi!", (), language, 1.0, 0.1)

    monkeypatch.setattr(bridge, "_caches", {})
    monkeypatch.setattr(bridge, "Transcriber", Heard)
    made = bridge.dispatch({"id": 9, "method": "tts", "params": {"text": "It costs $5. Bye!"}})
    clip = made["result"]
    params = {"path": clip["path"], "spoken": clip["spoken"]}
    result = bridge.dispatch({"id": 10, "method": "tts_check", "params": params})["result"]
    assert result["heard"] == "It costs $5. Hi!"
    assert result["wer"] == pytest.approx(1 / 5)


def test_tts_check_only_reads_ui_clips(fake_tts, tmp_path) -> None:
    other = tmp_path / "secret.wav"
    other.write_bytes(b"RIFF")
    response = bridge.dispatch({"id": 11, "method": "tts_check", "params": {"path": str(other)}})
    assert "output/ui-tts" in response["error"]["message"]


def test_unknown_method_returns_jsonrpc_style_error() -> None:
    response = bridge.dispatch({"id": 7, "method": "nope"})
    assert response["error"]["code"] == -32601


def test_process_line_handles_bad_json() -> None:
    out = bridge.process_line("not json")
    doc = json.loads(out)
    assert doc["error"]["code"] == -32700


def test_process_line_returns_empty_for_blank_lines() -> None:
    assert bridge.process_line("   ") == ""
