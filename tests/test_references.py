"""References generator: drafts each sample into data/references.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edith.references import REFERENCES_PATH, build, write
from edith.samples import Sample
from edith.stt import Segment, Transcript


class _FakeTranscriber:
    def __init__(self, model: str = "large-v3", device: str = "auto", compute_type: str = "auto") -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.calls: list[tuple] = []

    def transcribe(self, audio, language: str = "en", *, beam_size: int = 5):
        self.calls.append((audio, language, beam_size))
        return Transcript(
            text="hello world",
            segments=(Segment(0.0, 1.0, "hello"), Segment(1.0, 2.0, "world")),
            language="en",
            seconds=2.0,
            rtf=0.5,
        )


@pytest.fixture()
def fake_samples(monkeypatch) -> list[Sample]:
    samples = [
        Sample(path=Path("a.wav"), name="katie", style="e-learning & podcast host", seconds=2.0),
        Sample(path=Path("b.wav"), name="anne", style="", seconds=2.0),
    ]
    monkeypatch.setattr("edith.references.list_samples", lambda folder=None: samples)
    return samples


def test_build_marks_everything_as_draft(fake_samples) -> None:
    calls = _FakeTranscriber()
    doc = build(transcriber_factory=lambda: calls)

    assert doc["status"] == "draft"
    assert doc["reviewed"] is False
    assert doc["model"] == "large-v3"
    assert doc["beam_size"] == 5
    assert set(doc["samples"]) == {"katie-e-learning-podcast-host", "anne"}
    for entry in doc["samples"].values():
        assert entry["reviewed"] is False
        assert entry["text"] == "hello world"
        assert entry["segments"] == [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]


def test_build_uses_beam_size_and_language(fake_samples) -> None:
    calls = _FakeTranscriber()
    build(transcriber_factory=lambda: calls, beam_size=7)

    assert calls.calls == [
        (Path("a.wav"), "en", 7),
        (Path("b.wav"), "en", 7),
    ]


def test_build_raises_without_samples(monkeypatch) -> None:
    monkeypatch.setattr("edith.references.list_samples", lambda folder=None: [])
    with pytest.raises(RuntimeError):
        build(transcriber_factory=lambda: _FakeTranscriber())


def test_write_creates_json_file(tmp_path, fake_samples) -> None:
    calls = _FakeTranscriber()
    doc = build(transcriber_factory=lambda: calls)
    path = write(doc, tmp_path / "references.json")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["status"] == "draft"
    assert loaded["samples"]["anne"]["text"] == "hello world"
    assert REFERENCES_PATH.name == "references.json"
