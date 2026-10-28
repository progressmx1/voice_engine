"""Round trip: Edith speaks, Whisper listens, WER over the fixed sentences."""

from __future__ import annotations

import numpy as np
import pytest

from edith import models
from edith.roundtrip import MAX_WER, SENTENCES, Result, report, run, score, total_wer
from edith.stt import WHISPER_DIR


def test_twenty_sentences_cover_numbers_units_times_and_questions() -> None:
    assert len(SENTENCES) == 20 == len(set(SENTENCES))
    joined = " ".join(SENTENCES)
    for needle in ("$", "%", "°C", "kg", "km", "GB", ":", "?", "Dr."):
        assert needle in joined


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ("Your meeting with Dr. Patel starts at 3:30 p.m.", 0.0),
        ("Your meeting with Dr. Patel starts at 3.30 p.m.", 0.0),
        ("Your meeting with Dr. Patel starts at 4:30 p.m.", 1 / 11),  # 11 words, one wrong
    ],
)
def test_score_compares_words_not_formatting(heard: str, expected: float) -> None:
    spoken = "Your meeting with Doctor Patel starts at three thirty P M."
    assert score(spoken, heard) == pytest.approx(expected)


def test_score_reads_british_times_after_at() -> None:
    assert score("arriving at eight fifteen on Saturday.", "arriving at 8.15 on Saturday.") == 0.0


class EchoSpeaker:
    """'Speaks' by returning noise; the listener below knows the text."""

    rate = 24_000

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.last = text
        return np.zeros(self.rate // 2, np.float32), self.rate


class ScriptedListener:
    def __init__(self, speaker: EchoSpeaker, mistakes: dict[str, str]) -> None:
        self.speaker, self.mistakes = speaker, mistakes

    def transcribe(self, audio: np.ndarray, language: str = "en") -> object:
        assert len(audio) == 8_000  # resampled from 24 kHz to 16 kHz
        text = self.mistakes.get(self.speaker.last, self.speaker.last)
        return type("T", (), {"text": text})()


def test_run_scores_each_sentence_and_totals_by_words(tmp_path) -> None:
    speaker = EchoSpeaker()
    sentences = ["Set a timer for 15 minutes.", "Is it going to rain?"]
    listener = ScriptedListener(speaker, {sentences[1]: "Is it going to snow?"})
    results = run(speaker, listener, sentences, wav_dir=tmp_path)
    assert [r.wer for r in results] == [0.0, pytest.approx(0.2)]
    assert total_wer(results) == pytest.approx(1 / 11)
    assert (tmp_path / "01.wav").is_file() and (tmp_path / "02.wav").is_file()
    assert "| 2 | 20% |" in report(results)


def test_total_wer_of_nothing_is_zero() -> None:
    assert total_wer([]) == 0.0
    assert total_wer([Result("a", "a", "a", 0.0, 1, 0.1, 1.0)]) == 0.0


def _models_present() -> bool:
    kokoro = all(models.path(n).is_file() for n in ("kokoro-v1.0.onnx", "voices-v1.0.bin"))
    whisper = WHISPER_DIR.is_dir() and any("large-v3-turbo" in p.name for p in WHISPER_DIR.iterdir())
    return kokoro and whisper


@pytest.mark.models
@pytest.mark.skipif(not _models_present(), reason="Kokoro or Whisper large-v3-turbo not downloaded")
def test_round_trip_wer_is_at_most_5_percent() -> None:
    from edith.stt import Transcriber
    from edith.tts.engine import Synthesizer

    results = run(Synthesizer(), Transcriber("large-v3-turbo"))
    assert total_wer(results) <= MAX_WER, report(results)
