"""Intonation: pitch-range expansion keeps length and centre, widens the movement."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("parselmouth")

from edith.quality import voice_measures  # noqa: E402
from edith.tts.engine import Synthesizer  # noqa: E402
from edith.tts.prosody import expand_pitch, reference_pitch  # noqa: E402

RATE = 24_000


def gliding_voice(seconds: float = 2.0, lo: float = 160.0, hi: float = 240.0) -> np.ndarray:
    """A buzzy vowel whose pitch glides up and down between `lo` and `hi` Hz."""
    t = np.arange(int(seconds * RATE)) / RATE
    f0 = np.sqrt(lo * hi) * (hi / lo) ** (0.5 * np.sin(2 * np.pi * t / seconds))
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    wave = sum(np.sin(k * phase) / k for k in range(1, 15))
    return (0.2 * wave).astype(np.float32)


def test_factor_one_returns_audio_unchanged() -> None:
    x = gliding_voice()
    assert np.array_equal(expand_pitch(x, RATE, 1.0, 200.0), x)


def test_expansion_widens_pitch_spread_and_keeps_centre_and_length() -> None:
    x = gliding_voice()
    ref = reference_pitch(x, RATE)
    assert ref == pytest.approx(196, rel=0.05)
    before = voice_measures(x, RATE)
    y = expand_pitch(x, RATE, 1.5, ref)
    after = voice_measures(y, RATE)
    assert len(y) == len(x)
    assert after.f0_spread == pytest.approx(before.f0_spread * 1.5, rel=0.15)
    assert after.f0_median == pytest.approx(before.f0_median, rel=0.05)


def test_compression_narrows_pitch_spread() -> None:
    x = gliding_voice()
    y = expand_pitch(x, RATE, 0.5, reference_pitch(x, RATE))
    assert voice_measures(y, RATE).f0_spread < 0.7 * voice_measures(x, RATE).f0_spread


def test_silence_and_tiny_clips_pass_through() -> None:
    silence = np.zeros(RATE, np.float32)
    assert reference_pitch(silence, RATE) == 0.0
    assert np.array_equal(expand_pitch(silence, RATE, 1.5, 0.0), silence)
    tiny = gliding_voice()[:100]
    assert np.array_equal(expand_pitch(tiny, RATE, 1.5, 200.0), tiny)


class GlideBackend:
    rate = RATE

    def __init__(self) -> None:
        self.synthesized: list[str] = []

    def voices(self) -> list[str]:
        return ["glide"]

    def prepare(self, voice: object, utterance: str) -> None:
        return None

    def synthesize(self, text: str, prepared: object, speed: float) -> np.ndarray:
        self.synthesized.append(text)
        return gliding_voice(1.0)


def test_engine_calibrates_once_per_voice_and_widens_pitch() -> None:
    plain = Synthesizer(backend=GlideBackend())
    backend = GlideBackend()
    wide = Synthesizer(backend=backend, intonation=1.5)
    text = "One. Two. Three."
    a, _ = plain.synthesize(text)
    b, _ = wide.synthesize(text)
    wide.synthesize(text)
    calibrations = [t for t in backend.synthesized if t.startswith("Good morning")]
    assert len(calibrations) == 1
    assert voice_measures(b, RATE).f0_spread > 1.3 * voice_measures(a, RATE).f0_spread


def test_intonation_is_validated() -> None:
    with pytest.raises(ValueError):
        Synthesizer(backend=GlideBackend(), intonation=3.0)
