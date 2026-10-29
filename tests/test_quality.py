"""Quality measures: jitter and shimmer (Praat), join clicks and loudness steps, the report."""

from __future__ import annotations

import numpy as np
import pytest

from edith import models
from edith.quality import (
    JoinMeasures,
    Report,
    VoiceMeasures,
    active_level,
    in_range,
    join_measures,
    voice_measures,
)

pytest.importorskip("parselmouth")

RATE = 24_000


def pulse_voice(seconds: float, f0: float, period_jitter: float, amp_jitter: float,
                seed: int = 0) -> np.ndarray:
    """A vowel-like signal: decaying resonance per glottal pulse, with chosen random wobble."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * RATE) + RATE // 10, np.float32)
    t = 0.0
    ring = np.exp(-np.arange(200) / 40.0) * np.sin(2 * np.pi * 700 * np.arange(200) / RATE)
    while t < seconds:
        i = int(t * RATE)
        out[i : i + len(ring)] += (0.4 * (1 + amp_jitter * rng.standard_normal()) * ring)
        t += (1 + period_jitter * rng.standard_normal()) / f0
    return out[: int(seconds * RATE)]


def test_steady_voice_has_low_jitter_and_wobbly_voice_has_high() -> None:
    steady = voice_measures(pulse_voice(1.5, 150, 0.0, 0.0), RATE)
    wobbly = voice_measures(pulse_voice(1.5, 150, 0.03, 0.15), RATE)
    assert steady.jitter < 0.005 and steady.shimmer < 0.02
    assert wobbly.jitter > 0.015 and wobbly.shimmer > 0.05
    assert steady.f0_median == pytest.approx(150, rel=0.03)


def test_silence_has_no_pitch() -> None:
    m = voice_measures(np.zeros(RATE, np.float32), RATE)
    assert m.f0_median == 0.0


def test_active_level_ignores_silence() -> None:
    tone = 0.1 * np.sin(2 * np.pi * 200 * np.arange(RATE) / RATE).astype(np.float32)
    padded = np.concatenate([tone, np.zeros(RATE * 3, np.float32)])
    assert active_level(padded, RATE) == pytest.approx(active_level(tone, RATE), abs=0.5)


def tone(seconds: float, amp: float) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * 200 * np.arange(int(seconds * RATE)) / RATE)).astype(np.float32)


def test_join_measures_catch_a_click_and_a_loudness_step() -> None:
    a, b = tone(0.5, 0.3), tone(0.5, 0.3)
    clicky = np.concatenate([a, b + 0.5])  # DC step at the join
    m = join_measures(clicky, [len(a)], RATE)
    assert m.jump_ratio > MAX_RATIO_FOR_CLICK and not m.ok

    smooth = np.concatenate([a, tone(0.5, 0.3)])
    assert join_measures(smooth, [len(a)], RATE).jump_ratio <= 1.5

    steps = np.concatenate([tone(0.5, 0.3), tone(0.5, 0.075)])  # 12 dB quieter
    m = join_measures(steps, [RATE // 2], RATE)
    assert m.level_spread_db == pytest.approx(12.0, abs=0.5) and not m.ok


MAX_RATIO_FOR_CLICK = 5.0


def test_in_range() -> None:
    assert in_range(2.0, [1.0, 3.0]) and not in_range(3.5, [1.0, 3.0]) and not in_range(1.0, [])


def measures(jitter: float, shimmer: float, spread: float = 10.0) -> VoiceMeasures:
    return VoiceMeasures(jitter, shimmer, 200.0, spread)


def test_report_judges_wobble_as_ceiling_spread_as_range_and_joins() -> None:
    humans = {"a": measures(0.019, 0.07, 7.0), "b": measures(0.024, 0.10, 15.0)}
    good = Report(measures(0.02, 0.08), [measures(0.02, 0.08)], JoinMeasures(3, 0.5, 1.0), humans)
    assert good.ok and "Overall: **pass**" in good.markdown()
    steadier = Report(measures(0.015, 0.06), good.sentences, good.joins, humans)
    assert steadier.ok  # less wobble than people is fine
    wobbly = Report(measures(0.03, 0.08), good.sentences, good.joins, humans)
    assert not wobbly.jitter_ok and not wobbly.ok
    shaky = Report(measures(0.02, 0.12), good.sentences, good.joins, humans)
    assert not shaky.shimmer_ok and not shaky.ok
    flat = Report(measures(0.02, 0.08, 5.0), good.sentences, good.joins, humans)
    singsong = Report(measures(0.02, 0.08, 20.0), good.sentences, good.joins, humans)
    assert not flat.spread_ok and not singsong.spread_ok and not flat.ok
    clicky = Report(good.paragraph, good.sentences, JoinMeasures(3, 4.0, 1.0), humans)
    assert not clicky.ok and "FAIL" in clicky.markdown()


def _ready() -> bool:
    from edith.samples import list_samples

    kokoro = all(models.path(n).is_file() for n in ("kokoro-v1.0.onnx", "voices-v1.0.bin"))
    return kokoro and bool(list_samples())


@pytest.mark.models
@pytest.mark.skipif(not _ready(), reason="Kokoro files or sample voices missing")
def test_edith_voice_quality_passes() -> None:
    from edith.quality import measure
    from edith.roundtrip import SENTENCES
    from edith.tts.engine import Synthesizer

    report = measure(Synthesizer(), SENTENCES)
    assert report.joins.joins >= 19
    assert report.ok, report.markdown()
