"""Anti-jitter layer: chunk joins are click-free, DC-free and level-steady."""

from __future__ import annotations

import numpy as np
import pytest

from edith.tts.smooth import (
    Smoother,
    crossfade,
    fade_edges,
    join_jumps,
    level_step_db,
    max_jump,
    ms_to_samples,
    remove_dc,
    rms_dbfs,
    smooth_join,
)

RATE = 24_000


def sine(freq: float, seconds: float, amp: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def noise(seconds: float, dbfs: float, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal(int(seconds * RATE)).astype(np.float32)
    return x / np.sqrt(np.mean(x**2)) * 10 ** (dbfs / 20)


def join_points(lengths: list[int], overlap: int) -> list[int]:
    """Centre of each crossfade in the joined output."""
    points, start = [], 0
    for n in lengths[:-1]:
        start += n - overlap
        points.append(start + overlap // 2)
    return points


def abrupt_chunks() -> list[np.ndarray]:
    """Sines cut mid-cycle at different phases, levels and DC offsets: the worst case for clicks."""
    return [
        sine(220, 0.4, 0.5, phase=0.3) + 0.1,
        sine(180, 0.3, 0.15, phase=2.0) - 0.05,
        sine(260, 0.5, 0.4, phase=4.1),
    ]


def test_remove_dc_centres_signal() -> None:
    assert abs(float(remove_dc(sine(200, 0.5, 0.3) + 0.2).mean())) < 1e-6


def test_fade_edges_start_and_end_near_zero_and_keep_middle() -> None:
    x = np.ones(RATE // 10, dtype=np.float32)
    y = fade_edges(x, RATE, ms=5)
    assert y[0] < 0.01 and y[-1] < 0.01
    assert np.allclose(y[len(y) // 2], 1.0)


def test_crossfade_length_and_constant_power_for_uncorrelated_chunks() -> None:
    a, b = noise(0.5, -20, seed=1), noise(0.5, -20, seed=2)
    n = ms_to_samples(40, RATE)
    y = crossfade(a, b, n)
    assert len(y) == len(a) + len(b) - n
    mid = y[len(a) - n : len(a)]
    assert abs(rms_dbfs(mid) - (-20)) < 1.0


def test_naive_concatenation_clicks_but_smoothed_join_does_not() -> None:
    chunks = abrupt_chunks()
    overlap = ms_to_samples(10, RATE)
    radius = overlap

    naive = np.concatenate(chunks)
    naive_joins = np.cumsum([len(c) for c in chunks])[:-1]
    inside = max(max_jump(c) for c in chunks)
    assert max(join_jumps(naive, naive_joins, 2)) > 5 * inside  # the click the test must catch

    y = smooth_join(chunks, RATE, crossfade_ms=10)
    joins = join_points([len(c) for c in chunks], overlap)
    mask = np.ones(len(y) - 1, dtype=bool)
    for j in joins:
        mask[max(0, j - radius) : j + radius] = False
    interior = float(np.max(np.abs(np.diff(y))[mask]))
    for jump in join_jumps(y, joins, radius):
        assert jump <= 1.5 * interior


def test_output_starts_and_ends_silent() -> None:
    y = smooth_join(abrupt_chunks(), RATE)
    assert abs(y[0]) < 0.01 and abs(y[-1]) < 0.01


def test_loudness_step_between_chunks_is_reduced_below_3_db() -> None:
    a, b = noise(0.5, -14, seed=3), noise(0.5, -24, seed=4)
    naive = np.concatenate([a, b])
    assert level_step_db(naive, len(a), RATE) < -9.0

    overlap = ms_to_samples(10, RATE)
    y = smooth_join([a, b], RATE, crossfade_ms=10)
    join = join_points([len(a), len(b)], overlap)[0]
    assert abs(level_step_db(y, join, RATE)) <= 3.0


def test_gain_never_exceeds_ceiling() -> None:
    loud = sine(200, 0.3, 0.99)
    quiet_peaky = np.zeros(RATE // 4, dtype=np.float32)
    quiet_peaky[100] = 0.9  # low RMS, so the target asks for a big boost, but the peak forbids it
    y = smooth_join([loud, quiet_peaky], RATE, target_dbfs=-6.0)
    assert np.max(np.abs(y)) <= 0.98 + 1e-6


def test_silent_chunk_keeps_gain_and_stays_silent() -> None:
    y = smooth_join([noise(0.3, -20, 5), np.zeros(RATE // 5, np.float32)], RATE)
    assert np.all(np.isfinite(y))
    assert rms_dbfs(y[-RATE // 10 :]) < -80


def test_streaming_holds_back_only_the_crossfade_tail() -> None:
    s = Smoother(RATE, crossfade_ms=10)
    overlap = ms_to_samples(10, RATE)
    chunks = abrupt_chunks()
    out = [s.feed(chunks[0])]
    assert len(out[0]) == len(chunks[0]) - overlap
    out += [s.feed(c) for c in chunks[1:]]
    out.append(s.flush())
    total = sum(len(o) for o in out)
    assert total == sum(len(c) for c in chunks) - overlap * (len(chunks) - 1)
    assert np.allclose(np.concatenate(out), smooth_join(chunks, RATE, crossfade_ms=10))


def test_empty_chunk_is_ignored() -> None:
    s = Smoother(RATE)
    assert s.feed(np.zeros(0, np.float32)).size == 0
    assert s.flush().size == 0


@pytest.mark.parametrize("ms", [5.0, 10.0, 20.0])
def test_short_chunks_shorter_than_overlap_do_not_crash(ms: float) -> None:
    tiny = [sine(200, 0.002, 0.3), sine(300, 0.003, 0.3), sine(250, 0.2, 0.3)]
    y = smooth_join(tiny, RATE, crossfade_ms=ms)
    assert np.all(np.isfinite(y)) and len(y) > 0
