"""Anti-jitter layer: joins synthesized chunks into one continuous, click-free, level-steady signal.

Speech is synthesized clause by clause, and every chunk boundary is a chance for a click (a
sudden sample jump), a DC step or a loudness jump. `Smoother` fixes all three as chunks stream
in, holding back only a short tail to crossfade into the next chunk:

1. DC removal, so chunks don't sit at different offsets.
2. Loudness matching toward a target level, keeping part of the natural variation, with the gain
   ramped from the previous chunk's gain so it never steps.
3. A raised-cosine fade-in on the first chunk, equal-power crossfades between chunks and a
   fade-out on flush.

The metrics at the bottom (`max_jump`, `join_jumps`, `level_step_db`) are what the tests and
`edith.quality` use to prove there is no jitter at the joins.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

#: Below this level a chunk counts as silence and keeps the previous gain.
SILENCE_DBFS = -60.0


def db_to_gain(db: float) -> float:
    """Decibels to a linear amplitude factor."""
    return float(10.0 ** (db / 20.0))


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dB relative to full scale; -inf for silence."""
    if x.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")


def ms_to_samples(ms: float, rate: int) -> int:
    """Milliseconds to a whole number of samples."""
    return int(round(ms * rate / 1000.0))


def remove_dc(x: np.ndarray) -> np.ndarray:
    """Subtract the mean so the chunk is centred on zero."""
    return (x - x.mean()).astype(np.float32) if x.size else x.astype(np.float32)


def fade_curve(n: int) -> np.ndarray:
    """Raised-cosine ramp from 0 to 1 over `n` samples (smooth start and end slope)."""
    if n <= 0:
        return np.ones(0, dtype=np.float32)
    t = (np.arange(n, dtype=np.float64) + 0.5) / n
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def fade_edges(x: np.ndarray, rate: int, ms: float = 5.0) -> np.ndarray:
    """Fade both ends of `x` in and out over `ms` milliseconds."""
    y = x.astype(np.float32, copy=True)
    n = min(ms_to_samples(ms, rate), len(y) // 2)
    if n:
        ramp = fade_curve(n)
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return y


def crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Overlap the last `n` samples of `a` with the first `n` of `b` using equal-power curves.

    Equal power (sin/cos) keeps the level constant for uncorrelated material, which is what
    two separately synthesized chunks are. The result is `len(a) + len(b) - n` samples long.
    """
    n = min(n, len(a), len(b))
    if n == 0:
        return np.concatenate([a, b]).astype(np.float32)
    t = (np.arange(n, dtype=np.float64) + 0.5) / n
    fade_in = np.sin(0.5 * np.pi * t).astype(np.float32)
    fade_out = np.cos(0.5 * np.pi * t).astype(np.float32)
    mid = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([a[:-n], mid, b[n:]]).astype(np.float32)


class Smoother:
    """Streams chunks in and click-free, level-matched audio out.

    `feed(chunk)` returns the audio that is final so far; the last `crossfade_ms` are held back
    to blend with the next chunk. `flush()` returns the faded-out remainder. Use one Smoother
    per utterance.
    """

    def __init__(
        self,
        rate: int,
        crossfade_ms: float = 10.0,
        target_dbfs: float = -20.0,
        keep_variation: float = 0.25,
        ramp_ms: float = 30.0,
        ceiling: float = 0.98,
    ) -> None:
        """
        Args:
            rate: sample rate in Hz.
            crossfade_ms: overlap between neighbouring chunks.
            target_dbfs: level that chunks are pulled toward.
            keep_variation: share of each chunk's own level difference from the target that is
                kept (0 = every chunk exactly at the target, 1 = no loudness matching).
            ramp_ms: time over which the gain glides from the previous chunk's gain.
            ceiling: peak limit; a chunk's gain is lowered so it never exceeds this.
        """
        self.rate = rate
        self.overlap = ms_to_samples(crossfade_ms, rate)
        self.target_dbfs = target_dbfs
        self.keep_variation = keep_variation
        self.ramp = ms_to_samples(ramp_ms, rate)
        self.ceiling = ceiling
        self._gain: float | None = None
        self._tail = np.zeros(0, dtype=np.float32)
        self._started = False

    def _chunk_gain(self, x: np.ndarray) -> float:
        """Gain that moves this chunk toward the target level without clipping."""
        level = rms_dbfs(x)
        if level < SILENCE_DBFS:
            return self._gain if self._gain is not None else 1.0
        wanted = self.target_dbfs + self.keep_variation * (level - self.target_dbfs)
        gain = db_to_gain(wanted - level)
        peak = float(np.max(np.abs(x)))
        return min(gain, self.ceiling / peak) if peak > 0 else gain

    def _apply_gain(self, x: np.ndarray, gain: float) -> np.ndarray:
        """Multiply by `gain`, gliding from the previous gain over the ramp so it never steps."""
        start = self._gain if self._gain is not None else gain
        g = np.full(len(x), gain, dtype=np.float32)
        n = min(self.ramp, len(x))
        if n and start != gain:
            g[:n] = start + (gain - start) * fade_curve(n)
        self._gain = gain
        return x * g

    def feed(self, chunk: np.ndarray) -> np.ndarray:
        """Add a chunk; return the audio that will not change any more."""
        x = remove_dc(np.asarray(chunk, dtype=np.float32))
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        x = self._apply_gain(x, self._chunk_gain(x))
        if not self._started:
            n = min(self.overlap, len(x))
            x[:n] *= fade_curve(n)
            self._started = True
            joined = x
        else:
            joined = crossfade(self._tail, x, self.overlap)
        hold = min(self.overlap, len(joined))
        self._tail = joined[len(joined) - hold :].copy()
        return joined[: len(joined) - hold]

    def flush(self) -> np.ndarray:
        """Return the held tail with a fade-out; the Smoother is then empty."""
        tail = self._tail * fade_curve(len(self._tail))[::-1]
        self._tail = np.zeros(0, dtype=np.float32)
        return tail.astype(np.float32)


def smooth_join(chunks: Iterable[np.ndarray], rate: int, **options: float) -> np.ndarray:
    """Join whole chunks in one call (same processing as streaming them through a Smoother)."""
    smoother = Smoother(rate, **options)
    parts = [smoother.feed(c) for c in chunks]
    parts.append(smoother.flush())
    return np.concatenate(parts).astype(np.float32)


# --- Metrics -------------------------------------------------------------------------------


def max_jump(x: np.ndarray) -> float:
    """Largest difference between neighbouring samples (a click shows up as a large jump)."""
    return float(np.max(np.abs(np.diff(x)))) if len(x) > 1 else 0.0


def join_jumps(x: np.ndarray, joins: Iterable[int], radius: int) -> list[float]:
    """Largest sample jump within `radius` samples of each join index."""
    out = []
    for j in joins:
        lo, hi = max(0, j - radius), min(len(x), j + radius + 1)
        out.append(max_jump(x[lo:hi]))
    return out


def level_step_db(x: np.ndarray, join: int, rate: int, ms: float = 50.0) -> float:
    """Level of the window after `join` minus the window before it, in dB."""
    n = ms_to_samples(ms, rate)
    before, after = x[max(0, join - n) : join], x[join : join + n]
    return rms_dbfs(after) - rms_dbfs(before)
