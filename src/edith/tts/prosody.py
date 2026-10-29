"""Intonation: widen how far the pitch moves, keeping the words, timing and voice.

Kokoro's voices move their pitch across about 6-11 semitones when reading; people reading the
same way move across 13-17 (measured by `edith.quality` on the sample voices). `expand_pitch`
takes the pitch contour Kokoro produced, scales every point's distance from a reference pitch
by `factor` on a log scale (`ref * (f0 / ref) ** factor`), and resynthesizes with Praat's
overlap-add (PSOLA). Rises get higher and falls lower; the reference pitch itself is unchanged.

The reference must be the same for every chunk of a voice (`reference_pitch`), or chunks would
be scaled around different centres and the voice would jump between them.
"""

from __future__ import annotations

import numpy as np

#: Pitch analysis range (Hz) for Edith's voices.
F0_MIN, F0_MAX = 75.0, 600.0
#: Praat's pitch-analysis step for the manipulation.
TIME_STEP = 0.01


def reference_pitch(audio: np.ndarray, rate: int) -> float:
    """Median pitch (Hz) of voiced frames, or 0.0 if there are none."""
    import parselmouth

    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), float(rate))
    f0 = sound.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX).selected_array["frequency"]
    f0 = f0[f0 > 0]
    return float(np.median(f0)) if f0.size else 0.0


def expand_pitch(audio: np.ndarray, rate: int, factor: float, ref_hz: float) -> np.ndarray:
    """Scale the pitch movement of `audio` around `ref_hz` by `factor` (1.0 = unchanged)."""
    if factor == 1.0 or ref_hz <= 0 or len(audio) < int(0.05 * rate):
        return np.asarray(audio, dtype=np.float32)
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), float(rate))
    manipulation = call(sound, "To Manipulation", TIME_STEP, F0_MIN, F0_MAX)
    tier = call(manipulation, "Extract pitch tier")
    n = int(call(tier, "Get number of points"))
    if n == 0:
        return np.asarray(audio, dtype=np.float32)
    points = [(call(tier, "Get time from index", i), call(tier, "Get value at index", i))
              for i in range(1, n + 1)]
    call(tier, "Remove points between", 0, sound.duration)
    for t, f0 in points:
        call(tier, "Add point", t, ref_hz * (f0 / ref_hz) ** factor)
    call([tier, manipulation], "Replace pitch tier")
    out = call(manipulation, "Get resynthesis (overlap-add)").values[0]
    return out[: len(audio)].astype(np.float32)
