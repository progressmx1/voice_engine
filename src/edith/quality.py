"""Quality report for Edith's voice: jitter, shimmer, pitch and chunk joins, against real speech.

Jitter (cycle-to-cycle pitch wobble) and shimmer (cycle-to-cycle loudness wobble) are measured
with Praat (parselmouth), the standard tool for them. Clinical norms (jitter ≤ 1.04%, shimmer
≤ 3.81%) are for sustained vowels; running speech scores higher even for real people. So Edith
is judged against the sample voices measured the same way. They are real human speech used
only as a measuring stick, never for training (PLAN.md rule 6).

Joins are checked on a long paragraph: the largest sample jump near any join must stay within
1.5 times the largest jump elsewhere (no clicks), and the speech level of every chunk must stay
within 3 dB of the others (no loudness steps).

    python -m edith.quality      # writes output/quality.md and output/quality_paragraph.wav
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edith.audio import load_audio, save_wav
from edith.roundtrip import SENTENCES
from edith.tts.smooth import join_jumps, ms_to_samples, rms_dbfs

#: Praat's default pitch range for running speech.
F0_MIN, F0_MAX = 75.0, 500.0
MAX_JOIN_RATIO = 1.5
MAX_LEVEL_SPREAD_DB = 3.0
#: 20 ms frames above this count as speech when measuring a chunk's level.
ACTIVE_DBFS = -50.0


@dataclass(frozen=True)
class VoiceMeasures:
    """Praat measures of one recording. Jitter and shimmer are fractions (0.02 = 2%)."""

    jitter: float
    shimmer: float
    f0_median: float
    f0_spread: float  # semitones between the 10th and 90th percentile: how much intonation moves


@dataclass(frozen=True)
class JoinMeasures:
    joins: int
    jump_ratio: float  # worst jump near a join / largest jump elsewhere
    level_spread_db: float  # loudest chunk minus quietest chunk, speech frames only

    @property
    def ok(self) -> bool:
        return self.jump_ratio <= MAX_JOIN_RATIO and self.level_spread_db <= MAX_LEVEL_SPREAD_DB


def voice_measures(audio: np.ndarray, rate: int) -> VoiceMeasures:
    """Local jitter and shimmer (Praat defaults) and pitch statistics."""
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), float(rate))
    points = call(sound, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
    jitter = call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([sound, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    f0 = sound.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX).selected_array["frequency"]
    f0 = f0[f0 > 0]
    if f0.size == 0:
        return VoiceMeasures(float(jitter), float(shimmer), 0.0, 0.0)
    lo, hi = np.percentile(f0, [10, 90])
    return VoiceMeasures(float(jitter), float(shimmer), float(np.median(f0)),
                         float(12 * np.log2(hi / lo)))


def active_level(audio: np.ndarray, rate: int) -> float:
    """Level in dBFS of the 20 ms frames that contain speech (silence and pauses excluded)."""
    n = ms_to_samples(20, rate)
    frames = [audio[i : i + n] for i in range(0, len(audio) - n + 1, n)]
    active = [f for f in frames if rms_dbfs(f) > ACTIVE_DBFS]
    return rms_dbfs(np.concatenate(active)) if active else float("-inf")


def join_measures(audio: np.ndarray, joins: list[int], rate: int, radius_ms: float = 10.0) -> JoinMeasures:
    """Click and loudness-step checks at the given join positions."""
    radius = ms_to_samples(radius_ms, rate)
    diffs = np.abs(np.diff(audio))
    away = np.ones(len(diffs), dtype=bool)
    for j in joins:
        away[max(0, j - radius) : j + radius] = False
    elsewhere = float(diffs[away].max()) if away.any() else 0.0
    near = max(join_jumps(audio, joins, radius), default=0.0)
    ratio = near / elsewhere if elsewhere > 0 else 0.0
    edges = [0, *joins, len(audio)]
    levels = [active_level(audio[a:b], rate) for a, b in zip(edges, edges[1:])]
    levels = [lv for lv in levels if np.isfinite(lv)]
    spread = max(levels) - min(levels) if levels else 0.0
    return JoinMeasures(len(joins), ratio, spread)


def human_baseline(rate: int = 24_000) -> dict[str, VoiceMeasures]:
    """The same measures on every sample voice (real speech), keyed by sample key."""
    from edith.samples import list_samples

    return {s.key: voice_measures(load_audio(s.path, rate), rate) for s in list_samples()}


def in_range(value: float, values: list[float]) -> bool:
    return bool(values) and min(values) <= value <= max(values)


@dataclass(frozen=True)
class Report:
    paragraph: VoiceMeasures
    sentences: list[VoiceMeasures]
    joins: JoinMeasures
    humans: dict[str, VoiceMeasures]
    device: str = ""

    @property
    def jitter_ok(self) -> bool:
        return in_range(self.paragraph.jitter, [h.jitter for h in self.humans.values()])

    @property
    def shimmer_ok(self) -> bool:
        return in_range(self.paragraph.shimmer, [h.shimmer for h in self.humans.values()])

    @property
    def ok(self) -> bool:
        return self.jitter_ok and self.shimmer_ok and self.joins.ok

    def markdown(self) -> str:
        def row(name: str, m: VoiceMeasures) -> str:
            return (f"| {name} | {m.jitter:.2%} | {m.shimmer:.2%} | {m.f0_median:.0f} Hz "
                    f"| {m.f0_spread:.1f} st |")

        h = list(self.humans.values())
        rng = (lambda xs: f"{min(xs):.2%} to {max(xs):.2%}") if h else (lambda xs: "n/a")
        mark = {True: "pass", False: "FAIL"}
        jit = [m.jitter for m in self.sentences]
        shim = [m.shimmer for m in self.sentences]
        lines = [
            "# Edith voice quality",
            "",
            f"Overall: **{mark[self.ok]}**" + (f" (synthesized on {self.device})" if self.device else ""),
            "",
            "| Check | Edith | Target | Result |",
            "| ----- | ----- | ------ | ------ |",
            f"| Jitter, paragraph | {self.paragraph.jitter:.2%} | human range "
            f"{rng([x.jitter for x in h])} | {mark[self.jitter_ok]} |",
            f"| Shimmer, paragraph | {self.paragraph.shimmer:.2%} | human range "
            f"{rng([x.shimmer for x in h])} | {mark[self.shimmer_ok]} |",
            f"| Click at joins ({self.joins.joins} joins) | {self.joins.jump_ratio:.2f}× | "
            f"≤ {MAX_JOIN_RATIO}× largest jump elsewhere | "
            f"{mark[self.joins.jump_ratio <= MAX_JOIN_RATIO]} |",
            f"| Loudness spread between chunks | {self.joins.level_spread_db:.1f} dB | "
            f"≤ {MAX_LEVEL_SPREAD_DB:.0f} dB | "
            f"{mark[self.joins.level_spread_db <= MAX_LEVEL_SPREAD_DB]} |",
            "| Naturalness (UTMOS) | not measured | ≥ Kokoro baseline | needs PyTorch |",
            "",
            f"Per sentence ({len(self.sentences)}): jitter median {np.median(jit):.2%} "
            f"(max {max(jit):.2%}), shimmer median {np.median(shim):.2%} (max {max(shim):.2%}). "
            "Short clips vary more than the paragraph, so the paragraph is what is judged.",
            "",
            "| Voice | Jitter | Shimmer | Median pitch | Pitch spread |",
            "| ----- | ------ | ------- | ------------ | ------------ |",
            row("**Edith (paragraph)**", self.paragraph),
            *[row(k, m) for k, m in self.humans.items()],
            "",
            "Clinical norms (jitter ≤ 1.04%, shimmer ≤ 3.81%) are for sustained vowels and are "
            "not used: real people reading normally exceed them, as the table shows.",
        ]
        return "\n".join(lines) + "\n"


def measure(synth: object, sentences: list[str] = SENTENCES, wav: Path | None = None,
            humans: dict[str, VoiceMeasures] | None = None) -> Report:
    """Synthesize the sentences one by one and as one paragraph, and measure everything."""
    rate = synth.rate
    per_sentence = [voice_measures(synth.synthesize(s)[0], rate) for s in sentences]
    audio, _ = synth.synthesize(" ".join(sentences))
    joins = list(synth.last_joins)
    if wav is not None:
        save_wav(wav, audio, rate)
    return Report(
        paragraph=voice_measures(audio, rate),
        sentences=per_sentence,
        joins=join_measures(audio, joins, rate),
        humans=human_baseline(rate) if humans is None else humans,
        device=getattr(getattr(synth, "backend", None), "device", ""),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m edith.quality", description=__doc__.split("\n")[0])
    parser.add_argument("--voice", default="edith")
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args(argv)

    from edith.tts.engine import Synthesizer

    report = measure(Synthesizer(voice=args.voice), wav=args.out / "quality_paragraph.wav")
    text = report.markdown()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "quality.md").write_text(text, encoding="utf-8")
    print(f"{'pass' if report.ok else 'FAIL'}: wrote {args.out / 'quality.md'}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
