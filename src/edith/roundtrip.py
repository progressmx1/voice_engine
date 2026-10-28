"""Round trip: Edith speaks fixed sentences and Whisper transcribes them back (PLAN.md M4).

Speech-to-text checks the voice automatically: if Whisper cannot recover the words, a listener
probably can't either. Whisper writes numbers as digits ("$49.99", "3:30 p.m."), so both sides
go through `edith.text.normalize` before scoring, and WER compares words with words.

    python -m edith.roundtrip      # writes output/roundtrip.md and output/roundtrip/NN.wav
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from edith.audio import STT_RATE, resample, save_wav
from edith.text import normalize
from edith.wer import normalize as wer_normalize
from edith.wer import wer

#: Numbers, names, units, times, money and questions: the things a voice assistant says.
SENTENCES = [
    "Your meeting with Dr. Patel starts at 3:30 pm.",
    "The package weighs 2.5 kg and costs $49.99.",
    "Is it going to rain in London tomorrow?",
    "Set a timer for 15 minutes.",
    "The temperature outside is 22°C with light winds.",
    "Call Sarah back at 555-123-4567 before 6 pm.",
    "Your flight to Tokyo leaves on March 14th at 9:45 am.",
    "Battery is at 37%, so you should charge it soon.",
    "Would you like me to read your new messages?",
    "The project is due on Friday, October 31st.",
    "Michael ran 10 km in 48 minutes this morning.",
    "There are 1,250 unread emails in your inbox.",
    "Remind me to call Mom at 7:00 tonight.",
    "The download is 4.2 GB and will take about 3 minutes.",
    "Who won the World Cup in 2018?",
    "Add eggs, milk and 2 loaves of bread to the shopping list.",
    "The Eiffel Tower is about 330 m tall.",
    "Can you turn the volume down to 20 percent?",
    "Emma and James are arriving at 8:15 on Saturday.",
    "How long does it take to drive from Boston to New York?",
]

#: PLAN.md M4 target for the whole set.
MAX_WER = 0.05


class Speaker(Protocol):
    rate: int

    def synthesize(self, text: str) -> tuple[np.ndarray, int]: ...


class Listener(Protocol):
    def transcribe(self, audio: np.ndarray, language: str = "en") -> object: ...


@dataclass(frozen=True)
class Result:
    """One sentence: what was written, what Edith said, what Whisper heard, and the score."""

    sentence: str
    spoken: str
    heard: str
    wer: float
    words: int
    tts_seconds: float
    audio_seconds: float


def score(spoken: str, heard: str) -> float:
    """WER of what Whisper heard against what Edith was asked to say, both normalised.

    Whisper sometimes writes clock times British-style ("at 8.15"); those are read as times,
    not decimals, so formatting alone doesn't count as an error.
    """
    heard = re.sub(r"\b(at|by|until|from)\s(\d{1,2})\.([0-5]\d)\b", r"\1 \2:\3", heard)
    return wer(spoken, normalize(heard))


def run(speaker: Speaker, listener: Listener, sentences: list[str] = SENTENCES,
        wav_dir: Path | None = None) -> list[Result]:
    """Speak and transcribe each sentence; optionally save the audio as NN.wav."""
    results = []
    for i, sentence in enumerate(sentences, 1):
        start = time.perf_counter()
        audio, rate = speaker.synthesize(sentence)
        tts_seconds = time.perf_counter() - start
        if wav_dir is not None:
            save_wav(wav_dir / f"{i:02d}.wav", audio, rate)
        heard = getattr(listener.transcribe(resample(audio, rate, STT_RATE)), "text")
        spoken = normalize(sentence)
        results.append(Result(sentence, spoken, heard, score(spoken, heard),
                              len(wer_normalize(spoken).split()), tts_seconds, len(audio) / rate))
    return results


def total_wer(results: list[Result]) -> float:
    """Corpus WER: all word errors over all reference words."""
    words = sum(r.words for r in results)
    return sum(r.wer * r.words for r in results) / words if words else 0.0


def report(results: list[Result], device: str = "") -> str:
    """Markdown table of every sentence plus the totals."""
    tts = sum(r.tts_seconds for r in results)
    audio = sum(r.audio_seconds for r in results)
    lines = [
        "# Round trip: Edith speaks, Whisper listens",
        "",
        f"Total WER **{total_wer(results):.1%}** (target <= {MAX_WER:.0%}) over "
        f"{sum(r.words for r in results)} words. TTS RTF {tts / audio:.3f}"
        + (f" on {device}" if device else "") + ".",
        "",
        "| # | WER | Edith said | Whisper heard |",
        "| -- | --- | ---------- | ------------- |",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"| {i} | {r.wer:.0%} | {r.spoken} | {r.heard} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m edith.roundtrip", description=__doc__.split("\n")[0])
    parser.add_argument("--voice", default="edith")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model")
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args(argv)

    from edith.stt import Transcriber
    from edith.tts.engine import Synthesizer

    synth = Synthesizer(voice=args.voice)
    results = run(synth, Transcriber(args.model), wav_dir=args.out / "roundtrip")
    text = report(results, device=getattr(synth.backend, "device", ""))
    (args.out / "roundtrip.md").write_text(text, encoding="utf-8")
    print(text)
    return 0 if total_wer(results) <= MAX_WER else 1


if __name__ == "__main__":
    raise SystemExit(main())
