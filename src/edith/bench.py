"""Benchmark harness for Edith.

Run it as ``python -m edith.bench [stt|tts|all]``. Each stage is timed with
``time.perf_counter``; the harness reports the real-time factor
(RTF = processing time / audio duration) and end-to-end latency, prints a table
and writes ``output/bench.json``.

The measurement core is engine-agnostic so it can be tested with a dummy engine
that sleeps, and later reused unchanged by the STT (M2) and TTS (M4) stages.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# The NVIDIA CUDA DLL folders must be on the search path before faster-whisper/
# CTranslate2 can be imported by any stage. ``edith.stt`` does this on import
# too; doing it here as well keeps ``python -m edith.bench`` safe as an entry
# point regardless of import order.
try:
    from .stt import setup_cuda_dlls as _setup_cuda_dlls
except ImportError:  # faster-whisper (STT extra) not installed yet
    pass
else:
    _setup_cuda_dlls()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_OUT = OUTPUT_DIR / "bench.json"

#: Benchmarkable stages, in build order.
STAGES = ("stt", "tts")

#: Default speech-to-text model benchmarked by the ``stt`` stage.
DEFAULT_STT_MODEL = "large-v3-turbo"

#: Speech-to-text models benchmarked by the ``stt`` stage (M2 targets).
STT_MODELS = ("large-v3-turbo", "small.en")

#: Reference transcripts used to score WER (drafted by ``edith.references``).
REFERENCES_PATH = PROJECT_ROOT / "data" / "references.json"


@dataclass(frozen=True)
class Result:
    """One measured stage.

    ``latency_*`` is the wall-clock time of a single call (request to result).
    ``rtf_*`` is latency divided by the audio length the call processed.
    """

    stage: str
    engine: str
    device: str
    audio_seconds: float
    runs: int
    rtf_mean: float
    rtf_min: float
    rtf_max: float
    latency_mean: float
    latency_min: float
    latency_max: float
    meta: dict[str, Any] = field(default_factory=dict)
    wer: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable copy of this result."""
        return asdict(self)


def measure(
    stage: str,
    fn: Callable[[], object],
    audio_seconds: float,
    *,
    runs: int = 3,
    warmup: int = 0,
    engine: str = "",
    device: str = "cpu",
    meta: dict[str, Any] | None = None,
) -> Result:
    """Time ``fn`` (which processes ``audio_seconds`` of audio) and summarise.

    ``warmup`` calls are run first and not measured, so model loading and caches
    do not pollute the numbers.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be 0 or more")
    if audio_seconds <= 0:
        raise ValueError("audio_seconds must be positive")

    for _ in range(warmup):
        fn()

    latencies: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        latencies.append(time.perf_counter() - start)

    rtfs = [latency / audio_seconds for latency in latencies]
    return Result(
        stage=stage,
        engine=engine,
        device=device,
        audio_seconds=audio_seconds,
        runs=runs,
        rtf_mean=_mean(rtfs),
        rtf_min=min(rtfs),
        rtf_max=max(rtfs),
        latency_mean=_mean(latencies),
        latency_min=min(latencies),
        latency_max=max(latencies),
        meta=dict(meta or {}),
    )


def build_stage(
    stage: str,
    device: str,
    model: str = DEFAULT_STT_MODEL,
    compute_type: str = "auto",
) -> tuple[Callable[[], object], float, str, str] | None:
    """Build the callable to measure for ``stage``, or ``None`` if not ready.

    Returns ``(fn, audio_seconds, engine_label, device_label)``. M2 wires in the
    STT engine (``edith.stt.Transcriber``) and M4 the TTS engine
    (``edith.tts.Synthesizer``); until then the harness skips them honestly.
    """
    if stage == "stt":
        try:
            from .samples import list_samples
            from .stt import Transcriber
        except ImportError:
            return None
        samples = list_samples()
        if not samples:
            return None
        sample = samples[0]  # deterministic: list_samples sorts by key
        try:
            transcriber = Transcriber(
                model=model, device=device, compute_type=compute_type
            )
        except ImportError:
            return None

        def fn() -> object:
            return transcriber.transcribe(sample.path)

        return (
            fn,
            sample.seconds,
            f"faster-whisper/{transcriber.model}",
            transcriber.device,
        )
    if stage == "tts":
        try:
            from .tts import Synthesizer  # noqa: F401
        except ImportError:
            return None
        return None  # M4: synthesize a fixed sentence and report its duration.
    raise ValueError(f"unknown stage: {stage}")


def run_stage(
    stage: str,
    *,
    device: str = "auto",
    runs: int = 3,
    warmup: int = 0,
    model: str = DEFAULT_STT_MODEL,
    compute_type: str = "auto",
) -> Result | None:
    """Build and benchmark one stage, returning ``None`` if it is not ready."""
    built = build_stage(stage, device, model=model, compute_type=compute_type)
    if built is None:
        return None
    fn, audio_seconds, engine, device_used = built
    return measure(
        stage,
        fn,
        audio_seconds,
        runs=runs,
        warmup=warmup,
        engine=engine,
        device=device_used,
    )


def load_references(path: Path | None = None) -> dict[str, Any] | None:
    """Load ``data/references.json``, or ``None`` if it is missing/invalid."""
    ref_path = Path(path) if path else REFERENCES_PATH
    if not ref_path.is_file():
        return None
    try:
        doc = json.loads(ref_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return doc if isinstance(doc, dict) else None


def reference_text(key: str, doc: dict[str, Any] | None = None) -> str | None:
    """Reference transcript text for a sample key, or ``None`` if unavailable."""
    doc = doc if doc is not None else load_references()
    if not doc:
        return None
    entry = (doc.get("samples") or {}).get(key)
    if not isinstance(entry, dict):
        return None
    text = entry.get("text")
    return text if isinstance(text, str) and text.strip() else None


def run_stt(
    *,
    device: str = "auto",
    runs: int = 3,
    warmup: int = 0,
    model: str = DEFAULT_STT_MODEL,
    compute_type: str = "auto",
    limit: int | None = None,
) -> list[Result]:
    """Benchmark one STT model on the samples and score each against references.

    Returns one :class:`Result` per sample (empty when STT is not ready or no
    samples are found). ``limit`` restricts the run to the first ``limit``
    samples, which is the "quick" mode.
    """
    from .samples import list_samples
    from .stt import Transcriber
    from .wer import wer

    samples = list_samples()
    if limit is not None:
        samples = samples[: max(limit, 0)]
    if not samples:
        return []

    try:
        transcriber = Transcriber(
            model=model, device=device, compute_type=compute_type
        )
    except ImportError:
        return []

    results: list[Result] = []
    for sample in samples:
        def fn() -> object:
            return transcriber.transcribe(sample.path)

        # One unmeasured call warms the model and caches, and gives us WER text.
        transcript = fn()
        ref = reference_text(sample.key)
        sample_wer = wer(ref, transcript.text) if ref is not None else None

        result = measure(
            "stt",
            fn,
            sample.seconds,
            runs=runs,
            warmup=warmup,
            engine=f"faster-whisper/{transcriber.model}",
            device=transcriber.device,
            meta={
                "sample": sample.key,
                "model": model,
                "compute_type": transcriber.compute_type,
            },
        )
        results.append(_with_wer(result, sample_wer))
    return results


def _with_wer(result: Result, value: float | None) -> Result:
    """Return ``result`` with its WER set (frozen dataclass → copy)."""
    import dataclasses

    return dataclasses.replace(result, wer=value)


def format_table(results: Sequence[Result]) -> str:
    """Render results as a compact text table."""
    if not results:
        return "No stages were benchmarked.\n"
    header = (
        f"{'stage':<6} {'engine':<28} {'device':<8} {'audio_s':>8} "
        f"{'runs':>5} {'RTF':>8} {'WER':>7} {'latency_s':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        wer_s = f"{r.wer:>7.3f}" if r.wer is not None else f"{'n/a':>7}"
        lines.append(
            f"{r.stage:<6} {r.engine:<28} {r.device:<8} {r.audio_seconds:>8.3f} "
            f"{r.runs:>5} {r.rtf_mean:>8.3f} {wer_s} {r.latency_mean:>10.4f}"
        )
    return "\n".join(lines) + "\n"


def save_results(
    results: Sequence[Result], skipped: Sequence[str], path: Path
) -> Path:
    """Write results and the list of skipped stages to ``path`` as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": [result.to_dict() for result in results],
        "skipped": list(skipped),
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def main(
    argv: Sequence[str] | None = None, *, out: Path | None = None
) -> int:
    """CLI entry point: ``python -m edith.bench [stt|tts|all]``."""
    parser = argparse.ArgumentParser(
        prog="edith.bench",
        description="Benchmark Edith's stages (RTF and latency) and write output/bench.json.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=list(STAGES) + ["all"],
        help="stage to benchmark (default: all)",
    )
    parser.add_argument("--runs", type=int, default=3, help="measured runs per stage")
    parser.add_argument("--warmup", type=int, default=0, help="unmeasured warm-up runs")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument(
        "--model",
        default="",
        help=(
            "STT model to benchmark (default: all of "
            + ", ".join(STT_MODELS)
            + ")"
        ),
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help=(
            "faster-whisper compute type: auto, int8, int8_float16, float16, "
            "... (default: auto → float16 on CUDA, int8 on CPU)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N samples per model (quick mode)",
    )
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    out_path = out or DEFAULT_OUT
    stages = STAGES if args.stage == "all" else (args.stage,)
    results: list[Result] = []
    skipped: list[str] = []
    for stage in stages:
        if stage == "stt":
            models = [args.model] if args.model else list(STT_MODELS)
            for model in models:
                started = time.perf_counter()
                model_results = run_stt(
                    device=args.device,
                    runs=args.runs,
                    warmup=args.warmup,
                    model=model,
                    compute_type=args.compute_type,
                    limit=args.limit,
                )
                if not model_results:
                    skipped.append(f"stt:{model}")
                    print(f"stt ({model}): not implemented yet (skipped)")
                else:
                    results.extend(model_results)
                    mean_rtf = _mean([r.rtf_mean for r in model_results])
                    print(
                        f"stt ({model}): {len(model_results)} sample(s) in "
                        f"{time.perf_counter() - started:.1f}s "
                        f"(mean RTF {mean_rtf:.3f})"
                    )
                    # Save partial results after each model finishes so a
                    # crashed/full run never loses the models already done.
                    save_results(results, skipped, out_path)
        else:
            result = run_stage(
                stage,
                device=args.device,
                runs=args.runs,
                warmup=args.warmup,
                model=args.model or DEFAULT_STT_MODEL,
                compute_type=args.compute_type,
            )
            if result is None:
                skipped.append(stage)
                print(f"{stage}: not implemented yet (skipped)")
            else:
                results.append(result)

    print()
    print(format_table(results))
    path = save_results(results, skipped, out_path)
    print(f"Wrote {path}")
    return 0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
