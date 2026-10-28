"""Draft reference transcripts for the sample voices.

Run as ``python -m edith.references``: transcribes every sample voice with the
most accurate setting (``large-v3``, beam 5) and writes the result to
``data/references.json`` as **drafts** for Progress to review and correct.

The accuracy tests and the ``stt`` benchmark stage compare against this file,
so it must be reviewed before the WER numbers are meaningful. Every entry is
marked ``"reviewed": false`` until Progress has checked it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .samples import PROJECT_ROOT, list_samples

#: Where the (committed) reference transcripts live.
REFERENCES_PATH = PROJECT_ROOT / "data" / "references.json"

#: Most accurate transcription setting used for references.
DEFAULT_MODEL = "large-v3"
DEFAULT_BEAM_SIZE = 5


def build(
    folder: Path | None = None,
    *,
    model: str = DEFAULT_MODEL,
    beam_size: int = DEFAULT_BEAM_SIZE,
    transcriber_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Transcribe every sample and return the draft references document.

    ``transcriber_factory`` lets tests inject a fake transcriber instead of
    downloading ``large-v3``.
    """
    from .stt import Transcriber

    samples = list_samples(folder)
    if not samples:
        raise RuntimeError(
            "no sample voices found (set EDITH_SAMPLES if they are elsewhere)"
        )

    factory = transcriber_factory or (
        lambda: Transcriber(model=model, device="auto", compute_type="auto")
    )
    transcriber = factory()

    doc_samples: dict[str, dict[str, Any]] = {}
    for sample in samples:
        transcript = transcriber.transcribe(
            sample.path, language="en", beam_size=beam_size
        )
        doc_samples[sample.key] = {
            "name": sample.name,
            "style": sample.style,
            "seconds": sample.seconds,
            "language": transcript.language,
            "text": transcript.text,
            "segments": [asdict(seg) for seg in transcript.segments],
            "reviewed": False,
        }

    return {
        "status": "draft",
        "reviewed": False,
        "_note": (
            "Draft transcripts for Progress to review and correct. "
            "Set 'reviewed' to true on each entry (and top-level) once checked."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "beam_size": beam_size,
        "samples": doc_samples,
    }


def write(doc: dict[str, Any], path: Path = REFERENCES_PATH) -> Path:
    """Write a references document to ``data/references.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m edith.references``."""
    doc = build()
    path = write(doc)
    print(f"Drafted references for {len(doc['samples'])} samples -> {path}")
    print("Please review and correct data/references.json (it is marked as a draft).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
