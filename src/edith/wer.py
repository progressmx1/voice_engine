"""Word error rate for speech-to-text accuracy.

WER is measured with :mod:`jiwer` after a light, reproducible normalisation:
lower-case, punctuation removed and repeated spaces collapsed. A result of
``0.0`` means the transcript is identical after that normalisation; ``1.0`` is
the worst possible score.
"""

from __future__ import annotations

import jiwer

#: Normalisation applied to both the reference and the hypothesis before scoring.
_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)


def normalize(text: str) -> str:
    """Normalise text for scoring (case and punctuation only)."""
    return _TRANSFORM(text)


def wer(reference: str, hypothesis: str) -> float:
    """Return WER as a fraction in ``[0, 1]`` (lower is better)."""
    # Normalise first so the scoring does not depend on jiwer's keyword API,
    # which differs between jiwer 2.x and 3.x.
    return float(jiwer.wer(_TRANSFORM(reference), _TRANSFORM(hypothesis)))
