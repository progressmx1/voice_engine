"""WER helper: jiwer with case and punctuation normalisation."""

from __future__ import annotations

import pytest

from edith.wer import normalize, wer


def test_normalize_lowercases_and_strips_punctuation() -> None:
    assert normalize("Hello, world!") == "hello world"
    assert normalize("  One   TWO. three?! ") == "one two three"


def test_wer_zero_for_case_and_punctuation_only_differences() -> None:
    assert wer("Hello, world!", "hello world") == 0.0
    assert wer("It's fine.", "its fine") == 0.0


def test_wer_counts_substitutions() -> None:
    assert wer("a b c", "a b d") == pytest.approx(1.0 / 3.0)


def test_wer_perfect_transcript_is_zero() -> None:
    assert wer("the quick brown fox", "the quick brown fox") == 0.0
