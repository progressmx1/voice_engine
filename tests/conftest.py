"""Shared fixtures."""

from __future__ import annotations

import pytest

from edith.samples import Sample, list_samples


@pytest.fixture(scope="session")
def samples() -> list[Sample]:
    """The sample voices; tests that need them are skipped if the folder is missing."""
    found = list_samples()
    if not found:
        pytest.skip("sample voices folder not found (set EDITH_SAMPLES)")
    return found
