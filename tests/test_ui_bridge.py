"""Run the Electron bridge's Node unit tests (node --test), if Node is available.

The UI's ``bridge.js`` is plain Node (no Electron dependency), so it is tested
with Node's built-in test runner. This pytest wraps that so ``python -m pytest``
stays the single test command for the project.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parents[1] / "ui"


def test_node_bridge_tests_pass() -> None:
    if shutil.which("node") is None:
        pytest.skip("node is not installed; skipping the Electron bridge unit tests")
    result = subprocess.run(
        ["node", "--test", "test/bridge.test.js"],
        cwd=UI_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ui_javascript_files_parse() -> None:
    if shutil.which("node") is None:
        pytest.skip("node is not installed; skipping the UI JavaScript syntax check")
    for rel in ("bridge.js", "main.js", "preload.js", "renderer/renderer.js"):
        result = subprocess.run(
            ["node", "--check", rel],
            cwd=UI_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"{rel}: {result.stdout}{result.stderr}"
