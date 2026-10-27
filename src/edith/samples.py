"""The sample voices: real speech for testing speech-to-text and as similarity targets.

They are ElevenLabs voice-library previews of real voice actors. Never train on them or clone
them (see PLAN.md, rule 6).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .audio import duration

#: The project folder (…/Projects/edith).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_TYPES = {".mp3", ".wav", ".flac", ".ogg"}


def samples_dir() -> Path:
    """`EDITH_SAMPLES` if set, else the "sample voices" folder next to the project."""
    env = os.environ.get("EDITH_SAMPLES")
    return Path(env) if env else PROJECT_ROOT.parent / "sample voices"


@dataclass(frozen=True)
class Sample:
    """One sample clip. `name` is the speaker ("katie"); `style` is the description, if any."""

    path: Path
    name: str
    style: str
    seconds: float

    @property
    def key(self) -> str:
        """A stable id, e.g. "katie-e-learning-podcast-host"."""
        text = f"{self.name} {self.style}".lower().replace("&", " ")
        return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def parse_name(stem: str) -> tuple[str, str]:
    """ "voice_preview_katie - e-learning & podcast host" -> ("katie", "e-learning & podcast host")."""
    text = re.sub(r"^voice_preview_", "", stem).strip()
    name, _, style = text.partition(" - ")
    return name.strip().lower(), style.strip().lower()


def list_samples(folder: Path | None = None) -> list[Sample]:
    """All sample clips, sorted by key."""
    root = folder or samples_dir()
    if not root.is_dir():
        return []
    found = []
    for path in root.iterdir():
        if path.suffix.lower() in AUDIO_TYPES and path.is_file():
            name, style = parse_name(path.stem)
            found.append(Sample(path=path, name=name, style=style, seconds=duration(path)))
    return sorted(found, key=lambda s: s.key)
