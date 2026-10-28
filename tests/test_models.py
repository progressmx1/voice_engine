"""Model downloads: pinned size and SHA-256, atomic placement, clear errors."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from edith import models
from edith.models import ChecksumError, ModelFile, ModelMissing, download, ensure, sha256_of

PAYLOAD = b"edith model bytes " * 1000


def source(tmp_path: Path, sha: str | None, size: int = len(PAYLOAD)) -> ModelFile:
    src = tmp_path / "remote.bin"
    src.write_bytes(PAYLOAD)
    return ModelFile("model.bin", src.as_uri(), size, sha, "test")


def test_download_checks_hash_and_places_file(tmp_path: Path) -> None:
    sha = hashlib.sha256(PAYLOAD).hexdigest()
    seen: list[int] = []
    out = download(source(tmp_path, sha), root=tmp_path / "models",
                   progress=lambda done, total: seen.append(done))
    assert out.read_bytes() == PAYLOAD
    assert sha256_of(out) == sha
    assert seen[-1] == len(PAYLOAD)
    assert not list((tmp_path / "models").glob("*.part"))


def test_wrong_hash_is_rejected_and_nothing_is_left(tmp_path: Path) -> None:
    with pytest.raises(ChecksumError):
        download(source(tmp_path, "0" * 64), root=tmp_path / "models")
    assert not any((tmp_path / "models").iterdir())


def test_wrong_size_is_rejected(tmp_path: Path) -> None:
    sha = hashlib.sha256(PAYLOAD).hexdigest()
    with pytest.raises(ChecksumError):
        download(source(tmp_path, sha, size=5), root=tmp_path / "models")


def test_unpinned_file_is_refused_without_pin(tmp_path: Path) -> None:
    with pytest.raises(ChecksumError, match="no pinned SHA-256"):
        download(source(tmp_path, None), root=tmp_path / "models")


def test_pin_downloads_and_prints_hash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    download(source(tmp_path, None), root=tmp_path / "models", pin=True)
    assert hashlib.sha256(PAYLOAD).hexdigest() in capsys.readouterr().out


def test_existing_good_file_is_not_downloaded_again(tmp_path: Path) -> None:
    sha = hashlib.sha256(PAYLOAD).hexdigest()
    model = source(tmp_path, sha)
    first = download(model, root=tmp_path / "models")
    (tmp_path / "remote.bin").unlink()  # a second fetch would now fail
    assert download(model, root=tmp_path / "models") == first


def test_ensure_explains_how_to_download(tmp_path: Path) -> None:
    with pytest.raises(ModelMissing, match="python -m edith.models kokoro-v1.0.onnx"):
        ensure("kokoro-v1.0.onnx", root=tmp_path)


def test_registry_entries_are_https_with_sizes() -> None:
    for m in models.REGISTRY.values():
        assert m.url.startswith("https://") and m.size > 0 and m.license
        assert m.sha256 is None or len(m.sha256) == 64
