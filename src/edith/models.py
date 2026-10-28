"""Model files: pinned URLs and SHA-256, downloaded once into `models/` and checked.

`ensure("kokoro-v1.0.onnx")` returns the local path, or raises `ModelMissing` with the command
to download it. Downloads stream to a `.part` file while hashing, and are only renamed into
place when the size and SHA-256 match. A file whose hash is not pinned yet is refused unless
`pin=True`, which downloads it and prints the hash to paste into `REGISTRY`.

    python -m edith.models --list
    python -m edith.models kokoro-v1.0.onnx voices-v1.0.bin
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#: `models/` at the project root; EDITH_MODELS overrides it.
MODELS_DIR = Path(os.environ.get("EDITH_MODELS", Path(__file__).resolve().parents[2] / "models"))

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"


@dataclass(frozen=True)
class ModelFile:
    """One downloadable file. `sha256` is None until it has been pinned."""

    name: str
    url: str
    size: int
    sha256: str | None
    license: str


REGISTRY: dict[str, ModelFile] = {
    f.name: f
    for f in [
        ModelFile(
            "kokoro-v1.0.onnx", f"{KOKORO_RELEASE}/kokoro-v1.0.onnx", 325_532_387,
            "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
            "Apache-2.0 (Kokoro-82M)",
        ),
        ModelFile(
            "voices-v1.0.bin", f"{KOKORO_RELEASE}/voices-v1.0.bin", 28_214_398,
            "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
            "Apache-2.0 (Kokoro-82M)",
        ),
    ]
}


class ModelMissing(FileNotFoundError):
    """A model file is not downloaded yet."""


class ChecksumError(RuntimeError):
    """A download did not match its pinned size or SHA-256."""


def path(name: str, root: Path | None = None) -> Path:
    """Where `name` lives once downloaded."""
    return (root or MODELS_DIR) / name


def sha256_of(file: Path, block: int = 1 << 20) -> str:
    """SHA-256 of a file, read in blocks."""
    h = hashlib.sha256()
    with open(file, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def ensure(name: str, root: Path | None = None) -> Path:
    """Path to a downloaded model file; raises `ModelMissing` with the fix if it isn't there."""
    p = path(name, root)
    if not p.is_file():
        raise ModelMissing(f"{p} is missing. Download it with: python -m edith.models {name}")
    return p


def download(
    model: ModelFile,
    root: Path | None = None,
    pin: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download `model` into `root`, checking size and SHA-256; returns the final path.

    An existing file is kept when its size matches (and its hash, if pinned). Unpinned files are
    refused unless `pin` is set; then the computed hash is printed for pinning.
    """
    if model.sha256 is None and not pin:
        raise ChecksumError(
            f"{model.name} has no pinned SHA-256. Run `python -m edith.models --pin {model.name}` "
            "once, then paste the printed hash into edith.models.REGISTRY."
        )
    target = path(model.name, root)
    if target.is_file() and target.stat().st_size == model.size:
        if model.sha256 is None or sha256_of(target) == model.sha256:
            return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    h = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(model.url) as response, open(part, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, model.size)
        digest = h.hexdigest()
        if done != model.size:
            raise ChecksumError(f"{model.name}: got {done} bytes, expected {model.size}")
        if model.sha256 is not None and digest != model.sha256:
            raise ChecksumError(f"{model.name}: SHA-256 {digest} does not match {model.sha256}")
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    if model.sha256 is None:
        print(f"\nPIN {model.name} sha256={digest}")
    return target


def _bar(done: int, total: int) -> None:
    sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m edith.models", description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", help="files to download (default: all)")
    parser.add_argument("--list", action="store_true", help="show files and whether they are present")
    parser.add_argument("--pin", action="store_true", help="allow unpinned files and print their hash")
    args = parser.parse_args(argv)

    if args.list:
        for m in REGISTRY.values():
            state = "present" if path(m.name).is_file() else "missing"
            pinned = "pinned" if m.sha256 else "UNPINNED"
            print(f"{m.name:24} {m.size / 1e6:8.1f} MB  {state:8} {pinned:8} {m.license}")
        return 0

    unknown = [n for n in args.names if n not in REGISTRY]
    if unknown:
        parser.error(f"unknown model file(s): {', '.join(unknown)}")
    for name in args.names or list(REGISTRY):
        print(f"{name}:")
        download(REGISTRY[name], pin=args.pin, progress=_bar)
        print(f"\n  -> {path(name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
