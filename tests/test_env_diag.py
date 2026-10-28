"""CUDA library discovery smoke test (Windows GPU install).

These are presence checks only: they confirm the NVIDIA wheel ``bin`` folders
(cuBLAS/cuDNN) are discoverable and contain the libraries faster-whisper needs
for GPU float16. They do not load the (large) DLLs or touch the GPU, so they
stay fast, and they pass harmlessly when no NVIDIA wheels are installed.
"""

from __future__ import annotations

import os

from edith import stt


def test_nvidia_bin_dirs_contain_cuda_compute_libraries() -> None:
    if os.name != "nt":
        return
    dirs = stt._nvidia_bin_dirs()
    if not dirs:
        return

    found: set[str] = set()
    print("\nNVIDIA CUDA bin dirs and DLLs:")
    for bin_dir in dirs:
        files = sorted(p.name for p in bin_dir.iterdir() if p.is_file())
        print(f"  {bin_dir}")
        for name in files:
            print(f"    {name}")
            found.add(name)
    # The two libraries CTranslate2 loads for GPU float16 (cuBLAS and cuDNN 9).
    assert {"cublas64_12.dll", "cudnn64_9.dll"} <= found


def test_setup_cuda_dlls_prepends_nvidia_bin_dirs_to_path(monkeypatch) -> None:
    if os.name != "nt":
        return
    dirs = stt._nvidia_bin_dirs()
    if not dirs:
        return

    monkeypatch.setenv("PATH", "")
    stt.setup_cuda_dlls()

    parts = os.environ["PATH"].split(os.pathsep)
    for bin_dir in dirs:
        assert str(bin_dir) in parts
