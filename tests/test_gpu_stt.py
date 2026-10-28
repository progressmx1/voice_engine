"""GPU smoke test: one real, tiny transcription on CUDA float16 (small.en).

This proves the NVIDIA DLL setup actually works end-to-end: the model loads on
CUDA and a short clip is transcribed without falling back to CPU with a
cuBLAS/cuDNN warning. Skipped when CTranslate2 cannot see a CUDA device.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from edith.stt import Transcriber, _cuda_device_count


@pytest.mark.gpu
def test_tiny_gpu_transcription_small_en() -> None:
    if _cuda_device_count() == 0:
        pytest.skip("CUDA not available (CPU-only run)")

    transcriber = Transcriber(model="small.en", device="cuda", compute_type="float16")
    assert transcriber.device == "cuda"
    assert transcriber.compute_type == "float16"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # 1 s of silence still runs the encoder on the GPU (cuBLAS GEMMs).
        result = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    cuda_fail = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and ("cublas" in str(w.message).lower() or "cudnn" in str(w.message).lower())
    ]
    assert not cuda_fail, f"GPU transcription fell back: {[str(w.message) for w in cuda_fail]}"
    assert transcriber.device == "cuda"
    assert result.rtf >= 0.0
    print(
        f"GPU check OK: device={transcriber.device} "
        f"compute_type={transcriber.compute_type} rtf={result.rtf:.3f}"
    )
