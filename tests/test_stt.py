"""M2 STT: Transcriber and Transcript API (faster-whisper)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import edith.stt as stt
from edith.stt import Segment, Transcriber, Transcript


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class _FakeInfo:
    duration = 2.0
    language = "en"


class _FakeWhisperModel:
    instances: list["_FakeWhisperModel"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        _FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.audio = audio
        self.transcribe_kwargs = kwargs
        return [
            _FakeSegment(0.0, 1.0, "hello"),
            _FakeSegment(1.0, 2.0, "world"),
        ], _FakeInfo()


@pytest.fixture(autouse=True)
def _fresh_fake_model(monkeypatch) -> None:
    _FakeWhisperModel.instances.clear()
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisperModel)


def test_resolve_device_auto_prefers_cuda_then_cpu() -> None:
    assert stt._resolve_device("auto", 1) == "cuda"
    with pytest.warns(RuntimeWarning):
        assert stt._resolve_device("auto", 0) == "cpu"


def test_resolve_device_explicit_cuda_falls_back_with_warning() -> None:
    assert stt._resolve_device("cuda", 1) == "cuda"
    with pytest.warns(RuntimeWarning):
        assert stt._resolve_device("cuda", 0) == "cpu"
    assert stt._resolve_device("cpu", 1) == "cpu"
    with pytest.raises(ValueError):
        stt._resolve_device("tpu", 0)


def test_resolve_compute_type_auto_by_device() -> None:
    assert stt._resolve_compute_type("auto", "cuda") == "float16"
    assert stt._resolve_compute_type("auto", "cpu") == "int8"
    assert stt._resolve_compute_type("int8_float16", "cuda") == "int8_float16"


def test_transcript_dataclass() -> None:
    seg = Segment(0.0, 1.25, "hello")
    transcript = Transcript(
        text="hello world",
        segments=(seg,),
        language="en",
        seconds=2.0,
        rtf=0.5,
    )
    assert transcript.segments[0].end == 1.25
    assert transcript.segments[0] == Segment(0.0, 1.25, "hello")


def test_transcriber_defaults_to_gpu_float16_or_cpu_int8(monkeypatch) -> None:
    monkeypatch.setattr(stt, "_cuda_device_count", lambda: 0)
    cpu = Transcriber(model="tiny", device="auto")
    assert cpu.device == "cpu"
    assert cpu.compute_type == "int8"

    monkeypatch.setattr(stt, "_cuda_device_count", lambda: 1)
    gpu = Transcriber(model="tiny", device="auto")
    assert gpu.device == "cuda"
    assert gpu.compute_type == "float16"


def test_is_cuda_library_error() -> None:
    assert stt._is_cuda_library_error(
        RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
    )
    assert stt._is_cuda_library_error(
        RuntimeError("Library cudnn64_9.dll is not found or cannot be loaded")
    )
    assert not stt._is_cuda_library_error(RuntimeError("disk quota exceeded"))


def test_transcriber_falls_back_when_cuda_model_load_fails(monkeypatch) -> None:
    monkeypatch.setattr(stt, "_cuda_device_count", lambda: 1)

    class CudaLoadFailModel:
        def __init__(self, *args, **kwargs) -> None:
            if kwargs.get("device") == "cuda":
                raise RuntimeError(
                    "Library cublas64_12.dll is not found or cannot be loaded"
                )

    monkeypatch.setattr(stt.faster_whisper, "WhisperModel", CudaLoadFailModel)

    with pytest.warns(RuntimeWarning):
        transcriber = Transcriber(model="tiny", device="auto")

    assert transcriber.device == "cpu"
    assert transcriber.compute_type == "int8"


def test_transcribe_falls_back_to_cpu_when_cuda_library_missing(monkeypatch) -> None:
    monkeypatch.setattr(stt, "_cuda_device_count", lambda: 1)

    class FlakyCudaModel:
        calls = 0

        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

        def transcribe(self, audio, **kwargs):
            FlakyCudaModel.calls += 1
            if FlakyCudaModel.calls == 1:
                raise RuntimeError(
                    "Library cublas64_12.dll is not found or cannot be loaded"
                )
            return [_FakeSegment(0.0, 2.0, "hello world")], _FakeInfo()

    monkeypatch.setattr(stt.faster_whisper, "WhisperModel", FlakyCudaModel)

    transcriber = Transcriber(model="tiny", device="auto")
    assert transcriber.device == "cuda"
    assert transcriber.compute_type == "float16"

    with pytest.warns(RuntimeWarning):
        result = transcriber.transcribe(np.zeros(32000, dtype=np.float32))

    assert result.text == "hello world"
    assert transcriber.device == "cpu"
    assert transcriber.compute_type == "int8"
    assert FlakyCudaModel.calls == 2


def test_transcribe_returns_transcript_with_segments_and_rtf() -> None:
    transcriber = Transcriber(model="tiny", device="cpu", compute_type="int8")
    fake = _FakeWhisperModel.instances[-1]

    result = transcriber.transcribe(np.zeros(32000, dtype=np.float32))

    assert fake.kwargs == {
        "device": "cpu",
        "compute_type": "int8",
        "download_root": str(stt.WHISPER_DIR),
    }
    assert fake.transcribe_kwargs["language"] == "en"
    assert fake.transcribe_kwargs["beam_size"] == 5
    assert np.asarray(fake.audio).shape == (32000,)

    assert result.text == "hello world"
    assert result.language == "en"
    assert result.seconds == 2.0
    assert result.rtf >= 0.0
    assert result.segments == (
        Segment(0.0, 1.0, "hello"),
        Segment(1.0, 2.0, "world"),
    )


def test_transcribe_accepts_a_path(tmp_path: Path) -> None:
    transcriber = Transcriber(model="tiny", device="cpu", compute_type="int8")
    fake = _FakeWhisperModel.instances[-1]

    path = tmp_path / "clip.mp3"
    transcriber.transcribe(path)

    assert fake.audio == str(path)


def test_transcribe_normalises_stereo_to_mono_float32() -> None:
    transcriber = Transcriber(model="tiny", device="cpu", compute_type="int8")
    fake = _FakeWhisperModel.instances[-1]

    stereo = np.stack([np.ones(4), np.zeros(4)], axis=1)
    transcriber.transcribe(stereo)

    arr = np.asarray(fake.audio)
    assert arr.dtype == np.float32
    assert arr.shape == (4,)
    assert np.allclose(arr, 0.5)


def test_transcribe_rejects_bad_audio_type() -> None:
    transcriber = Transcriber(model="tiny", device="cpu", compute_type="int8")
    with pytest.raises(TypeError):
        transcriber.transcribe(123)


def test_as_float32_mono_rejects_3d() -> None:
    assert np.allclose(stt._as_float32_mono(np.ones((4, 2))), 1.0)
    with pytest.raises(ValueError):
        stt._as_float32_mono(np.zeros((2, 2, 2)))


@pytest.mark.gpu
def test_cuda_is_visible_to_ctranslate2() -> None:
    count = stt._cuda_device_count()
    if count == 0:
        pytest.skip("CUDA not available (CPU-only run)")
    assert count >= 1


def test_nvidia_bin_dirs_are_bin_folders() -> None:
    dirs = stt._nvidia_bin_dirs()
    assert all(d.name == "bin" for d in dirs)


def test_setup_cuda_dlls_does_not_raise() -> None:
    stt.setup_cuda_dlls()


def test_prepend_nvidia_bin_dirs_to_path(monkeypatch) -> None:
    existing = "C:\\alpha" if os.name == "nt" else "/usr/bin"
    monkeypatch.setenv("PATH", existing)
    first = Path("X:/nvidia/cublas/bin")
    second = Path("X:/nvidia/cudnn/bin")

    stt._prepend_nvidia_bin_dirs_to_path([first, second])

    parts = os.environ["PATH"].split(os.pathsep)
    assert parts[:2] == [str(first), str(second)]
    assert existing in parts


def test_prepend_nvidia_bin_dirs_to_path_is_idempotent(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "")
    bin_dir = Path("X:/nvidia/cublas/bin")

    stt._prepend_nvidia_bin_dirs_to_path([bin_dir])
    stt._prepend_nvidia_bin_dirs_to_path([bin_dir])

    assert os.environ["PATH"].split(os.pathsep).count(str(bin_dir)) == 1


def test_prepend_nvidia_bin_dirs_to_path_keeps_existing_entries(monkeypatch) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(["C:\\one", "C:\\two"]))
    stt._prepend_nvidia_bin_dirs_to_path([Path("C:\\nvidia\\cublas\\bin")])
    parts = os.environ["PATH"].split(os.pathsep)
    assert parts[0] == "C:\\nvidia\\cublas\\bin"
    assert parts[1:] == ["C:\\one", "C:\\two"]
