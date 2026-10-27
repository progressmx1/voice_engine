"""Audio loading, resampling and WAV output."""

from __future__ import annotations

import numpy as np

from edith.audio import STT_RATE, load_audio, resample, save_wav, to_mono
from edith.samples import Sample, parse_name


def tone(freq: float, rate: int, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def peak_frequency(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples))
    return float(np.fft.rfftfreq(len(samples), 1 / rate)[np.argmax(spectrum)])


def test_resample_keeps_pitch_and_length() -> None:
    out = resample(tone(440, 44_100), 44_100, STT_RATE)
    assert out.dtype == np.float32
    assert len(out) == STT_RATE
    assert abs(peak_frequency(out, STT_RATE) - 440) < 2


def test_to_mono_averages_channels() -> None:
    stereo = np.stack([np.ones(4), np.zeros(4)], axis=1)
    assert np.allclose(to_mono(stereo), 0.5)


def test_wav_round_trip(tmp_path) -> None:
    path = save_wav(tmp_path / "out" / "tone.wav", tone(300, 24_000), 24_000)
    back = load_audio(path, STT_RATE)
    assert len(back) == STT_RATE
    assert abs(peak_frequency(back, STT_RATE) - 300) < 2


def test_parse_sample_names() -> None:
    assert parse_name("voice_preview_katie - e-learning & podcast host") == (
        "katie",
        "e-learning & podcast host",
    )
    assert parse_name("voice_preview_anne") == ("anne", "")


def test_samples_load_as_16k_mono_speech(samples: list[Sample]) -> None:
    assert len(samples) >= 5
    for sample in samples:
        audio = load_audio(sample.path)
        assert audio.ndim == 1 and audio.dtype == np.float32
        assert abs(len(audio) / STT_RATE - sample.seconds) < 0.2
        assert sample.seconds > 3
        assert 0.01 < float(np.sqrt(np.mean(audio**2))) < 0.7  # real speech, not silence
    assert len({s.key for s in samples}) == len(samples)
