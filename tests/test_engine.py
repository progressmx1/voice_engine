"""TTS engine: normalisation, chunking, voices and smooth joins (fake backend; Kokoro if present)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from edith import models
from edith.tts import engine
from edith.tts.engine import Synthesizer, blend, load_blend, lock_style, split_chunks
from edith.tts.smooth import join_jumps, max_jump, ms_to_samples

RATE = 24_000


class FakeBackend:
    """Speaks each chunk as a 0.2 s tone that starts and stops abruptly (the worst case for clicks)."""

    rate = RATE

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.prepared_with: list[str] = []

    def voices(self) -> list[str]:
        return ["fake_a", "fake_b"]

    def prepare(self, voice: engine.Voice, utterance: str) -> float:
        self.prepared_with.append(utterance)
        return 0.4

    def synthesize(self, text: str, prepared: float, speed: float) -> np.ndarray:
        self.calls.append(text)
        n = int(0.2 * RATE / speed)
        phase = len(self.calls) * 1.3
        t = np.arange(n) / RATE
        return (prepared * np.sin(2 * np.pi * 200 * t + phase) + 0.05).astype(np.float32)


def fake_synth(**options: float) -> tuple[Synthesizer, FakeBackend]:
    backend = FakeBackend()
    return Synthesizer(backend=backend, **options), backend


def test_split_keeps_short_sentences_whole_with_pauses() -> None:
    assert split_chunks("Hello there. How are you? Fine!") == [
        ("Hello there.", 0.25), ("How are you?", 0.25), ("Fine!", 0.0),
    ]


def test_split_long_sentence_at_clauses_then_spaces() -> None:
    long = "one two three four, five six seven eight; nine ten eleven twelve."
    chunks = split_chunks(long, max_chars=25)
    assert [c for c, _ in chunks] == [
        "one two three four,", "five six seven eight;", "nine ten eleven twelve.",
    ]
    assert [p for _, p in chunks] == [0.1, 0.1, 0.0]
    words = " ".join(["word"] * 30) + "."
    assert all(len(c) <= 25 for c, _ in split_chunks(words, max_chars=25))


def test_split_empty_text() -> None:
    assert split_chunks("   ") == []


def test_backend_only_sees_normalised_text() -> None:
    synth, backend = fake_synth()
    synth.synthesize("It costs $12.50 at 3:30 pm. Dr. Lee agrees.")
    assert backend.calls == [
        "It costs twelve dollars and fifty cents at three thirty P M.",
        "Doctor Lee agrees.",
    ]
    assert not any(re.search(r"\d", c) for c in backend.calls)


def test_voice_is_prepared_once_per_utterance_with_the_whole_text() -> None:
    synth, backend = fake_synth()
    synth.synthesize("First sentence. Second sentence.")
    assert backend.prepared_with == ["First sentence. Second sentence."]


def test_stream_is_gapless_and_click_free() -> None:
    synth, backend = fake_synth(sentence_pause=0.0, crossfade_ms=10)
    parts = list(synth.stream("One. Two. Three."))
    assert len(parts) >= 3
    audio = np.concatenate(parts)
    overlap = ms_to_samples(10, RATE)
    chunk = int(0.2 * RATE)
    assert len(audio) == 3 * chunk - 2 * overlap
    joins = [chunk - overlap // 2, 2 * chunk - overlap - overlap // 2]
    interior = max_jump(audio[overlap * 3 : chunk - overlap * 2])
    assert max(join_jumps(audio, joins, overlap)) <= 1.5 * interior
    assert abs(audio[0]) < 0.01 and abs(audio[-1]) < 0.01


def test_pauses_follow_punctuation() -> None:
    synth, _ = fake_synth(sentence_pause=0.25, crossfade_ms=10)
    audio, rate = synth.synthesize("One. Two.")
    assert rate == RATE
    overlap = ms_to_samples(10, RATE)
    assert len(audio) == 2 * int(0.2 * RATE) + int(0.25 * RATE) - overlap


def test_speed_is_validated_and_passed_on() -> None:
    with pytest.raises(ValueError):
        Synthesizer(backend=FakeBackend(), speed=3.0)
    fast, _ = fake_synth(speed=2.0)
    slow, _ = fake_synth(speed=1.0)
    assert len(fast.synthesize("Hi.")[0]) < len(slow.synthesize("Hi.")[0])


def test_empty_text_gives_empty_audio() -> None:
    synth, backend = fake_synth()
    audio, _ = synth.synthesize("")
    assert audio.size == 0 and backend.calls == []


def test_save_writes_wav(tmp_path: Path) -> None:
    import soundfile as sf

    synth, _ = fake_synth()
    out = synth.save("Hello.", tmp_path / "hello.wav")
    data, rate = sf.read(str(out))
    assert rate == RATE and len(data) > 0


def test_voices_include_edith_and_backend_voices() -> None:
    synth, _ = fake_synth()
    assert {"edith", "fake_a", "fake_b"} <= set(synth.voices())


def test_blend_is_weighted_average() -> None:
    styles = {"a": np.ones((4, 1, 3), np.float32), "b": np.zeros((4, 1, 3), np.float32)}
    assert np.allclose(blend(styles, {"a": 3, "b": 1}), 0.75)
    with pytest.raises(ValueError):
        blend(styles, {"c": 1})
    with pytest.raises(ValueError):
        blend(styles, {})


def test_lock_style_uses_one_row_everywhere() -> None:
    style = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    locked = lock_style(style, 4)
    assert locked.shape == style.shape and np.all(locked == 3)
    assert np.all(lock_style(style, 999) == 9) and np.all(lock_style(style, 0) == 0)


def test_load_blend_reads_weights(tmp_path: Path) -> None:
    (tmp_path / "edith.json").write_text(json.dumps({"af_heart": 0.7, "af_bella": 0.3}))
    assert load_blend("edith", tmp_path) == {"af_heart": 0.7, "af_bella": 0.3}
    assert load_blend("nobody", tmp_path) is None


def _kokoro_present() -> bool:
    return all(models.path(n).is_file() for n in ("kokoro-v1.0.onnx", "voices-v1.0.bin"))


@pytest.mark.models
@pytest.mark.skipif(not _kokoro_present(), reason="Kokoro model files not downloaded")
def test_kokoro_speaks_a_sentence() -> None:
    synth = Synthesizer(device="cpu")
    audio, rate = synth.synthesize("Hello, I am Edith. It is 3:30 pm.")
    assert rate == 24_000
    assert 1.0 < len(audio) / rate < 8.0
    assert np.max(np.abs(audio)) <= 0.98 + 1e-6
    assert "af_heart" in synth.voices()
