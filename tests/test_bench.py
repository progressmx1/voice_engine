"""Benchmark harness: measurement, JSON output and CLI."""

from __future__ import annotations

import json
import time

import pytest

from edith import bench
from edith.bench import Result, format_table, measure, save_results


class DummyEngine:
    """An engine that sleeps a fixed amount, to time the harness itself."""

    def __init__(self, seconds: float = 0.05) -> None:
        self.seconds = seconds

    def run(self) -> str:
        time.sleep(self.seconds)
        return "done"


def test_measure_reports_rtf_and_latency_of_dummy_engine() -> None:
    result = measure(
        "stt",
        DummyEngine(0.05).run,
        audio_seconds=1.0,
        runs=3,
        engine="dummy",
        device="cpu",
    )

    assert result.stage == "stt"
    assert result.engine == "dummy"
    assert result.runs == 3
    assert result.audio_seconds == 1.0
    # The dummy sleeps 0.05 s per 1 s of audio, so RTF == latency ~= 0.05.
    assert result.rtf_mean == pytest.approx(0.05, abs=0.03)
    assert result.latency_mean == pytest.approx(0.05, abs=0.03)
    assert result.rtf_min <= result.rtf_mean <= result.rtf_max
    assert result.latency_min <= result.latency_mean <= result.latency_max


def test_measure_runs_warmup_without_timing_it() -> None:
    calls: list[float] = []

    def fn() -> None:
        calls.append(time.perf_counter())

    result = measure("tts", fn, audio_seconds=2.0, runs=2, warmup=1)

    assert len(calls) == 3
    assert result.runs == 2
    # A no-op call is essentially instantaneous, so RTF must be tiny.
    assert result.rtf_mean >= 0.0


def test_measure_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        measure("stt", lambda: None, audio_seconds=0.0)
    with pytest.raises(ValueError):
        measure("stt", lambda: None, audio_seconds=1.0, runs=0)
    with pytest.raises(ValueError):
        measure("stt", lambda: None, audio_seconds=1.0, warmup=-1)


def test_save_results_writes_json(tmp_path) -> None:
    result = Result(
        stage="stt",
        engine="dummy",
        device="cpu",
        audio_seconds=1.0,
        runs=2,
        rtf_mean=0.05,
        rtf_min=0.04,
        rtf_max=0.06,
        latency_mean=0.05,
        latency_min=0.04,
        latency_max=0.06,
    )

    path = save_results([result], ["tts"], tmp_path / "bench.json")
    doc = json.loads(path.read_text(encoding="utf-8"))

    assert doc["skipped"] == ["tts"]
    assert "generated_at" in doc
    assert doc["results"][0]["stage"] == "stt"
    assert doc["results"][0]["rtf_mean"] == 0.05
    assert doc["results"][0]["latency_max"] == 0.06


def test_format_table_includes_result_columns() -> None:
    result = Result(
        stage="stt",
        engine="dummy",
        device="cpu",
        audio_seconds=2.5,
        runs=3,
        rtf_mean=0.25,
        rtf_min=0.2,
        rtf_max=0.3,
        latency_mean=0.625,
        latency_min=0.5,
        latency_max=0.75,
    )

    text = format_table([result])

    assert "stt" in text
    assert "dummy" in text
    assert "0.250" in text
    assert "0.6250" in text


def test_main_writes_bench_json_and_reports_missing_stages(tmp_path, capsys) -> None:
    out = tmp_path / "bench.json"

    # TTS is still a later milestone, so it is skipped honestly.
    assert bench.main(["tts"], out=out) == 0

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["results"] == []
    assert doc["skipped"] == ["tts"]

    captured = capsys.readouterr()
    assert "not implemented" in captured.out
    assert "bench.json" in captured.out


def test_main_skips_stt_without_samples(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr("edith.samples.list_samples", lambda folder=None: [])
    out = tmp_path / "bench.json"

    assert bench.main(["stt"], out=out) == 0

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["results"] == []
    assert doc["skipped"] == ["stt:large-v3-turbo", "stt:small.en"]

    captured = capsys.readouterr()
    assert "not implemented" in captured.out


def test_load_references_returns_none_when_missing(tmp_path) -> None:
    assert bench.load_references(tmp_path / "missing.json") is None


def test_reference_text_looks_up_sample_key() -> None:
    doc = {
        "samples": {
            "katie-e-learning-podcast-host": {"text": "hello world"},
            "anne": {"text": " "},
            "bad": "not-a-dict",
        }
    }
    assert bench.reference_text("katie-e-learning-podcast-host", doc) == "hello world"
    assert bench.reference_text("anne", doc) is None
    assert bench.reference_text("bad", doc) is None
    assert bench.reference_text("missing", doc) is None


def test_run_stt_scores_wer_against_references(monkeypatch) -> None:
    from edith.samples import Sample
    from edith.stt import Segment, Transcript

    sample = Sample(path="clip.wav", name="katie", style="podcast", seconds=2.0)

    class FakeTranscriber:
        model = "large-v3-turbo"
        device = "cpu"
        compute_type = "int8"

        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe(self, audio, language="en", **kwargs):
            return Transcript(
                text="hello world",
                segments=(Segment(0.0, 2.0, "hello world"),),
                language="en",
                seconds=2.0,
                rtf=0.5,
            )

    monkeypatch.setattr("edith.samples.list_samples", lambda folder=None: [sample])
    monkeypatch.setattr("edith.stt.Transcriber", FakeTranscriber)
    monkeypatch.setattr(bench, "reference_text", lambda key, doc=None: "hello world")

    results = bench.run_stt(device="cpu", runs=2, warmup=0, model="large-v3-turbo")

    assert len(results) == 1
    result = results[0]
    assert result.wer == 0.0
    assert result.meta["sample"] == "katie-podcast"
    assert result.engine == "faster-whisper/large-v3-turbo"
    assert result.runs == 2


def test_run_stt_forwards_compute_type(monkeypatch) -> None:
    from edith.samples import Sample
    from edith.stt import Segment, Transcript

    sample = Sample(path="clip.wav", name="katie", style="podcast", seconds=2.0)
    seen: dict[str, str] = {}

    class FakeTranscriber:
        model = "large-v3-turbo"
        device = "cpu"
        compute_type = "int8"

        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

        def transcribe(self, audio, language="en", **kwargs):
            return Transcript(
                text="hello world",
                segments=(Segment(0.0, 2.0, "hello world"),),
                language="en",
                seconds=2.0,
                rtf=0.5,
            )

    monkeypatch.setattr("edith.samples.list_samples", lambda folder=None: [sample])
    monkeypatch.setattr("edith.stt.Transcriber", FakeTranscriber)
    monkeypatch.setattr(bench, "reference_text", lambda key, doc=None: "hello world")

    bench.run_stt(device="cpu", model="large-v3-turbo", compute_type="int8")

    assert seen == {"model": "large-v3-turbo", "device": "cpu", "compute_type": "int8"}


def test_build_stage_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError):
        bench.build_stage("asr", "cpu")


def test_run_stt_respects_limit(monkeypatch) -> None:
    from edith.samples import Sample
    from edith.stt import Segment, Transcript

    samples = [
        Sample(path="a.wav", name="anne", style="", seconds=2.0),
        Sample(path="b.wav", name="bob", style="", seconds=2.0),
        Sample(path="c.wav", name="cara", style="", seconds=2.0),
    ]

    class FakeTranscriber:
        model = "small.en"
        device = "cpu"
        compute_type = "int8"

        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe(self, audio, language="en", **kwargs):
            return Transcript(
                text="hello",
                segments=(Segment(0.0, 2.0, "hello"),),
                language="en",
                seconds=2.0,
                rtf=0.5,
            )

    monkeypatch.setattr("edith.samples.list_samples", lambda folder=None: samples)
    monkeypatch.setattr("edith.stt.Transcriber", FakeTranscriber)
    monkeypatch.setattr(bench, "reference_text", lambda key, doc=None: "hello")

    results = bench.run_stt(device="cpu", runs=1, model="small.en", limit=2)

    assert [r.meta["sample"] for r in results] == ["anne", "bob"]


def test_main_forwards_limit_and_model_and_saves_partial_results(
    tmp_path, capsys, monkeypatch
) -> None:
    out = tmp_path / "bench.json"
    result = Result(
        stage="stt",
        engine="faster-whisper/small.en",
        device="cpu",
        audio_seconds=1.0,
        runs=1,
        rtf_mean=0.5,
        rtf_min=0.5,
        rtf_max=0.5,
        latency_mean=0.5,
        latency_min=0.5,
        latency_max=0.5,
    )
    seen: list[dict] = []

    def fake_run_stt(**kwargs):
        seen.append(kwargs)
        return [result]

    monkeypatch.setattr(bench, "run_stt", fake_run_stt)

    saved: list[tuple[int, int]] = []
    real_save = bench.save_results

    def spy_save(results, skipped, path):
        saved.append((len(results), len(skipped)))
        return real_save(results, skipped, path)

    monkeypatch.setattr(bench, "save_results", spy_save)

    assert bench.main(["stt", "--model", "small.en", "--limit", "2"], out=out) == 0

    assert seen == [
        {
            "device": "auto",
            "runs": 3,
            "warmup": 0,
            "model": "small.en",
            "compute_type": "auto",
            "limit": 2,
        }
    ]

    # Partial save after the model finishes, then the final save.
    assert saved == [(1, 0), (1, 0)]

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["results"][0]["engine"] == "faster-whisper/small.en"

    captured = capsys.readouterr()
    assert "stt (small.en): 1 sample(s)" in captured.out


def test_main_rejects_limit_below_one(tmp_path) -> None:
    with pytest.raises(SystemExit):
        bench.main(["stt", "--limit", "0"], out=tmp_path / "bench.json")
