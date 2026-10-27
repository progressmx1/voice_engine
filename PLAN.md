# Edith: build plan

Edith is a fast, local voice engine for Friday: **speech-to-text first, then text-to-speech**.
This file is the brief for whoever builds it (Friday's Projects worker or a person). Work through
it one task at a time, in order. The same milestones and tasks are in `projects.yaml`; tick them
off there as they are done.

## Rules

1. **One task at a time, in order.** Read this file, `README.md` and the code you will touch
   before changing anything.
2. **Tests with every change.** Every task adds or extends tests in `tests/`. A task is done only
   when `test` passes. Never weaken, skip or delete a test to make it pass.
3. **Measure, don't guess.** Speed claims come from `edith.bench` (real-time factor, latency) and
   are written into the Status table in `README.md` with the device used (GPU or CPU).
4. **Dependencies** go in `pyproject.toml` only. After changing them, ask Progress to run the
   `install` command (it needs his approval). Prefer packages that ship Windows wheels for
   Python 3.10.
5. **Models** download into `models/` (git-ignored) from a pinned URL with a pinned SHA-256,
   through `edith.models`. Never commit weights.
6. **Sample voices** (`../sample voices/*.mp3`) are ElevenLabs voice-library previews of real
   voice actors. Use them to **test speech-to-text** and as **listening and similarity targets**
   when choosing Edith's voice. **Never train on them, fine-tune on them or clone them.** The
   actors haven't agreed to that and ElevenLabs' terms don't allow it. Edith's voices come from
   openly licensed models (Kokoro, Apache-2.0). A cloned voice is only ever made from a recording
   Progress has the rights to, such as his own.
7. **Local and private.** Nothing leaves the laptop at run time. A server, when there is one,
   binds to 127.0.0.1 only.
8. **Code style:** Python 3.10, type hints, small modules with docstrings, no global state
   except model caches.

## Hardware

NVIDIA RTX 4070 Laptop (8 GB), 15 GB RAM, Windows 11, Python 3.10. Targets are given for GPU and
CPU; everything must still work on CPU, just slower.

## Milestones

### M1. Foundations (done when scaffolded)

- [x] Audio loading: any file (WAV, MP3, FLAC) to mono float32 at a chosen rate (`edith.audio`).
- [x] Sample voices: discovery and metadata (`edith.samples`).
- [ ] **Benchmark harness** `edith/bench.py`, run as `python -m edith.bench [stt|tts|all]`. Times
  each stage with `time.perf_counter`, reports the real-time factor (RTF = processing time /
  audio duration) and latency, prints a table and writes `output/bench.json`. Test it with a
  dummy engine that sleeps.

### M2. Speech-to-text v1: fast offline transcription

- [ ] `edith/stt.py`: `Transcriber(model="large-v3-turbo", device="auto", compute_type="auto")`
  using **faster-whisper** (CTranslate2). `device="auto"` uses CUDA when it works (float16),
  otherwise CPU (int8). Load once and reuse. `transcribe(audio | path, language="en") ->
  Transcript(text, segments[(start, end, text)], language, seconds, rtf)`.
  - GPU on Windows needs CUDA 12 cuBLAS and cuDNN 9. They are installed by the `gpu` extra.
    Make their DLL folders visible before loading: call `os.add_dll_directory` on every
    `site-packages/nvidia/*/bin` folder. Checked on this laptop: CTranslate2 4.8 then sees the
    RTX 4070 with float16 and int8_float16. If CUDA fails, fall back to CPU with a clear warning.
- [ ] **References:** `python -m edith.references` transcribes each sample with the most accurate
  setting (large-v3, beam 5) into `data/references.json`. Progress checks and corrects them; the
  file is committed. Tests compare against it.
- [ ] Accuracy and speed tests on the samples: word error rate (jiwer, after normalising case and
  punctuation) ≤ 5% for `large-v3-turbo` and ≤ 10% for `small.en`. Benchmark both models on GPU
  and CPU.
- **Targets:** RTF ≤ 0.05 on GPU with `large-v3-turbo`; RTF ≤ 0.5 on CPU with `small.en` int8.

### M3. Speech-to-text v2: streaming with low latency

- [ ] `edith/stream.py`: `StreamingTranscriber` takes 16 kHz audio chunks (e.g. 32 ms), finds
  speech with Silero VAD (bundled with faster-whisper), emits **partial** text about every
  500 ms while someone speaks and a **final** text when they stop.
- [ ] A file simulator feeds a sample at real-time speed in chunks, so tests are repeatable
  without a microphone. A microphone source (`sounddevice`) for manual runs: `python -m
  edith.listen`.
- [ ] Tests: the final text matches the offline transcript (WER ≤ 5% against it), and timing is
  measured from the end of speech to the final text.
- **Targets:** final text ≤ 300 ms after the end of speech on GPU (≤ 800 ms on CPU); first
  partial ≤ 700 ms after speech starts.

### M4. Text-to-speech v1: Kokoro

- [ ] `edith/models.py`: downloads and checks the Kokoro-82M ONNX model and voices file (pinned
  URLs and SHA-256) into `models/`.
- [ ] `edith/tts.py`: `Synthesizer(voice="af_heart", speed=1.0, device="auto")` using
  **kokoro-onnx** (onnxruntime; CUDA provider when available). `synthesize(text) -> (samples,
  sample_rate)`, `save(text, path)`, `voices()`.
- [ ] **Round trip:** a fixed set of 20 test sentences (numbers, names, units, questions) is
  spoken by the synthesizer and transcribed by the M2 transcriber. WER ≤ 5%. This is why
  speech-to-text comes first: it checks the voice automatically.
- **Targets:** RTF ≤ 0.1 on CPU, ≤ 0.03 on GPU.
- The installed onnxruntime is CPU-only, and Kokoro is fast on CPU. Only if CPU misses the
  target, ask Progress to switch to `onnxruntime-gpu` (it replaces `onnxruntime`), then measure
  again.

### M5. Text-to-speech v2: Edith's voice

- [ ] `edith/similarity.py`: speaker embeddings (WeSpeaker ResNet34 ONNX via sherpa-onnx,
  downloaded through `edith.models`) and cosine similarity between two clips.
- [ ] `python -m edith.voice_search`: renders the same sentences with each Kokoro voice and with
  blends of two or three voices (weighted averages of voice style vectors), and scores them
  against the sample voices tagged "warm, friendly" (juliet, lauren, katie podcast). This only
  measures similarity to choose among licensed voices; nothing is trained on the samples.
  Writes `output/voice_search.md` with the top 5 and a WAV for each.
- [ ] Save the chosen blend as `voices/edith.json` (weights). The default voice becomes
  `"edith"`. Progress listens and approves (a task he ticks).

### M6. Text-to-speech v3: streaming and natural text

- [ ] `edith/text.py`: normalisation for speech (numbers, dates, times, currency, units,
  abbreviations, symbols) with thorough tests.
- [ ] `edith/speak.py`: splits text into sentences and clauses and synthesizes chunk by chunk,
  yielding audio as soon as the first chunk is ready; gapless playback with `sounddevice`
  (`python -m edith.say "text"`).
- **Targets:** first audio ≤ 150 ms after the text arrives on GPU (≤ 300 ms on CPU); no audible
  gaps between chunks (checked by a test on chunk boundaries).

### M7. Ready for Friday

- [ ] `edith/server.py`: a local server on 127.0.0.1 only. POST `/stt` takes WAV and returns
  JSON text; a WebSocket `/stt/stream` takes PCM chunks and sends partial/final text; POST
  `/tts` takes JSON text and voice and streams 16-bit PCM. Started with `python -m
  edith.server`. A random token in `output/server-token` is required on every request.
- [ ] `docs/protocol.md` documents the API with examples; end-to-end tests run the server on a
  random port.
- [ ] Final benchmark table in `README.md`. Wiring Edith into Friday itself happens in Friday's
  own repository, not here.
