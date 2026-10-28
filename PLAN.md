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
   actors haven't agreed to that and ElevenLabs' terms don't allow it. Edith's voice is trained
   only on openly licensed data: public-domain or CC-BY speech (LJSpeech, LibriTTS-R) and openly
   licensed starting weights (StyleTTS2 MIT, Kokoro Apache-2.0). A cloned voice is only ever made
   from a recording Progress has the rights to, such as his own.
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
- [x] **Benchmark harness** `edith/bench.py`, run as `python -m edith.bench [stt|tts|all]`. Times
  each stage with `time.perf_counter`, reports the real-time factor (RTF = processing time /
  audio duration) and latency, prints a table and writes `output/bench.json`. Test it with a
  dummy engine that sleeps.

### M2. Speech-to-text v1: fast offline transcription

- [x] `edith/stt.py`: `Transcriber(model="large-v3-turbo", device="auto", compute_type="auto")`
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

### TTS design: Edith's own engine

Edith's text-to-speech is **our own engine**: we write every stage that decides how speech
sounds (text normalisation, phonemes, prosody, chunking, joining, loudness and playback), and we
fine-tune our own acoustic model for Edith's voice. Training a model from scratch on an 8 GB
laptop GPU would take weeks and sound worse, so the acoustic model starts from openly licensed
StyleTTS2-class weights. StyleTTS2 is the architecture behind Kokoro and scores close to human
recordings in listening tests.

```
text -> normalise (text.py) -> phonemes -> chunker (clause boundaries, carried style)
     -> acoustic model + vocoder (backend: Kokoro baseline, then Edith's fine-tune)
     -> smoothing (tts/smooth.py: DC, edge fades, crossfades, loudness) -> gapless player
```

**No jitter** is a measured requirement. Each source of jitter has a fix and a test:

| Jitter source                         | Fix                                                        | Test |
| ------------------------------------- | ---------------------------------------------------------- | ---- |
| Clicks at chunk joins                 | DC removal, short edge fades, equal-power crossfade        | largest sample jump at a join ≤ 1.5 × the largest jump away from joins (naive joins are > 5 ×) |
| Loudness steps between chunks         | Per-chunk RMS matched to a running target with limited gain change | RMS of the 50 ms windows either side of a join within 3 dB |
| Prosody resets at each chunk          | Split at clause boundaries only; carry the style vector and the previous clause as context | pitch at a join stays within the speaker's range |
| Pitch and amplitude jitter from the vocoder | Measured on rendered speech; drives choosing and tuning the model | local F0 jitter ≤ 1.0 % and shimmer ≤ 3.5 % (normal human range) |
| Playback stutter (buffer underruns)   | Ring buffer with ~100 ms pre-roll, synthesis ahead of playback | underrun counter stays 0 in a 60 s run |

**Naturalness** is scored with UTMOS (an open predictor of human listening scores) on the
20 round-trip sentences. Progress's listening approval is the final check.

### M4. Text-to-speech v1: engine and baseline voice

- [x] `edith/tts/smooth.py`: the anti-jitter layer (DC removal, edge fades, equal-power
  crossfades, loudness matching) and the join metrics from the table above, with tests on
  synthetic signals. Needs only numpy.
- [x] `edith/text.py`: normalisation for speech (numbers, dates, times, currency, units,
  abbreviations, symbols) with thorough tests.
- [x] `edith/models.py`: downloads and checks model files (pinned URLs and SHA-256) into
  `models/`.
- [x] `edith/tts/engine.py`: `Synthesizer(voice="edith", speed=1.0, device="auto")` with a
  pluggable `Backend` (acoustic model + vocoder). `synthesize(text) -> (samples, rate)`,
  `stream(text)` yields smoothed chunks, `save(text, path)`, `voices()`. First backend:
  **Kokoro-82M ONNX** (kokoro-onnx) as the baseline, so the whole engine works before training.
- [x] **Round trip:** 20 fixed test sentences (numbers, names, units, questions) are spoken and
  transcribed by the M2 transcriber. WER ≤ 5%.
- [ ] `edith/quality.py`: UTMOS score, F0 jitter and shimmer, join metrics.
  `python -m edith.quality` writes `output/quality.md`.
- **Targets:** RTF ≤ 0.1 on CPU, ≤ 0.03 on GPU; all jitter tests pass; UTMOS recorded as the
  baseline that Edith's own voice must beat.

### M5. Text-to-speech v2: Edith's own voice

- [ ] Data: **LJSpeech** (public domain, 24 h, one speaker) downloaded through `edith.models`,
  or Progress's own recordings if he records them. `python -m edith.voice.prepare` resamples,
  trims silence, drops clips whose Whisper transcript disagrees with the text (WER > 10%), and
  writes phoneme manifests.
- [ ] Fine-tune a StyleTTS2-class model (MIT training code, openly licensed starting weights) on
  that data. Settings for 8 GB: mixed precision, batch 2, short max length, gradient
  checkpointing. `python -m edith.voice.train` resumes from checkpoints in `models/`. Ask
  Progress before long training runs; they keep the GPU busy for hours.
- [ ] Export to ONNX as a second backend `edith` and run the M4 round trip and quality report on
  it. It becomes the default only if WER ≤ 5%, the jitter metrics pass and its UTMOS is at least
  the Kokoro baseline. Progress listens and approves (a task he ticks).
- [ ] Optional: speaker-similarity check (WeSpeaker ONNX) against the "warm, friendly" sample
  voices, used only to compare candidate checkpoints. Nothing is trained on the samples.

### M6. Text-to-speech v3: streaming

- [ ] `edith/speak.py`: synthesizes clause by clause and yields audio as soon as the first clause
  is ready. Gapless playback with `sounddevice` through the ring buffer
  (`python -m edith.say "text"`).
- **Targets:** first audio ≤ 150 ms after the text arrives on GPU (≤ 300 ms on CPU); no
  underruns; all join tests pass on streamed output.

### M7. Ready for Friday

- [ ] `edith/server.py`: a local server on 127.0.0.1 only. POST `/stt` takes WAV and returns
  JSON text; a WebSocket `/stt/stream` takes PCM chunks and sends partial/final text; POST
  `/tts` takes JSON text and voice and streams 16-bit PCM. Started with `python -m
  edith.server`. A random token in `output/server-token` is required on every request.
- [ ] `docs/protocol.md` documents the API with examples; end-to-end tests run the server on a
  random port.
- [ ] Final benchmark table in `README.md`. Wiring Edith into Friday itself happens in Friday's
  own repository, not here.

## Worker notes (this session)

- Added `src/edith/wer.py` (jiwer WER, case/punctuation normalised) and
  `src/edith/references.py` (`python -m edith.references` → draft `data/references.json`, every
  entry marked `"reviewed": false` for Progress to check).
- `edith.bench` now benchmarks **both** STT models (`large-v3-turbo` and `small.en`) when run as
  `bench all`, and scores WER against `data/references.json` when it exists (new WER column).
- Added `src/edith/bridge.py`: a line-delimited JSON stdin/stdout bridge for the UI.
- Added `ui/` — an Electron test console (STT panel, benchmark panel, TTS placeholder) with
  `contextIsolation: true`, `nodeIntegration: false`, a preload allow-list, and a main process that
  spawns `python -m edith.bridge`. See `README.md` for launch steps.
- Pending approvals: model download + `references` command, the `bench` run, and `npm install` in
  `ui/`.
