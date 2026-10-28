# Edith

A fast, local voice engine for Friday: **speech-to-text first, then text-to-speech**, running on
this laptop (RTX 4070 or CPU). The step-by-step build plan, rules and targets are in
[PLAN.md](PLAN.md).

## Layout

- `src/edith/`: the engine package
  - `audio.py`: load any audio file as 16 kHz mono float32; write WAV
  - `samples.py`: the sample voices in `../sample voices` (testing and comparison only)
  - `stt.py`: offline transcription with faster-whisper (GPU float16 / CPU int8)
  - `references.py`: drafts `data/references.json` (large-v3, beam 5) for review
  - `wer.py`: word error rate (jiwer, case/punctuation normalised)
  - `text.py`: text normalisation for speech (numbers, currency, dates, times, units,
    abbreviations, symbols, markdown)
  - `models.py`: pinned model downloads into `models/` (`python -m edith.models --list`)
  - `tts/engine.py`: `Synthesizer` (normalise, clause chunks, backend, smooth joins);
    Kokoro-82M ONNX is the baseline backend
  - `roundtrip.py`: Edith speaks 20 fixed sentences, Whisper transcribes them, WER
    (`python -m edith.roundtrip` writes `output/roundtrip.md`)
  - `tts/smooth.py`: anti-jitter layer for TTS (DC removal, fades, equal-power crossfades,
    loudness matching) and join metrics
  - `bench.py`: benchmark harness (RTF, latency, WER; writes `output/bench.json`)
  - `bridge.py`: line-delimited JSON stdin/stdout bridge used by the UI
- `tests/`: pytest tests (`conftest.py` provides the samples); the Electron bridge's Node tests run
  from here too
- `ui/`: Electron test console (STT, benchmark, TTS placeholder)
- `data/references.json`: reviewed reference transcripts (committed; drafted by `edith.references`)
- `models/`, `output/`: downloaded models and generated files (not committed)

## Commands (from projects.yaml)

| Name    | Command                                                     | Approval    |
| ------- | ----------------------------------------------------------- | ----------- |
| test    | `.venv/Scripts/python -m pytest -q`                         | no          |
| setup   | `py -3.10 -m venv .venv`                                    | yes         |
| install | `.venv\Scripts\python -m pip install -e .[dev,stt,tts,audio,gpu]` | yes   |
| bench   | `.venv\Scripts\python -m edith.bench all`                   | yes         |
| gpu-tts | `.venv\Scripts\python -m pip uninstall -y onnxruntime` then `.venv\Scripts\python -m pip install --no-deps onnxruntime-gpu==1.23.2` (after `install`) | yes |

Note: `python -m edith.references` (draft reference transcripts) is not yet a named command in
projects.yaml — it needs Progress to add it (or approve a one-off run) before the first draft run.

## UI (Electron test console)

Launch:

    cd ui
    npm install      # needs Progress's approval (downloads Electron)
    npm start

The main process spawns `python -m edith.bridge` from the project `.venv` and exposes a small
allow-list to the renderer via a preload script (`contextIsolation` on, `nodeIntegration` off).

- **STT panel:** transcribe an audio file or a live microphone recording
  (model `large-v3-turbo`/`small.en`, device `auto`/`cpu`/`cuda`); shows transcript, segments,
  RTF and latency.
- **Benchmark panel:** reads `output/bench.json`.
- **TTS panel:** placeholder (M4).

Run the bridge's own unit tests with `node --test test/bridge.test.js` in `ui/` (also run by the
project `test` command when Node is installed).

## Status

| Milestone                          | State       | Measured                           |
| ---------------------------------- | ----------- | ---------------------------------- |
| M1 Foundations                     | done        | audio, samples and benchmark harness tested |
| M2 Speech-to-text v1               | in progress | Transcriber + Transcript API tested; WER harness ready; references module ready (draft run pending approval); UI STT panel wired |
| M3 Speech-to-text v2 (streaming)   | not started |                                    |
| M4 Text-to-speech v1 (own engine) | in progress | anti-jitter layer tested (joins ≤ 1.5× interior jump, level steps ≤ 3 dB); text normaliser tested; Kokoro baseline speaking through the engine. GPU (onnxruntime-gpu, cuDNN 9.8): RTF 0.086-0.092, first audio 245 ms (targets 0.03 and 150 ms not met yet). CPU (6 threads): RTF 0.24-0.30, first audio 445 ms. Real-speech joins: jump ≤ 0.0005, chunk levels within 0.4 dB. Round trip (Kokoro af_heart → Whisper large-v3-turbo, GPU): WER 0.0% over 20 sentences / 212 words |
| M5 Edith's own voice (fine-tune)  | not started |                                    |
| M6 Streaming speech                | not started |                                    |
| M7 Ready for Friday                | not started |                                    |
