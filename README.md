# Edith

A fast, local voice engine for Friday: **speech-to-text first, then text-to-speech**, running on
this laptop (RTX 4070 or CPU). The step-by-step build plan, rules and targets are in
[PLAN.md](PLAN.md).

## Layout

- `src/edith/`: the engine package
  - `audio.py`: load any audio file as 16 kHz mono float32; write WAV
  - `samples.py`: the sample voices in `../sample voices` (testing and comparison only)
- `tests/`: pytest tests (`conftest.py` provides the samples)
- `models/`, `output/`: downloaded models and generated files (not committed)

## Commands (from projects.yaml)

| Name    | Command                                                     | Approval    |
| ------- | ----------------------------------------------------------- | ----------- |
| test    | `.venv/Scripts/python -m pytest -q`                         | no          |
| setup   | `py -3.10 -m venv .venv`                                    | yes         |
| install | `.venv\Scripts\python -m pip install -e .[dev,stt,tts,audio,gpu]` | yes   |
| bench   | `.venv\Scripts\python -m edith.bench all`                   | yes         |

## Status

| Milestone                          | State       | Measured                           |
| ---------------------------------- | ----------- | ---------------------------------- |
| M1 Foundations                     | in progress | audio loading and samples tested   |
| M2 Speech-to-text v1               | not started |                                    |
| M3 Speech-to-text v2 (streaming)   | not started |                                    |
| M4 Text-to-speech v1 (Kokoro)      | not started |                                    |
| M5 Edith's voice                   | not started |                                    |
| M6 Streaming speech                | not started |                                    |
| M7 Ready for Friday                | not started |                                    |
