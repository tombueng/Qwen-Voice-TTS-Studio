# Qwen Voice TTS Studio 0.9 (Beta)
<img width="848" height="352" alt="image" src="https://github.com/user-attachments/assets/cbf18fef-cbc6-442b-90bd-0674fe3b5d35" />

A Windows-first, user-friendly Gradio GUI for **Qwen3-TTS**.

- Text-to-Speech
- Voice Cloning
- Voice Design
- Audio Library
- Saved cloned/designed voices for reuse
- MP3 voice samples for fast preview

Project GitHub:
https://github.com/Starnodes2024/Qwen-Voice-TTS-Studio

Upstream reference (Qwen3-TTS):
https://github.com/QwenLM/Qwen3-TTS

## Quick Start (Windows)

### 1) Setup

git clone https://github.com/Starnodes2024/Qwen-Voice-TTS-Studio

Run:

`01Setup.bat`

This will:

- Create `./venv`
- Install **PyTorch 2.9.0** (CUDA build if NVIDIA GPU is detected, otherwise CPU)
- Install dependencies from `requirements.txt`
- Install **FlashAttention 2** (Windows wheel) when an NVIDIA GPU is detected
- Optionally download the 3 Qwen3-TTS models into `./models`

### 2) Start

Run:

`02Start.bat`

The UI opens in your browser.

#### LAN Mode (use on phone / another PC)

If you enable LAN mode in the start script, it will:

- bind on `0.0.0.0` (so other devices can reach it)
- print the **real LAN URL** (e.g. `http://192.168.1.123:7860`) in purple

## Quick Start (Linux)

Run:

- `./01Setup.sh`
- `./02Start.sh`

**Not tested on Linux**. Please provide a PR if something breaks.

## Features

### Text-to-Speech

- Built-in personas
- Optional instructions to control style

### Voice Cloning

- Upload WAV or MP3 reference
- Save cloned voices for later reuse (no need to re-upload)

### Voice Design

- Create voices from text descriptions
- Save designed voices for later reuse

### Voice Samples

- Generate MP3 samples for all personas for fast preview
- Samples are stored in `./voicesamples`

### Render Device selector

In **Settings**, you can choose:

- Auto (default)
- GPU
- CPU

Changing it reloads models on the selected device.

## Performance Notes

- Fastest: **NVIDIA CUDA + FlashAttention2**
- If FlashAttention2 is unavailable, the app automatically falls back to `sdpa` / `eager` attention.
- Startup logs include the selected attention implementation.

## Requirements

- Windows: repo includes embedded Python **3.12** under `./312`
- Disk space:
  - models are ~10-12GB total
- SoX (required for audio processing). Install it and ensure `sox` is on PATH.

## Files / Folders

Key paths used by the app:

- `./models` (HF cache + downloaded models)
- `./outputs` (generated audio)
- `./cloned_voices` (saved cloned voice metadata)
- `./designed_voices` (saved designed voice metadata)
- `./voicesamples` (MP3 preview samples)

## License

This repository is licensed under the **Apache License 2.0** (see `LICENSE`).

Upstream reference:

- Qwen3-TTS repo: https://github.com/QwenLM/Qwen3-TTS
- The upstream codebase indicates Apache-2.0 licensing, and model weights may have separate terms.
