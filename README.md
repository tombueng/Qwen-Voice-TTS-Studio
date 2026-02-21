# Qwen Voice TTS Studio 1.1
<img width="848" height="352" alt="image" src="https://github.com/user-attachments/assets/cbf18fef-cbc6-442b-90bd-0674fe3b5d35" />

A Windows-first, user-friendly Gradio GUI for **Qwen3-TTS**.

- Text-to-Speech with 33 voices (built-in, designed, and celebrity clones)
- Voice Cloning
- Voice Design
- Voice ASR (Speech-to-Text)
- Conversations
- Audio Library
- Saved cloned/designed voices for reuse
- MP3 voice samples for fast preview
- Portable cloned voice references (stored in `./voiceinputs`)
- Save-name option for outputs (filename prefix + timestamp)
- Download outputs as WAV or MP3
- **No ffmpeg/sox required** — audio pipeline runs on soundfile + librosa only

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

The voice selector is a clickable table showing **Name**, **Type**, **Gender**, and **Description**.
Three voice types are available:

| Type | Count | Description |
|------|------:|-------------|
| Built-in | 9 | Named speaker IDs from the CustomVoice model (Aiden, Dylan, Eric, …) |
| Designed | 19 | Voices created from text style instructions (Emma, James, Rachel, …) |
| Clone | 5 | Celebrity voices from reference audio (Merkel, Trump, Scholz, …) |

The voice library is defined in `main_voices.json` — edit it to add, remove, or customise voices.

- Optional save name for output filename prefix

### Voice Cloning

- Upload WAV or MP3 reference (no ffmpeg required — soundfile reads both natively)
- Reference audio is automatically resampled to 24 kHz for the model
- Save cloned voices for later reuse (no need to re-upload)
- Reference audio is stored in `./voiceinputs` so cloned voices stay portable
- Optional save name for output filename prefix

### Voice Design

- Create voices from text descriptions
- Save designed voices for later reuse
- Optional save name for output filename prefix

### Conversations

- Select up to 3 voices
- Write a script using tags like `[Voice1]`, `[Voice2]`, `[Voice3]`
- Generates all parts and merges them into a single WAV output

### Voice ASR (Speech-to-Text)

- Upload audio or record from microphone
- Transcribe audio to text using Qwen3-ASR
- Auto-saves transcripts to `./outputs` (timestamped filenames)
- Optional timestamps output (requires Forced Aligner model)
- Optional re-voice: synthesize the transcript back into speech using any selected TTS voice

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

- **Python 3.12** — install from [python.org](https://www.python.org/downloads/) and check *"Add Python to PATH"* during setup
- Disk space: models are ~10-14 GB total
- **No ffmpeg, ffprobe, sox, or pydub** — the audio pipeline uses `soundfile` (native MP3/WAV/FLAC via libsndfile) and `librosa` only

## Model Download

Models are downloaded on first use and cached in `./models/`. To pre-download all models before running the UI:

```
python download_models.py
```

Options:

```
python download_models.py --custom    # CustomVoice model only
python download_models.py --base      # Base (voice cloning) only
python download_models.py --design    # VoiceDesign only
python download_models.py --asr       # ASR only
python download_models.py --models-dir D:/ml/models   # custom output directory
```

The app checks for a local `./models/<model-name>` directory first; it only hits the network if the directory is missing or empty.

## Screenshots
<img width="2256" height="1236" alt="Screenshot 2026-01-26 144955" src="https://github.com/user-attachments/assets/aeb48aa8-95be-4a1e-bbc9-e8a96b8a3bc1" />
<img width="2233" height="1281" alt="Screenshot 2026-01-26 145136" src="https://github.com/user-attachments/assets/f61f15eb-17ac-41e1-b3b8-0326168ce5e0" />
<img width="2249" height="1175" alt="Screenshot 2026-01-26 145308" src="https://github.com/user-attachments/assets/eca65bbe-00a2-4a18-a190-db32f9a88e87" />
<img width="2251" height="1301" alt="Screenshot 2026-01-26 145747" src="https://github.com/user-attachments/assets/25d6917c-4f5c-4b45-ba06-6c32e7c4cc05" />
<img width="2242" height="502" alt="Screenshot 2026-01-26 145811" src="https://github.com/user-attachments/assets/7a11dab4-ad7a-4898-9066-cb5fb7cb29d8" />
<img width="2256" height="1205" alt="Screenshot 2026-01-26 180324" src="https://github.com/user-attachments/assets/956608c8-2f7d-4928-8dbf-617aab1d901e" />
<img width="2265" height="1311" alt="Screenshot 2026-01-26 184549" src="https://github.com/user-attachments/assets/55b4c80d-091c-42f9-b81a-412c7f494083" />



## Files / Folders

Key paths used by the app:

- `./models` (downloaded models — pre-populate with `download_models.py`)
- `./outputs` (generated audio)
- `./voiceinputs` (reference audio for cloned voices, auto-resampled to 24 kHz)
- `./cloned_voices` (saved cloned voice metadata)
- `./designed_voices` (saved designed voice metadata)
- `./voicesamples` (MP3 preview samples)
- `./voices` (reference MP3s + transcripts for built-in clone voices)
- `main_voices.json` (voice library — edit to add/remove/customise voices)

## License

This repository is licensed under the **Apache License 2.0** (see `LICENSE`).

Upstream reference:

- Qwen3-TTS repo: https://github.com/QwenLM/Qwen3-TTS
- The upstream codebase indicates Apache-2.0 licensing, and model weights may have separate terms.
