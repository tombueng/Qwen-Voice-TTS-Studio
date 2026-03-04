# Qwen Dialogue Studio

**Multi-speaker narration, voice cloning, and synthesis powered by Qwen3-TTS**

Produce full audio productions — narrated stories, multi-character dialogues, podcasts, audiobooks — from a single JSON script. Every speaker can be a cloned voice (any uploaded audio), a built-in voice ID, or a voice described in plain text. Sound effects and background music are generated on the fly via Stable Audio and mixed in automatically. Everything merges into one polished WAV.

Single-voice modes (TTS, Clone, Design) are also available for quick generation.

Project GitHub: https://github.com/Starnodes2024/Qwen-Voice-TTS-Studio
Upstream (Qwen3-TTS): https://github.com/QwenLM/Qwen3-TTS

---

## Quick Start (Windows)

### 1) Prerequisites

Install **Python 3.12** from [python.org](https://www.python.org/downloads/).
Check *"Add Python to PATH"* during installation.

### 2) Clone and Setup

```
git clone https://github.com/Starnodes2024/Qwen-Voice-TTS-Studio
cd Qwen-Voice-TTS-Studio
01Setup.bat
```

`01Setup.bat` will:

- Detect Python 3.12 on your system
- Create `./venv` and install all dependencies
- Install **PyTorch 2.9.0** — CUDA build if an NVIDIA GPU is detected, otherwise CPU
- Install **FlashAttention 2** (Windows wheel, NVIDIA only) for maximum throughput
- Optionally download all four Qwen3 models (~10–14 GB total)
- Optionally generate preview samples for all built-in voices

### 3) Start

```
02Start.bat
```

The Gradio UI opens in your browser at `http://127.0.0.1:7860`.

#### LAN Mode

At startup you are asked whether to enable LAN mode. If yes, the server binds on `0.0.0.0` and prints the local network URL (e.g. `http://192.168.1.10:7860`) so you can use the studio from a phone or another PC on the same network.

---

## Quick Start (Linux)

```
./01Setup.sh
./02Start.sh
```

*Not tested on Linux — open a PR if something needs fixing.*

---

## Tabs Overview

| Tab | Purpose |
|-----|---------|
| 🎤 **TTS** | Single-voice text-to-speech from the full voice library |
| 🎯 **Clone** | Clone any voice from an uploaded audio reference |
| 🎨 **Design** | Synthesise speech with a voice described in plain text |
| 📜 **Script** | Multi-speaker dialogue engine — full JSON script runner |
| 📝 **ASR** | Transcribe audio to text (Qwen3-ASR) |
| 🔊 **Stable Audio** | Generate sound effects and music from text (Stable Audio Open 1.0) |
| ⚙️ **Settings** | HuggingFace token for gated models |

---

## 🎤 TTS Tab

Select a language, pick a voice, enter text, click **Generate**.

### Voice Library

All voices are defined in `main_voices.json`. Three built-in types exist:

| Type | Description | Routing |
|------|-------------|---------|
| **Built-in** | Named speaker IDs shipped with the CustomVoice model | Qwen3-TTS-CustomVoice |
| **Designed** | Voices created from a plain-text style description | Qwen3-TTS-VoiceDesign |
| **Clone** | Celebrity/reference voices from uploaded audio | Qwen3-TTS-Base (ICL) |

The dropdown also shows your **saved user voices** (clones or designs you have saved from the Clone/Design tabs) at the top, prefixed with `[My]`.

### Editing `main_voices.json`

Each entry follows one of these formats:

```json
{ "name": "Aiden",  "model": "custom",  "speaker": "aiden",  "gender": "Male", "age": 30, "desc": "Warm baritone" }
{ "name": "Emma",   "model": "design",  "instruct": "Female, 28, warm and friendly, calm measured pacing" }
{ "name": "Donald Trump", "model": "clone", "ref_audio": "voices/trump.mp3", "ref_text": "Full transcript of trump.mp3..." }
```

`ref_text` must match the **full content** of the reference audio for optimal clone quality.

---

## 🎯 Clone Tab

Clone any voice from a short audio clip and synthesise new speech with it.

1. Upload a WAV or MP3 reference (any sample rate — resampled automatically to 24 kHz)
2. Paste the text that is spoken in the reference clip into **Reference Text** (leave empty to use speaker embedding only)
3. Enter the **Text to Synthesise**
4. Click **Clone Voice**

### Saving a Cloned Voice

Expand **💾 Save this voice to My Voices**, give the voice a name and optional description, click **Save**. The reference audio is copied to `./uservoices/` and a record is written to `user_voices.json`. The voice then appears at the top of the TTS dropdown for reuse without re-uploading.

---

## 🎨 Design Tab

Create a voice by describing it in plain text, then synthesise speech with it.

1. Enter a **Voice Description** (e.g. *"Male, 55 years old, deep and authoritative, slow deliberate pacing, gravelly texture"*)
2. Enter the **Text to Synthesise**
3. Click **Design & Generate**

### Saving a Designed Voice

Expand **💾 Save this voice to My Voices**, give the voice a name, click **Save**. The description is stored in `user_voices.json` and the voice appears in the TTS dropdown.

---

## 📜 Script Tab — Multi-Speaker Dialogue Engine

The most powerful feature. Write a JSON script that defines speakers and scenes; the engine renders every line and merges them into a single WAV.

**Any combination of voice types can appear in the same script** — a cloned celebrity voice can speak alongside a designed character and a built-in narrator, all in one production. Sound effects and background music (via Stable Audio) can be woven in at any point.

### JSON Structure

```json
{
  "pause_between_lines":  1.0,
  "pause_between_scenes": 2.0,
  "speaker": [ ... ],
  "scenes":  [ ... ]
}
```

> The speaker list may be named **`speaker`** or **`speakers`** — both are accepted.

#### Root settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pause_between_lines` | float | `1.0` | Seconds of silence between every dialog line |
| `pause_between_scenes` | float | `2.0` | Seconds of silence between scenes |

---

### Speaker Definitions

Each entry in `speaker` defines one character. Four voice source modes are available — mix freely:

#### Mode 1 — Named Voice (`ref_name`)

```json
{ "ref_name": "My Saved Voice" }
```

- Looks up a voice by name from **My Voices** (`user_voices.json`) or the built-in library (`main_voices.json`)
- When `name` is omitted, `ref_name` doubles as the speaker's identity in dialogs (dialog lines reference it via `"speaker": "My Saved Voice"`)
- Simplest option — no extra files needed

#### Mode 2 — Voice Clone (from reference audio)

```json
{
  "name": "Donald Trump",
  "ref_audio": "trump.mp3",
  "ref_text": "Thank you very much. It's a privilege to be here..."
}
```

- `ref_audio` — filename inside the `./voices/` folder (relative, no `../`)
- `ref_text` — full transcript of the reference audio; must match the clip exactly for best quality
- Uses **Qwen3-TTS-Base** in ICL (In-Context Learning) mode

#### Mode 3 — Built-in Speaker ID

```json
{
  "name": "Narrator",
  "ref_speaker": "aiden"
}
```

- `ref_speaker` — any named speaker ID from the CustomVoice model (aiden, dylan, eric, ryan, serena, sohee, vivian, ono_anna, …)
- Uses **Qwen3-TTS-CustomVoice**

#### Mode 4 — Voice Design (plain-text description)

```json
{
  "name": "Character",
  "ref_description": "Female, 30 years old, warm and friendly voice, calm measured pacing"
}
```

- `ref_description` — natural language style instruction; describe age, gender, tone, pace, texture
- `voice` is accepted as an alias for `ref_description`
- Uses **Qwen3-TTS-VoiceDesign**

---

### Scene Definitions

```json
{
  "pos": 1,
  "title": "A Short Greeting",
  "pause_after": 3.0,
  "direction": [ ... ],
  "dialog": [ ... ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pos` | int | yes | Sort order for scenes |
| `title` | string | no | Label for your reference only; not rendered |
| `pause_after` | float | no | Overrides `pause_between_scenes` for the gap after this scene |
| `direction` | array | no | Scene-level background audio directions (see below) |
| `dialog` | array | yes | Ordered list of dialog lines |

> `direction` and `directions` are both accepted.

---

### Scene Directions (Background Audio)

Each entry in `direction` generates a Stable Audio clip the length of the entire scene and mixes it as background under all dialog in that scene:

```json
"direction": [
  {
    "type":        "ambience",
    "instruction": "quiet indoor room with faint traffic outside",
    "volume":      0.25,
    "steps":       20,
    "cfg_scale":   7.0
  },
  {
    "type":        "music",
    "instruction": "gentle orchestral underscore, warm and slow, children's fairy tale style"
  }
]
```

| Field | Default | Description |
|-------|---------|-------------|
| `type` | `"ambience"` | Category label — fed to Stable Audio as `type:instruction` |
| `instruction` | *(required)* | Natural-language description of the sound |
| `volume` | `0.3` | Background amplitude (0.0–1.0) |
| `steps` | `20` | Stable Audio diffusion steps |
| `cfg_scale` | `7.0` | Stable Audio guidance scale |

Results are cached by content hash — re-running with identical settings skips re-generation.

---

### Dialog Lines

```json
{
  "pos": 1,
  "speaker": "Narrator",
  "text": "The story begins on a quiet morning.",
  "connotation": "calm, measured",
  "pause_after": 0.5,
  "speed": 0.9
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pos` | int | yes | Sort order within the scene |
| `speaker` | string | yes | Must match a `name` or `ref_name` in `speaker` |
| `text` | string | yes | The text to synthesise (may contain inline markers, see below) |
| `connotation` | string | no | Style hint appended to the voice description (designed voices) |
| `pause_after` | float | no | Silence after this line; overrides `pause_between_lines` |
| `speed` | float | no | Playback speed factor (default `1.0`). Pitch-preserved via phase vocoder |

> `connotation` and `conotation` (legacy typo) are both accepted.

Lines with `speaker` set to `"direction"` or `"instruction"` are silently skipped — useful for embedding stage notes.

---

### Inline Audio Markers

Sound effects and background music can be embedded directly in the `text` field of any dialog line.

#### Sequential Effects — `[audio:...]`

Inserted at that exact position in the timeline; speech before and after pauses around it.

```
[audio:'crowd cheering loudly',duration:2.0,pause_before:0.3,pause_after:0.5,volume:0.8,steps:20,cfg_scale:7.0]
[audio:ref_file:'applause',duration:2.0,pause_after:0.5,volume:0.7]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration` | `2.0` | Length in seconds (Stable Audio) or playback hint (file ref) |
| `pause_before` | `0.0` | Silence inserted before the effect |
| `pause_after` | `0.0` | Silence inserted after the effect |
| `volume` | `1.0` | Amplitude scale of the effect |
| `steps` | `20` | Stable Audio diffusion steps |
| `cfg_scale` | `7.0` | Stable Audio guidance scale |

#### Background Effects — `[background-audio:...]`

Mixed *under* the surrounding speech — does **not** interrupt or pause the timeline.

```
[background-audio:'peaceful forest morning, birds and light breeze',duration:0,volume:0.3,steps:20]
[background-audio:ref_file:'rain_loop',volume:0.2]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration` | `0` | Length in seconds; `0` = auto-size to match the line's speech |
| `pause_before` | `0.0` | Silence added at the start of the foreground before mixing |
| `volume` | `0.3` | Background amplitude (0.0–1.0) |
| `steps` | `20` | Stable Audio diffusion steps (ignored for `ref_file`) |
| `cfg_scale` | `7.0` | Stable Audio guidance scale (ignored for `ref_file`) |

#### File References (`ref_file`)

Instead of generating audio with Stable Audio, you can reference a pre-recorded file from the `audio/` folder:

```
[audio:ref_file:'applause']                  → audio/applause.wav (or .mp3/.flac/.ogg)
[background-audio:ref_file:'soft_rain.wav']  → audio/soft_rain.wav
```

- Files must be placed in the `./audio/` folder at the project root
- Path traversal (`../`) is blocked
- If the file is not found, an error is reported (no silent Stable Audio fallback)

---

### Complete Script Example

```json
{
  "pause_between_lines": 1.0,
  "pause_between_scenes": 2.5,

  "speaker": [
    {
      "name": "Donald Trump",
      "ref_audio": "trump.mp3",
      "ref_text": "Thank you very much. It's a privilege to be here..."
    },
    {
      "name": "Narrator",
      "ref_speaker": "aiden"
    },
    {
      "name": "Anna",
      "ref_description": "Female, 32 years old, warm and slightly husky, slow deliberate pacing"
    },
    {
      "ref_name": "My Saved Voice"
    }
  ],

  "scenes": [
    {
      "pos": 1,
      "title": "Opening",
      "pause_after": 3.0,
      "direction": [
        {
          "type": "ambience",
          "instruction": "quiet conference room, faint air conditioning hum",
          "volume": 0.15,
          "steps": 20
        }
      ],
      "dialog": [
        {
          "pos": 1,
          "speaker": "Narrator",
          "text": "It was a cold Tuesday morning.",
          "connotation": "calm, neutral"
        },
        {
          "pos": 2,
          "speaker": "Donald Trump",
          "text": "Good morning, everyone. [audio:'crowd applause',duration:2.0,pause_after:0.5,volume:0.7] Tremendous day.",
          "speed": 0.95
        },
        {
          "pos": 3,
          "speaker": "Anna",
          "text": "[background-audio:'soft orchestral sting',duration:0,volume:0.25] Good morning. Shall we begin?",
          "pause_after": 0.3
        }
      ]
    },
    {
      "pos": 2,
      "title": "The Discussion",
      "dialog": [
        { "pos": 1, "speaker": "direction",     "text": "(characters sit down)" },
        { "pos": 2, "speaker": "Narrator",      "text": "They took their seats and the meeting started.", "speed": 1.05 },
        { "pos": 3, "speaker": "Anna",          "text": "We have a lot to cover today.", "connotation": "focused" }
      ]
    }
  ]
}
```

### Loading a Script

- **Upload JSON file** and click **Load from File** to populate the editor
- **Load Template** inserts a working example
- The **Parsed Preview** pane updates live as you type, showing any JSON errors immediately
- Click **▶ Run Script** to render all lines and download the merged WAV

---

## 🔊 Stable Audio Tab

Generate sound effects, ambience, or music from a text prompt using **Stable Audio Open 1.0** (stabilityai/stable-audio-open-1.0). The model (~3 GB) is downloaded from HuggingFace on first use — a HuggingFace access token is required (set it in the **⚙️ Settings** tab).

- **Duration** — 1–30 seconds
- **Steps** — more steps = higher quality, slower generation (20 steps ≈ good draft, 100 steps = full quality)
- **CFG Scale** — higher = follows prompt more strictly

Generated files are cached by content hash in `./outputs/` — re-running with identical settings returns the cached file instantly.

---

## 📝 ASR Tab

Transcribe any audio file to text using Qwen3-ASR.

- Upload WAV, MP3, or any format readable by libsndfile
- Select language (`auto`, `english`, `chinese`)
- Click **Transcribe**

---

## User Voices (`user_voices.json`)

Voices you save from the Clone or Design tabs are stored in `user_voices.json` at the repo root and their reference audio in `./uservoices/`. They are loaded on every startup and appear at the top of the TTS dropdown (prefixed `[My]`).

The file is a plain JSON array — you can edit, rename, or delete entries manually:

```json
{
  "voices": [
    {
      "name":      "My Clone",
      "desc":      "Warm male voice from uploaded sample",
      "model":     "clone",
      "ref_audio": "uservoices/My_Clone_20260222_143000.wav",
      "ref_text":  "The transcript of the reference clip..."
    },
    {
      "name":    "My Character",
      "desc":    "Older British narrator",
      "model":   "design",
      "instruct": "Male, 60 years old, distinguished British accent, slow authoritative delivery"
    }
  ]
}
```

---

## Models

| Model | Used for | Size |
|-------|---------|------|
| Qwen3-TTS-12Hz-1.7B-CustomVoice | TTS with named speaker IDs | ~3.5 GB |
| Qwen3-TTS-12Hz-1.7B-Base | Voice cloning (ICL mode) | ~3.5 GB |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | Voice design from text descriptions | ~3.5 GB |
| Qwen3-ASR-1.7B | Speech-to-text transcription | ~3.5 GB |
| stable-audio-open-1.0 | Sound effects and music generation | ~3 GB |

Models are downloaded on first use into `./models/` (Stable Audio into the HuggingFace hub cache). To pre-download Qwen3 models before launching:

```
python download_models.py
```

Options:

```
python download_models.py --custom              # CustomVoice only
python download_models.py --base                # Base (cloning) only
python download_models.py --design              # VoiceDesign only
python download_models.py --asr                 # ASR only
python download_models.py --models-dir D:/ml    # custom output path
```

---

## Performance Notes

- **Fastest**: NVIDIA GPU + CUDA + FlashAttention 2
- FlashAttention 2 is installed automatically by `01Setup.bat` when an NVIDIA GPU is detected
- Without FlashAttention 2 the app falls back to `sdpa` or `eager` attention automatically
- CPU mode works but is significantly slower for long texts or complex scripts
- Stable Audio on CPU takes ~37 minutes for 100 steps; use 20 steps for drafts
- Startup logs show the selected device, dtype, and attention implementation

---

## Requirements

- **Python 3.12** — [python.org](https://www.python.org/downloads/), check *"Add Python to PATH"*
- ~14–17 GB disk space for all models
- **No ffmpeg, ffprobe, sox, or pydub** — the audio pipeline uses `soundfile` (libsndfile, reads MP3/WAV/FLAC natively) and `librosa`

---

## Files & Folders

| Path | Contents |
|------|---------|
| `main_voices.json` | Built-in voice library — edit to add, remove, or customise voices |
| `user_voices.json` | User-saved voices (created by the Save buttons in Clone/Design tabs) |
| `./voices/` | Reference audio for built-in clone voices (referenced in `ref_audio`) |
| `./audio/` | Pre-recorded audio files for script `ref_file` references |
| `./uservoices/` | Reference audio for user-saved clone voices |
| `./models/` | Downloaded Qwen3 model weights |
| `./outputs/` | Generated WAV files (TTS, Clone, Script, Stable Audio) |
| `./voicesamples/` | Preview samples generated by `generate_samples.py` |
| `./cloned_voices/` | Raw output from the Clone tab (session files) |
| `./designed_voices/` | Raw output from the Design tab (session files) |

---

## License

This repository is licensed under the **Apache License 2.0** (see `LICENSE`).

- Qwen3-TTS upstream: https://github.com/QwenLM/Qwen3-TTS
- The upstream codebase is Apache-2.0; model weights may carry separate terms — check the HuggingFace model cards.
