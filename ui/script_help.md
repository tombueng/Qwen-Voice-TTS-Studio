Render a multi-speaker script from JSON.
All speakers are validated and all referenced files are checked **before** any audio is generated.

---

### Speaker definitions *(pick one per speaker — mutually exclusive)*

| Field | Resolves to | Example |
|---|---|---|
| **`ref_name`** | Any built-in or saved voice by name. When used without a `name` field, `ref_name` also serves as the dialog identity. | `{"ref_name": "Aiden"}` |
| **`ref_name`** + **`name`** | Same voice lookup, but dialogs use the `name` alias. | `{"name": "Narrator", "ref_name": "Emma"}` |
| **`ref_speaker`** | Built-in CustomVoice ID directly. | `{"name": "Host", "ref_speaker": "serena"}` |
| **`ref_audio`** + **`ref_text`** | Voice clone from a file in `voices/`. Extension optional — tries `.wav .mp3 .flac .ogg`. | `{"name": "Guest", "ref_audio": "myvoice", "ref_text": "..."}` |
| **`ref_description`** / **`voice`** | On-the-fly voice design from a text description. | `{"name": "Wizard", "voice": "Male, elderly, deep resonant voice"}` |

The top-level speakers array may be named `speaker` or `speakers` (both accepted).

---

### Per-line fields

| Field | Type | Notes |
|---|---|---|
| `speaker` | string | Must match a defined speaker name |
| `text` | string | Dialog text; may contain inline audio markers |
| `connotation` | string | Emotion/style hint for designed voices |
| `speed` | float | Playback speed multiplier (default `1.0`) |
| `pause_after` | float | Silence in seconds after this line |

---

### Scene directions *(AI-generated background, mixed under the full scene)*

```json
"direction": [
  {"type": "ambience", "instruction": "busy café, clinking cups", "volume": 0.25, "steps": 20},
  {"type": "music",   "instruction": "gentle orchestral underscore", "volume": 0.15}
]
```

Use `type: "info"` for a note that produces no audio.
Fields: `volume` (0–1, default 0.3), `steps` (default 20), `cfg_scale` (default 7.0).

---

### Inline audio markers

**Sequential** — speech pauses while the effect plays:
```
[audio:'crowd cheering',duration:2.0,pause_before:0.5,pause_after:1.0,volume:0.8,steps:20]
[audio:ref_file:'applause',duration:2.5,pause_after:0.6,volume:0.7]
```

**Background** — mixed *under* speech, does not interrupt the timeline:
```
[background-audio:'soft rain on leaves',duration:0,volume:0.3,steps:20]
[background-audio:ref_file:'birds',volume:0.2]
```

`ref_file:'name'` loads a pre-recorded file from the `audio/` folder — no AI generation.
Extension is optional (tries `.wav .mp3 .flac .ogg`).
`duration:0` on a background effect auto-sizes to the line's speech length.

Pre-installed files in `audio/`: **birds**, **rain**, **applause**, **cafe**.
