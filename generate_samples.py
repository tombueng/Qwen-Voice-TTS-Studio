"""Generate audio samples for all voices defined in main_voices.json.

Each voice gets a WAV file in ./voicesamples/{Voice_Name}.wav so users can
preview all voices without launching the full UI.

Usage:
    python generate_samples.py                         # all missing samples
    python generate_samples.py --force                 # regenerate all
    python generate_samples.py --voice "Aiden"         # one voice only
    python generate_samples.py --text "Custom text"    # override sample text
    python generate_samples.py --models-dir D:/models
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

# Default sample texts
_SAMPLE_EN = (
    "Hello! This is a voice sample from Qwen Voice TTS Studio. "
    "I hope you enjoy how I sound."
)
_SAMPLE_DE = (
    "Hallo! Dies ist eine Sprachprobe aus dem Qwen Voice TTS Studio. "
    "Ich hoffe, meine Stimme gefällt Ihnen."
)

# Voices whose ref_audio is in German → use German sample text
_GERMAN_STEMS = {"merkel", "merz", "scholz", "soeder", "söder"}


def _sample_text_for(voice: dict, override: str | None) -> str:
    if override:
        return override
    # Clone voices: match language to reference audio
    ref = Path(voice.get("ref_audio", "")).stem.lower()
    if ref in _GERMAN_STEMS:
        return _SAMPLE_DE
    return _SAMPLE_EN


def _safe_name(name: str) -> str:
    """Turn a voice name into a safe filename stem."""
    return name.strip().replace(" ", "_")


def _generate_one(app, voice: dict, sample_text: str, base_dir: Path,
                  out_path: Path) -> bool:
    """Generate a sample for *voice* and write it to *out_path*.

    Returns True on success.
    """
    model_type = voice.get("model", "")
    name = voice.get("name", "")

    if model_type == "custom":
        # generate_tts accepts an explicit output_file; write directly
        result = app.tts_generator.generate_tts(
            text=sample_text,
            language="auto",
            speaker=voice["speaker"],
            output_file=out_path,
            voice_name=name,
        )
        return out_path.exists()

    elif model_type == "design":
        result = app.voice_designer.design_voice(
            text_input=sample_text,
            instructions=voice.get("instruct", ""),
            voice_name=name,
        )
        if result and Path(result).exists():
            shutil.copy2(result, out_path)
            return True
        return False

    elif model_type == "clone":
        ref_audio = base_dir / voice["ref_audio"]
        ref_text = voice.get("ref_text", "")
        result = app.voice_cloner.generate_with_cloned_voice(
            text=sample_text,
            cloned_voice_config={"ref_audio": str(ref_audio), "ref_text": ref_text},
            voice_name=name,
        )
        if result and Path(result).exists():
            shutil.copy2(result, out_path)
            return True
        return False

    print(f"    Unknown model type: {model_type!r}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate voice samples for all voices in main_voices.json"
    )
    parser.add_argument("--force",      action="store_true",
                        help="Regenerate sample even if it already exists")
    parser.add_argument("--voice",      metavar="NAME",
                        help="Generate sample for a single named voice only")
    parser.add_argument("--text",       metavar="TEXT",
                        help="Override sample text (used for all voices)")
    parser.add_argument("--models-dir", default="./models", metavar="DIR",
                        help="Directory containing the downloaded models (default: ./models)")
    parser.add_argument("--output-dir", default="./voicesamples", metavar="DIR",
                        help="Directory for generated samples (default: ./voicesamples)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # ── Load voice list ───────────────────────────────────────────────────────
    voices_file = base_dir / "main_voices.json"
    try:
        voices = json.loads(voices_file.read_text(encoding="utf-8")).get("voices", [])
    except Exception as exc:
        print(f"ERROR: cannot load {voices_file}: {exc}")
        sys.exit(1)

    if not voices:
        print("No voices found in main_voices.json — nothing to do.")
        sys.exit(0)

    # Optional single-voice filter
    if args.voice:
        voices = [v for v in voices if v.get("name") == args.voice]
        if not voices:
            print(f"ERROR: voice '{args.voice}' not found in main_voices.json")
            sys.exit(1)

    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 52)
    print("  Qwen Voice TTS Studio — Sample Generator")
    print("=" * 52)
    print(f"  Voices    : {len(voices)}")
    print(f"  Output    : {output_dir}")
    print(f"  Models    : {args.models_dir}")
    print(f"  Force     : {args.force}")
    print("=" * 52)
    print()

    # ── Boot app ──────────────────────────────────────────────────────────────
    from utils import get_config_from_env
    from app import create_app

    config = get_config_from_env()
    config["models_dir"] = args.models_dir

    print("Initialising application...")
    app = create_app(config)
    print()

    # ── Generate ──────────────────────────────────────────────────────────────
    ok = skipped = failed = 0

    for voice in voices:
        name = voice.get("name", "")
        if not name:
            continue

        model_type = voice.get("model", "—")
        out_path   = output_dir / f"{_safe_name(name)}.wav"
        text       = _sample_text_for(voice, args.text)

        if out_path.exists() and not args.force:
            print(f"  [skip ]  {name}")
            skipped += 1
            continue

        label = f"[{model_type:7}]"
        print(f"  {label}  {name} ...", end="", flush=True)
        t0 = time.time()

        try:
            success = _generate_one(app, voice, text, base_dir, out_path)
            elapsed = time.time() - t0
            if success:
                print(f"  OK  ({elapsed:.1f}s)")
                ok += 1
            else:
                print(f"  FAILED (no output after {elapsed:.1f}s)")
                failed += 1
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  FAILED ({elapsed:.1f}s): {exc}")
            failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 52)
    print(f"  Generated : {ok}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    print("=" * 52)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
