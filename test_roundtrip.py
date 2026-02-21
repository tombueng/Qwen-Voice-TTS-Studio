"""Round-trip TTS > ASR quality test.

For every voice in main_voices.json:
  1. Generate audio with the test text via TTS
  2. Transcribe the result with Qwen3-ASR
  3. Compare the transcription to the input text and report WER

Usage:
    python test_roundtrip.py
    python test_roundtrip.py --text "Custom test sentence."
    python test_roundtrip.py --voice "Aiden"
    python test_roundtrip.py --force           # re-generate TTS even if cached
    python test_roundtrip.py --pass-wer 0.20   # accept up to 20% WER as PASS
    python test_roundtrip.py --models-dir D:/models
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows (logger uses emojis / box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

_DEFAULT_TEXT = "Ladies and gentlemen, thank you for joining us today. We face important challenges, but together we will find the right solutions."


# ── Text comparison helpers ───────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _wer(ref: str, hyp: str) -> float:
    """Word error rate (0.0 = perfect, 1.0 = totally wrong)."""
    ref_words = _normalize(ref).split()
    hyp_words = _normalize(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    # Word-level Levenshtein distance
    m, n = len(ref_words), len(hyp_words)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / m


# ── TTS generation (mirrors generate_samples.py) ─────────────────────────────

def _generate_audio(app, voice: dict, text: str, base_dir: Path, out_path: Path) -> bool:
    model_type = voice.get("model", "")
    name = voice.get("name", "")

    if model_type == "custom":
        app.tts_generator.generate_tts(
            text=text,
            language="auto",
            speaker=voice["speaker"],
            output_file=out_path,
            voice_name=name,
        )
        return out_path.exists()

    elif model_type == "design":
        result = app.voice_designer.design_voice(
            text_input=text,
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
            text=text,
            cloned_voice_config={"ref_audio": str(ref_audio), "ref_text": ref_text},
            voice_name=name,
        )
        if result and Path(result).exists():
            shutil.copy2(result, out_path)
            return True
        return False

    return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Round-trip TTS > ASR quality test for all voices in main_voices.json"
    )
    parser.add_argument("--text",       default=_DEFAULT_TEXT, metavar="TEXT",
                        help=f"Test text to synthesise and transcribe (default: {_DEFAULT_TEXT!r})")
    parser.add_argument("--voice",      metavar="NAME",
                        help="Test a single named voice only")
    parser.add_argument("--force",      action="store_true",
                        help="Re-generate TTS audio even if it already exists")
    parser.add_argument("--pass-wer",   type=float, default=0.10, metavar="N",
                        help="WER threshold for PASS (default: 0.10 = 10%%)")
    parser.add_argument("--models-dir", default="./models", metavar="DIR",
                        help="Directory containing downloaded models (default: ./models)")
    parser.add_argument("--output-dir", default="./test_roundtrip", metavar="DIR",
                        help="Directory for generated WAVs and report (default: ./test_roundtrip)")
    parser.add_argument("--model-type", metavar="TYPE",
                        help="Filter voices by model type: clone, design, custom")
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # ── Load voice list ───────────────────────────────────────────────────────
    voices_file = base_dir / "main_voices.json"
    try:
        voices = json.loads(voices_file.read_text(encoding="utf-8")).get("voices", [])
    except Exception as exc:
        print(f"ERROR: cannot load {voices_file}: {exc}")
        sys.exit(1)

    if args.voice:
        voices = [v for v in voices if v.get("name") == args.voice]
        if not voices:
            print(f"ERROR: voice '{args.voice}' not found in main_voices.json")
            sys.exit(1)

    if args.model_type:
        voices = [v for v in voices if v.get("model") == args.model_type]
        if not voices:
            print(f"ERROR: no voices with model type '{args.model_type}' found")
            sys.exit(1)

    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  Qwen Voice TTS Studio - Round-Trip Test  (TTS > ASR)")
    print("=" * 62)
    print(f"  Voices      : {len(voices)}")
    print(f"  Test text   : {args.text!r}")
    print(f"  PASS if WER : < {args.pass_wer:.0%}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Models dir  : {args.models_dir}")
    print("=" * 62)
    print()

    # ── Boot app ──────────────────────────────────────────────────────────────
    from utils import get_config_from_env
    from app import create_app

    config = get_config_from_env()
    config["models_dir"] = args.models_dir

    print("Initialising application...")
    app = create_app(config)
    print()

    # ── Load ASR model once ───────────────────────────────────────────────────
    print("Loading ASR model...")
    asr_status = app.model_manager.load_asr_model("Qwen/Qwen3-ASR-1.7B")
    if "Error" in asr_status or "error" in asr_status.lower():
        print(f"  {asr_status}")
        sys.exit(1)
    print(f"  {asr_status}")
    print()

    # ── Run tests ─────────────────────────────────────────────────────────────
    results = []
    safe = lambda name: name.strip().replace(" ", "_")

    for voice in voices:
        name = voice.get("name", "")
        if not name:
            continue

        model_type = voice.get("model", "—")
        wav_path   = output_dir / f"{safe(name)}.wav"
        label      = f"[{model_type:7}]"

        print(f"  {label}  {name}")

        # Step 1 — TTS
        t0 = time.time()
        if wav_path.exists() and not args.force:
            print(f"             TTS : (cached)")
        else:
            try:
                ok = _generate_audio(app, voice, args.text, base_dir, wav_path)
                t_tts = time.time() - t0
                if not ok:
                    print(f"             TTS : FAILED ({t_tts:.1f}s)")
                    results.append({"name": name, "model": model_type,
                                    "status": "TTS_FAILED", "wer": None, "transcription": ""})
                    print()
                    continue
                print(f"             TTS : OK ({t_tts:.1f}s)")
            except Exception as exc:
                t_tts = time.time() - t0
                print(f"             TTS : FAILED ({t_tts:.1f}s) — {exc}")
                results.append({"name": name, "model": model_type,
                                "status": "TTS_FAILED", "wer": None, "transcription": ""})
                print()
                continue

        # Step 2 — ASR
        t1 = time.time()
        try:
            asr_results = app.model_manager.asr_model.transcribe(
                audio=str(wav_path),
                language=None,
                return_time_stamps=False,
            )
            transcription = (asr_results[0].text if asr_results else "") or ""
            t_asr = time.time() - t1
        except Exception as exc:
            print(f"             ASR : FAILED — {exc}")
            results.append({"name": name, "model": model_type,
                            "status": "ASR_FAILED", "wer": None, "transcription": ""})
            print()
            continue

        # Step 3 — Compare
        wer    = _wer(args.text, transcription)
        passed = wer <= args.pass_wer
        status = "PASS" if passed else "FAIL"
        mark   = "PASS" if passed else "FAIL"

        print(f"             ASR : {transcription!r}  ({t_asr:.1f}s)")
        print(f"             WER : {wer:.1%}  ->  {mark}")
        print()

        results.append({
            "name":          name,
            "model":         model_type,
            "status":        mark,
            "wer":           round(wer, 4),
            "transcription": transcription,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass   = sum(1 for r in results if r["status"] == "PASS")
    n_fail   = sum(1 for r in results if r["status"] == "FAIL")
    n_error  = sum(1 for r in results if "FAILED" in r["status"])

    print("=" * 62)
    print(f"  PASS  : {n_pass}")
    print(f"  FAIL  : {n_fail}")
    print(f"  ERROR : {n_error}")
    print("=" * 62)

    if n_fail:
        print("\n  Failing voices:")
        for r in (r for r in results if r["status"] == "FAIL"):
            print(f"    [{r['model']:7}]  {r['name']:<22}  WER={r['wer']:.1%}")
            print(f"              got : {r['transcription']!r}")

    if n_error:
        print("\n  Errored voices:")
        for r in (r for r in results if "FAILED" in r["status"]):
            print(f"    [{r['model']:7}]  {r['name']:<22}  {r['status']}")

    # Save JSON report
    report = {
        "text":               args.text,
        "pass_wer_threshold": args.pass_wer,
        "summary":            {"pass": n_pass, "fail": n_fail, "error": n_error},
        "results":            results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Report saved : {report_path}")
    print()

    if n_fail or n_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
