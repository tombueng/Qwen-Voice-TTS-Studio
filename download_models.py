"""One-time model download script.

Downloads all Qwen TTS/ASR models into the local ./models/ directory so the
app never hits the network again on subsequent runs.

Usage:
    python download_models.py                     # download all models
    python download_models.py --custom            # CustomVoice only
    python download_models.py --base              # Base (voice cloning) only
    python download_models.py --design            # VoiceDesign only
    python download_models.py --asr               # ASR only
    python download_models.py --models-dir D:/ml/models   # custom output dir
"""

import argparse
import sys
from pathlib import Path

MODELS = {
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "base":   "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "asr":    "Qwen/Qwen3-ASR-1.7B",
}


def download(repo_id: str, local_dir: Path):
    from huggingface_hub import snapshot_download
    name = local_dir.name
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  ✓ {name}  already present — skipping")
        return
    print(f"  ↓ downloading {repo_id}  →  {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print(f"  ✓ {name}  done")


def main():
    parser = argparse.ArgumentParser(description="Download Qwen TTS/ASR models locally")
    parser.add_argument("--models-dir", default="./models",
                        help="Directory to store models (default: ./models)")
    parser.add_argument("--custom", action="store_true", help="Download CustomVoice model")
    parser.add_argument("--base",   action="store_true", help="Download Base model")
    parser.add_argument("--design", action="store_true", help="Download VoiceDesign model")
    parser.add_argument("--asr",    action="store_true", help="Download ASR model")
    args = parser.parse_args()

    # No flags → download everything
    selected = {k for k in MODELS if getattr(args, k, False)}
    if not selected:
        selected = set(MODELS.keys())

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {models_dir.resolve()}\n")

    try:
        for key in ("custom", "base", "design", "asr"):
            if key in selected:
                repo_id = MODELS[key]
                local_dir = models_dir / repo_id.split("/")[-1]
                download(repo_id, local_dir)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)

    print("\nAll done. Set MODELS_DIR if you used a non-default path:")
    print(f"  set MODELS_DIR={models_dir.resolve()}")


if __name__ == "__main__":
    main()
