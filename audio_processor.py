"""Audio processing utilities for TTS.

No external binaries required — soundfile 0.13+ reads MP3/WAV/FLAC natively
via libsndfile, and librosa uses soundfile as its backend on this installation.
ffmpeg, ffprobe, and sox are NOT needed.
"""
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


class AudioProcessor:
    """Handles audio file operations and processing."""

    @staticmethod
    def get_dialog_cache_hash(dialog: dict) -> str:
        """Generate hash for dialog cache."""
        dialog_str = json.dumps(dialog, sort_keys=True)
        return hashlib.md5(dialog_str.encode()).hexdigest()

    @staticmethod
    def merge_audio_files(audio_files: list, output_path: Path) -> bool:
        """Concatenate multiple audio files into one WAV.

        All input files must share the same sample rate (they will if they were
        all produced by the TTS generators, which write at the model's native rate).
        """
        try:
            if not audio_files:
                return False

            segments = []
            target_sr = None
            for f in audio_files:
                data, sr = sf.read(str(f), dtype="float32", always_2d=False)
                if target_sr is None:
                    target_sr = sr
                elif sr != target_sr:
                    # Resample to match the first file's rate
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                segments.append(data)

            combined = np.concatenate(segments)
            sf.write(str(output_path), combined, target_sr)
            return True
        except Exception:
            return False

    @staticmethod
    def convert_audio_format(input_path: Path, output_path: Path) -> bool:
        """Re-encode audio to WAV using soundfile (preserves original sample rate)."""
        try:
            data, sr = sf.read(str(input_path), dtype="float32", always_2d=False)
            sf.write(str(output_path), data, sr)
            return True
        except Exception:
            return False

    @staticmethod
    def get_audio_duration(audio_path: Path) -> Optional[float]:
        """Get duration of audio file in seconds."""
        try:
            data, samplerate = sf.read(str(audio_path))
            return len(data) / samplerate
        except Exception:
            return None

    @staticmethod
    def resample_audio(input_path: Path, output_path: Path,
                       target_sr: int = 24000,
                       max_seconds: Optional[float] = None) -> bool:
        """Load any audio format, resample to target_sr, write as mono WAV.

        Uses librosa (which delegates to soundfile on this installation — no ffmpeg needed).
        If max_seconds is given, the audio is clipped to that length after resampling.
        """
        try:
            import librosa
            y, sr = librosa.load(str(input_path), sr=None, mono=True)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            if max_seconds is not None:
                max_samples = int(max_seconds * target_sr)
                y = y[:max_samples]
            sf.write(str(output_path), y, target_sr)
            return True
        except Exception:
            return False
