"""Audio processing utilities for TTS.

No external binaries required — soundfile 0.13+ reads MP3/WAV/FLAC natively
via libsndfile, and librosa uses soundfile as its backend on this installation.
ffmpeg, ffprobe, and sox are NOT needed.
"""
import hashlib
import json
import shutil
import time as _time
from pathlib import Path
from typing import Optional

import math
import numpy as np
import soundfile as sf

from logger import log, dim, bold


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample *data* from *orig_sr* to *target_sr*.

    Uses scipy.signal.resample_poly (polyphase filter, fast, good quality).
    Falls back to librosa only if scipy is unavailable.
    """
    if orig_sr == target_sr:
        return data
    try:
        from scipy.signal import resample_poly
        g = math.gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        resampled = resample_poly(data, up, down)
        return resampled.astype(np.float32)
    except ImportError:
        import librosa
        return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)


class AudioProcessor:
    """Handles audio file operations and processing."""

    # ── Cache (class-level, set once at startup) ───────────────────────────────

    _cache_dir: Optional[Path] = None   # None = caching disabled

    @classmethod
    def set_cache_dir(cls, path: Path) -> None:
        """Enable audio-processing result caching into *path*.

        Call once at application startup, e.g. from QwenVoiceStudio._setup_paths.
        """
        cls._cache_dir = Path(path)
        cls._cache_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"  Audio proc cache: {dim(str(cls._cache_dir))}")

    @classmethod
    def _cache_slot(cls, prefix: str, *key_parts) -> Optional[Path]:
        """Return the cache file path for these key parts, or None if cache disabled."""
        if cls._cache_dir is None:
            return None
        raw = "|".join(str(p) for p in key_parts)
        h   = hashlib.md5(raw.encode()).hexdigest()[:16]
        return cls._cache_dir / f"{prefix}_{h}.wav"

    @staticmethod
    def _file_hash(path) -> str:
        """MD5 of full file content — stable, content-addressed cache key."""
        return hashlib.md5(Path(path).read_bytes()).hexdigest()

    # ── Misc helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def get_dialog_cache_hash(dialog: dict) -> str:
        """Generate hash for dialog cache."""
        dialog_str = json.dumps(dialog, sort_keys=True)
        return hashlib.md5(dialog_str.encode()).hexdigest()

    @staticmethod
    def make_silence(duration_s: float, sample_rate: int = 24000) -> np.ndarray:
        """Return a float32 silence array of the given duration (clamped to ≥ 0)."""
        return np.zeros(max(0, int(duration_s * sample_rate)), dtype=np.float32)

    @staticmethod
    def apply_fade(data: np.ndarray, sr: int,
                   fade_in_s: float = 0.0,
                   fade_out_s: float = 0.0) -> np.ndarray:
        """Apply linear fade-in and/or fade-out ramps to *data* in-place.

        Values <= 0 skip the corresponding ramp.  If the combined ramps would
        exceed the array length, each is clamped to half the total length so
        they never overlap.  Mutates and returns *data*.
        """
        n = len(data)
        if n == 0:
            return data
        max_fade   = n // 2
        fi_samples = min(int(fade_in_s  * sr), max_fade)
        fo_samples = min(int(fade_out_s * sr), max_fade)
        if fi_samples > 0:
            data[:fi_samples] *= np.linspace(0.0, 1.0, fi_samples, dtype=np.float32)
        if fo_samples > 0:
            data[n - fo_samples:] *= np.linspace(1.0, 0.0, fo_samples, dtype=np.float32)
        return data

    # ── Merge ─────────────────────────────────────────────────────────────────

    @staticmethod
    def merge_audio_files(audio_files: list, output_path: Path) -> bool:
        """Concatenate audio files/arrays into one WAV.

        Items may be Path/str (loaded from disk) or np.ndarray (float32 silence
        arrays at the same sample rate as the surrounding audio).
        """
        try:
            if not audio_files:
                return False

            n_files  = sum(1 for f in audio_files if not isinstance(f, np.ndarray))
            n_arrays = sum(1 for f in audio_files if isinstance(f, np.ndarray))
            log.info(
                f"  Merge  {bold(str(len(audio_files)))} segment(s)"
                f"  ({n_files} file(s)  {n_arrays} array(s))"
                f"  →  {bold(Path(output_path).name)}"
            )
            t0 = _time.time()

            def _to_mono_1d(arr: np.ndarray) -> np.ndarray:
                """Collapse any stereo/multi-channel array to mono 1-D float32."""
                if arr.ndim == 2:
                    arr = arr.mean(axis=1)
                return arr.astype(np.float32)

            segments  = []
            target_sr = None
            for f in audio_files:
                if isinstance(f, np.ndarray):
                    arr = _to_mono_1d(f)
                    arr_dur = len(arr) / (target_sr or 24000)
                    if arr_dur >= 0.01:                          # skip zero-length pads
                        log.debug(f"    silence  {arr_dur:.3f}s  ({len(arr)} samples)")
                    segments.append(arr)
                    continue

                data, sr = sf.read(str(f), dtype="float32", always_2d=False)
                data = _to_mono_1d(data)
                dur = len(data) / sr
                try:
                    size_kb = Path(f).stat().st_size // 1024
                except OSError:
                    size_kb = 0
                log.debug(f"    read  {bold(Path(f).name)}  {dur:.2f}s  {sr} Hz  {size_kb} KB")

                if target_sr is None:
                    target_sr = sr
                elif sr != target_sr:
                    log.warning(
                        f"    Resample  {Path(f).name}  {sr} Hz → {target_sr} Hz  ({dur:.2f}s)…"
                    )
                    t_rs = _time.time()
                    data = _resample(data, sr, target_sr)
                    log.ok(f"    Resample done  ({_time.time() - t_rs:.1f}s)")

                segments.append(data)

            if target_sr is None:
                target_sr = 24000   # silence-only edge case

            combined  = np.concatenate(segments)
            total_dur = len(combined) / target_sr
            sf.write(str(output_path), combined, target_sr)
            try:
                size_kb = Path(output_path).stat().st_size // 1024
            except OSError:
                size_kb = 0
            log.ok(
                f"  Merged  →  {bold(Path(output_path).name)}"
                f"  {total_dur:.2f}s  {target_sr} Hz  {size_kb} KB"
                f"  ({_time.time() - t0:.1f}s)"
            )
            return True
        except Exception as exc:
            log.error(f"  merge_audio_files failed: {exc}")
            return False

    # ── Format conversion ─────────────────────────────────────────────────────

    @staticmethod
    def convert_audio_format(input_path: Path, output_path: Path) -> bool:
        """Re-encode audio to WAV using soundfile (preserves original sample rate)."""
        try:
            data, sr = sf.read(str(input_path), dtype="float32", always_2d=False)
            dur = len(data) / sr
            log.info(
                f"  Convert  {bold(Path(input_path).name)} → {Path(output_path).name}"
                f"  {dur:.2f}s  {sr} Hz"
            )
            sf.write(str(output_path), data, sr)
            log.ok(f"  Converted  →  {Path(output_path).name}")
            return True
        except Exception as exc:
            log.error(f"  convert_audio_format failed: {exc}")
            return False

    # ── Time-stretch ──────────────────────────────────────────────────────────
    # Uses Spotify's pedalboard library (native DSP, transient-preserving,
    # no phase-vocoder artifacts).  Falls back to librosa if unavailable.

    @staticmethod
    def _do_stretch(y: np.ndarray, sr: int, speed: float) -> np.ndarray:
        """Pitch-preserving time-stretch via pedalboard (falls back to librosa).

        speed > 1.0 = faster / shorter,  speed < 1.0 = slower / longer.
        pedalboard.time_stretch expects stretch_factor = output_len / input_len
        = 1 / speed.
        """
        try:
            import pedalboard
            # pedalboard expects (channels, samples) float32
            mono = y if y.ndim == 1 else y.mean(axis=1)
            stretched = pedalboard.time_stretch(
                mono.reshape(1, -1).astype(np.float32),
                samplerate=float(sr),
                stretch_factor=1.0 / speed,
            )
            return stretched.squeeze().astype(np.float32)
        except ImportError:
            import librosa
            return librosa.effects.time_stretch(y, rate=float(speed))

    @classmethod
    def time_stretch_file(cls, audio_path: Path, speed: float) -> bool:
        """Apply pitch-preserving time stretch to a WAV file in-place.

        speed > 1.0 = faster (shorter output),  speed < 1.0 = slower (longer output).
        Returns True on success (or when speed ≈ 1.0 and nothing needs doing).
        """
        if abs(speed - 1.0) < 0.001:
            return True
        try:
            slot = cls._cache_slot("stretch2", cls._file_hash(audio_path), f"spd:{speed:.6f}")
            if slot and slot.exists():
                log.ok(f"  cache hit  [stretch]  {dim(slot.name)}")
                shutil.copy2(str(slot), str(audio_path))
                return True

            y, sr   = sf.read(str(audio_path), dtype="float32", always_2d=False)
            dur_in  = len(y) / sr
            dur_out = dur_in / speed
            log.info(
                f"  Time-stretch  {bold(Path(audio_path).name)}"
                f"  ×{speed:.3f}  {dur_in:.2f}s → {dur_out:.2f}s"
            )
            t0 = _time.time()
            stretched = cls._do_stretch(y, sr, speed)
            sf.write(str(audio_path), stretched, sr)
            log.ok(f"  Time-stretch done  ({_time.time() - t0:.1f}s)")

            if slot:
                shutil.copy2(str(audio_path), str(slot))
                log.debug(f"  cached  [stretch]  {dim(slot.name)}")
            return True
        except Exception as exc:
            log.error(f"  time_stretch_file failed: {exc}")
            return False

    @classmethod
    def time_stretch_to_file(cls, src: Path, dst: Path, speed: float) -> bool:
        """Load src, apply pitch-preserving time stretch, write result to dst.

        src is never modified — dst receives the (possibly) stretched audio.
        Returns True on success.
        """
        try:
            slot = cls._cache_slot("stretch2", cls._file_hash(src), f"spd:{speed:.6f}")
            if slot and slot.exists():
                log.ok(f"  cache hit  [stretch]  {dim(slot.name)}")
                shutil.copy2(str(slot), str(dst))
                return True

            y, sr  = sf.read(str(src), dtype="float32", always_2d=False)
            dur_in = len(y) / sr
            if abs(speed - 1.0) >= 0.001:
                dur_out = dur_in / speed
                log.info(
                    f"  Time-stretch  {bold(Path(src).name)} → {Path(dst).name}"
                    f"  ×{speed:.3f}  {dur_in:.2f}s → {dur_out:.2f}s"
                )
                t0 = _time.time()
                y  = cls._do_stretch(y, sr, speed)
                log.ok(f"  Time-stretch done  ({_time.time() - t0:.1f}s)")
            else:
                log.info(
                    f"  Copy (speed ≈ 1.0)  {bold(Path(src).name)} → {Path(dst).name}"
                    f"  {dur_in:.2f}s  {sr} Hz"
                )
            sf.write(str(dst), y, sr)

            if slot:
                shutil.copy2(str(dst), str(slot))
                log.debug(f"  cached  [stretch]  {dim(slot.name)}")
            return True
        except Exception as exc:
            log.error(f"  time_stretch_to_file failed: {exc}")
            return False

    @classmethod
    def time_stretch_array(cls, audio_path: Path, speed: float):
        """Load audio, apply pitch-preserving time stretch, return (ndarray, sample_rate).

        Used when the caller needs a numpy array instead of a file (e.g. in-memory
        script assembly).  Returns (None, None) on error.
        """
        try:
            slot = cls._cache_slot("stretch2", cls._file_hash(audio_path), f"spd:{speed:.6f}")
            if slot and slot.exists():
                log.ok(f"  cache hit  [stretch]  {dim(slot.name)}")
                y_cached, sr_cached = sf.read(str(slot), dtype="float32", always_2d=False)
                return y_cached, sr_cached

            y, sr  = sf.read(str(audio_path), dtype="float32", always_2d=False)
            dur_in = len(y) / sr
            if abs(speed - 1.0) >= 0.001:
                dur_out = dur_in / speed
                log.info(
                    f"  Time-stretch array  {bold(Path(audio_path).name)}"
                    f"  ×{speed:.3f}  {dur_in:.2f}s → {dur_out:.2f}s"
                )
                t0 = _time.time()
                y  = cls._do_stretch(y, sr, speed)
                log.ok(f"  Time-stretch done  ({_time.time() - t0:.1f}s)")

            if slot:
                sf.write(str(slot), y, sr)
                log.debug(f"  cached  [stretch]  {dim(slot.name)}")
            return y, sr
        except Exception as exc:
            log.error(f"  time_stretch_array failed: {exc}")
            return None, None

    # ── Duration / measurement ────────────────────────────────────────────────

    @staticmethod
    def get_audio_duration(audio_path: Path) -> Optional[float]:
        """Get duration of audio file in seconds."""
        try:
            data, samplerate = sf.read(str(audio_path))
            return len(data) / samplerate
        except Exception:
            return None

    @staticmethod
    def measure_segments_duration(segments: list, default_sr: int = 24000) -> float:
        """Return total duration in seconds of a mixed list of audio segments (Path/str/ndarray).

        ndarray items are assumed to share the sample rate of the last seen file segment.
        """
        total  = 0.0
        sr_ref = None
        for seg in segments:
            if isinstance(seg, np.ndarray):
                total += len(seg) / (sr_ref or default_sr)
            else:
                try:
                    data, sr = sf.read(str(seg), dtype="float32", always_2d=False)
                    sr_ref = sr
                    total += len(data) / sr
                except Exception:
                    pass
        return total

    # ── Mix ───────────────────────────────────────────────────────────────────

    @classmethod
    def mix_with_background(cls, foreground_path: Path, background_path: Path,
                            bg_volume: float, output_path: Path) -> bool:
        """Overlay *background_path* (looped/trimmed) under *foreground_path*.

        The background is scaled by *bg_volume* (0.0–1.0 typical) before
        addition.  The result is clamped to [-1, 1] and written to *output_path*
        as a mono WAV at the foreground's sample rate.

        Results are cached by content hash of both input files + volume.
        Returns True on success.
        """
        try:
            slot = cls._cache_slot(
                "mix",
                cls._file_hash(foreground_path),
                cls._file_hash(background_path),
                f"vol:{bg_volume:.6f}",
            )
            if slot and slot.exists():
                log.ok(f"  cache hit  [mix]  {dim(slot.name)}")
                shutil.copy2(str(slot), str(output_path))
                return True

            t0 = _time.time()
            fg,     fg_sr = sf.read(str(foreground_path), dtype="float32", always_2d=False)
            bg_raw, bg_sr = sf.read(str(background_path), dtype="float32", always_2d=False)
            fg_dur = len(fg)     / fg_sr
            bg_dur = len(bg_raw) / bg_sr
            log.info(
                f"  Mix  fg={bold(Path(foreground_path).name)}  {fg_dur:.2f}s  {fg_sr} Hz"
            )
            log.info(
                f"       bg={bold(Path(background_path).name)}  {bg_dur:.2f}s  {bg_sr} Hz"
                f"  vol={bg_volume:.2f}"
            )

            # Resample background to foreground rate if needed.
            # This step is cached independently (keyed on bg content + SR pair)
            # so it only runs once per unique background file regardless of fg.
            if bg_sr != fg_sr:
                bg_slot = cls._cache_slot(
                    "bg_rs",
                    cls._file_hash(background_path),
                    f"{bg_sr}to{fg_sr}",
                )
                if bg_slot and bg_slot.exists():
                    log.ok(
                        f"    Resample cache hit  {bg_sr} Hz → {fg_sr} Hz"
                        f"  {dim(bg_slot.name)}"
                    )
                    bg_raw, _ = sf.read(str(bg_slot), dtype="float32", always_2d=False)
                else:
                    log.warning(
                        f"    Resampling background  {bg_sr} Hz → {fg_sr} Hz"
                        f"  ({bg_dur:.2f}s of audio — may take a moment)…"
                    )
                    t_rs = _time.time()
                    bg_raw = _resample(bg_raw, bg_sr, fg_sr)
                    log.ok(f"    Resample done  ({_time.time() - t_rs:.1f}s)")
                    if bg_slot:
                        sf.write(str(bg_slot), bg_raw, fg_sr)
                        log.debug(f"    cached  [bg_rs]  {dim(bg_slot.name)}")

            # Collapse to mono
            if fg.ndim == 2:
                log.debug(f"    fg stereo → mono  ({fg.shape[1]} ch)")
                fg = fg.mean(axis=1)
            if bg_raw.ndim == 2:
                log.debug(f"    bg stereo → mono  ({bg_raw.shape[1]} ch)")
                bg_raw = bg_raw.mean(axis=1)

            if len(bg_raw) == 0:
                log.error(f"    Background has zero samples — aborting mix")
                return False

            # Loop background to cover foreground length
            n = len(fg)
            if len(bg_raw) < n:
                repeats = (n // len(bg_raw)) + 1
                log.info(
                    f"    Loop background ×{repeats}"
                    f"  ({bg_dur:.2f}s → covers {fg_dur:.2f}s foreground)"
                )
                bg_raw = np.tile(bg_raw, repeats)

            bg    = bg_raw[:n] * float(bg_volume)
            mixed = np.clip(fg + bg, -1.0, 1.0)
            sf.write(str(output_path), mixed, fg_sr)

            try:
                size_kb = Path(output_path).stat().st_size // 1024
            except OSError:
                size_kb = 0
            log.ok(
                f"    Mixed  →  {bold(Path(output_path).name)}"
                f"  {fg_dur:.2f}s  {fg_sr} Hz  {size_kb} KB"
                f"  ({_time.time() - t0:.1f}s total)"
            )

            if slot:
                shutil.copy2(str(output_path), str(slot))
                log.debug(f"    cached  [mix]  {dim(slot.name)}")
            return True
        except Exception as exc:
            log.error(f"    mix_with_background failed: {exc}")
            return False

    # ── Resample ──────────────────────────────────────────────────────────────

    @classmethod
    def resample_audio(cls, input_path: Path, output_path: Path,
                       target_sr: int = 24000,
                       max_seconds: Optional[float] = None) -> bool:
        """Load any audio format, resample to target_sr, write as mono WAV.

        Uses librosa (which delegates to soundfile on this installation — no ffmpeg needed).
        If max_seconds is given, the audio is clipped to that length after resampling.
        Results are cached by content hash of the input file + params.
        """
        try:
            slot = cls._cache_slot(
                "resample",
                cls._file_hash(input_path),
                f"sr:{target_sr}",
                f"max:{max_seconds}",
            )
            if slot and slot.exists():
                log.ok(f"  cache hit  [resample]  {dim(slot.name)}")
                shutil.copy2(str(slot), str(output_path))
                return True

            y, sr  = sf.read(str(input_path), dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            dur_in = len(y) / sr
            log.info(
                f"  Resample  {bold(Path(input_path).name)}"
                f"  {dur_in:.2f}s  {sr} Hz → {target_sr} Hz"
            )
            t0 = _time.time()
            if sr != target_sr:
                y = _resample(y, sr, target_sr)
                log.ok(f"  Resample done  ({_time.time() - t0:.1f}s)")

            if max_seconds is not None:
                max_samples = int(max_seconds * target_sr)
                if len(y) > max_samples:
                    log.info(f"    Clip to {max_seconds:.2f}s")
                    y = y[:max_samples]

            sf.write(str(output_path), y, target_sr)
            dur_out = len(y) / target_sr
            try:
                size_kb = Path(output_path).stat().st_size // 1024
            except OSError:
                size_kb = 0
            log.ok(
                f"  Resampled  →  {bold(Path(output_path).name)}"
                f"  {dur_out:.2f}s  {target_sr} Hz  {size_kb} KB"
            )

            if slot:
                shutil.copy2(str(output_path), str(slot))
                log.debug(f"  cached  [resample]  {dim(slot.name)}")
            return True
        except Exception as exc:
            log.error(f"  resample_audio failed: {exc}")
            return False
