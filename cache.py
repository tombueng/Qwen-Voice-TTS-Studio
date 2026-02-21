"""Audio generation cache — shared by all generator modules.

Provides a single AudioCache class that:
  - Derives stable, content-addressed filenames via MD5
  - When metadata is supplied, builds a human-readable filename:
        {prefix}_{speaker}_{YYYYMMDD-HHMMSS}_{text_slug}_{hash16}.wav
    The hash still guarantees uniqueness and enables cache-hit detection via
    a glob search (needed because the timestamp part varies).
  - Writes a .json sidecar file alongside each new audio file, containing all
    parameters needed to reproduce the generation.
  - Checks for existing files before inference (cache hit).
  - Logs every hit/miss with the fancy project logger.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from logger import log, dim, bold


class AudioCache:
    """Content-addressed file cache for generated audio."""

    def __init__(self, output_dir: Path, prefix: str):
        """
        Args:
            output_dir: Directory where cached .wav files live.
            prefix:     Short label prepended to filenames, e.g. "tts", "dialog".
        """
        self.output_dir = output_dir
        self.prefix = prefix

    # ── Public API ────────────────────────────────────────────────────────────

    def path(self, *key_parts, meta: dict = None) -> Path:
        """Return the canonical cache path for the given key parts (no I/O).

        If *meta* is provided the filename is human-readable:
            {prefix}_{speaker}_{YYYYMMDD-HHMMSS}_{text_slug}_{hash16}.wav
        Otherwise the legacy format is used:
            {prefix}_{hash16}.wav
        """
        hash16 = self._md5(*key_parts)
        if meta:
            name = self._build_slug(meta, hash16)
        else:
            name = f"{self.prefix}_{hash16}"
        return self.output_dir / f"{name}.wav"

    def get(self, *key_parts) -> Path | None:
        """Return cached path if it exists, else None.  Logs the outcome.

        Searches for both the legacy hash-only filename AND any readable
        filename that ends with _{hash16}.wav (necessary because the readable
        filename embeds a generation timestamp that we cannot reconstruct).
        """
        hash16 = self._md5(*key_parts)

        # 1. Legacy format: prefix_hash16.wav
        legacy = self.output_dir / f"{self.prefix}_{hash16}.wav"
        if legacy.exists():
            log.ok(f"cache hit  [{self.prefix}]  {dim(legacy.name)}")
            return legacy

        # 2. Readable format: prefix_*_{hash16}.wav
        matches = sorted(self.output_dir.glob(f"{self.prefix}_*_{hash16}.wav"))
        if matches:
            p = matches[-1]  # newest if somehow duplicated
            log.ok(f"cache hit  [{self.prefix}]  {dim(p.name)}")
            return p

        log.debug(f"cache miss [{self.prefix}]  hash={hash16}")
        return None

    def put(self, path: Path, meta: dict = None) -> Path:
        """Record that *path* has just been written; write sidecar if meta given.

        The sidecar is a .json file alongside the audio with all generation
        parameters plus 'generated_at' and 'hash' fields for reproducibility.

        Returns *path* unchanged.
        """
        if meta is not None:
            self._write_sidecar(path, meta)
        log.debug(f"cache save [{self.prefix}]  {dim(path.name)}")
        return path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_slug(self, meta: dict, hash16: str) -> str:
        """Build a human-readable filename stem from meta + hash."""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = [self.prefix]

        speaker = (
            meta.get("speaker")
            or meta.get("voice_name")
            or ""
        )
        if speaker:
            parts.append(self._sanitize(speaker, 24))

        parts.append(ts)

        text = meta.get("text", "")
        if text:
            parts.append(self._text_slug(text, 32))

        parts.append(hash16)
        return "_".join(parts)

    @staticmethod
    def _sanitize(s: str, max_len: int) -> str:
        """Replace non-alphanumeric chars with underscores, truncate."""
        s = re.sub(r"[^\w]", "_", s.strip())
        s = re.sub(r"_+", "_", s).strip("_")
        return s[:max_len].rstrip("_")

    @staticmethod
    def _text_slug(text: str, max_len: int) -> str:
        """Turn the first few words of *text* into a filename-safe slug."""
        # Strip leading/trailing whitespace, take first 2× max_len chars
        text = text.strip()[: max_len * 2]
        # Remove anything that's not a word character or whitespace
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse whitespace to hyphens
        text = re.sub(r"\s+", "-", text.strip())
        return text[:max_len].rstrip("-")

    def _write_sidecar(self, audio_path: Path, meta: dict) -> None:
        """Write a .json sidecar next to *audio_path*."""
        # Extract the hash from the filename (last _-separated segment before .wav)
        stem = audio_path.stem  # e.g. "tts_Aiden_20260221-143022_Hello_abc123"
        hash_part = stem.rsplit("_", 1)[-1]

        data = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "hash": hash_part,
            **meta,
        }
        sidecar = audio_path.with_suffix(".json")
        try:
            sidecar.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning(f"Could not write sidecar {sidecar.name}: {exc}")

    @staticmethod
    def _md5(*parts) -> str:
        combined = "|".join(str(p) for p in parts)
        return hashlib.md5(combined.encode()).hexdigest()[:16]
