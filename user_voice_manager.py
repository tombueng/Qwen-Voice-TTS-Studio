"""Manages user-saved voices in user_voices.json / uservoices/ folder."""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class UserVoiceManager:
    """Saves and loads user-created voices (cloned or designed)."""

    def __init__(self, base_dir: Path):
        self.user_voices_dir  = base_dir / "uservoices"
        self.user_voices_json = base_dir / "user_voices.json"
        self.user_voices_dir.mkdir(parents=True, exist_ok=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> list:
        if self.user_voices_json.exists():
            try:
                return json.loads(
                    self.user_voices_json.read_text(encoding="utf-8")
                ).get("voices", [])
            except Exception:
                return []
        return []

    def _save(self, voices: list):
        self.user_voices_json.write_text(
            json.dumps({"voices": voices}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Add / update ──────────────────────────────────────────────────────────

    def add_design_voice(self, name: str, desc: str, instruct: str) -> str:
        name = name.strip()
        if not name:
            return "✗ Voice name is required"
        voices = self.load()
        entry = {"name": name, "desc": desc.strip(), "model": "design", "instruct": instruct.strip()}
        for i, v in enumerate(voices):
            if v.get("name") == name:
                voices[i] = entry
                self._save(voices)
                return f"✓ Updated '{name}'"
        voices.append(entry)
        self._save(voices)
        return f"✓ Saved '{name}'"

    def add_clone_voice(self, name: str, desc: str,
                        ref_audio_src: str, ref_text: str) -> str:
        name = name.strip()
        if not name:
            return "✗ Voice name is required"
        if not ref_audio_src:
            return "✗ Reference audio is required"

        # Copy audio file into uservoices/ with a stable name
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c for c in name if c.isalnum() or c in " _-").strip().replace(" ", "_")
        src  = Path(ref_audio_src)
        dest = self.user_voices_dir / f"{safe}_{ts}{src.suffix}"
        try:
            shutil.copy2(src, dest)
        except Exception as e:
            return f"✗ Could not copy audio: {e}"

        voices = self.load()
        entry = {
            "name":      name,
            "desc":      desc.strip(),
            "model":     "clone",
            "ref_audio": str(dest),
            "ref_text":  ref_text.strip(),
        }
        for i, v in enumerate(voices):
            if v.get("name") == name:
                voices[i] = entry
                self._save(voices)
                return f"✓ Updated '{name}'"
        voices.append(entry)
        self._save(voices)
        return f"✓ Saved '{name}'"

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_voice(self, name: str) -> Optional[dict]:
        for v in self.load():
            if v.get("name") == name:
                return v
        return None

    def get_choices(self) -> list:
        """Return [(label, value), ...] for a Gradio Dropdown."""
        return [(f"[My] {v['name']}", f"user:{v['name']}") for v in self.load() if v.get("name")]
