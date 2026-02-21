"""Voice cloning functionality module."""
import json
import base64
from pathlib import Path
from typing import Optional, Dict
import soundfile as sf
from audio_processor import AudioProcessor

from cache import AudioCache


class VoiceCloner:
    """Handles voice cloning operations."""

    def __init__(self, model_manager, audio_processor: AudioProcessor,
                 output_dir: Path, voice_inputs_dir: Path):
        self.model_manager = model_manager
        self.audio_processor = audio_processor
        self.output_dir = output_dir
        self.voice_inputs_dir = voice_inputs_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice_inputs_dir.mkdir(parents=True, exist_ok=True)
        self._cache        = AudioCache(output_dir,       "cloned")
        self._dialog_cache = AudioCache(output_dir,       "dialog")
        self._gen_cache    = AudioCache(output_dir,       "generated")
        self._decoded_cache = AudioCache(voice_inputs_dir, "decoded")

    def clone_voice(self, json_config: Optional[str] = None, audio_file: Optional[Path] = None,
                    ref_text: Optional[str] = None) -> Optional[Path]:
        """Clone a voice from reference audio — exceptions propagate so the caller can log them."""
        if json_config:
            return self._clone_from_json(json_config)
        elif audio_file and ref_text:
            return self._clone_from_audio(audio_file, ref_text)
        return None

    # ── JSON mode ─────────────────────────────────────────────────────────────

    def _clone_from_json(self, json_config: str) -> Optional[Path]:
        """Clone voice using JSON configuration (full dialog)."""
        try:
            cached = self._cache.get(json_config)
            if cached:
                return cached
            output_file = self._cache.path(json_config)

            config = json.loads(json_config)
            speaker_map = self._setup_json_speaker_map(config)

            audio_files = []
            for scene in config.get("scenes", []):
                for dialog in scene.get("dialog", []):
                    speaker_name = dialog.get("speaker", "")
                    if not speaker_name or speaker_name.lower() in ("direction", "instruction"):
                        continue
                    f = self._process_dialog_line(dialog, speaker_map)
                    if f:
                        audio_files.append(f)

            if audio_files:
                if self.audio_processor.merge_audio_files(audio_files, output_file):
                    return self._cache.put(output_file)
            return None
        except Exception:
            return None

    # ── Ref-text helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_ref_text(ref_text: Optional[str]) -> Optional[str]:
        """If ref_text looks like a file path and the file exists, load its content."""
        if not ref_text:
            return ref_text
        p = Path(ref_text.strip())
        if p.suffix.lower() == ".txt" and p.exists():
            return p.read_text(encoding="utf-8").strip() or None
        return ref_text

    # ── Single-audio mode ─────────────────────────────────────────────────────

    def _clone_from_audio(self, audio_file: Path, ref_text: str) -> Optional[Path]:
        """Clone voice from a single reference audio file."""
        cached = self._cache.get(str(audio_file.resolve()), ref_text)
        if cached:
            return cached
        output_file = self._cache.path(str(audio_file.resolve()), ref_text)

        load_result = self.model_manager.load_base_model()
        if "Error" in load_result:
            raise RuntimeError(load_result)

        model = self.model_manager.base_model
        if model is None:
            raise RuntimeError("Base model is None after loading")

        wavs, sr = model.generate_voice_clone(
            text=ref_text,
            language=None,
            ref_audio=str(audio_file),
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=not bool(ref_text),
        )
        sf.write(str(output_file), wavs[0], sr)
        return self._cache.put(output_file)

    # ── Speaker map ───────────────────────────────────────────────────────────

    def _setup_json_speaker_map(self, config: Dict) -> Dict:
        """Build speaker → {ref_audio, ref_text} map from JSON config."""
        speaker_map = {}
        for speaker in config.get("speaker", []):
            name = speaker.get("name", "")
            ref_audio_data = speaker.get("ref_audio")
            ref_text = speaker.get("ref_text", "")

            if isinstance(ref_audio_data, str) and ref_audio_data.startswith("data:"):
                audio_file = self._decode_base64_audio(ref_audio_data)
            else:
                audio_file = Path(ref_audio_data) if ref_audio_data else None

            speaker_map[name] = {"ref_audio": audio_file, "ref_text": self._resolve_ref_text(ref_text)}
        return speaker_map

    # ── Dialog line ───────────────────────────────────────────────────────────

    def _process_dialog_line(self, dialog: Dict, speaker_map: Dict) -> Optional[Path]:
        """Generate a single dialog line with voice cloning and connotation."""
        try:
            speaker_name = dialog.get("speaker")
            text = dialog.get("text", "")
            instruct = dialog.get("conotation", "").strip() or None

            if speaker_name not in speaker_map:
                return None

            speaker_info = speaker_map[speaker_name]
            ref_audio = speaker_info["ref_audio"]
            ref_text = speaker_info.get("ref_text", "") or None

            if not ref_audio:
                return None

            ref_audio_str = str(Path(ref_audio).resolve())

            cached = self._dialog_cache.get(
                speaker_name, text, ref_audio_str, ref_text or "", instruct or ""
            )
            if cached:
                return cached
            output_file = self._dialog_cache.path(
                speaker_name, text, ref_audio_str, ref_text or "", instruct or ""
            )

            model = self.model_manager.base_model
            if model is None:
                return None

            wavs, sr = model.generate_voice_clone(
                text=text,
                language=None,
                ref_audio=str(ref_audio),
                ref_text=ref_text if ref_text else None,
                x_vector_only_mode=not bool(ref_text),
                instruct=instruct,
            )
            sf.write(str(output_file), wavs[0], sr)
            return self._dialog_cache.put(output_file)
        except Exception:
            return None

    # ── Base64 decode ─────────────────────────────────────────────────────────

    def _decode_base64_audio(self, base64_data: str) -> Optional[Path]:
        """Decode a data-URI base64 audio blob to a .wav file."""
        try:
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]

            cached = self._decoded_cache.get(base64_data)
            if cached:
                return cached
            output_file = self._decoded_cache.path(base64_data)

            audio_bytes = base64.b64decode(base64_data)
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
            return self._decoded_cache.put(output_file)
        except Exception:
            return None

    # ── Generate with saved voice ─────────────────────────────────────────────

    def generate_with_cloned_voice(self, text: str, cloned_voice_config: Dict,
                                   json_mode: bool = False,
                                   voice_name: Optional[str] = None) -> Optional[Path]:
        """Generate speech using a previously saved cloned voice — exceptions propagate to the caller."""
        ref_audio_path = Path(cloned_voice_config.get("ref_audio", ""))
        ref_text = self._resolve_ref_text(cloned_voice_config.get("ref_text", ""))

        if not ref_audio_path.exists():
            raise RuntimeError(f"Reference audio not found: {ref_audio_path}")
        if not ref_text:
            raise RuntimeError("Reference text is required for voice cloning")

        cached = self._gen_cache.get(text, str(ref_audio_path.resolve()), ref_text)
        if cached:
            return cached

        meta = {
            "type":       "clone",
            "text":       text,
            "voice_name": voice_name or "",
            "ref_audio":  cloned_voice_config.get("ref_audio", ""),
            "ref_text":   ref_text,
            "model":      "Qwen3-TTS-12Hz-1.7B-Base",
        }
        output_file = self._gen_cache.path(text, str(ref_audio_path.resolve()), ref_text, meta=meta)

        load_result = self.model_manager.load_base_model()
        if "Error" in load_result:
            raise RuntimeError(load_result)

        model = self.model_manager.base_model
        if model is None:
            raise RuntimeError("Base model is None after loading")

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=None,
            ref_audio=str(ref_audio_path),
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=not bool(ref_text),
        )
        sf.write(str(output_file), wavs[0], sr)
        return self._gen_cache.put(output_file, meta=meta)
