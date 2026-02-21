"""Voice design functionality module."""
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
from audio_processor import AudioProcessor

from cache import AudioCache


class VoiceDesigner:
    """Handles voice design and custom voice creation."""

    def __init__(self, model_manager, audio_processor: AudioProcessor, output_dir: Path):
        self.model_manager = model_manager
        self.audio_processor = audio_processor
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._cache        = AudioCache(output_dir, "designed")
        self._scene_cache  = AudioCache(output_dir, "scene")

    def design_voice(self, json_config: Optional[str] = None, text_input: Optional[str] = None,
                     instructions: Optional[str] = None,
                     voice_name: Optional[str] = None) -> Optional[Path]:
        """Design a custom voice — exceptions propagate so the caller can log them."""
        if json_config:
            return self._design_from_json(json_config)
        elif text_input:
            return self._design_from_text(text_input, instructions or "", voice_name=voice_name)
        return None

    # ── JSON mode ─────────────────────────────────────────────────────────────

    def _design_from_json(self, json_config: str) -> Optional[Path]:
        """Design voice using JSON configuration."""
        try:
            cached = self._cache.get(json_config)
            if cached:
                return cached
            output_file = self._cache.path(json_config)

            config = json.loads(json_config)
            audio_files = []
            for scene in config.get("scenes", []):
                f = self._process_design_scene(scene)
                if f:
                    audio_files.append(f)

            if audio_files:
                if self.audio_processor.merge_audio_files(audio_files, output_file):
                    return self._cache.put(output_file)
            return None
        except Exception:
            return None

    # ── Text mode ─────────────────────────────────────────────────────────────

    def _design_from_text(self, text_input: str, instructions: str,
                          voice_name: Optional[str] = None) -> Optional[Path]:
        """Design voice from plain text with instructions."""
        cached = self._cache.get(text_input, instructions)
        if cached:
            return cached
        meta = {
            "type":         "design",
            "text":         text_input,
            "voice_name":   voice_name or "",
            "instructions": instructions,
            "model":        "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        }
        output_file = self._cache.path(text_input, instructions, meta=meta)

        load_result = self.model_manager.load_design_model()
        if "Error" in load_result:
            raise RuntimeError(load_result)

        model = self.model_manager.design_model
        if model is None:
            raise RuntimeError("VoiceDesign model is None after loading")

        wavs, sr = model.generate_voice_design(
            text=text_input,
            language=None,
            instruct=instructions or None,
        )
        sf.write(str(output_file), wavs[0], sr)
        return self._cache.put(output_file, meta=meta)

    # ── Scene helper ──────────────────────────────────────────────────────────

    def _process_design_scene(self, scene: Dict) -> Optional[Path]:
        """Process a single design scene."""
        try:
            text = scene.get("text", "")
            instructions = scene.get("instructions", "")
            if not text:
                return None

            cached = self._scene_cache.get(text, instructions)
            if cached:
                return cached
            output_file = self._scene_cache.path(text, instructions)

            model = self.model_manager.design_model
            if model is None:
                return None

            wavs, sr = model.generate_voice_design(
                text=text,
                language=None,
                instruct=instructions or None,
            )
            sf.write(str(output_file), wavs[0], sr)
            return self._scene_cache.put(output_file)
        except Exception:
            return None

    async def design_voice_async(self, text_input: str, instructions: str) -> Optional[Path]:
        """Async version of voice design."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._design_from_text, text_input, instructions
        )
