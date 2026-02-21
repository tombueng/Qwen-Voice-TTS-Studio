"""TTS generation and speech synthesis module."""
import asyncio
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import numpy as np

from cache import AudioCache


class TTSGenerator:
    """Handles text-to-speech generation."""

    def __init__(self, model_manager, output_dir: Path):
        self.model_manager = model_manager
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._cache = AudioCache(output_dir, "tts")

    def generate_tts(self, text: str, language: str, speaker: str,
                     output_file: Optional[Path] = None,
                     voice_name: Optional[str] = None) -> Optional[Path]:
        """Generate speech from text using CustomVoice model."""
        key_lang = language or "auto"
        key_spk  = speaker  or "default"
        meta = {
            "type":     "tts",
            "text":     text,
            "language": key_lang,
            "speaker":  voice_name or speaker or "default",
            "model":    "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        }

        write_meta: Optional[dict] = None
        if output_file is None:
            cached = self._cache.get(text, key_lang, key_spk)
            if cached:
                return cached
            output_file = self._cache.path(text, key_lang, key_spk, meta=meta)
            write_meta = meta   # sidecar written only for freshly created files

        # Load model — raises on error so caller can log the real reason
        load_result = self.model_manager.load_custom_model()
        if "Error" in load_result:
            raise RuntimeError(load_result)

        if self.model_manager.custom_model is None:
            raise RuntimeError("CustomVoice model is None after loading")

        wavs, sr = self.model_manager.custom_model.generate_custom_voice(
            text=text,
            language=None if (not language or language == "auto") else language,
            speaker=speaker or None,
            instruct=None,
        )

        sf.write(str(output_file), wavs[0], sr)
        return self._cache.put(output_file, meta=write_meta)

    def generate_voice_sample(self, language: str, speaker: str,
                              sample_text: Optional[str] = None) -> Optional[Path]:
        """Generate a sample of a voice."""
        if sample_text is None:
            sample_text = {
                "english": "Hello, this is a voice sample.",
                "chinese": "您好，这是一个语音样本。",
            }.get(language, "Hello, this is a voice sample.")
        return self.generate_tts(sample_text, language, speaker)

    async def generate_tts_async(self, text: str, language: str, speaker: str) -> Optional[Path]:
        """Async version of TTS generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.generate_tts, text, language, speaker, None
        )

    def batch_generate_tts(self, texts: List[str], language: str, speaker: str) -> List[Optional[Path]]:
        """Generate multiple TTS samples."""
        return [self.generate_tts(t, language, speaker) for t in texts]
