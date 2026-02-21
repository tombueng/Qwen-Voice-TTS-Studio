"""ASR (Automatic Speech Recognition) module."""
import asyncio
from pathlib import Path
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf


class ASRHandler:
    """Handles speech-to-text recognition."""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def transcribe_audio(self, audio_file: Path, language: Optional[str] = None,
                        use_forced_aligner: bool = False) -> Optional[str]:
        """Transcribe audio file to text."""
        try:
            # Load model
            result = self.model_manager.load_asr_model(
                "Qwen/Qwen3-ASR-1.7B",
                use_forced_aligner=use_forced_aligner
            )
            if "Error" in result:
                return None
            
            model = self.model_manager.asr_model
            if model is None:
                return None
            
            # Load audio
            audio, sr = sf.read(str(audio_file))
            
            # Prepare prompt
            prompt = {}
            if language:
                prompt["language"] = language
            
            # Transcribe
            result = model.generate(audio, sample_rate=sr, prompt=prompt if prompt else None)
            
            if isinstance(result, dict):
                return result.get("text", "")
            
            return str(result)
        except Exception:
            return None
    
    def transcribe_with_timestamps(self, audio_file: Path, language: Optional[str] = None) -> Optional[Dict]:
        """Transcribe audio with word-level timestamps."""
        try:
            result = self.model_manager.load_asr_model(
                "Qwen/Qwen3-ASR-1.7B",
                use_forced_aligner=True
            )
            if "Error" in result:
                return None
            
            model = self.model_manager.asr_model
            if model is None:
                return None
            
            audio, sr = sf.read(str(audio_file))
            
            prompt = {}
            if language:
                prompt["language"] = language
            
            result = model.generate(audio, sample_rate=sr, prompt=prompt if prompt else None)
            
            if isinstance(result, dict):
                return result
            
            return {"text": str(result)}
        except Exception:
            return None
    
    async def transcribe_async(self, audio_file: Path, language: Optional[str] = None) -> Optional[str]:
        """Async transcription."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.transcribe_audio,
            audio_file,
            language
        )
    
    def set_asr_model(self, model_id: str) -> str:
        """Change ASR model."""
        result = self.model_manager.load_asr_model(model_id)
        return result
