"""Voice profile management for cloned and designed voices."""
import json
from pathlib import Path
from typing import Dict, List, Optional


class VoiceManager:
    """Manages saving and loading of voice profiles."""

    def __init__(self, designed_voices_dir: Path):
        self.designed_voices_dir = designed_voices_dir
        self.designed_voices_index = designed_voices_dir / "voices.json"
        self.designed_voices_dir.mkdir(parents=True, exist_ok=True)

    # ==================== Designed Voices ====================
    
    def load_designed_voices(self) -> Dict:
        """Load designed voices from index."""
        if self.designed_voices_index.exists():
            try:
                with open(self.designed_voices_index, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save_designed_voices(self, voices_data: Dict) -> bool:
        """Save designed voices to index."""
        try:
            with open(self.designed_voices_index, 'w', encoding='utf-8') as f:
                json.dump(voices_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def add_designed_voice(self, voice_name: str, voice_data: Dict) -> bool:
        """Add a new designed voice."""
        try:
            voices = self.load_designed_voices()
            voices[voice_name] = voice_data
            return self.save_designed_voices(voices)
        except Exception:
            return False
    
    def delete_designed_voice(self, voice_name: str) -> bool:
        """Delete a designed voice."""
        try:
            voices = self.load_designed_voices()
            if voice_name in voices:
                del voices[voice_name]
                return self.save_designed_voices(voices)
            return False
        except Exception:
            return False
    
    def get_designed_voices_list(self) -> List[str]:
        """Get list of all designed voice names."""
        voices = self.load_designed_voices()
        return list(voices.keys())
    
    def get_designed_voice(self, voice_name: str) -> Optional[Dict]:
        """Get a specific designed voice."""
        voices = self.load_designed_voices()
        return voices.get(voice_name)
    
