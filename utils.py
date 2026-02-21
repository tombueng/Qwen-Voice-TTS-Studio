"""Utility functions for the application."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib


def parse_json_config(config_str: str) -> Optional[Dict]:
    """Parse and validate JSON configuration."""
    try:
        return json.loads(config_str)
    except json.JSONDecodeError:
        return None


def load_json_file(file_path: Path) -> Optional[Dict]:
    """Load JSON from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def save_json_file(file_path: Path, data: Dict) -> bool:
    """Save JSON to file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def generate_hash(data: str, hash_type: str = "md5") -> str:
    """Generate hash of a string."""
    if hash_type == "md5":
        return hashlib.md5(data.encode()).hexdigest()
    elif hash_type == "sha256":
        return hashlib.sha256(data.encode()).hexdigest()
    else:
        return hashlib.md5(data.encode()).hexdigest()


def validate_audio_file(file_path: Path) -> bool:
    """Validate if file is a valid audio file."""
    if not file_path.exists():
        return False
    
    valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    return file_path.suffix.lower() in valid_extensions


def validate_text_input(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
    """Validate text input."""
    if not isinstance(text, str):
        return False
    
    text_len = len(text.strip())
    return min_length <= text_len <= max_length


def parse_dialog_script(script: str) -> List[Dict]:
    """Parse dialog script into scenes."""
    try:
        lines = script.strip().split('\n')
        scenes = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse format: "Speaker: Text"
            if ':' in line:
                speaker, text = line.split(':', 1)
                scenes.append({
                    "speaker": speaker.strip(),
                    "text": text.strip()
                })
        
        return scenes
    except Exception:
        return []


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def validate_voice_name(name: str) -> bool:
    """Validate voice name."""
    if not isinstance(name, str):
        return False
    
    name = name.strip()
    if not name or len(name) > 100:
        return False
    
    import re
    # Allow alphanumeric, underscore, hyphen, and spaces
    pattern = r'^[a-zA-Z0-9_\-\s]+$'
    return bool(re.match(pattern, name))


def ensure_directory(dir_path: Path) -> bool:
    """Ensure directory exists."""
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def get_file_list(directory: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """Get list of files in directory with optional extension filter."""
    if not directory.exists():
        return []
    
    files = list(directory.iterdir())
    
    if extensions:
        extensions_lower = [ext.lower() for ext in extensions]
        files = [f for f in files if f.suffix.lower() in extensions_lower]
    
    return sorted(files)


def get_config_from_env() -> Dict[str, Any]:
    """Get application configuration from environment variables."""
    import os
    
    return {
        "device": os.getenv("DEVICE", "cuda" if os.getenv("FORCE_CPU") != "1" else "cpu"),
        "dtype": os.getenv("DTYPE", "bfloat16"),
        "models_dir": os.getenv("MODELS_DIR", "./models"),
        "outputs_dir": os.getenv("OUTPUTS_DIR", "./outputs"),
        "voice_inputs_dir": os.getenv("VOICE_INPUTS_DIR", "./voiceinputs"),
        "cloned_voices_dir": os.getenv("CLONED_VOICES_DIR", "./cloned_voices"),
        "designed_voices_dir": os.getenv("DESIGNED_VOICES_DIR", "./designed_voices"),
        "voice_samples_dir": os.getenv("VOICE_SAMPLES_DIR", "./voicesamples"),
        "server_port": int(os.getenv("PORT", 7860)),
        "server_name": os.getenv("LISTEN_IP", "127.0.0.1"),
        "server_share": os.getenv("SHARE", "false").lower() == "true",
    }
