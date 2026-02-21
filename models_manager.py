"""Model management for Qwen TTS models."""
import torch
from pathlib import Path
from transformers.utils.import_utils import is_flash_attn_2_available


class ModelManager:
    """Manages loading and caching of TTS models."""
    
    def __init__(self, models_dir: Path, device: str, dtype: torch.dtype):
        self.models_dir = models_dir
        self.device = device
        self.dtype = dtype
        self.custom_model = None
        self.base_model = None
        self.design_model = None
        self.asr_model = None
        self.asr_model_id = None
    
    def _get_attn_implementation(self) -> str:
        """Get appropriate attention implementation."""
        if self.device.startswith("cuda") and is_flash_attn_2_available():
            return "flash_attention_2"
        if self.device.startswith("cuda"):
            return "sdpa"
        return "eager"
    
    def load_custom_model(self):
        """Load CustomVoice model."""
        if self.custom_model is None:
            try:
                from qwen_tts import Qwen3TTSModel
                local_model_path = self.models_dir / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
                model_path = str(local_model_path) if local_model_path.exists() else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
                
                self.custom_model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map=self.device,
                    dtype=self.dtype,
                    attn_implementation=self._get_attn_implementation(),
                )
                return "✓ CustomVoice model loaded successfully"
            except Exception as e:
                return f"✗ Error loading CustomVoice model: {str(e)}"
        return "✓ CustomVoice model already loaded"
    
    def load_base_model(self):
        """Load Base voice cloning model."""
        if self.base_model is None:
            try:
                from qwen_tts import Qwen3TTSModel
                local_model_path = self.models_dir / "Qwen3-TTS-12Hz-1.7B-Base"
                model_path = str(local_model_path) if local_model_path.exists() else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
                
                self.base_model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map=self.device,
                    dtype=self.dtype,
                    attn_implementation=self._get_attn_implementation(),
                )
                return "✓ Base model loaded successfully"
            except Exception as e:
                return f"✗ Error loading Base model: {str(e)}"
        return "✓ Base model already loaded"
    
    def load_design_model(self):
        """Load VoiceDesign model."""
        if self.design_model is None:
            try:
                from qwen_tts import Qwen3TTSModel
                local_model_path = self.models_dir / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                model_path = str(local_model_path) if local_model_path.exists() else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                
                self.design_model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map=self.device,
                    dtype=self.dtype,
                    attn_implementation=self._get_attn_implementation(),
                )
                return "✓ VoiceDesign model loaded successfully"
            except Exception as e:
                return f"✗ Error loading VoiceDesign model: {str(e)}"
        return "✓ VoiceDesign model already loaded"
    
    def load_asr_model(self, model_choice: str, use_forced_aligner: bool = False):
        """Load ASR model."""
        try:
            from qwen_asr import Qwen3ASRModel

            model_choice = (model_choice or "Qwen/Qwen3-ASR-1.7B").strip()
            local_dir_name = model_choice.split("/")[-1]
            local_model_path = self.models_dir / local_dir_name
            model_path = str(local_model_path) if local_model_path.exists() else model_choice

            aligner_id = None
            aligner_kwargs = None
            if use_forced_aligner:
                aligner_local = self.models_dir / "Qwen3-ForcedAligner-0.6B"
                aligner_id = str(aligner_local) if aligner_local.exists() else "Qwen/Qwen3-ForcedAligner-0.6B"
                aligner_kwargs = dict(
                    dtype=self.dtype,
                    device_map=self.device,
                    attn_implementation=self._get_attn_implementation(),
                )

            desired_id = f"{model_path}|aligner={bool(use_forced_aligner)}|device={self.device}|dtype={str(self.dtype)}"
            if self.asr_model is not None and self.asr_model_id == desired_id:
                return "✓ ASR model already loaded"

            self.asr_model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=self.dtype,
                device_map=self.device,
                attn_implementation=self._get_attn_implementation(),
                max_inference_batch_size=8,
                max_new_tokens=256,
                forced_aligner=aligner_id,
                forced_aligner_kwargs=aligner_kwargs,
            )
            self.asr_model_id = desired_id
            return "✓ ASR model loaded successfully"
        except ImportError:
            return "✗ qwen-asr is not installed. Please run setup again or install it in the venv: pip install -U qwen-asr"
        except Exception as e:
            return f"✗ Error loading ASR model: {str(e)}"
    
    def get_supported_speakers(self) -> list | None:
        """Return supported speaker names from the loaded CustomVoice model, or None if not yet loaded."""
        if self.custom_model is not None:
            return self.custom_model.get_supported_speakers()
        return None

    def get_supported_languages(self) -> list | None:
        """Return supported language names from the loaded CustomVoice model, or None if not yet loaded."""
        if self.custom_model is not None:
            return self.custom_model.get_supported_languages()
        return None

    def unload_all(self):
        """Unload all models."""
        self.custom_model = None
        self.base_model = None
        self.design_model = None
        self.asr_model = None
        self.asr_model_id = None
