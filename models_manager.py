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
        self.stableaudio_model = None
        self.stableaudio_sample_rate = None
        self.stableaudio_sample_size = None
    
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

    def load_stableaudio_model(self):
        """Load Stable Audio model via diffusers StableAudioPipeline."""
        if self.stableaudio_model is not None:
            return "✓ Stable Audio model already loaded"

        # flash_attn_2_cuda is a compiled CUDA extension; on CPU it raises
        # OSError ("DLL load failed") which diffusers does not catch (only
        # ImportError is caught).  Pre-stub it so the import chain succeeds.
        import sys, types  # noqa: E401
        if "flash_attn_2_cuda" not in sys.modules:
            try:
                import flash_attn_2_cuda  # noqa: F401
            except Exception:
                sys.modules["flash_attn_2_cuda"] = types.ModuleType("flash_attn_2_cuda")

        # Scope the ImportError check tightly so that any ImportError raised
        # by lazy imports inside from_pretrained is reported verbatim.
        try:
            from diffusers import StableAudioPipeline
        except ImportError:
            return "✗ diffusers is not installed. Run: pip install diffusers"

        try:
            local_model_path = self.models_dir / "stable-audio-open-1.0"
            model_id = str(local_model_path) if local_model_path.exists() else "stabilityai/stable-audio-open-1.0"

            pipe_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=pipe_dtype)
            pipe = pipe.to(self.device)

            self.stableaudio_model = pipe
            self.stableaudio_sample_rate = pipe.vae.sampling_rate
            self.stableaudio_sample_size = None  # not used with diffusers
            return "✓ Stable Audio model loaded successfully"
        except Exception as e:
            return f"✗ Error loading Stable Audio model: {str(e)}"
    
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
        self.stableaudio_model = None
        self.stableaudio_sample_rate = None
        self.stableaudio_sample_size = None
