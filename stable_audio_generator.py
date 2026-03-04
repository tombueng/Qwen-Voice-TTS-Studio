"""Stable Audio generation module — text-to-sound-effects via diffusers StableAudioPipeline."""
from pathlib import Path
from typing import Optional

import soundfile as sf

from cache import AudioCache


class StableAudioGenerator:
    """Generates sound effects and music from text prompts using Stable Audio Open."""

    def __init__(self, model_manager, output_dir: Path):
        self.model_manager = model_manager
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cache = AudioCache(output_dir, "sfx")

    def generate(self, prompt: str, duration: float = 10.0,
                 steps: int = 100, cfg_scale: float = 7.0) -> Optional[Path]:
        """Generate audio from a text prompt.

        Args:
            prompt:    Natural-language description of the sound to generate.
            duration:  Desired output length in seconds (1–30).
            steps:     Diffusion steps — more steps = higher quality but slower.
            cfg_scale: Classifier-free guidance scale — higher = closer to prompt.

        Returns:
            Path to the generated WAV file (cached on repeated calls with identical args).
        """
        # Normalise key components so float precision doesn't break cache hits
        key_duration  = str(round(float(duration),  3))
        key_steps     = str(int(steps))
        key_cfg_scale = str(round(float(cfg_scale), 3))

        cached = self._cache.get(prompt, key_duration, key_steps, key_cfg_scale)
        if cached:
            return cached

        load_result = self.model_manager.load_stableaudio_model()
        if not load_result.startswith("✓"):
            raise RuntimeError(load_result)

        pipe = self.model_manager.stableaudio_model
        if pipe is None:
            raise RuntimeError("Stable Audio model is None after loading")

        sample_rate = self.model_manager.stableaudio_sample_rate

        import torch
        generator = torch.Generator(self.model_manager.device)

        output = pipe(
            prompt,
            num_inference_steps=int(steps),
            audio_end_in_s=float(duration),
            guidance_scale=float(cfg_scale),
            num_waveforms_per_prompt=1,
            generator=generator,
        )

        # output.audios shape: (num_waveforms, channels, samples) — take first
        audio = output.audios[0].T.float().cpu().numpy()  # → (samples, channels)
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio[:, 0]  # mono squeeze

        meta = {
            "type":      "sfx",
            "text":      prompt,   # drives the human-readable filename slug
            "duration":  duration,
            "steps":     steps,
            "cfg_scale": cfg_scale,
        }
        out_file = self._cache.path(prompt, key_duration, key_steps, key_cfg_scale, meta=meta)
        sf.write(str(out_file), audio, sample_rate)
        return self._cache.put(out_file, meta=meta)
