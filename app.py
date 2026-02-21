"""Main application module for Qwen Voice TTS Studio.

This module orchestrates all components of the application including
voice generation, cloning, design, and the web interface.
"""

import json
import os
import time
from pathlib import Path
import torch

from logger import log, section, section_end, bold, dim
from models_manager import ModelManager
from audio_processor import AudioProcessor
from voice_manager import VoiceManager
from tts_generator import TTSGenerator
from voice_cloner import VoiceCloner
from voice_designer import VoiceDesigner
from asr_handler import ASRHandler
from interface_builder import InterfaceBuilder
from user_voice_manager import UserVoiceManager


class QwenVoiceStudio:
    """Main application class coordinating all voice synthesis and cloning operations."""

    def __init__(self, config: dict = None):
        """Initialize the Qwen Voice Studio application."""
        log.info(bold("QwenVoiceStudio") + " — initialising")
        self.config = config or {}
        self._setup_paths()
        self._setup_device()
        self._initialize_components()
        log.ok("Initialisation complete")

    def _setup_paths(self):
        """Setup all required directories."""
        self.base_dir = Path(self.config.get("base_dir", "."))
        self.models_dir = Path(self.config.get("models_dir", "./models"))
        self.outputs_dir = Path(self.config.get("outputs_dir", "./outputs"))
        self.voice_inputs_dir = Path(self.config.get("voice_inputs_dir", "./voiceinputs"))
        self.cloned_voices_dir = Path(self.config.get("cloned_voices_dir", "./cloned_voices"))
        self.designed_voices_dir = Path(self.config.get("designed_voices_dir", "./designed_voices"))
        self.voice_samples_dir = Path(self.config.get("voice_samples_dir", "./voicesamples"))

        # Create all directories
        for directory in [self.outputs_dir, self.voice_inputs_dir, self.cloned_voices_dir,
                         self.designed_voices_dir, self.voice_samples_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        log.debug(f"Paths ready — outputs={self.outputs_dir}  models={self.models_dir}")

    def _setup_device(self):
        """Setup compute device and dtype."""
        # Priority: config → DEVICE env var → auto-detect
        env_device = os.environ.get("DEVICE", "").strip().lower()
        default = env_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.config.get("device", default)

        # Determine dtype based on device
        if self.device.startswith("cuda"):
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            self.dtype = torch.float32

        log.info(f"Device: {bold(self.device)}   dtype: {dim(str(self.dtype))}")

    def _initialize_components(self):
        """Initialize all application components."""
        log.info("Initialising components…")

        # Core managers
        self.model_manager = ModelManager(self.models_dir, self.device, self.dtype)
        self.audio_processor = AudioProcessor()
        self.voice_manager = VoiceManager(self.designed_voices_dir)
        self.user_voice_manager = UserVoiceManager(self.base_dir)

        # Feature handlers
        self.tts_generator = TTSGenerator(self.model_manager, self.voice_samples_dir)
        self.voice_cloner = VoiceCloner(
            self.model_manager,
            self.audio_processor,
            self.cloned_voices_dir,
            self.voice_inputs_dir
        )
        self.voice_designer = VoiceDesigner(
            self.model_manager,
            self.audio_processor,
            self.designed_voices_dir
        )
        self.asr_handler = ASRHandler(self.model_manager)

        # UI builder
        self.interface_builder = InterfaceBuilder("Qwen Voice TTS Studio")

        # Load voice personas from main_voices.json
        self._load_voices()

        log.ok("All components initialised")

    # ==================== TTS Methods ====================

    def generate_tts(self, text: str, language: str, speaker: str):
        """Generate speech from text."""
        return self.tts_generator.generate_tts(text, language, speaker)

    def generate_voice_sample(self, language: str, speaker: str):
        """Generate a voice sample."""
        return self.tts_generator.generate_voice_sample(language, speaker)

    # ==================== Voice Cloning Methods ====================

    def clone_voice(self, json_config: str = None, audio_file: Path = None, ref_text: str = None):
        """Clone a voice."""
        voice_path = self.voice_cloner.clone_voice(json_config, audio_file, ref_text)
        return voice_path

    # ==================== Voice Design Methods ====================

    def design_voice(self, json_config: str = None, text_input: str = None, instructions: str = None):
        """Design a custom voice."""
        return self.voice_designer.design_voice(json_config, text_input, instructions)

    def save_designed_voice(self, name: str, voice_data: dict) -> bool:
        """Save a designed voice."""
        return self.voice_manager.add_designed_voice(name, voice_data)

    def delete_designed_voice(self, name: str) -> bool:
        """Delete a designed voice."""
        return self.voice_manager.delete_designed_voice(name)

    def get_designed_voices(self):
        """Get list of designed voices."""
        return self.voice_manager.get_designed_voices_list()

    # ==================== ASR Methods ====================

    def transcribe_audio(self, audio_file: Path, language: str = None):
        """Transcribe audio to text."""
        return self.asr_handler.transcribe_audio(audio_file, language)

    def transcribe_with_timestamps(self, audio_file: Path, language: str = None):
        """Transcribe audio with timestamps."""
        return self.asr_handler.transcribe_with_timestamps(audio_file, language)

    # ==================== Model Methods ====================

    def load_custom_model(self):
        """Load CustomVoice model."""
        log.info("🤖 Loading CustomVoice model…")
        t0 = time.time()
        result = self.model_manager.load_custom_model()
        log.timing(f"CustomVoice model ready in {time.time()-t0:.1f}s")
        return result

    def load_base_model(self):
        """Load Base model."""
        log.info("🤖 Loading Base model…")
        t0 = time.time()
        result = self.model_manager.load_base_model()
        log.timing(f"Base model ready in {time.time()-t0:.1f}s")
        return result

    def load_design_model(self):
        """Load VoiceDesign model."""
        log.info("🤖 Loading VoiceDesign model…")
        t0 = time.time()
        result = self.model_manager.load_design_model()
        log.timing(f"VoiceDesign model ready in {time.time()-t0:.1f}s")
        return result

    def load_asr_model(self, model_id: str, use_forced_aligner: bool = False):
        """Load ASR model."""
        log.info(f"🤖 Loading ASR model: {model_id}…")
        t0 = time.time()
        result = self.model_manager.load_asr_model(model_id, use_forced_aligner)
        log.timing(f"ASR model ready in {time.time()-t0:.1f}s")
        return result

    # ==================== Voice Personas ====================

    def _load_voices(self):
        """Load voice personas from main_voices.json into self._voice_list and self._voice_personas."""
        voices_file = self.base_dir / "main_voices.json"
        try:
            with open(voices_file, encoding="utf-8") as f:
                entries = json.load(f).get("voices", [])
            self._voice_list = [e["name"] for e in entries if "name" in e]
            self._voice_personas = {e["name"]: e for e in entries if "name" in e}
            log.ok(f"Loaded {len(self._voice_list)} voices from {voices_file.name}")
        except Exception as e:
            log.warning(f"Could not load {voices_file}: {e}")
            self._voice_list = []
            self._voice_personas = {}

    # ==================== UI Methods ====================

    def _get_voice_names(self) -> list:
        """Return ordered voice display names."""
        return self._voice_list

    def create_interface(self):
        """Create and return the Gradio interface."""
        voice_names = self._get_voice_names()
        voices_data = [self._voice_personas[n] for n in voice_names if n in self._voice_personas]
        log.debug(f"Voice list ({len(voice_names)}): {', '.join(voice_names)}")

        return self.interface_builder.create_interface(
            on_tts_generate=self._ui_generate_tts,
            on_clone_voice=self._ui_clone_voice,
            on_design_voice=self._ui_design_voice,
            on_script_run=self._ui_run_script,
            on_transcribe=self._ui_transcribe,
            on_save_clone=self._ui_save_clone_voice,
            on_save_design=self._ui_save_design_voice,
            speakers_list=voices_data,
            user_voice_choices=self.user_voice_manager.get_choices(),
        )

    def launch_interface(self, share: bool = False, server_port: int = 7860, server_name: str = "127.0.0.1"):
        """Launch the web interface."""
        self.create_interface()
        self.interface_builder.launch(
            share=share,
            server_port=server_port,
            server_name=server_name,
        )

    # ==================== UI Callbacks ====================

    def _ui_generate_tts(self, text: str, language: str, voice_name: str):
        """UI callback for TTS generation — routes to CustomVoice or VoiceDesign based on persona."""
        section("TTS Generate")
        log.info(f"voice={bold(voice_name)}  language={bold(language)}")
        log.info(f"text: {dim(text[:120])}")
        t0 = time.time()
        try:
            result = None

            # ── User-saved voices (prefix "user:") ───────────────────────────
            if voice_name and voice_name.startswith("user:"):
                uname = voice_name[5:]
                uv = self.user_voice_manager.get_voice(uname)
                if uv is None:
                    raise ValueError(f"User voice '{uname}' not found in user_voices.json")
                model_type = uv.get("model", "")
                if model_type == "design":
                    log.debug(f"User voice → VoiceDesign  instruct={dim(uv.get('instruct','')[:60])}")
                    result = self.voice_designer.design_voice(
                        text_input=text, instructions=uv["instruct"], voice_name=uname
                    )
                elif model_type == "clone":
                    log.debug(f"User voice → VoiceCloner  ref={dim(uv.get('ref_audio',''))}")
                    result = self.voice_cloner.generate_with_cloned_voice(
                        text,
                        {"ref_audio": uv["ref_audio"], "ref_text": uv.get("ref_text", "")},
                        voice_name=uname,
                    )
                else:
                    raise ValueError(f"Unknown model type '{model_type}' for user voice '{uname}'")

            # ── Built-in voices ───────────────────────────────────────────────
            else:
                persona = self._voice_personas.get(voice_name)
                if persona:
                    model_type = persona.get("model", "")
                    if model_type == "custom":
                        log.debug(f"Routing to CustomVoice  speaker={dim(persona['speaker'])}")
                        result = self.tts_generator.generate_tts(
                            text, language, persona["speaker"], voice_name=voice_name
                        )
                    elif model_type == "design":
                        log.debug(f"Routing to VoiceDesign  instruct={dim(persona['instruct'][:60])}")
                        result = self.voice_designer.design_voice(
                            text_input=text, instructions=persona["instruct"], voice_name=voice_name
                        )
                    elif model_type == "clone":
                        ref_audio = self.base_dir / persona["ref_audio"]
                        ref_text = persona["ref_text"]
                        log.debug(f"Routing to VoiceCloner  ref={dim(persona['ref_audio'])}")
                        result = self.voice_cloner.generate_with_cloned_voice(
                            text,
                            {"ref_audio": str(ref_audio), "ref_text": ref_text},
                            voice_name=voice_name,
                        )
                    else:
                        raise ValueError(f"Unknown model type '{model_type}' for voice '{voice_name}'")
                else:
                    raise ValueError(f"Unknown voice '{voice_name}' — not found in main_voices.json")

            if result is None:
                log.error("TTS generation returned no audio — model may not be loaded or failed silently")
                section_end("failed")
                return None, "✗ Generation failed — model returned no audio"
            log.ok("TTS complete")
            elapsed = time.time() - t0
            log.timing(f"Elapsed: {elapsed:.1f}s  →  {dim(str(result))}")
            section_end("done")
            return str(result), f"✓ Done ({elapsed:.1f}s)"
        except Exception as e:
            log.error(f"TTS failed: {e}")
            section_end("error")
            return None, f"✗ Error: {e}"

    def _ui_clone_voice(self, audio_file, ref_text: str, target_text: str):
        """UI callback for voice cloning (single audio mode)."""
        section("Clone Voice")
        log.info(f"ref_text={dim(ref_text[:80])}")
        t0 = time.time()
        try:
            result = self.voice_cloner.generate_with_cloned_voice(
                text=target_text,
                cloned_voice_config={"ref_audio": str(audio_file), "ref_text": ref_text},
            )
            if result:
                log.ok(f"Clone generated: {Path(result).name}")
                log.timing(f"Elapsed: {time.time()-t0:.1f}s")
                section_end("done")
                return str(result), f"✓ Done ({time.time()-t0:.1f}s)"
            log.warning("clone_voice returned no path")
            section_end("failed")
            return None, "✗ Failed to clone voice"
        except Exception as e:
            log.error(f"Clone voice error: {e}")
            section_end("error")
            return None, f"✗ Error: {str(e)}"

    def _ui_save_clone_voice(self, name: str, desc: str, ref_audio, ref_text: str) -> str:
        """UI callback to persist a cloned voice to user_voices.json."""
        if ref_audio is None:
            return "✗ No reference audio — upload a file first"
        return self.user_voice_manager.add_clone_voice(name, desc, str(ref_audio), ref_text or "")

    def _ui_save_design_voice(self, name: str, desc: str, instructions: str) -> str:
        """UI callback to persist a designed voice to user_voices.json."""
        if not instructions or not instructions.strip():
            return "✗ No voice description — enter a description first"
        return self.user_voice_manager.add_design_voice(name, desc, instructions)

    # ── Script helpers ────────────────────────────────────────────────────────

    def _resolve_ref_audio(self, raw: str) -> tuple:
        """Resolve a ref_audio value to an absolute path inside voices/.

        Rules:
          - Absolute paths are rejected.
          - A leading 'voices/' component is stripped (assumed default).
          - Path traversal (../) is blocked via resolve() + relative_to().
          - Returns (resolved_Path, None) on success or (None, error_str) on failure.
        """
        if not raw or not raw.strip():
            return None, "ref_audio is empty"

        p = Path(raw.strip().replace("\\", "/"))

        if p.is_absolute():
            return None, f"ref_audio must be a relative filename, not an absolute path: '{raw}'"

        # Strip optional leading 'voices/' that the user may have included
        if p.parts and p.parts[0].lower() == "voices":
            p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(".")

        voices_dir = (self.base_dir / "voices").resolve()
        resolved   = (voices_dir / p).resolve()

        # Block path traversal
        try:
            resolved.relative_to(voices_dir)
        except ValueError:
            return None, f"ref_audio escapes the voices directory: '{raw}'"

        if not resolved.exists():
            return None, f"ref_audio file not found: 'voices/{p}'"

        return resolved, None

    def _validate_script_speakers(self, speakers: list) -> tuple:
        """Validate all speaker definitions before any model inference.

        Returns (speaker_map, errors) where errors is a list of strings.
        speaker_map maps name → resolved info dict (ref_audio already an abs Path str).
        """
        speaker_map: dict = {}
        errors: list = []

        for sp in speakers:
            name = sp.get("name", "").strip()
            if not name:
                errors.append("A speaker entry is missing the 'name' field")
                continue

            if sp.get("ref_audio"):
                resolved, err = self._resolve_ref_audio(sp["ref_audio"])
                if err:
                    errors.append(f"Speaker '{name}': {err}")
                    continue
                ref_text = sp.get("ref_text", "").strip()
                if not ref_text:
                    errors.append(f"Speaker '{name}': ref_text is required with ref_audio")
                    continue
                speaker_map[name] = {
                    "type":      "clone",
                    "ref_audio": str(resolved),
                    "ref_text":  ref_text,
                }

            elif sp.get("ref_speaker"):
                speaker_id = sp["ref_speaker"].strip()
                if not speaker_id:
                    errors.append(f"Speaker '{name}': ref_speaker is empty")
                    continue
                speaker_map[name] = {
                    "type":    "custom",
                    "speaker": speaker_id,
                }

            elif sp.get("ref_description") is not None:
                desc = sp["ref_description"].strip()
                if not desc:
                    errors.append(f"Speaker '{name}': ref_description is empty")
                    continue
                speaker_map[name] = {
                    "type":         "design",
                    "instructions": desc,
                }

            else:
                errors.append(
                    f"Speaker '{name}': must have ref_audio+ref_text, ref_speaker, or ref_description"
                )

        return speaker_map, errors

    def _ui_run_script(self, json_config: str):
        """UI callback for the Script tab.

        Validates all speakers up-front and reports every error before touching
        any model.  Returns (audio_path, status_message).

        Speaker definition options (mutually exclusive):
          ref_audio + ref_text  → VoiceCloner   (file resolved inside voices/)
          ref_speaker           → TTSGenerator  (built-in CustomVoice speaker ID)
          ref_description       → VoiceDesigner (text-based voice design)
        """
        import json as _json
        from datetime import datetime

        section("Run Script")
        t0 = time.time()
        _fail = lambda msg: (None, msg)

        # ── Parse ─────────────────────────────────────────────────────────────
        try:
            config = _json.loads(json_config or "{}")
        except _json.JSONDecodeError as exc:
            log.error(f"Invalid JSON: {exc}")
            section_end("error")
            return _fail(f"✗ Invalid JSON: {exc}")

        # ── Validate speakers (fast, no model loading) ────────────────────────
        speaker_map, validation_errors = self._validate_script_speakers(
            config.get("speakers", [])
        )

        if validation_errors:
            msg = "✗ Script validation failed:\n" + "\n".join(
                f"  • {e}" for e in validation_errors
            )
            log.error(msg)
            section_end("error")
            return _fail(msg)

        if not speaker_map:
            section_end("failed")
            return _fail("✗ No valid speakers defined in 'speakers' array")

        # ── Render dialog lines ───────────────────────────────────────────────
        audio_files: list = []
        errors: list = []

        for scene in sorted(config.get("scenes", []), key=lambda s: s.get("pos", 0)):
            for line in sorted(scene.get("dialog", []), key=lambda d: d.get("pos", 0)):
                speaker_name = line.get("speaker", "").strip()
                text         = line.get("text", "").strip()
                conotation   = line.get("conotation", "").strip() or None

                if not text or speaker_name.lower() in ("direction", "instruction", ""):
                    continue

                sp_info = speaker_map.get(speaker_name)
                if not sp_info:
                    log.warning(f"Unknown speaker: '{speaker_name}'")
                    errors.append(f"Unknown speaker '{speaker_name}' (not in speakers list)")
                    continue

                try:
                    result = None
                    sp_type = sp_info["type"]

                    if sp_type == "clone":
                        # ref_audio is already a resolved absolute path string
                        result = self.voice_cloner.generate_with_cloned_voice(
                            text=text,
                            cloned_voice_config={
                                "ref_audio": sp_info["ref_audio"],
                                "ref_text":  sp_info["ref_text"],
                            },
                            voice_name=speaker_name,
                        )

                    elif sp_type == "custom":
                        result = self.tts_generator.generate_tts(
                            text=text,
                            language="auto",
                            speaker=sp_info["speaker"],
                            voice_name=speaker_name,
                        )

                    elif sp_type == "design":
                        instruct = sp_info["instructions"]
                        if conotation:
                            instruct = f"{instruct}, {conotation}"
                        result = self.voice_designer.design_voice(
                            text_input=text,
                            instructions=instruct,
                            voice_name=speaker_name,
                        )

                    if result:
                        audio_files.append(Path(result))
                    else:
                        errors.append(f"No audio for [{speaker_name}]: {text[:40]}")

                except Exception as exc:
                    log.error(f"Line [{speaker_name}]: {exc}")
                    errors.append(f"Error [{speaker_name}]: {exc}")

        # ── Merge ─────────────────────────────────────────────────────────────
        if not audio_files:
            msg = "✗ No audio was generated.\n" + "\n".join(errors)
            section_end("failed")
            return _fail(msg)

        ts       = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_file = self.outputs_dir / f"script_{ts}.wav"
        if not self.audio_processor.merge_audio_files(audio_files, out_file):
            section_end("failed")
            return _fail("✗ Failed to merge audio lines")

        elapsed = time.time() - t0
        log.ok(f"Script complete — {len(audio_files)} lines")
        log.timing(f"Elapsed: {elapsed:.1f}s  →  {dim(str(out_file))}")
        section_end("done")

        status_lines = [f"✓ {len(audio_files)} lines merged → {out_file.name}  ({elapsed:.1f}s)"]
        if errors:
            status_lines.append(f"⚠ {len(errors)} line(s) skipped:")
            status_lines.extend(f"  {e}" for e in errors)
        return str(out_file), "\n".join(status_lines)

    def _ui_design_voice(self, text_input: str, instructions: str):
        """UI callback for voice design (text mode)."""
        section("Design Voice")
        log.info(f"instructions: {dim(instructions[:80])}")
        log.info(f"text: {dim(text_input[:80])}")
        t0 = time.time()
        try:
            result = self.design_voice(text_input=text_input, instructions=instructions)
            if result is None:
                log.error("Voice design returned no audio — model may not be loaded or failed silently")
                section_end("failed")
                return None, "✗ Design failed — model returned no audio"
            log.ok("Design complete")
            elapsed = time.time() - t0
            log.timing(f"Elapsed: {elapsed:.1f}s  →  {dim(str(result))}")
            section_end("done")
            return str(result), f"✓ Done ({elapsed:.1f}s)"
        except Exception as e:
            log.error(f"Design voice error: {e}")
            section_end("error")
            return None, f"✗ Error: {e}"

    def _ui_design_voice_json(self, json_config: str):
        """UI callback for voice design (JSON mode)."""
        section("Design Voice (JSON)")
        t0 = time.time()
        try:
            result = self.design_voice(json_config=json_config)
            if result is None:
                log.error("Voice design (JSON) returned no audio — model may not be loaded or failed silently")
                section_end("failed")
                return None
            log.ok("Design complete")
            log.timing(f"Elapsed: {time.time()-t0:.1f}s  →  {dim(str(result))}")
            section_end("done")
            return str(result)
        except Exception as e:
            log.error(f"Design voice JSON error: {e}")
            section_end("error")
            return f"✗ Error: {str(e)}"

    def _ui_transcribe(self, audio_file, language: str):
        """UI callback for transcription."""
        section("Transcribe")
        log.info(f"language={bold(language or 'auto')}  file={dim(str(audio_file))}")
        t0 = time.time()
        try:
            result = self.transcribe_audio(Path(audio_file), language)
            if result is None:
                log.error("Transcription returned no text — model may not be loaded or failed silently")
                section_end("failed")
                return None
            log.ok("Transcription complete")
            log.timing(f"Elapsed: {time.time()-t0:.1f}s")
            section_end("done")
            return result
        except Exception as e:
            log.error(f"Transcription error: {e}")
            section_end("error")
            return None

def create_app(config: dict = None) -> QwenVoiceStudio:
    """Factory function to create the application."""
    return QwenVoiceStudio(config)


def main():
    """Main entry point."""
    # Initialize application
    app = create_app()

    # Launch web interface
    app.launch_interface(server_port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    main()
