"""Main application module for Qwen Voice TTS Studio.

This module orchestrates all components of the application including
voice generation, cloning, design, and the web interface.
"""

import json
import os
import re
import threading
import time
from pathlib import Path
import torch

from logger import (log, section, subsection, section_end, bold, dim,
                    ui_capture_start, ui_capture_drain, ui_capture_stop)
from models_manager import ModelManager
from audio_processor import AudioProcessor
from voice_manager import VoiceManager
from tts_generator import TTSGenerator
from voice_cloner import VoiceCloner
from voice_designer import VoiceDesigner
from asr_handler import ASRHandler
from interface_builder import InterfaceBuilder
from user_voice_manager import UserVoiceManager
from stable_audio_generator import StableAudioGenerator

# [audio:'<prompt>',key:value,...]          sequential inline effect (Stable Audio)
_EFFECT_RE      = re.compile(r"\[audio:'([^']+)'((?:,[a-z_]+:[\d.]+)*)\]")
# [audio:ref_file:'<name>',key:value,...]   sequential inline effect (file in audio/)
_EFFECT_FILE_RE = re.compile(r"\[audio:ref_file:'([^']+)'((?:,[a-z_]+:[\d.]+)*)\]")
# [background-audio:'<prompt>',...]         background effect (Stable Audio)
_BGAUDIO_RE      = re.compile(r"\[background-audio:'([^']+)'((?:,[a-z_]+:[\d.]+)*)\]")
# [background-audio:ref_file:'<name>',...]  background effect (file in audio/)
_BGAUDIO_FILE_RE = re.compile(r"\[background-audio:ref_file:'([^']+)'((?:,[a-z_]+:[\d.]+)*)\]")


class QwenVoiceStudio:
    """Main application class coordinating all voice synthesis and cloning operations."""

    def __init__(self, config: dict = None):
        """Initialize the Qwen Voice Studio application."""
        log.info(bold("QwenVoiceStudio") + " — initialising")
        self.config = config or {}
        self._abort_event = threading.Event()  # set to request abort of running job
        self._setup_paths()
        self._load_config_file()
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
        self.config_file = self.base_dir / "config.json"

        self.audio_ref_dir = self.base_dir / "audio"

        self.proc_cache_dir = self.outputs_dir / "proc_cache"

        # Create all directories
        for directory in [self.outputs_dir, self.voice_inputs_dir, self.cloned_voices_dir,
                         self.designed_voices_dir, self.voice_samples_dir, self.audio_ref_dir,
                         self.proc_cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        AudioProcessor.set_cache_dir(self.proc_cache_dir)
        log.debug(f"Paths ready — outputs={self.outputs_dir}  models={self.models_dir}")

    def _load_config_file(self):
        """Load config.json and apply persisted settings (e.g. HuggingFace token)."""
        self._saved_config: dict = {}
        if not self.config_file.exists():
            return
        try:
            with open(self.config_file, encoding="utf-8") as f:
                self._saved_config = json.load(f)
            token = self._saved_config.get("hf_token", "").strip()
            if token:
                self._apply_hf_token(token)
                log.ok("HuggingFace token loaded from config.json")
        except Exception as e:
            log.warning(f"Could not load config.json: {e}")
            self._saved_config = {}

    def _apply_hf_token(self, token: str):
        """Apply a HuggingFace token to the current process session."""
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)

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
        self.stable_audio_generator = StableAudioGenerator(self.model_manager, self.outputs_dir)

        # UI builder
        self.interface_builder = InterfaceBuilder("Qwen Dialogue Studio")

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

    def design_voice(self, json_config: str = None, text_input: str = None,
                     instructions: str = None):
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
            on_tts_generate=self._stream(self._ui_generate_tts),
            on_clone_voice=self._stream(self._ui_clone_voice),
            on_design_voice=self._stream(self._ui_design_voice),
            on_script_run=self._stream(self._ui_run_script),
            on_transcribe=self._ui_transcribe,
            on_save_clone=self._ui_save_clone_voice,
            on_save_design=self._ui_save_design_voice,
            on_voice_speed_change=self._ui_get_voice_speed,
            on_stable_audio_generate=self._stream(self._ui_generate_stable_audio),
            on_save_hf_token=self._ui_save_hf_token,
            on_abort=self._abort_job,
            hf_token=self._saved_config.get("hf_token", ""),
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

    # ==================== UI Helpers ====================

    def _stream(self, fn):
        """Return a generator *function* that Gradio can recognise as streaming.

        A plain ``lambda *a: self._ui_stream(fn, a)`` is a regular function
        that merely *returns* a generator object — Gradio then wraps the object
        itself as the single return value and raises a "not enough outputs"
        error.  Wrapping with ``yield from`` makes the outer callable a real
        generator function so Gradio iterates it correctly.
        """
        def _g(*args):
            yield from self._ui_stream(fn, args)
        return _g

    def _ui_stream(self, fn, args):
        """Run fn(*args) in a background thread and stream log lines to the UI.

        Yields (audio_or_None, log_text) pairs — use as a Gradio generator for
        callbacks that return (audio_path, status_str).
        The status Textbox shows a live rolling log while the operation runs,
        then the final status line appended at the bottom when done.
        """
        import queue, threading

        result_q  = queue.SimpleQueue()
        tid_q     = queue.SimpleQueue()

        def _worker():
            self._abort_event.clear()          # fresh start for every job
            tid = ui_capture_start()
            tid_q.put(tid)
            _placed = False
            try:
                result_q.put(("ok", fn(*args)))
                _placed = True
            except Exception as exc:
                import traceback
                result_q.put(("err", f"{exc}\n{traceback.format_exc()}"))
                _placed = True
            finally:
                ui_capture_stop(tid)
                if not _placed:                         # BaseException / KeyboardInterrupt
                    result_q.put(("err", "Worker thread terminated unexpectedly"))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        try:
            tid = tid_q.get(timeout=15.0)
        except queue.Empty:
            yield None, "✗ Worker thread did not start in time"
            return

        log_lines: list = []
        while t.is_alive():
            new = ui_capture_drain(tid)
            if new:
                log_lines.extend(new)
                yield None, "\n".join(log_lines[-400:])
            time.sleep(0.2)

        # Final drain after thread ends
        log_lines.extend(ui_capture_drain(tid))

        try:
            kind, payload = result_q.get(timeout=10.0)
        except Exception:
            kind, payload = "err", "Worker result was not returned (timeout)"

        if kind == "ok":
            audio_out, final_msg = payload
            log_lines.append("")
            log_lines.append("─" * 50)
            log_lines.append(final_msg)
        else:
            audio_out = None
            log_lines.append("")
            log_lines.append(f"✗ {payload}")

        yield audio_out, "\n".join(log_lines[-400:])

    # ==================== UI Callbacks ====================

    def _apply_speed_to_output(self, result_path: str, speed: float) -> str:
        """If speed != 1.0, write a time-stretched copy to outputs/ and return its path.

        The original cached file is never modified.
        """
        if abs(speed - 1.0) < 0.001:
            return result_path
        from datetime import datetime
        ts  = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        dst = self.outputs_dir / f"speed_{ts}.wav"
        if AudioProcessor.time_stretch_to_file(Path(result_path), dst, speed):
            return str(dst)
        log.warning(f"Time stretch failed — returning audio at original speed")
        return result_path

    def _ui_get_voice_speed(self, voice_name: str) -> float:
        """Return the saved default speed for a voice (used to update the speed slider on voice change)."""
        if voice_name and voice_name.startswith("user:"):
            uv = self.user_voice_manager.get_voice(voice_name[5:])
            if uv:
                return float(uv.get("speed", 1.0))
        else:
            persona = self._voice_personas.get(voice_name)
            if persona:
                return float(persona.get("speed", 1.0))
        return 1.0

    def _ui_generate_tts(self, text: str, language: str, voice_name: str, speed: float = 1.0):
        """UI callback for TTS generation — routes to CustomVoice or VoiceDesign based on persona."""
        section("TTS Generate")
        log.info(f"voice={bold(voice_name)}  language={bold(language)}  speed={speed:.2f}")
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
                            text_input=text, instructions=persona["instruct"],
                            voice_name=voice_name
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
            result = self._apply_speed_to_output(str(result), speed)
            log.ok("TTS complete")
            elapsed = time.time() - t0
            log.timing(f"Elapsed: {elapsed:.1f}s  →  {dim(str(result))}")
            section_end("done")
            return str(result), f"✓ Done ({elapsed:.1f}s)"
        except Exception as e:
            log.error(f"TTS failed: {e}")
            section_end("error")
            return None, f"✗ Error: {e}"

    def _ui_clone_voice(self, audio_file, ref_text: str, target_text: str, speed: float = 1.0):
        """UI callback for voice cloning (single audio mode)."""
        section("Clone Voice")
        log.info(f"ref_text={dim(ref_text[:80])}  speed={speed:.2f}")
        t0 = time.time()
        try:
            result = self.voice_cloner.generate_with_cloned_voice(
                text=target_text,
                cloned_voice_config={"ref_audio": str(audio_file), "ref_text": ref_text},
            )
            if result:
                result = self._apply_speed_to_output(str(result), speed)
                elapsed = time.time() - t0
                log.ok(f"Clone generated: {Path(result).name}")
                log.timing(f"Elapsed: {elapsed:.1f}s")
                section_end("done")
                return str(result), f"✓ Done ({elapsed:.1f}s)"
            log.warning("clone_voice returned no path")
            section_end("failed")
            return None, "✗ Failed to clone voice"
        except Exception as e:
            log.error(f"Clone voice error: {e}")
            section_end("error")
            return None, f"✗ Error: {str(e)}"

    def _ui_save_clone_voice(self, name: str, desc: str, ref_audio, ref_text: str,
                              speed: float = 1.0) -> str:
        """UI callback to persist a cloned voice to user_voices.json."""
        if ref_audio is None:
            return "✗ No reference audio — upload a file first"
        return self.user_voice_manager.add_clone_voice(
            name, desc, str(ref_audio), ref_text or "", speed=speed
        )

    def _ui_save_design_voice(self, name: str, desc: str, instructions: str,
                               speed: float = 1.0) -> str:
        """UI callback to persist a designed voice to user_voices.json."""
        if not instructions or not instructions.strip():
            return "✗ No voice description — enter a description first"
        return self.user_voice_manager.add_design_voice(name, desc, instructions, speed=speed)

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

        # Try with and without common audio extensions
        candidates = (
            [p] if p.suffix
            else [p.with_suffix(ext) for ext in (".wav", ".mp3", ".flac", ".ogg")] + [p]
        )
        for candidate in candidates:
            resolved = (voices_dir / candidate).resolve()
            try:
                resolved.relative_to(voices_dir)
            except ValueError:
                return None, f"ref_audio escapes the voices directory: '{raw}'"
            if resolved.exists():
                return resolved, None

        return None, f"ref_audio file not found: 'voices/{p}'"

    def _validate_script_speakers(self, speakers: list) -> tuple:
        """Validate all speaker definitions before any model inference.

        Returns (speaker_map, errors) where errors is a list of strings.
        speaker_map maps name → resolved info dict (ref_audio already an abs Path str).
        """
        speaker_map: dict = {}
        errors: list = []

        for sp in speakers:
            # "name" is canonical; fall back to "ref_name" when it doubles as the identity
            name = (sp.get("name") or sp.get("ref_name") or "").strip()
            if not name:
                errors.append("A speaker entry is missing 'name' (or 'ref_name') field")
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

            elif sp.get("ref_description") is not None or sp.get("voice") is not None:
                # "voice" is an accepted alias for "ref_description"
                desc = (sp.get("ref_description") or sp.get("voice") or "").strip()
                if not desc:
                    errors.append(f"Speaker '{name}': ref_description/voice is empty")
                    continue
                speaker_map[name] = {
                    "type":         "design",
                    "instructions": desc,
                }

            elif sp.get("ref_name") is not None:
                ref_name = sp["ref_name"].strip()
                if not ref_name:
                    errors.append(f"Speaker '{name}': ref_name is empty")
                    continue

                # ── 1. Built-in voices (main_voices.json) ──────────────────
                persona = self._voice_personas.get(ref_name)
                if persona:
                    model_type = persona.get("model", "")
                    if model_type == "custom":
                        speaker_map[name] = {"type": "custom", "speaker": persona["speaker"]}
                    elif model_type == "design":
                        speaker_map[name] = {"type": "design", "instructions": persona["instruct"]}
                    elif model_type == "clone":
                        resolved, err = self._resolve_ref_audio(persona["ref_audio"])
                        if err:
                            errors.append(f"Speaker '{name}': ref_name '{ref_name}' — {err}")
                            continue
                        ref_text = persona.get("ref_text", "").strip()
                        if not ref_text:
                            errors.append(f"Speaker '{name}': ref_name '{ref_name}' has no ref_text")
                            continue
                        speaker_map[name] = {
                            "type":      "clone",
                            "ref_audio": str(resolved),
                            "ref_text":  ref_text,
                        }
                    else:
                        errors.append(
                            f"Speaker '{name}': ref_name '{ref_name}' has unknown model type '{model_type}'"
                        )
                    continue

                # ── 2. User voices (user_voices.json) ──────────────────────
                uv = self.user_voice_manager.get_voice(ref_name)
                if uv:
                    model_type = uv.get("model", "")
                    if model_type == "design":
                        speaker_map[name] = {"type": "design", "instructions": uv["instruct"]}
                    elif model_type == "clone":
                        ref_audio_path = uv.get("ref_audio", "")
                        if not Path(ref_audio_path).exists():
                            errors.append(
                                f"Speaker '{name}': ref_name '{ref_name}' ref_audio not found: '{ref_audio_path}'"
                            )
                            continue
                        speaker_map[name] = {
                            "type":      "clone",
                            "ref_audio": ref_audio_path,
                            "ref_text":  uv.get("ref_text", ""),
                        }
                    else:
                        errors.append(
                            f"Speaker '{name}': ref_name '{ref_name}' (user voice) has unknown model type '{model_type}'"
                        )
                    continue

                errors.append(
                    f"Speaker '{name}': ref_name '{ref_name}' not found in built-in voices or user voices"
                )

            else:
                errors.append(
                    f"Speaker '{name}': must have ref_audio+ref_text, ref_speaker, "
                    f"ref_description/voice, or ref_name"
                )

        return speaker_map, errors

    def _validate_script(self, config: dict, speaker_map: dict) -> list:
        """Full pre-render validation of scenes, dialog, and all file references.

        Checks:
          - scenes array exists and is non-empty
          - top-level numeric fields (pause_between_lines, pause_between_scenes)
          - each scene direction: type + instruction present, numeric fields valid
          - each dialog line: speaker known, text non-empty, numeric fields valid
          - every inline ref_file:'...' resolved against audio/ before rendering

        Returns a list of error strings (empty list = valid).
        """
        errors: list = []
        known_speakers = set(speaker_map.keys())
        skip_speakers  = {"direction", "instruction"}
        audio_ref_dir  = self.audio_ref_dir.resolve()
        _audio_exts    = (".wav", ".mp3", ".flac", ".ogg")

        # ── Top-level numeric fields ──────────────────────────────────────────
        for field in ("pause_between_lines", "pause_between_scenes"):
            val = config.get(field)
            if val is not None:
                try:
                    float(val)
                except (TypeError, ValueError):
                    errors.append(f"'{field}' must be a number, got: {val!r}")

        # ── Scenes ────────────────────────────────────────────────────────────
        scenes = config.get("scenes")
        if scenes is None:
            errors.append("Missing 'scenes' array")
            return errors
        if not isinstance(scenes, list):
            errors.append("'scenes' must be an array")
            return errors
        if not scenes:
            errors.append("'scenes' is empty — no content to render")
            return errors

        for scene_idx, scene in enumerate(scenes):
            title = scene.get("title", "")
            scene_label = f"Scene {scene_idx + 1}" + (f" '{title}'" if title else "")

            # ── Scene-level pause_after ───────────────────────────────────────
            val = scene.get("pause_after")
            if val is not None:
                try:
                    float(val)
                except (TypeError, ValueError):
                    errors.append(f"{scene_label}: 'pause_after' must be a number, got: {val!r}")

            # ── Directions ───────────────────────────────────────────────────
            directions = scene.get("directions") or scene.get("direction") or []
            if not isinstance(directions, list):
                errors.append(f"{scene_label}: 'direction'/'directions' must be an array")
            else:
                for d_idx, d in enumerate(directions):
                    d_type = str(d.get("type", "")).strip()
                    d_inst = str(d.get("instruction", "")).strip()
                    d_label = f"{scene_label} direction[{d_idx + 1}]"

                    if not d_type:
                        errors.append(f"{d_label}: missing 'type' field")
                    if d_type.lower() != "info" and not d_inst:
                        errors.append(f"{d_label}: 'instruction' is required for type '{d_type or '?'}'")

                    for field in ("volume", "steps", "cfg_scale"):
                        val = d.get(field)
                        if val is not None:
                            try:
                                float(val)
                            except (TypeError, ValueError):
                                errors.append(
                                    f"{d_label}: '{field}' must be a number, got: {val!r}"
                                )

            # ── Dialog lines ─────────────────────────────────────────────────
            dialog = scene.get("dialog", [])
            if not isinstance(dialog, list):
                errors.append(f"{scene_label}: 'dialog' must be an array")
                continue

            for line_idx, line in enumerate(dialog):
                speaker     = str(line.get("speaker", "")).strip()
                text        = str(line.get("text", "")).strip()
                line_label  = f"{scene_label} line[{line_idx + 1}]"

                if not speaker:
                    errors.append(f"{line_label}: missing 'speaker' field")
                elif speaker.lower() in skip_speakers:
                    continue  # direction/instruction placeholder lines are intentionally skipped
                elif speaker not in known_speakers:
                    errors.append(
                        f"{line_label}: speaker '{speaker}' is not defined in the speakers list"
                    )

                if not text:
                    errors.append(f"{line_label}: 'text' is empty")
                    continue

                # Numeric per-line fields
                for field in ("speed", "pause_after"):
                    val = line.get(field)
                    if val is not None:
                        try:
                            float(val)
                        except (TypeError, ValueError):
                            errors.append(
                                f"{line_label}: '{field}' must be a number, got: {val!r}"
                            )

                # Inline ref_file references — resolve against audio/ dir now
                for seg in self._parse_inline_effects(text):
                    ref_file = seg.get("ref_file")
                    if not ref_file:
                        continue

                    p = Path(ref_file.strip())
                    if p.is_absolute() or ".." in p.parts:
                        marker_type = "background-audio" if seg["type"] == "background" else "audio"
                        errors.append(
                            f"{line_label}: [{marker_type}:ref_file:'{ref_file}'] — "
                            f"path traversal not allowed"
                        )
                        continue

                    candidates = (
                        [p] if p.suffix
                        else [p.with_suffix(ext) for ext in _audio_exts] + [p]
                    )
                    found = False
                    for candidate in candidates:
                        resolved = (audio_ref_dir / candidate).resolve()
                        try:
                            resolved.relative_to(audio_ref_dir)
                        except ValueError:
                            continue
                        if resolved.exists():
                            found = True
                            break

                    if not found:
                        marker_type = "background-audio" if seg["type"] == "background" else "audio"
                        if p.suffix:
                            tried = str(p)
                        else:
                            tried = ", ".join(str(p.with_suffix(e)) for e in _audio_exts)
                        errors.append(
                            f"{line_label}: [{marker_type}:ref_file:'{ref_file}'] — "
                            f"not found in audio/  (tried: {tried})"
                        )

        return errors

    @staticmethod
    def _parse_inline_effects(text: str) -> list:
        """Split dialog text into plain-text, sequential-effect, and background segments.

        Supported marker syntax (single-quoted prompt, optional key:value pairs):

          Sequential (takes space in timeline, speech pauses):
            [audio:'<prompt>',duration:2.0,pause_before:0.5,pause_after:1.0,steps:20,cfg_scale:7.0]

          Background (mixed under speech, does not interrupt):
            [background-audio:'<prompt>',duration:8.0,pause_before:1.0,volume:0.3,steps:20,cfg_scale:7.0]
            duration=0 → auto-sized to match the line's speech length.

        Returns a list where each item is one of:
          {"type": "text",       "value": str}
          {"type": "effect",     "prompt": str, "duration": float,
           "pause_before": float, "pause_after": float,
           "steps": int, "cfg_scale": float}
          {"type": "background", "prompt": str, "duration": float,
           "pause_before": float, "volume": float,
           "steps": int, "cfg_scale": float}

        A line without any markers returns [{"type": "text", "value": text}].
        """
        # Collect all matches from all four patterns, tagged with their kind
        all_matches: list = []
        for m in _EFFECT_RE.finditer(text):
            all_matches.append(("effect", m))
        for m in _EFFECT_FILE_RE.finditer(text):
            all_matches.append(("effect_file", m))
        for m in _BGAUDIO_RE.finditer(text):
            all_matches.append(("background", m))
        for m in _BGAUDIO_FILE_RE.finditer(text):
            all_matches.append(("background_file", m))
        all_matches.sort(key=lambda x: x[1].start())

        segments: list = []
        last_end = 0
        for kind, m in all_matches:
            before = text[last_end:m.start()]
            if before.strip():
                segments.append({"type": "text", "value": before.strip()})

            params: dict = {}
            for kv in re.finditer(r",([a-z_]+):([\d.]+)", m.group(2)):
                params[kv.group(1)] = float(kv.group(2))

            if kind == "effect":
                segments.append({
                    "type":         "effect",
                    "prompt":       m.group(1),
                    "duration":     params.get("duration",     2.0),
                    "pause_before": params.get("pause_before", 0.0),
                    "pause_after":  params.get("pause_after",  0.0),
                    "volume":       params.get("volume",        1.0),
                    "fade_in":      params.get("fade_in",       0.0),
                    "fade_out":     params.get("fade_out",      0.0),
                    "steps":        int(params.get("steps",    20)),
                    "cfg_scale":    params.get("cfg_scale",    7.0),
                })
            elif kind == "effect_file":
                segments.append({
                    "type":         "effect",
                    "ref_file":     m.group(1),
                    "duration":     params.get("duration",     0.0),
                    "pause_before": params.get("pause_before", 0.0),
                    "pause_after":  params.get("pause_after",  0.0),
                    "volume":       params.get("volume",        1.0),
                    "fade_in":      params.get("fade_in",       0.0),
                    "fade_out":     params.get("fade_out",      0.0),
                })
            elif kind == "background":
                segments.append({
                    "type":         "background",
                    "prompt":       m.group(1),
                    "duration":     params.get("duration",     0.0),  # 0 = auto
                    "pause_before": params.get("pause_before", 0.0),
                    "volume":       params.get("volume",       0.3),
                    "fade_in":      params.get("fade_in",      0.0),
                    "fade_out":     params.get("fade_out",     0.0),
                    "steps":        int(params.get("steps",    20)),
                    "cfg_scale":    params.get("cfg_scale",    7.0),
                })
            else:  # background_file
                segments.append({
                    "type":         "background",
                    "ref_file":     m.group(1),
                    "duration":     params.get("duration",     0.0),  # 0 = auto
                    "pause_before": params.get("pause_before", 0.0),
                    "volume":       params.get("volume",       0.3),
                    "fade_in":      params.get("fade_in",      0.0),
                    "fade_out":     params.get("fade_out",     0.0),
                })
            last_end = m.end()

        after = text[last_end:]
        if after.strip():
            segments.append({"type": "text", "value": after.strip()})

        if not segments:
            segments.append({"type": "text", "value": text})
        return segments

    def _apply_scene_directions(self, scene_segs: list, directions: list,
                                errors: list) -> list:
        """Generate one Stable Audio background per direction entry and mix under the scene.

        Steps:
          1. Measure total scene duration from existing segments.
          2. Merge scene segments into a temp WAV.
          3. For each direction generate (or cache-hit) a background clip at that duration.
          4. Mix each background under the scene audio at the specified volume.
          5. Return the final mixed file as a one-element list.

        Falls back to the original scene_segs on any error.

        Direction JSON fields:
          type        string  e.g. "ambience", "music", "sfx"           (default "ambience")
          instruction string  natural-language sound description          (required)
          volume      float   background volume, 0.0–1.0                 (default 0.3)
          steps       int     Stable Audio diffusion steps               (default 20)
          cfg_scale   float   guidance scale                             (default 7.0)
        """
        from datetime import datetime as _dt

        duration = AudioProcessor.measure_segments_duration(scene_segs)
        if duration <= 0:
            return scene_segs

        ts = _dt.now().strftime("%Y%m%d-%H%M%S-%f")
        scene_path = self.outputs_dir / f"scene_bg_tmp_{ts}.wav"
        if not self.audio_processor.merge_audio_files(scene_segs, scene_path):
            return scene_segs

        current_path = scene_path
        for idx, direction in enumerate(directions):
            d_type = str(direction.get("type", "ambience")).strip()
            d_inst = str(direction.get("instruction", "")).strip()
            if not d_inst:
                continue
            prompt    = f"{d_type}:{d_inst}"
            bg_volume = float(direction.get("volume",    0.3))
            steps     = int(direction.get("steps",       20))
            cfg_scale = float(direction.get("cfg_scale", 7.0))
            try:
                log.info(f"  Direction [{d_type}] — {dim(d_inst[:60])}  {duration:.1f}s")
                bg_path = self.stable_audio_generator.generate(
                    prompt=prompt, duration=duration, steps=steps, cfg_scale=cfg_scale,
                )
                mixed_path = self.outputs_dir / f"scene_bg_{ts}_{idx}_{d_type}.wav"
                if AudioProcessor.mix_with_background(
                    current_path, Path(bg_path), bg_volume, mixed_path
                ):
                    current_path = mixed_path
            except Exception as exc:
                log.warning(f"  Direction '{prompt}': {exc}")
                errors.append(f"Direction '{prompt}': {exc}")

        return [current_path]

    def _resolve_audio_ref(self, seg: dict, duration: float, errors: list):
        """Resolve audio for an inline effect or background segment.

        Routing:
          seg has "ref_file"  → load the named file from the audio/ directory.
                                 Path traversal is blocked.  Error if not found.
          seg has "prompt"    → generate via Stable Audio at *duration* seconds.

        *duration* is only used for Stable Audio (ignored for file references).
        Returns a Path on success, None on failure.
        """
        ref_file = seg.get("ref_file")
        if ref_file:
            p = Path(ref_file.strip())
            if p.is_absolute() or ".." in p.parts:
                errors.append(f"ref_file path not allowed: '{ref_file}'")
                return None
            # Try with and without common audio extensions
            candidates = (
                [p] if p.suffix
                else [p.with_suffix(ext) for ext in (".wav", ".mp3", ".flac", ".ogg")] + [p]
            )
            for candidate in candidates:
                resolved = (self.audio_ref_dir / candidate).resolve()
                try:
                    resolved.relative_to(self.audio_ref_dir.resolve())
                except ValueError:
                    continue
                if resolved.exists():
                    size_kb = resolved.stat().st_size // 1024
                    log.info(f"  Audio ref → file: {resolved.name}  ({size_kb} KB)")
                    return resolved
            errors.append(f"ref_file not found in audio/: '{ref_file}'")
            return None

        # Stable Audio generation
        prompt = seg.get("prompt", "")
        if not prompt:
            errors.append("Audio segment has neither 'prompt' nor 'ref_file'")
            return None
        actual_duration = max(float(duration), 1.0)
        steps     = int(seg.get("steps",     20))
        cfg_scale = float(seg.get("cfg_scale", 7.0))
        try:
            path = self.stable_audio_generator.generate(
                prompt=prompt, duration=actual_duration, steps=steps, cfg_scale=cfg_scale,
            )
            return Path(path)
        except Exception as exc:
            errors.append(f"Audio '{prompt}': {exc}")
            return None

    def _mix_line_backgrounds(self, line_segs: list, background_segs: list,
                               errors: list) -> list:
        """Generate/load background audio and mix it under the line's speech segments.

        Each background_seg dict fields:
          prompt       str    Stable Audio text OR filename in audio/ dir
          duration     float  0 = auto-size to match speech duration
          pause_before float  silence prepended to the foreground before mixing
          volume       float  background amplitude scale (0.0–1.0)
          steps        int    Stable Audio diffusion steps
          cfg_scale    float  Stable Audio guidance scale

        Returns the mixed result as a one-element list, or the original line_segs
        on any fatal error.
        """
        from datetime import datetime as _dt

        speech_duration = AudioProcessor.measure_segments_duration(line_segs)
        if speech_duration <= 0:
            return line_segs

        ts = _dt.now().strftime("%Y%m%d-%H%M%S-%f")
        line_path = self.outputs_dir / f"line_bg_tmp_{ts}.wav"
        if not self.audio_processor.merge_audio_files(line_segs, line_path):
            return line_segs

        current_path = line_path
        for idx, bg_seg in enumerate(background_segs):
            duration     = bg_seg["duration"] if bg_seg["duration"] > 0 else speech_duration
            bg_volume    = float(bg_seg.get("volume",       0.3))
            pause_before = float(bg_seg.get("pause_before", 0.0))

            # Optionally pad the foreground with leading silence
            if pause_before > 0:
                padded_segs = [AudioProcessor.make_silence(pause_before), current_path]
                padded_path = self.outputs_dir / f"line_bg_pad_{ts}_{idx}.wav"
                if self.audio_processor.merge_audio_files(padded_segs, padded_path):
                    current_path = padded_path

            try:
                bg_path = self._resolve_audio_ref(bg_seg, duration, errors)
                if bg_path is None:
                    continue
                bg_id    = bg_seg.get("ref_file") or bg_seg.get("prompt", "?")
                fade_in  = float(bg_seg.get("fade_in",  0.0))
                fade_out = float(bg_seg.get("fade_out", 0.0))
                if fade_in > 0 or fade_out > 0:
                    bg_data, bg_sr = _sf.read(str(bg_path), dtype="float32", always_2d=False)
                    if bg_data.ndim == 2:
                        bg_data = bg_data.mean(axis=1)
                    log.info(f"  Background fade '{bg_id}'  in={fade_in:.2f}s  out={fade_out:.2f}s")
                    AudioProcessor.apply_fade(bg_data, bg_sr, fade_in, fade_out)
                    faded_path = self.outputs_dir / f"line_bg_faded_{ts}_{idx}.wav"
                    _sf.write(str(faded_path), bg_data, bg_sr)
                    bg_path = faded_path
                mixed_path = self.outputs_dir / f"line_bg_{ts}_{idx}.wav"
                log.info(f"  Background '{bg_id}'  vol={bg_volume:.2f}  speech={speech_duration:.1f}s")
                if AudioProcessor.mix_with_background(current_path, bg_path, bg_volume, mixed_path):
                    current_path = mixed_path
            except Exception as exc:
                bg_id = bg_seg.get("prompt") or bg_seg.get("ref_file", "?")
                log.warning(f"  Background seg {idx}: {exc}")
                errors.append(f"Background '{bg_id}': {exc}")

        return [current_path]

    # ── Abort helper ─────────────────────────────────────────────────────────

    def _abort_job(self) -> None:
        """Signal the currently running job to stop at the next safe checkpoint."""
        self._abort_event.set()
        log.warning("Abort requested — will stop after the current operation completes…")

    # ── Shutdown helper ───────────────────────────────────────────────────────

    @staticmethod
    def _shutdown_machine(delay_seconds: int = 60) -> None:
        """Schedule an OS shutdown *delay_seconds* from now.

        Windows : ``shutdown /s /t <delay>``  — cancel with ``shutdown /a``
        Linux   : ``shutdown -h +<minutes>``  — cancel with ``shutdown -c``
        macOS   : ``shutdown -h +<minutes>``  — cancel with ``kill <PID>``
        """
        import platform, subprocess
        system = platform.system()
        if system == "Windows":
            cmd        = ["shutdown", "/s", "/t", str(delay_seconds)]
            cancel_tip = "shutdown /a"
        else:
            # Linux / macOS: shutdown takes whole minutes; round up
            minutes    = max(1, (delay_seconds + 59) // 60)
            cmd        = ["shutdown", "-h", f"+{minutes}"]
            cancel_tip = "shutdown -c"

        log.warning(
            f"Shutdown scheduled in {delay_seconds}s "
            f"— to cancel open a terminal and run:  {cancel_tip}"
        )
        try:
            subprocess.run(cmd, check=True)
        except Exception as exc:
            log.warning(f"Shutdown command failed: {exc}")

    def _ui_run_script(self, json_config: str, shutdown_after: bool = False):
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

        raw_speakers = config.get("speakers") or config.get("speaker") or []
        scenes_raw   = config.get("scenes", [])
        log.info(
            f"Script parsed — "
            f"{bold(str(len(raw_speakers)))} speaker(s)  "
            f"{bold(str(len(scenes_raw)))} scene(s)"
        )

        # ── Validate speakers + full script (fast, no model loading) ─────────
        log.info("Validating speakers and scene structure…")
        speaker_map, validation_errors = self._validate_script_speakers(raw_speakers)

        # Run full scene/dialog/ref_file validation even if speaker errors exist,
        # so the user sees ALL problems at once before any rendering starts.
        validation_errors += self._validate_script(config, speaker_map)

        if validation_errors:
            log.error(f"Validation failed — {len(validation_errors)} error(s):")
            for e in validation_errors:
                log.error(f"  • {e}")
            msg = "✗ Script validation failed:\n" + "\n".join(
                f"  • {e}" for e in validation_errors
            )
            section_end("error")
            return _fail(msg)

        if not speaker_map:
            section_end("failed")
            return _fail("✗ No valid speakers defined in 'speakers'/'speaker' array")

        # Count renderable dialog lines for the progress display
        _n_lines = sum(
            1 for sc in scenes_raw for ln in sc.get("dialog", [])
            if ln.get("text", "").strip()
            and ln.get("speaker", "").strip().lower() not in ("direction", "instruction", "")
        )
        sp_types = {}
        for info in speaker_map.values():
            sp_types[info["type"]] = sp_types.get(info["type"], 0) + 1
        sp_summary = "  ".join(
            f"{cnt} {t}" for t, cnt in sorted(sp_types.items())
        )
        log.ok(
            f"Validation passed — "
            f"{bold(str(len(speaker_map)))} speaker(s) [{dim(sp_summary)}]  "
            f"{bold(str(len(scenes_raw)))} scene(s)  "
            f"{bold(str(_n_lines))} line(s)"
        )

        # ── Pause configuration ───────────────────────────────────────────────
        pause_between_lines  = float(config.get("pause_between_lines",  1.0))
        pause_between_scenes = float(config.get("pause_between_scenes", 2.0))
        log.info(
            f"Pauses — between lines: {pause_between_lines:.1f}s  "
            f"between scenes: {pause_between_scenes:.1f}s"
        )

        # ── Voice clone prompts — built lazily on first cache miss per speaker ──
        # If all lines for a speaker are already cached, the prompt (and model
        # load) is never needed, so we skip it entirely.
        voice_clone_prompts: dict = {}  # speaker_name → List[VoiceClonePromptItem] | None (failed)

        # ── Render dialog lines ───────────────────────────────────────────────
        audio_files: list = []
        errors: list = []
        pending_pause = None   # inter-scene pause, inserted before the next scene

        scenes     = list(scenes_raw)
        n_scenes   = len(scenes)
        line_count = 0  # running total across all scenes for progress display

        for scene_idx, scene in enumerate(scenes):
            if self._abort_event.is_set():
                log.warning(f"Job aborted before scene {scene_idx + 1} — stopping")
                break

            is_last_scene = (scene_idx == n_scenes - 1)
            scene_segs: list = []
            pending_line_pause = None  # intra-scene pause, inserted before the next line

            scene_title = scene.get("title", "")
            subsection(
                f"Scene {scene_idx + 1}/{n_scenes}"
                + (f"  —  {scene_title}" if scene_title else "")
            )

            valid_lines = [
                l for l in scene.get("dialog", [])
                if l.get("text", "").strip()
                and l.get("speaker", "").strip().lower() not in ("direction", "instruction", "")
            ]

            for line_idx, line in enumerate(valid_lines):
                if self._abort_event.is_set():
                    log.warning(
                        f"Job aborted at line {line_count + 1}/{_n_lines} "
                        f"— {line_count} line(s) already rendered"
                    )
                    break

                is_last_line = (line_idx == len(valid_lines) - 1)
                speaker_name = line.get("speaker", "").strip()
                text         = line.get("text", "").strip()
                # Accept both spellings; "connotation" is canonical
                connotation  = (line.get("connotation") or line.get("conotation") or "").strip() or None
                speed        = float(line.get("speed", 1.0))
                line_count  += 1

                log.info(
                    f"  [{line_count}/{_n_lines}]  {bold(speaker_name)}"
                    + (f"  {dim(connotation)}" if connotation else "")
                    + (f"  ×{speed:.2f}" if abs(speed - 1.0) >= 0.01 else "")
                    + f"  —  {dim(text[:70])}"
                )

                # Insert pending intra-scene pause before this line's audio
                if pending_line_pause is not None and scene_segs:
                    scene_segs.append(AudioProcessor.make_silence(pending_line_pause))
                pending_line_pause = None

                sp_info = speaker_map.get(speaker_name)

                # Parse all inline markers and split into sequential vs background
                effect_segs          = self._parse_inline_effects(text)
                background_segs_line = [s for s in effect_segs if s["type"] == "background"]
                sequential_segs      = [s for s in effect_segs if s["type"] != "background"]
                is_effect_only       = (
                    bool(sequential_segs)
                    and all(s["type"] == "effect" for s in sequential_segs)
                )

                if sp_info is None and not is_effect_only and not background_segs_line:
                    log.warning(f"  Unknown speaker: '{speaker_name}' — skipping line")
                    errors.append(f"Unknown speaker '{speaker_name}' (not in speakers list)")
                    continue

                try:
                    import soundfile as _sf
                    import numpy as _np

                    sp_type        = sp_info["type"] if sp_info else None
                    line_segs: list = []
                    line_had_audio  = False

                    for seg in sequential_segs:

                        if seg["type"] == "text":
                            if sp_info is None:
                                continue
                            seg_text = seg["value"]
                            if not seg_text:
                                continue

                            result = None
                            if sp_type == "clone":
                                # Build the clone prompt lazily: only on the first
                                # cache miss for this speaker.  If every line is
                                # already cached we never touch the model.
                                if speaker_name not in voice_clone_prompts:
                                    if not self.voice_cloner.is_speech_cached(
                                        seg_text,
                                        sp_info["ref_audio"],
                                        sp_info["ref_text"],
                                    ):
                                        try:
                                            log.info(f"  Building clone prompt — {bold(speaker_name)}…")
                                            voice_clone_prompts[speaker_name] = \
                                                self.voice_cloner.build_voice_clone_prompt(
                                                    ref_audio=sp_info["ref_audio"],
                                                    ref_text=sp_info["ref_text"],
                                                )
                                            log.ok(f"  Clone prompt ready — {bold(speaker_name)}")
                                        except Exception as _e:
                                            log.warning(f"  Clone prompt failed for {speaker_name}: {_e}")
                                            voice_clone_prompts[speaker_name] = None

                                prompt_items = voice_clone_prompts.get(speaker_name)
                                if prompt_items is not None:
                                    result = self.voice_cloner.generate_with_voice_clone_prompt(
                                        text=seg_text,
                                        prompt_items=prompt_items,
                                        ref_audio=sp_info["ref_audio"],
                                        ref_text=sp_info["ref_text"],
                                        voice_name=speaker_name,
                                    )
                                else:
                                    result = self.voice_cloner.generate_with_cloned_voice(
                                        text=seg_text,
                                        cloned_voice_config={
                                            "ref_audio": sp_info["ref_audio"],
                                            "ref_text":  sp_info["ref_text"],
                                        },
                                        voice_name=speaker_name,
                                    )
                            elif sp_type == "custom":
                                result = self.tts_generator.generate_tts(
                                    text=seg_text,
                                    language="auto",
                                    speaker=sp_info["speaker"],
                                    voice_name=speaker_name,
                                )
                            elif sp_type == "design":
                                instruct = sp_info["instructions"]
                                if connotation:
                                    instruct = f"{instruct}, {connotation}"
                                result = self.voice_designer.design_voice(
                                    text_input=seg_text,
                                    instructions=instruct,
                                    voice_name=speaker_name,
                                )

                            if result:
                                if abs(speed - 1.0) >= 0.001:
                                    arr, _ = AudioProcessor.time_stretch_array(Path(result), speed)
                                    line_segs.append(arr if arr is not None else Path(result))
                                else:
                                    line_segs.append(Path(result))
                                line_had_audio = True
                            else:
                                errors.append(f"No audio for [{speaker_name}]: {seg_text[:40]}")

                        elif seg["type"] == "effect":
                            if seg["pause_before"] > 0:
                                line_segs.append(AudioProcessor.make_silence(seg["pause_before"]))
                            try:
                                eff_id = seg.get("ref_file") or seg.get("prompt", "?")
                                log.info(f"  Loading sequential effect '{eff_id}'  vol={float(seg.get('volume', 1.0)):.2f}  dur={seg['duration']:.1f}s")
                                effect_path = self._resolve_audio_ref(
                                    seg, seg["duration"], errors,
                                )
                                if effect_path is not None:
                                    vol      = float(seg.get("volume",   1.0))
                                    dur      = float(seg.get("duration", 0))
                                    fade_in  = float(seg.get("fade_in",  0.0))
                                    fade_out = float(seg.get("fade_out", 0.0))
                                    needs_read = (abs(vol - 1.0) >= 0.001) or (dur > 0) or (fade_in > 0) or (fade_out > 0)
                                    if needs_read:
                                        data, sr = _sf.read(str(effect_path), dtype="float32",
                                                            always_2d=False)
                                        if data.ndim == 2:          # collapse stereo → mono
                                            data = data.mean(axis=1).astype(_np.float32)
                                        if dur > 0:
                                            max_samples = int(dur * sr)
                                            if len(data) > max_samples:
                                                log.info(f"    Trim '{eff_id}'  {len(data)/sr:.1f}s → {dur:.1f}s")
                                                data = data[:max_samples]
                                        if abs(vol - 1.0) >= 0.001:
                                            data = _np.clip(data * vol, -1.0, 1.0)
                                        if fade_in > 0 or fade_out > 0:
                                            log.info(f"    Fade '{eff_id}'  in={fade_in:.2f}s  out={fade_out:.2f}s")
                                            AudioProcessor.apply_fade(data, sr, fade_in, fade_out)
                                        line_segs.append(data)
                                    else:
                                        line_segs.append(Path(effect_path))
                                    line_had_audio = True
                            except Exception as exc:
                                seg_id = seg.get("prompt") or seg.get("ref_file", "?")
                                errors.append(f"Effect '{seg_id}': {exc}")
                            if seg["pause_after"] > 0:
                                line_segs.append(AudioProcessor.make_silence(seg["pause_after"]))

                    # Apply inline background audio under the line's speech
                    if background_segs_line:
                        if line_segs:
                            line_segs = self._mix_line_backgrounds(
                                line_segs, background_segs_line, errors
                            )
                        else:
                            # Background with no speech: insert each clip standalone
                            for bg_seg in background_segs_line:
                                dur = bg_seg["duration"] if bg_seg["duration"] > 0 else 3.0
                                if bg_seg.get("pause_before", 0) > 0:
                                    line_segs.append(
                                        AudioProcessor.make_silence(bg_seg["pause_before"])
                                    )
                                try:
                                    bg_path = self._resolve_audio_ref(
                                        bg_seg, dur, errors,
                                    )
                                    if bg_path is not None:
                                        line_segs.append(bg_path)
                                        line_had_audio = True
                                except Exception as exc:
                                    bg_id = bg_seg.get("prompt") or bg_seg.get("ref_file", "?")
                                    errors.append(f"Background '{bg_id}': {exc}")

                    if line_segs:
                        scene_segs.extend(line_segs)

                    if line_had_audio:
                        if "pause_after" in line:
                            pending_line_pause = float(line["pause_after"])
                        elif not is_last_line:
                            pending_line_pause = pause_between_lines
                    elif not is_effect_only and not background_segs_line and sp_info is not None:
                        errors.append(f"No audio for [{speaker_name}]: {text[:40]}")

                except Exception as exc:
                    log.error(f"Line [{speaker_name}]: {exc}")
                    errors.append(f"Error [{speaker_name}]: {exc}")

            # ── Scene directions — generate Stable Audio bg, mix under dialog ──
            # Accept both "directions" (canonical) and "direction" (used in sample files)
            directions = scene.get("directions") or scene.get("direction") or []
            if directions and scene_segs:
                non_info = [d for d in directions if str(d.get("type", "")).lower() != "info"]
                if non_info:
                    log.info(
                        f"  Applying {len(non_info)} direction(s) to scene — "
                        f"generating Stable Audio background…"
                    )
                scene_segs = self._apply_scene_directions(scene_segs, directions, errors)

            # Insert inter-scene pause BEFORE this scene's audio
            if pending_pause is not None and audio_files:
                audio_files.append(AudioProcessor.make_silence(pending_pause))
            pending_pause = None

            audio_files.extend(scene_segs)

            # Carry inter-scene pause (last line's pause_after or scene.pause_after)
            if scene_segs and not is_last_scene:
                pending_pause = (
                    pending_line_pause
                    if pending_line_pause is not None
                    else float(scene.get("pause_after", pause_between_scenes))
                )

        # ── Merge ─────────────────────────────────────────────────────────────
        was_aborted = self._abort_event.is_set()

        if not audio_files:
            msg = ("⏹ Aborted — no audio was rendered yet" if was_aborted
                   else "✗ No audio was generated.\n" + "\n".join(errors))
            section_end("aborted" if was_aborted else "failed")
            return _fail(msg)

        ts       = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_file = self.outputs_dir / f"script_{ts}.wav"
        log.info(f"Final merge — {bold(str(len(audio_files)))} segment(s)  →  {out_file.name}")
        if not self.audio_processor.merge_audio_files(audio_files, out_file):
            section_end("failed")
            return _fail("✗ Failed to merge audio lines")

        elapsed = time.time() - t0
        if was_aborted:
            log.warning(
                f"Script aborted — "
                f"{bold(str(line_count))}/{_n_lines} line(s) rendered  "
                f"{bold(str(len(audio_files)))} segment(s) merged"
            )
        else:
            log.ok(
                f"Script complete — "
                f"{bold(str(line_count))} line(s) rendered  "
                f"{bold(str(len(audio_files)))} segment(s) merged"
            )
        if errors:
            log.warning(f"  {len(errors)} non-fatal error(s) during render:")
            for e in errors:
                log.warning(f"    • {e}")
        log.timing(f"Elapsed: {elapsed:.1f}s  →  {dim(str(out_file))}")
        section_end("aborted" if was_aborted else "done")

        prefix = f"⏹ Aborted ({line_count}/{_n_lines} lines)" if was_aborted else f"✓ {line_count} line(s)"
        status_lines = [f"{prefix} · {len(audio_files)} segment(s) → {out_file.name}  ({elapsed:.1f}s)"]
        if errors:
            status_lines.append(f"⚠ {len(errors)} item(s) skipped:")
            status_lines.extend(f"  • {e}" for e in errors)

        if shutdown_after:
            self._shutdown_machine(delay_seconds=60)
            status_lines.append("")
            status_lines.append("⏻  Shutdown scheduled in 60 s — cancel: shutdown /a  (Win) · shutdown -c  (Linux)")

        return str(out_file), "\n".join(status_lines)

    def _ui_design_voice(self, text_input: str, instructions: str, speed: float = 1.0):
        """UI callback for voice design (text mode)."""
        section("Design Voice")
        log.info(f"instructions: {dim(instructions[:80])}  speed={speed:.2f}")
        log.info(f"text: {dim(text_input[:80])}")
        t0 = time.time()
        try:
            result = self.design_voice(text_input=text_input, instructions=instructions)
            if result is None:
                log.error("Voice design returned no audio — model may not be loaded or failed silently")
                section_end("failed")
                return None, "✗ Design failed — model returned no audio"
            result = self._apply_speed_to_output(str(result), speed)
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

    def _ui_generate_stable_audio(self, prompt: str, duration: float,
                                   steps: int, cfg_scale: float):
        """UI callback for Stable Audio sound-effects generation."""
        section("Stable Audio")
        log.info(f"prompt: {dim(prompt[:100])}  duration={duration}s  steps={int(steps)}  cfg={cfg_scale}")
        t0 = time.time()
        try:
            if not prompt or not prompt.strip():
                section_end("skipped")
                return None, "✗ Please enter a prompt describing the sound to generate"
            result = self.stable_audio_generator.generate(
                prompt=prompt.strip(),
                duration=float(duration),
                steps=int(steps),
                cfg_scale=float(cfg_scale),
            )
            elapsed = time.time() - t0
            log.ok(f"Stable Audio generated: {Path(result).name}")
            log.timing(f"Elapsed: {elapsed:.1f}s")
            section_end("done")
            return str(result), f"✓ Done ({elapsed:.1f}s)  →  {Path(result).name}"
        except Exception as e:
            log.error(f"Stable Audio error: {e}")
            section_end("error")
            return None, f"✗ Error: {e}"

    def _ui_save_hf_token(self, token: str) -> str:
        """UI callback — persist HuggingFace token to config.json and apply it."""
        token = (token or "").strip()
        try:
            existing: dict = {}
            if self.config_file.exists():
                with open(self.config_file, encoding="utf-8") as f:
                    existing = json.load(f)
            existing["hf_token"] = token
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            self._saved_config = existing
            if token:
                self._apply_hf_token(token)
                log.ok("HuggingFace token saved and applied")
                return "✓ Token saved and applied — gated models are now accessible"
            else:
                log.info("HuggingFace token cleared")
                return "✓ Token cleared"
        except Exception as e:
            log.error(f"Failed to save HF token: {e}")
            return f"✗ Failed to save token: {e}"

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
