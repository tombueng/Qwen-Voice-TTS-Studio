import gradio as gr
import torch
import soundfile as sf
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from pydub import AudioSegment
import tempfile
import shutil
import subprocess
from transformers.utils.import_utils import is_flash_attn_2_available
import socket

os.environ['HF_HOME'] = str(Path(__file__).parent / "models")

class QwenVoiceGUI:
    def __init__(self):
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.cloned_voices_dir = Path("cloned_voices")
        self.cloned_voices_dir.mkdir(exist_ok=True)
        self.voiceinputs_dir = Path("voiceinputs")
        self.voiceinputs_dir.mkdir(exist_ok=True)
        self.downloads_dir = self.output_dir / "_downloads"
        self.downloads_dir.mkdir(exist_ok=True)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.voicesamples_dir = Path("voicesamples")
        self.voicesamples_dir.mkdir(exist_ok=True)
        self.designed_voices_dir = Path("designed_voices")
        self.designed_voices_dir.mkdir(exist_ok=True)
        
        self.custom_model = None
        self.base_model = None
        self.design_model = None
        self.asr_model = None
        self.asr_model_id = None
        
        self.audio_history = []
        self.cloned_voices = self.load_cloned_voices()
        self.migrate_cloned_voice_paths()
        self.designed_voices = self.load_designed_voices()
        self.main_voices = self.load_main_voices()
        
        self.render_device = "Auto"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.voice_personas = self.get_voice_personas()

    def sanitize_filename_part(self, name: str) -> str:
        name = (name or "").strip()
        if not name:
            return "audio"
        cleaned = "".join(c for c in name if c.isalnum() or c in ("_", "-", " ")).strip()
        cleaned = cleaned.replace(" ", "_")
        return cleaned or "audio"

    def make_output_filename(self, prefix: str, timestamp: str, ext: str = "wav") -> str:
        safe_prefix = self.sanitize_filename_part(prefix)
        safe_ext = (ext or "wav").lstrip(".")
        return f"{safe_prefix}_{timestamp}.{safe_ext}"

    def prepare_download_file(self, filename: str, fmt: str):
        if not filename:
            return gr.update(value=None, visible=False), "Please select a file"

        src_path = (self.output_dir / filename).resolve()
        if not src_path.exists():
            return gr.update(value=None, visible=False), "✗ File not found"

        fmt = (fmt or "wav").lower()
        if fmt not in ("wav", "mp3"):
            fmt = "wav"

        if fmt == "wav":
            return gr.update(value=str(src_path), visible=True), f"✓ File ready: {filename}"

        dst_name = f"{src_path.stem}.mp3"
        dst_path = (self.downloads_dir / dst_name).resolve()

        try:
            audio = AudioSegment.from_wav(str(src_path))
            audio.export(str(dst_path), format="mp3", bitrate="192k")
            return gr.update(value=str(dst_path), visible=True), f"✓ File ready: {dst_name}"
        except Exception as e:
            return gr.update(value=None, visible=False), f"✗ Error: {str(e)}"

    def update_library_tiles(self):
        files = self.get_audio_library_files()
        file_names = [f["name"] for f in files][:20]

        label_updates = []
        name_updates = []
        check_updates = []
        audio_updates = []

        dl_wav_updates = []
        dl_mp3_updates = []
        dl_file_updates = []

        for i in range(20):
            if i < len(file_names):
                name = file_names[i]
                audio_path = str((self.output_dir / name).resolve())
                label_updates.append(gr.update(value=f"**{name}**", visible=True))
                name_updates.append(gr.update(value=name, visible=False))
                check_updates.append(gr.update(value=False, visible=True))
                audio_updates.append(gr.update(value=audio_path, visible=True))

                dl_wav_updates.append(gr.update(visible=True))
                dl_mp3_updates.append(gr.update(visible=True))
                dl_file_updates.append(gr.update(value=None, visible=False))
            else:
                label_updates.append(gr.update(value="", visible=False))
                name_updates.append(gr.update(value="", visible=False))
                check_updates.append(gr.update(value=False, visible=False))
                audio_updates.append(gr.update(value=None, visible=False))

                dl_wav_updates.append(gr.update(visible=False))
                dl_mp3_updates.append(gr.update(visible=False))
                dl_file_updates.append(gr.update(value=None, visible=False))

        return file_names, *label_updates, *name_updates, *check_updates, *audio_updates, *dl_wav_updates, *dl_mp3_updates, *dl_file_updates

    def delete_selected_tile_files(self, file_names, selected_flags):
        try:
            if not file_names:
                return "No files available", None

            selected = []
            for i, name in enumerate(file_names):
                if i < len(selected_flags) and selected_flags[i]:
                    selected.append(name)

            status, _ = self.delete_selected_files(selected)
            updated = self.get_audio_library_files()
            return status, [f["name"] for f in updated]
        except Exception as e:
            return f"✗ Error: {str(e)}", None

    def parse_conversation_script(self, script: str):
        script = script or ""
        lines = script.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        turns = []
        current_slot = None
        buffer = []

        def flush():
            nonlocal buffer
            if current_slot and buffer:
                text = "\n".join(buffer).strip()
                if text:
                    turns.append((current_slot, text))
            buffer = []

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            lower = line.lower()
            if lower in ("[voice1]", "[voice2]", "[voice3]"):
                flush()
                current_slot = lower.strip("[]")
                continue

            if line.startswith("[") and line.endswith("]"):
                return None, f"Unknown voice tag: {line}. Use [Voice1], [Voice2], [Voice3]."

            if current_slot is None:
                return None, "Conversation must start with a voice tag like [Voice1]."

            buffer.append(raw)

        flush()

        if not turns:
            return None, "No conversation lines found. Add at least one block under [Voice1]/[Voice2]/[Voice3]."

        return turns, None

    def synthesize_text(self, text: str, language: str, speaker: str, instruct: str = "", progress=None):
        if progress is None:
            progress = gr.Progress()

        voice_info = self.voice_personas.get(speaker, {})

        if voice_info:
            model_type = voice_info.get("model", "custom")

            if model_type == "custom":
                status = self.load_custom_model()
                if "Error" in status:
                    return None, None, status
                wavs, sr = self.custom_model.generate_custom_voice(
                    text=text,
                    language=language if language != "Auto" else None,
                    speaker=voice_info.get("speaker", speaker),
                    instruct=instruct if (instruct or "").strip() else None,
                )
                return wavs, sr, None

            status = self.load_design_model()
            if "Error" in status:
                return None, None, status

            persona_instruct = voice_info.get("instruct", "")
            combined_instruct = persona_instruct
            if (instruct or "").strip():
                combined_instruct = f"{persona_instruct}. {instruct}"

            wavs, sr = self.design_model.generate_voice_design(
                text=text,
                language=language if language != "Auto" else None,
                instruct=combined_instruct,
            )
            return wavs, sr, None

        if speaker in self.cloned_voices:
            status = self.load_base_model()
            if "Error" in status:
                return None, None, status

            voice_data = self.cloned_voices[speaker]
            prompt_items = self.base_model.create_voice_clone_prompt(
                ref_audio=self.resolve_audio_path(voice_data.get("ref_audio")),
                ref_text=voice_data.get("ref_text"),
                x_vector_only_mode=not voice_data.get("ref_text"),
            )
            wavs, sr = self.base_model.generate_voice_clone(
                text=text,
                language=language if language != "Auto" else None,
                voice_clone_prompt=prompt_items,
                instruct=instruct if (instruct or "").strip() else None,
            )
            return wavs, sr, None

        if speaker in self.designed_voices:
            status = self.load_design_model()
            if "Error" in status:
                return None, None, status

            voice_data = self.designed_voices[speaker]
            wavs, sr = self.design_model.generate_voice_design(
                text=text,
                language=language if language != "Auto" else None,
                instruct=voice_data.get("instruct", ""),
            )
            return wavs, sr, None

        return None, None, f"✗ Voice not found: {speaker}"

    def generate_conversation(self, voice1, voice2, voice3, script, language, save_name=None, progress=gr.Progress()):
        try:
            turns, err = self.parse_conversation_script(script)
            if err:
                return None, err

            slot_map = {
                "voice1": voice1,
                "voice2": voice2,
                "voice3": voice3,
            }

            if any(slot_map[k] is None for k in ("voice1", "voice2", "voice3")):
                pass

            parts = []
            sample_rate = None

            total = len(turns)
            for i, (slot, text) in enumerate(turns, 1):
                speaker = slot_map.get(slot)
                if not speaker:
                    return None, f"No voice selected for [{slot.capitalize()}]."

                progress(i / (total + 1), desc=f"Generating {speaker}... ({i}/{total})")
                wavs, sr, status = self.synthesize_text(text=text, language=language, speaker=speaker, instruct="")
                if status:
                    return None, status

                if sr is None:
                    return None, "✗ Failed to generate audio"

                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    return None, "✗ Sample rate mismatch between turns"

                chunk = wavs[0]
                parts.append(chunk)

                silence_len = int(sample_rate * 0.25)
                parts.append(np.zeros(silence_len, dtype=chunk.dtype))

            if not parts:
                return None, "No audio generated"

            merged = np.concatenate(parts)

            progress((total + 1) / (total + 1), desc="Saving audio...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.make_output_filename(save_name or "conversation", timestamp, "wav")
            filepath = self.output_dir / filename

            sf.write(str(filepath), merged, sample_rate)

            self.audio_history.append({
                "filename": filename,
                "filepath": str(filepath),
                "timestamp": timestamp,
                "type": "Conversation"
            })

            progress(1.0, desc="Complete!")
            return str(filepath), f"✓ Conversation generated: {filename}"

        except Exception as e:
            return None, f"✗ Error: {str(e)}"

    def set_render_device(self, mode: str):
        mode = (mode or "Auto").strip()
        if mode not in ("Auto", "GPU", "CPU"):
            mode = "Auto"

        self.render_device = mode

        if mode == "CPU":
            self.device = "cpu"
            self.dtype = torch.float32
        elif mode == "GPU":
            if torch.cuda.is_available():
                self.device = "cuda:0"
                self.dtype = torch.bfloat16
            else:
                self.device = "cpu"
                self.dtype = torch.float32
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.custom_model = None
        self.base_model = None
        self.design_model = None
        self.asr_model = None
        self.asr_model_id = None

        return self.get_system_info_markdown()

    def download_asr_models(self, download_forced_aligner: bool = False):
        try:
            from huggingface_hub import snapshot_download

            self.models_dir.mkdir(exist_ok=True)
            asr_local_dir = self.models_dir / "Qwen3-ASR-1.7B"
            snapshot_download(
                repo_id="Qwen/Qwen3-ASR-1.7B",
                local_dir=str(asr_local_dir),
                local_dir_use_symlinks=False,
            )

            if download_forced_aligner:
                aligner_local_dir = self.models_dir / "Qwen3-ForcedAligner-0.6B"
                snapshot_download(
                    repo_id="Qwen/Qwen3-ForcedAligner-0.6B",
                    local_dir=str(aligner_local_dir),
                    local_dir_use_symlinks=False,
                )

            self.asr_model = None
            self.asr_model_id = None
            return "✓ ASR model(s) downloaded to ./models"
        except Exception as e:
            return f"✗ Error downloading ASR model(s): {str(e)}"

    def load_asr_model(self, model_choice: str, use_forced_aligner: bool = False):
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

    def transcribe_audio(
        self,
        audio_file,
        model_choice: str,
        language: str,
        save_prefix: str,
        recreate_voice: bool,
        tts_voice: str,
        tts_language: str,
        return_timestamps: bool,
        progress=gr.Progress(),
    ):
        try:
            if audio_file is None:
                return "", "", None, "Please provide audio (upload or microphone)."

            progress(0.1, desc="Loading ASR model...")
            status = self.load_asr_model(model_choice=model_choice, use_forced_aligner=return_timestamps)
            if status.startswith("✗"):
                return "", "", None, status

            lang = None
            if (language or "Auto").strip() not in ("", "Auto"):
                lang = language

            progress(0.5, desc="Transcribing...")
            results = self.asr_model.transcribe(
                audio=audio_file,
                language=lang,
                return_time_stamps=bool(return_timestamps),
            )
            text = (results[0].text if results else "") or ""

            ts_text = ""
            ts_data = None
            if return_timestamps and results:
                ts_data = getattr(results[0], "time_stamps", None)
                if ts_data:
                    try:
                        ts_text = json.dumps(ts_data, ensure_ascii=False, indent=2)
                    except Exception:
                        ts_text = str(ts_data)

            progress(0.75, desc="Saving transcript...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = Path(audio_file).stem if audio_file else "audio"
            prefix = save_prefix.strip() if (save_prefix or "").strip() else f"asr_{base}"
            filename = self.make_output_filename(prefix, timestamp, "txt")
            filepath = (self.output_dir / filename).resolve()
            filepath.parent.mkdir(exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

            ts_path = None
            if return_timestamps:
                ts_filename = self.make_output_filename(f"{prefix}_timestamps", timestamp, "json")
                ts_path = (self.output_dir / ts_filename).resolve()
                with open(ts_path, "w", encoding="utf-8") as f:
                    f.write(ts_text or "[]")

            tts_out = None
            extra = f"✓ Transcript saved: {filepath}"
            if return_timestamps:
                if ts_path:
                    extra = extra + f"\n✓ Timestamps saved: {ts_path}"
                if not ts_text.strip():
                    extra = extra + "\n✗ No timestamps returned (check that Forced Aligner is downloaded and 'Return timestamps' is enabled)."

            if recreate_voice:
                if not text.strip():
                    return text, ts_text, None, extra + "\n✗ Cannot recreate voice from empty transcript."

                progress(0.85, desc="Recreating voice...")
                wavs, sr, err = self.synthesize_text(
                    text=text,
                    language=tts_language,
                    speaker=tts_voice,
                    instruct="",
                    progress=progress,
                )
                if err:
                    return text, ts_text, None, extra + f"\n{err}"

                ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_filename = self.make_output_filename(f"asr_revoice_{tts_voice}", ts2, "wav")
                audio_path = (self.output_dir / audio_filename).resolve()
                sf.write(str(audio_path), wavs[0], sr)
                tts_out = str(audio_path)
                extra = extra + f"\n✓ Re-voiced audio saved: {audio_path}"

            progress(1.0, desc="Complete!")
            return text, ts_text, tts_out, extra
        except Exception as e:
            return "", "", None, f"✗ Error: {str(e)}"

    def _get_attn_implementation(self):
        if self.device.startswith("cuda") and is_flash_attn_2_available():
            return "flash_attention_2"
        if self.device.startswith("cuda"):
            return "sdpa"
        return "eager"

    def get_system_info_markdown(self):
        return """
                    ### System Info:
                    - Render Device: {}
                    - Device: {}
                    - Attention: {}
                    - PyTorch: {}
                    - CUDA Available: {}
                    - FlashAttn2 Available: {}
                    """.format(
            self.render_device,
            self.device,
            self._get_attn_implementation(),
            torch.__version__,
            torch.cuda.is_available(),
            is_flash_attn_2_available(),
        )
        
    def get_voice_personas(self):
        # Voice personas with their model type and instructions
        return {
            # Built-in speakers (use CustomVoice model)
            "Vivian": {"age": 28, "gender": "Female", "desc": "Warm female voice, versatile for various content", "model": "custom", "speaker": "vivian"},
            "Serena": {"age": 32, "gender": "Female", "desc": "Professional female voice, clear and authoritative", "model": "custom", "speaker": "serena"},
            "Ono Anna": {"age": 25, "gender": "Female", "desc": "Youthful female voice, energetic and friendly", "model": "custom", "speaker": "ono_anna"},
            "Sohee": {"age": 30, "gender": "Female", "desc": "Expressive female voice, dynamic and engaging", "model": "custom", "speaker": "sohee"},
            "Ryan": {"age": 35, "gender": "Male", "desc": "Technical male voice, precise for explanations", "model": "custom", "speaker": "ryan"},
            "Aiden": {"age": 27, "gender": "Male", "desc": "Friendly male voice, approachable and warm", "model": "custom", "speaker": "aiden"},
            "Dylan": {"age": 34, "gender": "Male", "desc": "Expressive male voice, dynamic for storytelling", "model": "custom", "speaker": "dylan"},
            "Eric": {"age": 42, "gender": "Male", "desc": "Trustworthy male voice, deep and authoritative", "model": "custom", "speaker": "eric"},
            "Uncle Fu": {"age": 58, "gender": "Male", "desc": "Mature male voice, wise and experienced", "model": "custom", "speaker": "uncle_fu"},
            
            # Designed personas (use VoiceDesign model)
            "Emma": {"age": 36, "gender": "Female", "desc": "Warm long-form narrator, 180-210 Hz, perfect for audiobooks and documentaries", "model": "design", "instruct": "Female, 36 years old, warm and rich voice, honey-toned with slight husky undertone, slow pacing, emotionally warm but controlled"},
            "Lily": {"age": 25, "gender": "Female", "desc": "Friendly digital assistant, 220-250 Hz, bright and energetic for UI interactions", "model": "design", "instruct": "Female, 25 years old, bright and clear voice, sparkly and energetic, quick pacing, cheerful and helpful tone"},
            "Sophia": {"age": 44, "gender": "Female", "desc": "Professional analyst, 190-220 Hz, authoritative for business and technical content", "model": "design", "instruct": "Female, 44 years old, firm authoritative voice, neutral and clinical precision, measured pacing, composed and factual tone"},
            "Ava": {"age": 32, "gender": "Female", "desc": "Expressive story performer, 200-240 Hz, theatrical and character-capable", "model": "design", "instruct": "Female, 32 years old, theatrical expressive voice, adaptable resonance, dynamic pacing, full emotional range"},
            "Grace": {"age": 52, "gender": "Female", "desc": "Empathetic mentor, 170-195 Hz, nurturing and reassuring for educational content", "model": "design", "instruct": "Female, 52 years old, warm deep motherly voice, velvety with mature richness, slow patient pacing, nurturing and reassuring tone"},
            "Mia": {"age": 21, "gender": "Female", "desc": "Youthful social voice, 235-270 Hz, playful and contemporary for social media", "model": "design", "instruct": "Female, 21 years old, playful bubbly voice, bright and youthful, fast energetic pacing, cheerful and casual tone"},
            "Claire": {"age": 39, "gender": "Female", "desc": "British RP elegant voice, 195-225 Hz, sophisticated for luxury brands", "model": "design", "instruct": "Female, 39 years old, British RP elegant voice, sophisticated and polished, measured refined pacing, composed elegance"},
            "Nina": {"age": 30, "gender": "Female", "desc": "Technical explainer, 205-230 Hz, precise and intelligent for tutorials", "model": "design", "instruct": "Female, 30 years old, clear direct voice, precise and analytical, slightly fast pacing, focused and competent tone"},
            "Rachel": {"age": 47, "gender": "Female", "desc": "News anchor, 185-210 Hz, authoritative and credible for broadcasts", "model": "design", "instruct": "Female, 47 years old, strong authoritative voice, professional and commanding, steady measured pacing, composed objectivity"},
            "Olivia": {"age": 28, "gender": "Female", "desc": "Soft ASMR & wellness voice, 195-220 Hz, calming for meditation and relaxation", "model": "design", "instruct": "Female, 28 years old, soft breathy voice, gentle and soothing, very slow pacing, calming and peaceful tone"},
            "James": {"age": 42, "gender": "Male", "desc": "Trustworthy documentary narrator, 105-125 Hz, deep and authoritative", "model": "design", "instruct": "Male, 42 years old, deep warm baritone voice, rich resonance, measured pacing, trustworthy and knowledgeable tone"},
            "Ethan": {"age": 27, "gender": "Male", "desc": "Friendly conversational assistant, 115-140 Hz, approachable and warm", "model": "design", "instruct": "Male, 27 years old, warm friendly voice, approachable and natural, conversational pacing, helpful and engaged tone"},
            "Michael": {"age": 54, "gender": "Male", "desc": "Corporate authority voice, 95-115 Hz, commanding for executive communications", "model": "design", "instruct": "Male, 54 years old, deep commanding voice, powerful executive presence, deliberate pacing, formal authoritative tone"},
            "Lucas": {"age": 34, "gender": "Male", "desc": "Expressive story voice, 110-135 Hz, dynamic and theatrical for audiobooks", "model": "design", "instruct": "Male, 34 years old, dynamic theatrical voice, full range capability, variable pacing, emotionally expressive tone"},
            "David": {"age": 48, "gender": "Male", "desc": "Calm educational guide, 105-125 Hz, patient and encouraging for e-learning", "model": "design", "instruct": "Male, 48 years old, calm reassuring voice, patient and supportive, unhurried pacing, encouraging and understanding tone"},
            "Noah": {"age": 20, "gender": "Male", "desc": "Youthful casual voice, 125-150 Hz, energetic for gaming and social content", "model": "design", "instruct": "Male, 20 years old, casual energetic voice, bright and youthful, fast pacing, relaxed and enthusiastic tone"},
            "Henry": {"age": 62, "gender": "Male", "desc": "British gentleman narrator, 95-115 Hz, distinguished for heritage content", "model": "design", "instruct": "Male, 62 years old, British RP distinguished voice, deep resonant, measured dignified pacing, cultivated warm gravitas"},
            "Thomas": {"age": 58, "gender": "Male", "desc": "Serious news & authority voice, 90-110 Hz, commanding for breaking news", "model": "design", "instruct": "Male, 58 years old, very deep authoritative voice, commanding gravitas, measured serious pacing, restrained factual tone"},
            "Leo": {"age": 38, "gender": "Male", "desc": "Smooth radio & podcast host, 105-130 Hz, charismatic and engaging", "model": "design", "instruct": "Male, 38 years old, warm charismatic voice, smooth radio quality, flowing rhythmic pacing, engaging and personable tone"}
        }
    
    def load_cloned_voices(self):
        voices_file = self.cloned_voices_dir / "voices.json"
        if voices_file.exists():
            try:
                if voices_file.stat().st_size == 0:
                    return {}
                with open(voices_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except Exception:
                try:
                    with open(voices_file, 'w', encoding='utf-8') as f:
                        json.dump({}, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                return {}
        return {}
    
    def save_cloned_voices(self):
        voices_file = self.cloned_voices_dir / "voices.json"
        with open(voices_file, 'w', encoding='utf-8') as f:
            json.dump(self.cloned_voices, f, indent=2, ensure_ascii=False)

    def resolve_audio_path(self, audio_path: str) -> str:
        if not audio_path:
            return audio_path
        p = Path(audio_path)
        if p.is_absolute():
            return str(p)
        return str((Path.cwd() / p).resolve())

    def migrate_cloned_voice_paths(self):
        # Make cloned voice ref_audio portable by pointing it into voiceinputs/
        # if the user already placed the files there.
        changed = False
        for voice_name, voice_data in (self.cloned_voices or {}).items():
            if not isinstance(voice_data, dict):
                continue
            ref_audio = voice_data.get("ref_audio")
            if not ref_audio:
                continue

            ref_basename = Path(ref_audio).name
            candidate = self.voiceinputs_dir / ref_basename
            if candidate.exists():
                portable_path = str((self.voiceinputs_dir / ref_basename).as_posix())
                if voice_data.get("ref_audio") != portable_path:
                    voice_data["ref_audio"] = portable_path
                    changed = True

        if changed:
            self.save_cloned_voices()
    
    def load_designed_voices(self):
        voices_file = self.designed_voices_dir / "voices.json"
        if voices_file.exists():
            try:
                if voices_file.stat().st_size == 0:
                    return {}
                with open(voices_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except Exception:
                try:
                    with open(voices_file, 'w', encoding='utf-8') as f:
                        json.dump({}, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                return {}
        return {}
    
    def save_designed_voices(self):
        voices_file = self.designed_voices_dir / "voices.json"
        with open(voices_file, 'w', encoding='utf-8') as f:
            json.dump(self.designed_voices, f, indent=2, ensure_ascii=False)
    
    def load_main_voices(self):
        voices_file = Path("main_voices.json")
        if voices_file.exists():
            with open(voices_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {"voices": list(self.voice_personas.keys())}
        return {"voices": list(self.get_voice_personas().keys())}
    
    def save_main_voices(self):
        voices_file = Path("main_voices.json")
        with open(voices_file, 'w', encoding='utf-8') as f:
            json.dump(self.main_voices, f, indent=2, ensure_ascii=False)
    
    def load_custom_model(self):
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
    
    def get_supported_speakers(self):
        return self.main_voices.get("voices", list(self.voice_personas.keys()))
    
    def get_supported_languages(self):
        if self.custom_model is None:
            self.load_custom_model()
        if self.custom_model:
            return self.custom_model.get_supported_languages()
        return ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]
    
    def generate_tts(self, text, language, speaker, instruct, save_name=None, progress=gr.Progress()):
        try:
            if not text.strip():
                return None, "Please enter text to synthesize"
            
            # Check voice type: persona, cloned, or designed
            voice_info = self.voice_personas.get(speaker, {})
            
            if voice_info:
                # Built-in persona voice
                model_type = voice_info.get("model", "custom")
                
                progress(0.2, desc="Loading model...")
                
                if model_type == "custom":
                    # Use CustomVoice model for built-in speakers
                    status = self.load_custom_model()
                    if "Error" in status:
                        return None, status
                    
                    progress(0.5, desc="Generating speech...")
                    wavs, sr = self.custom_model.generate_custom_voice(
                        text=text,
                        language=language if language != "Auto" else None,
                        speaker=voice_info.get("speaker", speaker),
                        instruct=instruct if instruct.strip() else None,
                    )
                else:
                    # Use VoiceDesign model for designed personas
                    status = self.load_design_model()
                    if "Error" in status:
                        return None, status
                    
                    progress(0.5, desc="Generating speech...")
                    # Combine persona instructions with user instructions
                    persona_instruct = voice_info.get("instruct", "")
                    combined_instruct = persona_instruct
                    if instruct.strip():
                        combined_instruct = f"{persona_instruct}. {instruct}"
                    
                    wavs, sr = self.design_model.generate_voice_design(
                        text=text,
                        language=language if language != "Auto" else None,
                        instruct=combined_instruct,
                    )
            elif speaker in self.cloned_voices:
                # Cloned voice - use Base model
                progress(0.2, desc="Loading model...")
                status = self.load_base_model()
                if "Error" in status:
                    return None, status
                
                voice_data = self.cloned_voices[speaker]
                progress(0.4, desc="Loading voice profile...")
                prompt_items = self.base_model.create_voice_clone_prompt(
                    ref_audio=self.resolve_audio_path(voice_data.get("ref_audio")),
                    ref_text=voice_data.get("ref_text"),
                    x_vector_only_mode=not voice_data.get("ref_text"),
                )
                
                progress(0.5, desc="Generating speech...")
                wavs, sr = self.base_model.generate_voice_clone(
                    text=text,
                    language=language if language != "Auto" else None,
                    voice_clone_prompt=prompt_items,
                    instruct=instruct if instruct.strip() else None,
                )
            elif speaker in self.designed_voices:
                # Designed voice - use VoiceDesign model
                progress(0.2, desc="Loading model...")
                status = self.load_design_model()
                if "Error" in status:
                    return None, status
                
                voice_data = self.designed_voices[speaker]
                progress(0.5, desc="Generating speech...")
                wavs, sr = self.design_model.generate_voice_design(
                    text=text,
                    language=language if language != "Auto" else None,
                    instruct=voice_data.get("instruct", ""),
                )
            else:
                return None, f"✗ Voice not found: {speaker}"
            
            progress(0.8, desc="Saving audio...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.make_output_filename(save_name or speaker, timestamp, "wav")
            filepath = self.output_dir / filename
            
            sf.write(str(filepath), wavs[0], sr)
            
            self.audio_history.append({
                "filename": filename,
                "filepath": str(filepath),
                "text": text,
                "speaker": speaker,
                "language": language,
                "timestamp": timestamp,
                "type": "TTS"
            })
            
            progress(1.0, desc="Complete!")
            return str(filepath), f"✓ Audio generated successfully: {filename}"
            
        except Exception as e:
            return None, f"✗ Error: {str(e)}"
    
    def clone_voice(self, audio_file, ref_text, target_text, language, voice_name, save_name=None, progress=gr.Progress()):
        try:
            if audio_file is None:
                return None, "Please upload a reference audio file", gr.update()
            
            if not target_text.strip():
                return None, "Please enter text to synthesize", gr.update()
            
            progress(0.2, desc="Loading model...")
            status = self.load_base_model()
            if "Error" in status:
                return None, status, gr.update()
            
            progress(0.4, desc="Processing reference audio...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_voice_name = "".join(c for c in (voice_name or "voice") if c.isalnum() or c in ("_", "-", " ")).strip() or "voice"
            ref_filename = f"{safe_voice_name}_{timestamp}.wav"
            ref_dest = self.voiceinputs_dir / ref_filename

            src_suffix = Path(audio_file).suffix.lower()
            if src_suffix == ".wav":
                shutil.copy2(audio_file, ref_dest)
            else:
                audio = AudioSegment.from_file(audio_file)
                audio.export(str(ref_dest), format="wav")

            audio_path = str(ref_dest)
            
            progress(0.6, desc="Cloning voice...")
            wavs, sr = self.base_model.generate_voice_clone(
                text=target_text,
                language=language if language != "Auto" else None,
                ref_audio=audio_path,
                ref_text=ref_text if ref_text.strip() else None,
                x_vector_only_mode=not ref_text.strip(),
            )
            
            progress(0.8, desc="Saving audio...")
            filename = self.make_output_filename(save_name or voice_name or "clone", timestamp, "wav")
            filepath = self.output_dir / filename
            
            sf.write(str(filepath), wavs[0], sr)
            
            self.audio_history.append({
                "filename": filename,
                "filepath": str(filepath),
                "text": target_text,
                "timestamp": timestamp,
                "type": "Clone"
            })
            
            if voice_name.strip():
                progress(0.9, desc="Saving voice profile...")
                prompt_items = self.base_model.create_voice_clone_prompt(
                    ref_audio=audio_path,
                    ref_text=ref_text if ref_text.strip() else None,
                    x_vector_only_mode=not ref_text.strip(),
                )
                
                voice_data = {
                    "name": voice_name,
                    "ref_audio": str((self.voiceinputs_dir / ref_filename).as_posix()),
                    "ref_text": ref_text,
                    "timestamp": timestamp
                }
                
                self.cloned_voices[voice_name] = voice_data
                self.save_cloned_voices()
            
            progress(1.0, desc="Complete!")
            cloned_list = list(self.cloned_voices.keys())
            return str(filepath), f"✓ Voice cloned successfully: {filename}", gr.update(choices=cloned_list, value=cloned_list[0] if cloned_list else None)
            
        except Exception as e:
            return None, f"✗ Error: {str(e)}", gr.update()
    
    def generate_with_cloned_voice(self, voice_name, text, language, save_name=None, progress=gr.Progress()):
        try:
            if not voice_name:
                return None, "Please select a cloned voice"
            
            if not text.strip():
                return None, "Please enter text to synthesize"
            
            if voice_name not in self.cloned_voices:
                return None, "Selected voice not found"
            
            progress(0.2, desc="Loading model...")
            status = self.load_base_model()
            if "Error" in status:
                return None, status
            
            voice_data = self.cloned_voices[voice_name]
            ref_audio = self.resolve_audio_path(voice_data.get("ref_audio"))
            
            progress(0.4, desc="Loading voice profile...")
            prompt_items = self.base_model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=voice_data.get("ref_text"),
                x_vector_only_mode=not voice_data.get("ref_text"),
            )
            
            progress(0.6, desc="Generating speech...")
            wavs, sr = self.base_model.generate_voice_clone(
                text=text,
                language=language if language != "Auto" else None,
                voice_clone_prompt=prompt_items,
            )
            
            progress(0.8, desc="Saving audio...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.make_output_filename(save_name or voice_name, timestamp, "wav")
            filepath = self.output_dir / filename
            
            sf.write(str(filepath), wavs[0], sr)
            
            self.audio_history.append({
                "filename": filename,
                "filepath": str(filepath),
                "text": text,
                "voice": voice_name,
                "timestamp": timestamp,
                "type": "Cloned Voice"
            })
            
            progress(1.0, desc="Complete!")
            return str(filepath), f"✓ Audio generated with cloned voice: {filename}"
            
        except Exception as e:
            return None, f"✗ Error: {str(e)}"
    
    def design_voice(self, text, language, instruct, voice_name, save_name=None, progress=gr.Progress()):
        try:
            if not text.strip():
                return None, "Please enter text to synthesize", gr.update()
            
            if not instruct.strip():
                return None, "Please provide voice design instructions", gr.update()
            
            progress(0.2, desc="Loading model...")
            status = self.load_design_model()
            if "Error" in status:
                return None, status, gr.update()
            
            progress(0.5, desc="Designing voice...")
            wavs, sr = self.design_model.generate_voice_design(
                text=text,
                language=language if language != "Auto" else None,
                instruct=instruct,
            )
            
            progress(0.8, desc="Saving audio...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.make_output_filename(save_name or voice_name or "design", timestamp, "wav")
            filepath = self.output_dir / filename
            
            sf.write(str(filepath), wavs[0], sr)
            
            self.audio_history.append({
                "filename": filename,
                "filepath": str(filepath),
                "text": text,
                "instruct": instruct,
                "timestamp": timestamp,
                "type": "Voice Design"
            })
            
            if voice_name.strip():
                progress(0.9, desc="Saving voice profile...")
                voice_data = {
                    "name": voice_name,
                    "instruct": instruct,
                    "timestamp": timestamp
                }
                self.designed_voices[voice_name] = voice_data
                self.save_designed_voices()
            
            progress(1.0, desc="Complete!")
            designed_list = list(self.designed_voices.keys())
            return str(filepath), f"✓ Voice designed successfully: {filename}", gr.update(choices=designed_list, value=designed_list[0] if designed_list else None)
            
        except Exception as e:
            return None, f"✗ Error: {str(e)}", gr.update()
    
    def get_audio_library_files(self):
        files = []
        if self.output_dir.exists():
            for file in sorted(self.output_dir.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True):
                files.append({
                    "name": file.name,
                    "path": str(file),
                    "size": file.stat().st_size,
                    "modified": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
        return files
    
    def get_audio_history(self):
        files = self.get_audio_library_files()
        if not files:
            return "No audio files generated yet"
        
        history_text = "## Generated Audio Files\n\n"
        for i, file in enumerate(files[:20], 1):
            size_mb = file['size'] / (1024 * 1024)
            history_text += f"**{i}. {file['name']}**\n"
            history_text += f"- Size: {size_mb:.2f} MB\n"
            history_text += f"- Modified: {file['modified']}\n\n"
        
        return history_text
    
    def delete_audio(self, filename):
        try:
            if not filename:
                return "Please enter a filename to delete", gr.update()
            
            filepath = self.output_dir / filename
            if filepath.exists():
                os.remove(filepath)
                self.audio_history = [item for item in self.audio_history if item['filename'] != filename]
                return f"✓ Deleted: {filename}", gr.update(value=self.get_audio_history())
            else:
                return f"✗ File not found: {filename}", gr.update()
        except Exception as e:
            return f"✗ Error: {str(e)}", gr.update()
    
    def delete_selected_files(self, selected_files):
        try:
            if not selected_files:
                return "No files selected", gr.update()
            
            deleted = []
            for filename in selected_files:
                filepath = self.output_dir / filename
                if filepath.exists():
                    os.remove(filepath)
                    deleted.append(filename)
                    self.audio_history = [item for item in self.audio_history if item.get('filename') != filename]
            
            if deleted:
                return f"✓ Deleted {len(deleted)} file(s): {', '.join(deleted)}", gr.update(value=self.get_audio_history())
            else:
                return "✗ No files were deleted", gr.update()
        except Exception as e:
            return f"✗ Error: {str(e)}", gr.update()
    
    def generate_all_samples(self, progress=gr.Progress()):
        try:
            all_voices = list(self.voice_personas.keys())
            total = len(all_voices)
            results = []
            
            for i, voice_name in enumerate(all_voices):
                progress((i + 1) / total, desc=f"Generating {voice_name}... ({i+1}/{total})")
                _, status = self.generate_voice_sample(voice_name)
                results.append(f"{voice_name}: {status}")
            
            return "\n".join(results)
        except Exception as e:
            return f"✗ Error: {str(e)}"
    
    def delete_cloned_voice(self, voice_name):
        try:
            if not voice_name:
                return "Please select a voice to delete", gr.update()
            
            if voice_name in self.cloned_voices:
                del self.cloned_voices[voice_name]
                self.save_cloned_voices()
                
                if voice_name in self.main_voices.get("voices", []):
                    self.main_voices["voices"].remove(voice_name)
                    self.save_main_voices()
                
                sample_path = self.voicesamples_dir / f"{voice_name}.mp3"
                if sample_path.exists():
                    os.remove(sample_path)
                
                cloned_list = list(self.cloned_voices.keys())
                return f"✓ Deleted voice: {voice_name}", gr.update(choices=cloned_list, value=cloned_list[0] if cloned_list else None)
            else:
                return f"✗ Voice not found: {voice_name}", gr.update()
        except Exception as e:
            return f"✗ Error: {str(e)}", gr.update()
    
    def get_voice_info(self, voice_name):
        if not voice_name:
            return ""
        
        if voice_name in self.voice_personas:
            info = self.voice_personas[voice_name]
            return f"**{voice_name}** ({info['gender']}, Age {info['age']})\n\n{info['desc']}"
        elif voice_name in self.cloned_voices:
            return f"**{voice_name}** (Cloned Voice)\n\nCustom cloned voice from audio sample"
        elif voice_name in self.designed_voices:
            info = self.designed_voices[voice_name]
            return f"**{voice_name}** (Designed Voice)\n\n{info.get('instruct', 'Custom designed voice')}"
        return ""
    
    def get_sample_path(self, voice_name):
        if not voice_name:
            return gr.update(visible=False)
        sample_path = self.voicesamples_dir / f"{voice_name}.mp3"
        if sample_path.exists():
            return gr.update(value=str(sample_path), visible=True)
        return gr.update(visible=False)
    
    def play_or_generate_sample(self, voice_name):
        """Check if sample exists, return it or prompt to generate"""
        if not voice_name:
            return gr.update(visible=False), "Please select a voice first"
        
        sample_path = self.voicesamples_dir / f"{voice_name}.mp3"
        if sample_path.exists():
            return gr.update(value=str(sample_path), visible=True), f"▶️ Playing sample for {voice_name}"
        else:
            return gr.update(visible=False), f"⚠️ Sample not found for '{voice_name}'. Click 'Generate Sample' to create it."
    
    def generate_voice_sample(self, voice_name, progress=gr.Progress()):
        try:
            if not voice_name:
                return None, "Please select a voice"
            
            sample_text = "Welcome to Qwen TTS Studio. have fun with voices! hehe"
            
            progress(0.2, desc="Loading model...")
            
            if voice_name in self.voice_personas:
                voice_info = self.voice_personas[voice_name]
                model_type = voice_info.get("model", "custom")
                
                if model_type == "custom":
                    # Use CustomVoice model for built-in speakers
                    status = self.load_custom_model()
                    if "Error" in status:
                        return None, status
                    
                    progress(0.5, desc="Generating sample...")
                    wavs, sr = self.custom_model.generate_custom_voice(
                        text=sample_text,
                        language=None,
                        speaker=voice_info.get("speaker", voice_name),
                        instruct=None,
                    )
                else:
                    # Use VoiceDesign model for designed personas
                    status = self.load_design_model()
                    if "Error" in status:
                        return None, status
                    
                    progress(0.5, desc="Generating sample...")
                    wavs, sr = self.design_model.generate_voice_design(
                        text=sample_text,
                        language=None,
                        instruct=voice_info.get("instruct", ""),
                    )
            elif voice_name in self.cloned_voices:
                status = self.load_base_model()
                if "Error" in status:
                    return None, status
                
                voice_data = self.cloned_voices[voice_name]
                progress(0.4, desc="Loading voice profile...")
                prompt_items = self.base_model.create_voice_clone_prompt(
                    ref_audio=self.resolve_audio_path(voice_data.get("ref_audio")),
                    ref_text=voice_data.get("ref_text"),
                    x_vector_only_mode=not voice_data.get("ref_text"),
                )
                
                progress(0.5, desc="Generating sample...")
                wavs, sr = self.base_model.generate_voice_clone(
                    text=sample_text,
                    language=None,
                    voice_clone_prompt=prompt_items,
                )
            elif voice_name in self.designed_voices:
                status = self.load_design_model()
                if "Error" in status:
                    return None, status
                
                voice_data = self.designed_voices[voice_name]
                progress(0.5, desc="Generating sample...")
                wavs, sr = self.design_model.generate_voice_design(
                    text=sample_text,
                    language=None,
                    instruct=voice_data.get("instruct", ""),
                )
            else:
                return None, f"✗ Voice not found: {voice_name}"
            
            progress(0.8, desc="Converting to MP3...")
            wav_path = self.voicesamples_dir / f"{voice_name}_temp.wav"
            mp3_path = self.voicesamples_dir / f"{voice_name}.mp3"
            
            sf.write(str(wav_path), wavs[0], sr)
            
            audio = AudioSegment.from_wav(str(wav_path))
            audio.export(str(mp3_path), format="mp3", bitrate="192k")
            
            if wav_path.exists():
                os.remove(wav_path)
            
            progress(1.0, desc="Complete!")
            return gr.update(value=str(mp3_path), visible=True), f"✓ Sample generated: {voice_name}.mp3"
            
        except Exception as e:
            return gr.update(visible=False), f"✗ Error: {str(e)}"
    
    def add_to_main_voices(self, voice_name, voice_type):
        try:
            if not voice_name:
                return "Please select a voice"
            
            if voice_name in self.main_voices.get("voices", []):
                return f"✓ {voice_name} is already in main voices"
            
            if "voices" not in self.main_voices:
                self.main_voices["voices"] = []
            
            self.main_voices["voices"].append(voice_name)
            self.save_main_voices()
            
            return f"✓ Added {voice_name} to main voices"
        except Exception as e:
            return f"✗ Error: {str(e)}"
    
    def delete_designed_voice(self, voice_name):
        try:
            if not voice_name:
                return "Please select a voice to delete", gr.update()
            
            if voice_name in self.designed_voices:
                del self.designed_voices[voice_name]
                self.save_designed_voices()
                
                if voice_name in self.main_voices.get("voices", []):
                    self.main_voices["voices"].remove(voice_name)
                    self.save_main_voices()
                
                sample_path = self.voicesamples_dir / f"{voice_name}.mp3"
                if sample_path.exists():
                    os.remove(sample_path)
                
                designed_list = list(self.designed_voices.keys())
                return f"✓ Deleted voice: {voice_name}", gr.update(choices=designed_list, value=designed_list[0] if designed_list else None)
            else:
                return f"✗ Voice not found: {voice_name}", gr.update()
        except Exception as e:
            return f"✗ Error: {str(e)}", gr.update()
    
    def create_interface(self):
        with gr.Blocks(title="Qwen Voice TTS Studio") as app:
            library_refresh_token = gr.State(0)
            gr.Markdown("# 🎙️ Qwen Voice TTS Studio")
            gr.Markdown("Generate high-quality speech with multiple voice options")
            
            with gr.Tabs():
                with gr.Tab("🎤 Text-to-Speech"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            tts_text = gr.Textbox(
                                label="Text to Synthesize",
                                placeholder="Enter the text you want to convert to speech...",
                                lines=5
                            )
                            
                            with gr.Row():
                                tts_language = gr.Dropdown(
                                    label="Language",
                                    choices=self.get_supported_languages(),
                                    value="Auto",
                                    allow_custom_value=True
                                )
                                tts_speaker = gr.Dropdown(
                                    label="Speaker Voice",
                                    choices=self.get_supported_speakers(),
                                    value=self.get_supported_speakers()[0] if self.get_supported_speakers() else None
                                )
                            
                            tts_voice_info = gr.Markdown(value="", label="Voice Info")
                            
                            with gr.Row():
                                tts_sample_audio = gr.Audio(label="Voice Sample", type="filepath", visible=True)
                                tts_play_sample_btn = gr.Button("▶️ Play Sample", size="sm", variant="secondary")
                                tts_generate_sample_btn = gr.Button("🔊 Generate Sample", size="sm")
                            
                            tts_instruct = gr.Textbox(
                                label="Voice Instructions (Optional)",
                                placeholder="e.g., 'Speak in a happy tone' or 'Use an angry voice'",
                                lines=2
                            )

                            tts_save_name = gr.Textbox(
                                label="Save name (Optional)",
                                placeholder="If set: used as filename prefix (timestamp will be appended)",
                                lines=1
                            )
                            
                            tts_generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            tts_audio_output = gr.Audio(label="Generated Audio", type="filepath")
                            tts_status = gr.Textbox(label="Status", lines=3)
                    
                    tts_speaker.change(
                        fn=lambda v: (self.get_voice_info(v), self.get_sample_path(v)),
                        inputs=[tts_speaker],
                        outputs=[tts_voice_info, tts_sample_audio]
                    )

                    tts_play_sample_btn.click(
                        fn=self.play_or_generate_sample,
                        inputs=[tts_speaker],
                        outputs=[tts_sample_audio, tts_status]
                    )

                    tts_generate_sample_btn.click(
                        fn=self.generate_voice_sample,
                        inputs=[tts_speaker],
                        outputs=[tts_sample_audio, tts_status]
                    )

                    tts_generate_btn.click(
                        fn=self.generate_tts,
                        inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_save_name],
                        outputs=[tts_audio_output, tts_status]
                    )

                    tts_generate_btn.click(
                        fn=lambda x: (x or 0) + 1,
                        inputs=[library_refresh_token],
                        outputs=[library_refresh_token]
                    )
                
                with gr.Tab("🎭 Voice Cloning"):
                    gr.Markdown("### Clone a voice from an audio sample")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Step 1: Upload Reference Audio")
                            clone_audio = gr.Audio(
                                label="Reference Audio (WAV or MP3)",
                                type="filepath"
                            )
                            clone_ref_text = gr.Textbox(
                                label="Reference Text (Optional but recommended)",
                                placeholder="Transcript of the reference audio...",
                                lines=3
                            )
                            clone_voice_name = gr.Textbox(
                                label="Save Voice As (Optional)",
                                placeholder="Enter a name to save this voice for reuse"
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### Step 2: Generate with Cloned Voice")
                            clone_target_text = gr.Textbox(
                                label="Text to Synthesize",
                                placeholder="Enter text to speak in the cloned voice...",
                                lines=5
                            )
                            clone_language = gr.Dropdown(
                                label="Language",
                                choices=self.get_supported_languages(),
                                value="Auto",
                                allow_custom_value=True
                            )
                            clone_save_name = gr.Textbox(
                                label="Save name (Optional)",
                                placeholder="If set: used as output filename prefix (timestamp will be appended)",
                                lines=1
                            )
                            clone_generate_btn = gr.Button("🎭 Clone & Generate", variant="primary", size="lg")
                    
                    with gr.Row():
                        clone_audio_output = gr.Audio(label="Generated Audio", type="filepath")
                        clone_status = gr.Textbox(label="Status", lines=3)
                    
                    gr.Markdown("---")
                    gr.Markdown("### Use Saved Cloned Voices")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            saved_voice_dropdown = gr.Dropdown(
                                label="Select Saved Voice",
                                choices=list(self.cloned_voices.keys()),
                                value=list(self.cloned_voices.keys())[0] if self.cloned_voices else None
                            )
                            
                            clone_voice_info = gr.Markdown(value="", label="Voice Info")
                            
                            with gr.Row():
                                clone_sample_audio = gr.Audio(label="Voice Sample", type="filepath", visible=True)
                                clone_play_sample_btn = gr.Button("▶️ Play Sample", size="sm", variant="secondary")
                                clone_generate_sample_btn = gr.Button("🔊 Generate Sample", size="sm")
                            
                            saved_voice_text = gr.Textbox(
                                label="Text to Synthesize",
                                placeholder="Enter text to speak...",
                                lines=4
                            )
                            saved_voice_language = gr.Dropdown(
                                label="Language",
                                choices=self.get_supported_languages(),
                                value="Auto",
                                allow_custom_value=True
                            )

                            saved_voice_save_name = gr.Textbox(
                                label="Save name (Optional)",
                                placeholder="If set: used as output filename prefix (timestamp will be appended)",
                                lines=1
                            )
                            
                            with gr.Row():
                                saved_voice_generate_btn = gr.Button("🎵 Generate", variant="primary")
                                clone_add_to_main_btn = gr.Button("➕ Add to Main Voices", variant="secondary")
                                saved_voice_delete_btn = gr.Button("🗑️ Delete Voice", variant="stop")
                        
                        with gr.Column(scale=1):
                            saved_voice_audio_output = gr.Audio(label="Generated Audio", type="filepath")
                            saved_voice_status = gr.Textbox(label="Status", lines=3)
                    
                    clone_generate_btn.click(
                        fn=self.clone_voice,
                        inputs=[clone_audio, clone_ref_text, clone_target_text, clone_language, clone_voice_name, clone_save_name],
                        outputs=[clone_audio_output, clone_status, saved_voice_dropdown]
                    )

                    clone_generate_btn.click(
                        fn=lambda x: (x or 0) + 1,
                        inputs=[library_refresh_token],
                        outputs=[library_refresh_token]
                    )
                    
                    saved_voice_dropdown.change(
                        fn=lambda v: (self.get_voice_info(v), self.get_sample_path(v)),
                        inputs=[saved_voice_dropdown],
                        outputs=[clone_voice_info, clone_sample_audio]
                    )
                    
                    clone_play_sample_btn.click(
                        fn=self.play_or_generate_sample,
                        inputs=[saved_voice_dropdown],
                        outputs=[clone_sample_audio, saved_voice_status]
                    )
                    
                    clone_generate_sample_btn.click(
                        fn=self.generate_voice_sample,
                        inputs=[saved_voice_dropdown],
                        outputs=[clone_sample_audio, saved_voice_status]
                    )
                    
                    saved_voice_generate_btn.click(
                        fn=self.generate_with_cloned_voice,
                        inputs=[saved_voice_dropdown, saved_voice_text, saved_voice_language, saved_voice_save_name],
                        outputs=[saved_voice_audio_output, saved_voice_status]
                    )

                    saved_voice_generate_btn.click(
                        fn=lambda x: (x or 0) + 1,
                        inputs=[library_refresh_token],
                        outputs=[library_refresh_token]
                    )
                    
                    clone_add_to_main_btn.click(
                        fn=lambda v: self.add_to_main_voices(v, "cloned"),
                        inputs=[saved_voice_dropdown],
                        outputs=[saved_voice_status]
                    )
                    
                    saved_voice_delete_btn.click(
                        fn=self.delete_cloned_voice,
                        inputs=[saved_voice_dropdown],
                        outputs=[saved_voice_status, saved_voice_dropdown]
                    )
                
                with gr.Tab("🎨 Voice Design"):
                    gr.Markdown("### Design a custom voice with natural language instructions")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            design_text = gr.Textbox(
                                label="Text to Synthesize",
                                placeholder="Enter the text you want to speak...",
                                lines=5
                            )
                            design_language = gr.Dropdown(
                                label="Language",
                                choices=self.get_supported_languages(),
                                value="Auto",
                                allow_custom_value=True
                            )
                            design_instruct = gr.Textbox(
                                label="Voice Design Instructions",
                                placeholder="Describe the voice characteristics you want (e.g., 'Male, 30 years old, deep voice, confident tone, slightly raspy')",
                                lines=4
                            )
                            design_voice_name = gr.Textbox(
                                label="Save Voice As (Optional)",
                                placeholder="Enter a name to save this designed voice for reuse"
                            )

                            design_save_name = gr.Textbox(
                                label="Save name (Optional)",
                                placeholder="If set: used as output filename prefix (timestamp will be appended)",
                                lines=1
                            )
                            
                            gr.Markdown("""
                            **Examples:**
                            - "Female, 25 years old, soft and gentle voice, warm and friendly tone"
                            - "Male, 40 years old, authoritative voice, professional broadcaster style"
                            - "Young female, cheerful and energetic, high-pitched voice with excitement"
                            """)
                            
                            design_generate_btn = gr.Button("🎨 Design Voice", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            design_audio_output = gr.Audio(label="Generated Audio", type="filepath")
                            design_status = gr.Textbox(label="Status", lines=3)
                    
                    gr.Markdown("---")
                    gr.Markdown("### Use Saved Designed Voices")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            designed_voice_dropdown = gr.Dropdown(
                                label="Select Saved Voice",
                                choices=list(self.designed_voices.keys()),
                                value=list(self.designed_voices.keys())[0] if self.designed_voices else None
                            )
                            
                            design_voice_info = gr.Markdown(value="", label="Voice Info")
                            
                            with gr.Row():
                                design_sample_audio = gr.Audio(label="Voice Sample", type="filepath", visible=True)
                                design_play_sample_btn = gr.Button("▶️ Play Sample", size="sm", variant="secondary")
                                design_generate_sample_btn = gr.Button("🔊 Generate Sample", size="sm")
                            
                            with gr.Row():
                                design_add_to_main_btn = gr.Button("➕ Add to Main Voices", variant="secondary")
                                designed_voice_delete_btn = gr.Button("🗑️ Delete Voice", variant="stop")
                        
                        with gr.Column(scale=1):
                            designed_voice_status = gr.Textbox(label="Status", lines=3)
                    
                    design_generate_btn.click(
                        fn=self.design_voice,
                        inputs=[design_text, design_language, design_instruct, design_voice_name, design_save_name],
                        outputs=[design_audio_output, design_status, designed_voice_dropdown]
                    )

                    design_generate_btn.click(
                        fn=lambda x: (x or 0) + 1,
                        inputs=[library_refresh_token],
                        outputs=[library_refresh_token]
                    )
                    
                    designed_voice_dropdown.change(
                        fn=lambda v: (self.get_voice_info(v), self.get_sample_path(v)),
                        inputs=[designed_voice_dropdown],
                        outputs=[design_voice_info, design_sample_audio]
                    )
                    
                    design_play_sample_btn.click(
                        fn=self.play_or_generate_sample,
                        inputs=[designed_voice_dropdown],
                        outputs=[design_sample_audio, designed_voice_status]
                    )
                    
                    design_generate_sample_btn.click(
                        fn=self.generate_voice_sample,
                        inputs=[designed_voice_dropdown],
                        outputs=[design_sample_audio, designed_voice_status]
                    )
                    
                    design_add_to_main_btn.click(
                        fn=lambda v: self.add_to_main_voices(v, "designed"),
                        inputs=[designed_voice_dropdown],
                        outputs=[designed_voice_status]
                    )
                    
                    designed_voice_delete_btn.click(
                        fn=self.delete_designed_voice,
                        inputs=[designed_voice_dropdown],
                        outputs=[designed_voice_status, designed_voice_dropdown]
                    )

                with gr.Tab("💬 Conversations"):
                    gr.Markdown("### Generate multi-voice conversations from a script")

                    with gr.Row():
                        with gr.Column(scale=1):
                            conv_voice1 = gr.Dropdown(
                                label="Voice 1",
                                choices=self.get_supported_speakers(),
                                value=self.get_supported_speakers()[0] if self.get_supported_speakers() else None,
                            )
                            conv_voice2 = gr.Dropdown(
                                label="Voice 2",
                                choices=self.get_supported_speakers(),
                                value=self.get_supported_speakers()[1] if len(self.get_supported_speakers()) > 1 else (self.get_supported_speakers()[0] if self.get_supported_speakers() else None),
                            )
                            conv_voice3 = gr.Dropdown(
                                label="Voice 3",
                                choices=self.get_supported_speakers(),
                                value=self.get_supported_speakers()[2] if len(self.get_supported_speakers()) > 2 else (self.get_supported_speakers()[0] if self.get_supported_speakers() else None),
                            )

                            conv_language = gr.Dropdown(
                                label="Language",
                                choices=self.get_supported_languages(),
                                value="Auto",
                                allow_custom_value=True,
                            )

                            conv_save_name = gr.Textbox(
                                label="Save name (Optional)",
                                placeholder="If set: used as output filename prefix (timestamp will be appended)",
                                lines=1,
                            )

                            conv_script = gr.Textbox(
                                label="Conversation Script",
                                placeholder="[Voice1]\nHey who are you?\n[Voice2]\nWho me?\n[Voice1]\nYes you!\n[Voice3]\nHi, I'm Sophie!\n",
                                lines=14,
                            )

                            conv_generate_btn = gr.Button("💬 Generate Conversation", variant="primary", size="lg")

                        with gr.Column(scale=1):
                            conv_audio_output = gr.Audio(label="Generated Conversation", type="filepath")
                            conv_status = gr.Textbox(label="Status", lines=3)
                            conv_help = gr.Markdown(
                                value=(
                                    "**How to write a conversation script**\n\n"
                                    "Use tags to switch speakers. Valid tags are:\n\n"
                                    "- `[Voice1]`\n"
                                    "- `[Voice2]`\n"
                                    "- `[Voice3]`\n\n"
                                    "**Example (copy/paste):**\n\n"
                                    "```\n"
                                    "[Voice1]\n"
                                    "Hey who are you?\n"
                                    "[Voice2]\n"
                                    "Who me?\n"
                                    "[Voice1]\n"
                                    "Yes you!\n"
                                    "[Voice3]\n"
                                    "Hi, I'm Sophie!\n"
                                    "[Voice1]\n"
                                    "Not you!\n"
                                    "```\n"
                                )
                            )

                    conv_generate_btn.click(
                        fn=self.generate_conversation,
                        inputs=[conv_voice1, conv_voice2, conv_voice3, conv_script, conv_language, conv_save_name],
                        outputs=[conv_audio_output, conv_status],
                    )

                    conv_generate_btn.click(
                        fn=lambda x: (x or 0) + 1,
                        inputs=[library_refresh_token],
                        outputs=[library_refresh_token]
                    )

                with gr.Tab("🎙️ Voice ASR"):
                    gr.Markdown("### Transcribe audio to text (Qwen3-ASR)")

                    with gr.Row():
                        with gr.Column(scale=2):
                            asr_audio = gr.Audio(
                                label="Audio Input (Upload or Microphone)",
                                type="filepath",
                                sources=["upload", "microphone"],
                            )

                            with gr.Row():
                                asr_model_choice = gr.Dropdown(
                                    label="ASR Model",
                                    choices=["Qwen/Qwen3-ASR-1.7B"],
                                    value="Qwen/Qwen3-ASR-1.7B",
                                )
                                asr_language = gr.Dropdown(
                                    label="Language",
                                    choices=["Auto", "English", "Chinese"],
                                    value="Auto",
                                    allow_custom_value=True,
                                )

                            asr_save_prefix = gr.Textbox(
                                label="Save name (Optional)",
                                placeholder="If set: used as transcript filename prefix (timestamp will be appended)",
                                lines=1,
                            )

                            with gr.Row():
                                asr_download_aligner = gr.Checkbox(
                                    label="Also download Forced Aligner (timestamps model)",
                                    value=False,
                                )
                                asr_return_timestamps = gr.Checkbox(
                                    label="Return timestamps (requires Forced Aligner)",
                                    value=False,
                                )

                            with gr.Row():
                                asr_download_btn = gr.Button("⬇ Download ASR Model(s)", variant="secondary")
                                asr_transcribe_btn = gr.Button("📝 Transcribe", variant="primary")

                            gr.Markdown("---")
                            asr_recreate_voice = gr.Checkbox(
                                label="Recreate voice with selected TTS voice",
                                value=False,
                            )

                            with gr.Row():
                                asr_tts_voice = gr.Dropdown(
                                    label="TTS Voice",
                                    choices=self.get_supported_speakers(),
                                    value=self.get_supported_speakers()[0] if self.get_supported_speakers() else None,
                                )
                                asr_tts_language = gr.Dropdown(
                                    label="TTS Language",
                                    choices=self.get_supported_languages(),
                                    value="Auto",
                                    allow_custom_value=True,
                                )

                        with gr.Column(scale=1):
                            asr_text_output = gr.Textbox(label="Transcript", lines=12)
                            asr_timestamps_output = gr.Textbox(label="Timestamps (JSON)", lines=8)
                            asr_revoice_audio = gr.Audio(label="Re-voiced Audio (Optional)", type="filepath")
                            asr_status = gr.Textbox(label="Status", lines=6)

                    asr_download_btn.click(
                        fn=self.download_asr_models,
                        inputs=[asr_download_aligner],
                        outputs=[asr_status],
                    )

                    asr_transcribe_btn.click(
                        fn=self.transcribe_audio,
                        inputs=[
                            asr_audio,
                            asr_model_choice,
                            asr_language,
                            asr_save_prefix,
                            asr_recreate_voice,
                            asr_tts_voice,
                            asr_tts_language,
                            asr_return_timestamps,
                        ],
                        outputs=[asr_text_output, asr_timestamps_output, asr_revoice_audio, asr_status],
                    )
                
                with gr.Tab("📂 Audio Library"):
                    gr.Markdown("### Manage your generated audio files")
                    
                    with gr.Row():
                        refresh_library_btn = gr.Button("🔄 Refresh List", size="sm")

                    library_file_names_state = gr.State([])

                    library_status = gr.Textbox(label="Status", lines=2)

                    with gr.Row():
                        delete_selected_btn = gr.Button("🗑️ Delete Selected Files", variant="stop")

                    tile_labels = []
                    tile_names = []
                    tile_checks = []
                    tile_audios = []
                    tile_dl_wav_btns = []
                    tile_dl_mp3_btns = []
                    tile_dl_files = []

                    for row in range(5):
                        with gr.Row():
                            for col in range(4):
                                with gr.Column():
                                    lbl = gr.Markdown(value="", visible=False)
                                    name = gr.Textbox(value="", visible=False)
                                    chk = gr.Checkbox(label="Select", value=False, visible=False)
                                    aud = gr.Audio(label="", type="filepath", visible=False)
                                    with gr.Row():
                                        dl_wav = gr.Button("⬇ WAV", size="sm")
                                        dl_mp3 = gr.Button("⬇ MP3", size="sm")
                                    dl_file = gr.File(label="Download", visible=False)
                                    tile_labels.append(lbl)
                                    tile_names.append(name)
                                    tile_checks.append(chk)
                                    tile_audios.append(aud)
                                    tile_dl_wav_btns.append(dl_wav)
                                    tile_dl_mp3_btns.append(dl_mp3)
                                    tile_dl_files.append(dl_file)

                    def refresh_tiles(_token=None):
                        return self.update_library_tiles()

                    library_refresh_token.change(
                        fn=refresh_tiles,
                        inputs=[library_refresh_token],
                        outputs=[
                            library_file_names_state,
                            *tile_labels,
                            *tile_names,
                            *tile_checks,
                            *tile_audios,
                            *tile_dl_wav_btns,
                            *tile_dl_mp3_btns,
                            *tile_dl_files,
                        ]
                    )

                    app.load(
                        fn=refresh_tiles,
                        inputs=[],
                        outputs=[
                            library_file_names_state,
                            *tile_labels,
                            *tile_names,
                            *tile_checks,
                            *tile_audios,
                            *tile_dl_wav_btns,
                            *tile_dl_mp3_btns,
                            *tile_dl_files,
                        ]
                    )

                    refresh_library_btn.click(
                        fn=lambda x: (x or 0) + 1,
                        inputs=[library_refresh_token],
                        outputs=[library_refresh_token]
                    )

                    for i in range(20):
                        tile_dl_wav_btns[i].click(
                            fn=lambda n: self.prepare_download_file(n, "wav"),
                            inputs=[tile_names[i]],
                            outputs=[tile_dl_files[i], library_status]
                        )
                        tile_dl_mp3_btns[i].click(
                            fn=lambda n: self.prepare_download_file(n, "mp3"),
                            inputs=[tile_names[i]],
                            outputs=[tile_dl_files[i], library_status]
                        )

                    def delete_from_tiles(file_names, *flags):
                        status, updated_names = self.delete_selected_tile_files(file_names, list(flags))
                        return status, updated_names

                    delete_selected_btn.click(
                        fn=delete_from_tiles,
                        inputs=[library_file_names_state, *tile_checks],
                        outputs=[library_status, library_file_names_state]
                    )

                    delete_selected_btn.click(
                        fn=lambda x: (x or 0) + 1,
                        inputs=[library_refresh_token],
                        outputs=[library_refresh_token]
                    )
                
                with gr.Tab("⚙️ Settings"):
                    gr.Markdown("### Voice Sample Management")

                    render_device = gr.Dropdown(
                        label="Render Device",
                        choices=["Auto", "GPU", "CPU"],
                        value=self.render_device,
                    )
                    system_info = gr.Markdown(value=self.get_system_info_markdown())
                    
                    gr.Markdown("""
                    Generate MP3 samples for all voice personas. These samples use the text:
                    **"Welcome to Qwen TTS Studio. have fun with voices! hehe"**
                    
                    Samples are saved in the `voicesamples` folder and can be played when selecting voices.
                    """)
                    
                    with gr.Row():
                        generate_all_samples_btn = gr.Button("🎵 Generate Samples for All Voices", variant="primary", size="lg")
                    
                    samples_status = gr.Textbox(label="Status", lines=10)
                    
                    gr.Markdown("### Voice Management")
                    
                    main_voices_list = gr.Textbox(
                        label="Main Voices (Available in Text-to-Speech)",
                        value=", ".join(self.main_voices.get("voices", [])),
                        lines=5,
                        interactive=False
                    )
                    
                    gr.Markdown("""
                    **Note:** Use the "Add to Main Voices" button in Voice Cloning and Voice Design tabs 
                    to add custom voices to the main Text-to-Speech dropdown.
                    """)
                    
                    generate_all_samples_btn.click(
                        fn=self.generate_all_samples,
                        inputs=[],
                        outputs=[samples_status]
                    )

                    render_device.change(
                        fn=self.set_render_device,
                        inputs=[render_device],
                        outputs=[system_info]
                    )
                
                with gr.Tab("ℹ️ Info"):
                    gr.Markdown("""
                    ## Qwen Voice TTS Studio 1.1
                    
                    ### Features:
                    - **Text-to-Speech**: Generate speech using pre-built voice profiles
                    - **Voice Cloning**: Clone any voice from a short audio sample
                    - **Voice Design**: Create custom voices using natural language descriptions
                    - **Conversations**: Script multi-voice dialogs with `[Voice1]`, `[Voice2]`, `[Voice3]`
                    - **Audio Library**: Tile-based library with preview and per-file WAV/MP3 download
                    - **Save name**: Optional output filename prefix (timestamp is appended)
                    - **Portable cloned voices**: Reference audio is stored in `voiceinputs/`
                    
                    ### Supported Languages:
                    Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
                    
                    ### Tips:
                    - For best voice cloning results, provide a clear reference audio (3-10 seconds)
                    - Include the transcript of reference audio for better quality
                    - Save cloned voices to reuse them without re-uploading
                    - Use voice instructions to control emotion and speaking style
                    - All generated files are saved in the `outputs` folder
                    
                    {}
                    
                    ### Models:
                    - CustomVoice: Pre-built speaker voices
                    - Base: Voice cloning capabilities
                    - VoiceDesign: Custom voice creation
                    
                    ---
                    Powered by Qwen3-TTS | [Documentation](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
                    """.format(
                        self.get_system_info_markdown()
                    ))
        
        return app

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--listen", nargs="?", const="0.0.0.0", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    server_name = "127.0.0.1" if args.listen is None else args.listen
    server_port = 7860 if args.port is None else args.port

    def _get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    if server_name == "0.0.0.0":
        local_ip = _get_local_ip()
        url = f"http://{local_ip}:{server_port}"
        purple = "\033[35m"
        reset = "\033[0m"
        print(f"LAN access URL: {purple}{url}{reset}")

    gui = QwenVoiceGUI()
    try:
        print(f"Render Device: {gui.render_device} | Device: {gui.device} | Attention: {gui._get_attn_implementation()} | FlashAttn2 Available: {is_flash_attn_2_available()}")
    except Exception:
        pass
    app = gui.create_interface()
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft()
    )

if __name__ == "__main__":
    main()
