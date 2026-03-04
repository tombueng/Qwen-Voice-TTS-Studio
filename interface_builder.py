"""Gradio UI interface builder module."""
from pathlib import Path
from typing import Optional, Callable
import gradio as gr

_UI_DIR = Path(__file__).parent / "ui"


def _load_ui_file(filename: str, fallback: str = "") -> str:
    """Read a file from the ui/ directory next to this module.

    Returns *fallback* (and prints a warning) if the file is missing.
    """
    path = _UI_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[InterfaceBuilder] WARNING: ui/{filename} not found — using fallback", flush=True)
        return fallback


class InterfaceBuilder:
    """Builds the Gradio UI interface."""

    def __init__(self, title: str = "Qwen Voice TTS Studio"):
        self.title = title
        self.demo  = None
        # Load external UI assets once at startup
        self._script_template = _load_ui_file(
            "script_template.json",
            fallback='{\n  "speaker": [],\n  "scenes": []\n}',
        )
        self._script_help = _load_ui_file(
            "script_help.md",
            fallback="*(script_help.md not found — see ui/ directory)*",
        )
    
    def create_interface(self,
                        on_tts_generate: Callable,
                        on_clone_voice: Callable,
                        on_design_voice: Callable,
                        on_script_run: Callable,
                        on_transcribe: Callable,
                        on_save_clone: Callable,
                        on_save_design: Callable,
                        on_voice_speed_change: Optional[Callable] = None,
                        on_stable_audio_generate: Optional[Callable] = None,
                        on_save_hf_token: Optional[Callable] = None,
                        on_abort: Optional[Callable] = None,
                        hf_token: str = "",
                        speakers_list: Optional[list] = None,
                        user_voice_choices: Optional[list] = None) -> gr.Blocks:
        """Create the main Gradio interface.

        speakers_list: ordered list of voice dicts from main_voices.json
        user_voice_choices: list of (label, value) tuples for user-saved voices
        """
        with gr.Blocks(title=self.title) as demo:
            with gr.Row():
                gr.Markdown(f"# 🎙️ {self.title}")
                abort_btn = gr.Button(
                    "⏹ Abort",
                    variant="stop",
                    scale=0,
                    min_width=110,
                    elem_id="global-abort-btn",
                )
            abort_btn.click(on_abort or (lambda: None), inputs=[], outputs=[])

            with gr.Tabs():
                with gr.Tab("🎤 TTS"):
                    self._create_tts_tab(on_tts_generate, speakers_list or [],
                                         user_voice_choices or [],
                                         on_voice_speed_change=on_voice_speed_change)

                with gr.Tab("🎯 Clone"):
                    self._create_cloning_tab(on_clone_voice, on_save_clone)

                with gr.Tab("🎨 Design"):
                    self._create_design_tab(on_design_voice, on_save_design)

                with gr.Tab("📜 Script"):
                    self._create_script_tab(on_script_run)

                with gr.Tab("📝 ASR"):
                    self._create_asr_tab(on_transcribe)

                with gr.Tab("🔊 Stable Audio"):
                    self._create_stable_audio_tab(on_stable_audio_generate)

                with gr.Tab("⚙️ Settings"):
                    self._create_settings_tab(on_save_hf_token=on_save_hf_token,
                                              hf_token=hf_token)

                with gr.Tab("ℹ️ Info"):
                    self._create_info_tab()

        self.demo = demo
        return demo
    
    # ── Voice label helpers ───────────────────────────────────────────────────

    _TYPE_ICON = {
        "custom": "🔧",   # built-in speaker ID
        "design": "🎨",   # voice design
        "clone":  "🎭",   # voice clone
    }
    _GENDER_SYMBOL = {
        "Male":   "♂",
        "Female": "♀",
    }
    _TYPE_LABEL = {
        "custom": "Built-in",
        "design": "Designed",
        "clone":  "Clone",
    }

    @classmethod
    def _voice_label(cls, v: dict) -> str:
        """Build a rich dropdown label for one voice dict."""
        name        = v.get("name", "")
        type_icon   = cls._TYPE_ICON.get(v.get("model", ""), "❓")
        type_label  = cls._TYPE_LABEL.get(v.get("model", ""), "")
        gender_sym  = cls._GENDER_SYMBOL.get(v.get("gender", ""), "")
        age         = v.get("age", "")
        desc        = v.get("desc", "")

        meta = "  ".join(filter(None, [
            f"{type_icon} {type_label}" if type_label else type_icon,
            f"{gender_sym}" if gender_sym else "",
            f"{age} y/o" if age else "",
        ]))
        if desc:
            return f"{name}   {meta}   ·   {desc}"
        return f"{name}   {meta}"

    def _create_tts_tab(self, callback: Callable, voices_data: list,
                        user_voice_choices: list = [],
                        on_voice_speed_change: Optional[Callable] = None):
        """Create Text-to-Speech tab.

        voices_data: ordered list of voice dicts from main_voices.json.
        user_voice_choices: list of (label, value) tuples for user-saved voices.
        """
        gr.Markdown("## 🎤 Text-to-Speech Generation")

        # User voices first, then built-in voices
        builtin_choices = [
            (self._voice_label(v), v["name"])
            for v in voices_data
            if v.get("name")
        ]
        choices = user_voice_choices + builtin_choices
        default_voice = choices[0][1] if choices else None

        language = gr.Dropdown(
            choices=["auto", "english", "chinese"],
            value="auto",
            label="🌍 Language",
        )

        selected_voice = gr.Dropdown(
            choices=choices,
            value=default_voice,
            label="👤 Voice",
        )

        speed = gr.Slider(
            minimum=0.5, maximum=2.0, step=0.05, value=1.0,
            label="⚡ Speed  (1.0 = normal · 0.8 = 20% slower · 1.2 = 20% faster)",
        )

        if on_voice_speed_change:
            selected_voice.change(
                on_voice_speed_change,
                inputs=[selected_voice],
                outputs=[speed],
            )

        text_input = gr.TextArea(
            label="📝 Text Input",
            value="Hello, I'm Qwen text to speech AI model.",
            placeholder="Enter text to convert to speech...",
        )

        generate_btn = gr.Button("🎵 Generate Speech", variant="primary")
        audio_output = gr.Audio(label="🔊 Generated Audio")
        status_out   = gr.Textbox(label="📋 Log", interactive=False,
                                   lines=10, max_lines=200)

        generate_btn.click(
            callback,
            inputs=[text_input, language, selected_voice, speed],
            outputs=[audio_output, status_out],
        )
    
    def _create_cloning_tab(self, callback: Callable, on_save: Callable):
        """Create Voice Cloning tab (single-audio mode only; batch JSON lives in Script tab)."""
        gr.Markdown("## 🎯 Voice Cloning")
        gr.Markdown(
            "Upload a reference audio clip to clone that voice and synthesise new speech with it. "
            "For multi-speaker scripts use the **📜 Script** tab."
        )

        with gr.Row():
            ref_audio = gr.File(label="🔊 Reference Audio", file_types=["audio"])
            ref_text  = gr.Textbox(label="📝 Reference Text (spoken in the audio)", lines=4)

        target_text  = gr.Textbox(label="📝 Text to Synthesise", lines=3,
                                   value="Hello, I'm Qwen text to speech AI model.")
        speed        = gr.Slider(
            minimum=0.5, maximum=2.0, step=0.05, value=1.0,
            label="⚡ Speed  (1.0 = normal · 0.8 = 20% slower · 1.2 = 20% faster)",
        )
        clone_btn    = gr.Button("✨ Clone Voice", variant="primary")
        audio_output = gr.Audio(label="🔊 Output")
        status       = gr.Textbox(label="📋 Log", interactive=False,
                                   lines=10, max_lines=200)

        clone_btn.click(callback, inputs=[ref_audio, ref_text, target_text, speed],
                        outputs=[audio_output, status])

        with gr.Accordion("💾 Save this voice to My Voices", open=False):
            with gr.Row():
                save_name = gr.Textbox(label="Voice Name", placeholder="e.g. My Clone")
                save_desc = gr.Textbox(label="Description (optional)",
                                       placeholder="Brief description of the voice")
            gr.Markdown("*The current Speed value above will be stored as this voice's default.*")
            save_btn    = gr.Button("💾 Save")
            save_status = gr.Textbox(label="", interactive=False, show_label=False)
            save_btn.click(
                on_save,
                inputs=[save_name, save_desc, ref_audio, ref_text, speed],
                outputs=[save_status],
            )
    
    def _create_design_tab(self, callback: Callable, on_save: Callable):
        """Create Voice Design tab (single-voice mode; multi-speaker scripts live in Script tab)."""
        gr.Markdown("## 🎨 Voice Design")
        gr.Markdown(
            "Describe a voice in plain text and synthesise speech with it. "
            "For multi-speaker scripts use the **📜 Script** tab."
        )

        instructions = gr.Textbox(
            label="🗣️ Voice Description",
            placeholder="e.g. Female, 35 years old, warm and slightly husky, slow pacing, emotionally warm tone",
            lines=3,
        )
        text_input = gr.TextArea(
            label="📝 Text to Synthesise",
            placeholder="Enter the text you want to speak with the designed voice...",
            value="Hello, I'm Qwen text to speech AI model.",
        )
        speed = gr.Slider(
            minimum=0.5, maximum=2.0, step=0.05, value=1.0,
            label="⚡ Speed  (1.0 = normal · 0.8 = 20% slower · 1.2 = 20% faster)",
        )
        design_btn   = gr.Button("✨ Design & Generate", variant="primary")
        audio_output = gr.Audio(label="🔊 Generated Audio")
        status_out   = gr.Textbox(label="📋 Log", interactive=False,
                                   lines=10, max_lines=200)

        design_btn.click(callback, inputs=[text_input, instructions, speed],
                         outputs=[audio_output, status_out])

        with gr.Accordion("💾 Save this voice to My Voices", open=False):
            with gr.Row():
                save_name = gr.Textbox(label="Voice Name", placeholder="e.g. My Designed Voice")
                save_desc = gr.Textbox(label="Description (optional)",
                                       placeholder="Brief description of the voice")
            gr.Markdown("*The current Speed value above will be stored as this voice's default.*")
            save_btn    = gr.Button("💾 Save")
            save_status = gr.Textbox(label="", interactive=False, show_label=False)
            save_btn.click(
                on_save,
                inputs=[save_name, save_desc, instructions, speed],
                outputs=[save_status],
            )
    def _create_script_tab(self, callback: Callable):
        """Create the unified multi-speaker Script tab."""
        gr.Markdown("## 📜 Script")
        with gr.Accordion("📖 Reference — speakers, directions, inline audio markers", open=False):
            gr.Markdown(self._script_help)

        with gr.Row():
            json_file   = gr.File(label="📁 Upload Script JSON", file_types=[".json"])
            with gr.Column():
                load_btn     = gr.Button("📥 Load from File")
                template_btn = gr.Button("📋 Load Template")
                clear_btn    = gr.Button("🗑️ Clear")

        json_editor = gr.Code(
            label="✏️ Script JSON",
            language="json",
            lines=25,
            interactive=True,
            value=self._script_template,
        )

        json_preview = gr.JSON(label="👁️ Parsed Preview")

        with gr.Row():
            run_btn       = gr.Button("▶ Run Script", variant="primary", scale=3)
            shutdown_chk  = gr.Checkbox(
                label="⏻  Shutdown machine when done",
                value=False,
                scale=1,
                info="Schedules OS shutdown ~60 s after the job finishes. Cancel: shutdown /a (Win) · shutdown -c (Linux)",
            )

        status_out   = gr.Textbox(label="📋 Log", interactive=False,
                                   lines=20, max_lines=400)
        audio_output = gr.Audio(label="🔊 Merged Output")

        # ── helpers ──────────────────────────────────────────────────────────

        def _load_from_file(file):
            if file is None:
                return "", {}
            import json
            try:
                with open(file.name, "r", encoding="utf-8") as fh:
                    content = json.load(fh)
                return json.dumps(content, indent=2, ensure_ascii=False), content
            except Exception as exc:
                return f"// Error loading file: {exc}", {}

        # ── event wiring ─────────────────────────────────────────────────────

        load_btn.click(_load_from_file, inputs=[json_file], outputs=[json_editor, json_preview])

        template_btn.click(
            lambda: (self._script_template, self._parse_json_for_preview(self._script_template)),
            outputs=[json_editor, json_preview],
        )

        clear_btn.click(
            lambda: ('{\n  "speakers": [],\n  "scenes": []\n}', {}),
            outputs=[json_editor, json_preview],
        )

        json_editor.change(
            lambda x: self._parse_json_for_preview(x),
            inputs=[json_editor],
            outputs=[json_preview],
        )

        run_btn.click(callback, inputs=[json_editor, shutdown_chk], outputs=[audio_output, status_out])

    def _create_asr_tab(self, callback: Callable):
        """Create Speech Recognition tab."""
        gr.Markdown("## 📝 Automatic Speech Recognition")
        
        audio_input = gr.File(label="🔊 Upload Audio", file_types=["audio"])
        language = gr.Dropdown(
            choices=["auto", "english", "chinese"],
            value="auto",
            label="🌍 Language"
        )
        
        transcribe_btn = gr.Button("🎙️ Transcribe")
        text_output = gr.Textbox(label="📄 Transcription")
        
        transcribe_btn.click(
            callback,
            inputs=[audio_input, language],
            outputs=[text_output]
        )
    
    def _create_stable_audio_tab(self, callback: Optional[Callable]):
        """Create the Stable Audio sound-effects generation tab."""
        gr.Markdown("## 🔊 Stable Audio — Sound Effects & Music")
        gr.Markdown(
            "Generate sound effects, ambience, or music from a text description using "
            "**Stable Audio Open 1.0** (stabilityai/stable-audio-open-1.0). "
            "The model (~3 GB) is downloaded from HuggingFace on first use."
        )

        prompt = gr.Textbox(
            label="🎵 Prompt",
            placeholder=(
                "e.g. Heavy rain on a metal roof, distant rolling thunder, wind howling through trees"
            ),
            lines=3,
        )

        with gr.Row():
            duration = gr.Slider(
                minimum=1.0, maximum=30.0, step=0.5, value=10.0,
                label="⏱️ Duration (seconds)",
            )
            steps = gr.Slider(
                minimum=20, maximum=150, step=10, value=100,
                label="🔁 Steps  (more = higher quality, slower)",
            )
            cfg_scale = gr.Slider(
                minimum=1.0, maximum=15.0, step=0.5, value=7.0,
                label="🎯 CFG Scale  (higher = follows prompt more strictly)",
            )

        generate_btn = gr.Button("🔊 Generate Sound", variant="primary")
        audio_output = gr.Audio(label="🎵 Generated Audio")
        status_out   = gr.Textbox(label="📋 Log", interactive=False,
                                   lines=10, max_lines=200)

        if callback:
            generate_btn.click(
                callback,
                inputs=[prompt, duration, steps, cfg_scale],
                outputs=[audio_output, status_out],
            )
        else:
            generate_btn.click(
                lambda *_: (None, "✗ Stable Audio callback not wired"),
                inputs=[prompt, duration, steps, cfg_scale],
                outputs=[audio_output, status_out],
            )

    def _create_settings_tab(self, on_save_hf_token: Optional[Callable] = None,
                             hf_token: str = ""):
        """Create Settings tab."""
        gr.Markdown("## ⚙️ Settings")

        with gr.Accordion("🔑 HuggingFace Token", open=not hf_token):
            gr.Markdown(
                "Some models (e.g. **Stable Audio Open 1.0**) are gated and require you to "
                "accept their license on HuggingFace and provide an access token.  \n"
                "Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) "
                "(read access is sufficient).  \n"
                "The token is saved locally in `config.json` and applied on every startup."
            )
            token_input = gr.Textbox(
                value=hf_token,
                label="Access Token",
                placeholder="hf_...",
                type="password",
            )
            save_token_btn = gr.Button("💾 Save Token", variant="primary")
            token_status   = gr.Textbox(label="", interactive=False, show_label=False)

            if on_save_hf_token:
                save_token_btn.click(on_save_hf_token, inputs=[token_input],
                                     outputs=[token_status])
            else:
                save_token_btn.click(lambda _: "✗ Callback not wired",
                                     inputs=[token_input], outputs=[token_status])
    
    def _create_info_tab(self):
        """Create Information tab."""
        gr.Markdown("## About")
        gr.Markdown("""
        # Qwen Voice TTS Studio
        
        A comprehensive voice synthesis and cloning application powered by Qwen3-TTS models.
        
        ### Features
        - 🎤 Text-to-Speech synthesis
        - 🎯 Voice cloning
        - 🎨 Voice design
        - 📝 Speech recognition
        - 📚 Voice library management
        
        ### Models
        - Qwen3-TTS-12Hz-1.7B-CustomVoice
        - Qwen3-TTS-12Hz-1.7B-Base
        - Qwen3-TTS-12Hz-1.7B-VoiceDesign
        - Qwen3-ASR-1.7B
        
        ### Documentation
        For more information and documentation, visit the GitHub repository.
        """)
    
    def _parse_json_for_preview(self, json_str: str) -> dict:
        """Parse JSON string and return for preview display.
        
        Args:
            json_str: JSON string from editor
            
        Returns:
            Parsed JSON dict for gr.JSON() display, or empty dict on error
        """
        import json
        try:
            if not json_str or not json_str.strip():
                return {}
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Return partial valid JSON or empty dict on parse error
            return {"error": "Invalid JSON syntax"}
        except Exception:
            return {"error": "Error parsing JSON"}
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Launch the Gradio interface."""
        if self.demo:
            self.demo.queue()
            self.demo.launch(
                share=share,
                server_name=server_name,
                server_port=server_port,
                theme=gr.themes.Soft()
            )
