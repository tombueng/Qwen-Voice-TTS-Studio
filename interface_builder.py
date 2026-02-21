"""Gradio UI interface builder module."""
from pathlib import Path
from typing import Optional, Callable
import gradio as gr


class InterfaceBuilder:
    """Builds the Gradio UI interface."""
    
    def __init__(self, title: str = "Qwen Voice TTS Studio"):
        self.title = title
        self.demo = None
    
    def create_interface(self,
                        on_tts_generate: Callable,
                        on_clone_voice: Callable,
                        on_design_voice: Callable,
                        on_script_run: Callable,
                        on_transcribe: Callable,
                        on_save_clone: Callable,
                        on_save_design: Callable,
                        speakers_list: Optional[list] = None,
                        user_voice_choices: Optional[list] = None) -> gr.Blocks:
        """Create the main Gradio interface.

        speakers_list: ordered list of voice dicts from main_voices.json
        user_voice_choices: list of (label, value) tuples for user-saved voices
        """
        with gr.Blocks(title=self.title) as demo:
            gr.Markdown(f"# 🎙️ {self.title}")

            with gr.Tabs():
                with gr.Tab("🎤 TTS"):
                    self._create_tts_tab(on_tts_generate, speakers_list or [],
                                         user_voice_choices or [])

                with gr.Tab("🎯 Clone"):
                    self._create_cloning_tab(on_clone_voice, on_save_clone)

                with gr.Tab("🎨 Design"):
                    self._create_design_tab(on_design_voice, on_save_design)

                with gr.Tab("📜 Script"):
                    self._create_script_tab(on_script_run)

                with gr.Tab("📝 ASR"):
                    self._create_asr_tab(on_transcribe)

                with gr.Tab("⚙️ Settings"):
                    self._create_settings_tab()

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
                        user_voice_choices: list = []):
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

        text_input = gr.TextArea(
            label="📝 Text Input",
            value="Hello, I'm Qwen text to speech AI model.",
            placeholder="Enter text to convert to speech...",
        )

        generate_btn = gr.Button("🎵 Generate Speech", variant="primary")
        audio_output = gr.Audio(label="🔊 Generated Audio")
        status_out   = gr.Textbox(label="📋 Status", interactive=False)

        generate_btn.click(
            callback,
            inputs=[text_input, language, selected_voice],
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
        clone_btn    = gr.Button("✨ Clone Voice", variant="primary")
        audio_output = gr.Audio(label="🔊 Output")
        status       = gr.Textbox(label="📋 Status", interactive=False)

        clone_btn.click(callback, inputs=[ref_audio, ref_text, target_text], outputs=[audio_output, status])

        with gr.Accordion("💾 Save this voice to My Voices", open=False):
            with gr.Row():
                save_name = gr.Textbox(label="Voice Name", placeholder="e.g. My Clone")
                save_desc = gr.Textbox(label="Description (optional)",
                                       placeholder="Brief description of the voice")
            save_btn    = gr.Button("💾 Save")
            save_status = gr.Textbox(label="", interactive=False, show_label=False)
            save_btn.click(
                on_save,
                inputs=[save_name, save_desc, ref_audio, ref_text],
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
        design_btn   = gr.Button("✨ Design & Generate", variant="primary")
        audio_output = gr.Audio(label="🔊 Generated Audio")
        status_out   = gr.Textbox(label="📋 Status", interactive=False)

        design_btn.click(callback, inputs=[text_input, instructions], outputs=[audio_output, status_out])

        with gr.Accordion("💾 Save this voice to My Voices", open=False):
            with gr.Row():
                save_name = gr.Textbox(label="Voice Name", placeholder="e.g. My Designed Voice")
                save_desc = gr.Textbox(label="Description (optional)",
                                       placeholder="Brief description of the voice")
            save_btn    = gr.Button("💾 Save")
            save_status = gr.Textbox(label="", interactive=False, show_label=False)
            save_btn.click(
                on_save,
                inputs=[save_name, save_desc, instructions],
                outputs=[save_status],
            )
    
    # ── Script tab template ───────────────────────────────────────────────────

    _SCRIPT_TEMPLATE = """{
  "speakers": [
    {
      "name": "Donald Trump",
      "ref_audio": "trump.mp3",
      "ref_text": "Thank you very much. It's a privilege to be here at this forum where leaders in business science art diplomacy and world affairs have gathered for many many years to discuss how we can advance prosperity security and peace. I'm here today to represent the interests of the American people and to affirm America's friendship and partnership in building a better world. Like all nations represented at this great forum America hopes for a future in which everyone can prosper. And every child can grow up free from violence poverty and fear. Over the past year we have made extraordinary strides in the US. We are lifting up forgotten communities creating exciting new opportunities and helping every American find their path to the American dream. The dream of a great job a safe home and a better life for their children after years of stagnation. The United States is once again experiencing strong economic growth. The stock market is smashing one record after another and has added more than 7 trillion dollars in new wealth since my election."
    },
    {
      "name": "Narrator",
      "ref_speaker": "aiden"
    },
    {
      "name": "Character",
      "ref_description": "Female, 30 years old, warm and friendly voice, calm measured pacing"
    }
  ],
  "scenes": [
    {
      "pos": 1,
      "title": "A Short Greeting",
      "dialog": [
        {"pos": 1, "speaker": "Narrator",      "text": "The story begins on a quiet morning.",        "conotation": "calm, measured"},
        {"pos": 2, "speaker": "Donald Trump",  "text": "Good morning! It is wonderful to see you.",   "conotation": "warm, friendly"},
        {"pos": 3, "speaker": "Character",     "text": "Likewise. Shall we get started?",             "conotation": "gentle, focused"}
      ]
    }
  ]
}"""

    def _create_script_tab(self, callback: Callable):
        """Create the unified multi-speaker Script tab."""
        gr.Markdown("## 📜 Script")
        gr.Markdown(
            "Write a multi-speaker script in JSON. "
            "Each speaker can be defined by **reference audio** (voice clone — file must be in the `voices/` folder, "
            "just use the filename e.g. `alice.mp3`), "
            "a **built-in speaker ID** (`ref_speaker`), "
            "or a **text description** (`ref_description`). "
            "All dialog lines are rendered and merged into a single WAV output."
        )

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
            value=self._SCRIPT_TEMPLATE,
        )

        json_preview = gr.JSON(label="👁️ Parsed Preview")

        run_btn      = gr.Button("▶ Run Script", variant="primary")
        status_out   = gr.Textbox(label="📋 Status", interactive=False, lines=3)
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
            lambda: (self._SCRIPT_TEMPLATE, self._parse_json_for_preview(self._SCRIPT_TEMPLATE)),
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

        run_btn.click(callback, inputs=[json_editor], outputs=[audio_output, status_out])

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
    
    def _create_settings_tab(self):
        """Create Settings tab."""
        gr.Markdown("## ⚙️ Settings")
        
        device = gr.Dropdown(
            choices=["cuda", "cpu"],
            value="cuda",
            label="💻 Compute Device"
        )
        
        model_dtype = gr.Dropdown(
            choices=["bfloat16", "float32"],
            value="bfloat16",
            label="🔢 Model Dtype"
        )
        
        save_btn = gr.Button("💾 Save Settings")
        status = gr.Textbox(label="📋 Status")
        
        save_btn.click(
            lambda d, dt: "Settings saved ✓",
            inputs=[device, model_dtype],
            outputs=[status]
        )
    
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
