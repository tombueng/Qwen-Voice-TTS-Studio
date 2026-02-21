"""Main launcher script for Qwen Voice TTS Studio.

This script serves as the entry point for the application,
handling initialization and launching the web interface.
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Activate logging (ANSI setup + log singleton) before anything else
from logger import log, section, section_end, bold, dim


from app import create_app, QwenVoiceStudio
from utils import get_config_from_env


def print_banner():
    print("\033[1;96m╔══════════════════════════════════════════════════╗\033[0m", flush=True)
    print("\033[1;96m║\033[0m  \033[1;97m🎙️  Qwen Voice TTS Studio\033[0m\033[1;96m                        ║\033[0m", flush=True)
    print("\033[1;96m║\033[0m  \033[2;37m    logging active — ANSI colours enabled\033[0m\033[1;96m       ║\033[0m", flush=True)
    print("\033[1;96m╚══════════════════════════════════════════════════╝\033[0m", flush=True)


def print_system_info(app: QwenVoiceStudio):
    """Print system information."""
    section("System Configuration")
    log.info(f"Device:      {bold(app.device)}")
    log.info(f"Dtype:       {dim(str(app.dtype))}")
    log.info(f"Models dir:  {app.models_dir}")
    log.info(f"Outputs dir: {app.outputs_dir}")
    section_end()


def verify_dependencies() -> bool:
    """Verify required packages are installed."""
    required_packages = [
        "torch",
        "gradio",
        "transformers",
        "soundfile",
        "librosa",
        "numpy",
    ]

    section("Dependency Check")
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            log.ok(f"  {package}")
        except ImportError:
            log.error(f"  {package} — MISSING")
            missing_packages.append(package)

    if missing_packages:
        log.error(f"Missing packages: {', '.join(missing_packages)}")
        log.error("Install them with:  pip install " + " ".join(missing_packages))
        section_end("FAILED")
        return False

    section_end("ok")
    return True


def main():
    """Main entry point."""
    print_banner()

    # Verify dependencies
    if not verify_dependencies():
        sys.exit(1)

    # Get configuration from environment
    config = get_config_from_env()

    log.info("Initialising Qwen Voice Studio…")

    try:
        # Create application
        app = create_app(config)
        print_system_info(app)

        # Pre-load models if requested
        if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
            section("Pre-loading Models")
            app.load_custom_model()
            app.load_base_model()
            app.load_design_model()
            section_end("done")

        # Launch web interface
        server_port = config.get("server_port", 7860)
        server_name = config.get("server_name", "127.0.0.1")
        server_share = config.get("server_share", False)

        log.info(f"Launching web interface on port {bold(str(server_port))}  listen={bold(server_name)}")
        if server_share:
            log.info("Share mode enabled — public URL will be generated")

        app.launch_interface(
            share=server_share,
            server_port=server_port,
            server_name=server_name,
        )

    except KeyboardInterrupt:
        print("\n", flush=True)
        log.warning("Shutting down…")
        sys.exit(0)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
