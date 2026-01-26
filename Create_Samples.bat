@echo off
echo ========================================
echo  Qwen Voice TTS Studio 1.0  - Sample Generator
echo ========================================
echo.
echo This will generate MP3 samples for all voice personas.
echo Sample text: "Welcome to Qwen TTS Studio. have fun with voices! hehe"
echo.
echo Samples will be saved in the voicesamples folder.
echo This process may take several minutes
echo.
pause

echo.
echo Activating virtual environment
call venv\Scripts\activate.bat

echo.
echo Starting sample generation
python -c "from qwen_voice_gui import QwenVoiceGUI; import gradio as gr; gui = QwenVoiceGUI(); print('Generating samples for all voices'); result = gui.generate_all_samples(progress=lambda x, desc='': print(f'{desc}')); print(result)"

echo.
echo ========================================
echo Sample generation complete!
echo Check the voicesamples folder for MP3 files.
echo ========================================
echo.
pause
