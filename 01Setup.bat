@echo off
setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
echo ========================================
echo Qwen Voice TTS Studio 1.1  - Setup Script
echo ========================================
echo.

set "PYTHON_PATH=%SCRIPT_DIR%312\python.exe"
set "VENV_PATH=%SCRIPT_DIR%venv"

echo Checking Python installation
if not exist "%PYTHON_PATH%" (
    echo ERROR: Python not found at %PYTHON_PATH%
    echo Please ensure Python 3.12+ is installed at the specified location.
    pause
    exit /b 1
)

echo Python found: %PYTHON_PATH%
echo.

echo Creating virtual environment
if exist "%VENV_PATH%" (
    choice /C RN /N /M "Virtual environment already exists. (R)euse or create (N)ew?: "
    if errorlevel 2 (
        echo.
        echo Removing existing virtual environment
        rmdir /S /Q "%VENV_PATH%"
        if errorlevel 1 (
            echo ERROR: Failed to remove existing virtual environment
            pause
            exit /b 1
        )
        echo.
        echo Creating virtual environment
        "%PYTHON_PATH%" -m venv "%VENV_PATH%"
        if errorlevel 1 (
            echo ERROR: Failed to create virtual environment
            pause
            exit /b 1
        )
        echo Virtual environment created successfully.
    ) else (
        echo Using existing virtual environment.
    )
) else (
    "%PYTHON_PATH%" -m venv "%VENV_PATH%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)
echo.

echo Activating virtual environment
call "%VENV_PATH%\Scripts\activate.bat"
echo.

echo Upgrading pip
"%VENV_PATH%\Scripts\python.exe" -m pip install --upgrade pip
echo.

echo Detecting NVIDIA GPU
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    set "HAS_NVIDIA_GPU=0"
) else (
    set "HAS_NVIDIA_GPU=1"
)

echo Installing PyTorch
if "%HAS_NVIDIA_GPU%"=="1" (
    echo NVIDIA GPU detected - attempting CUDA-enabled PyTorch install (cu128)
    "%VENV_PATH%\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
    if errorlevel 1 (
        echo.
        echo WARNING: CUDA-enabled PyTorch install failed. Falling back to CPU-only.
        "%VENV_PATH%\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
    )
) else (
    echo No NVIDIA GPU detected - installing CPU-only PyTorch.
    "%VENV_PATH%\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
)

echo.
echo Installing requirements
"%VENV_PATH%\Scripts\python.exe" -m pip install -r "%SCRIPT_DIR%requirements.txt"
echo.

echo.
echo Installing pytz (required by gradio)
"%VENV_PATH%\Scripts\python.exe" -m pip install -U pytz
if errorlevel 1 (
    echo.
    echo WARNING: pytz install failed. Gradio may fail to import.
)
echo.

echo.
echo Installing qwen-asr (for Voice ASR tab)
"%VENV_PATH%\Scripts\python.exe" -m pip install -U qwen-asr
if errorlevel 1 (
    echo.
    echo WARNING: qwen-asr install failed. Voice ASR tab will prompt you to install it manually.
)
echo.

echo.
echo Pinning transformers==4.57.3 (required by qwen-tts)
"%VENV_PATH%\Scripts\python.exe" -m pip install --upgrade --no-deps transformers==4.57.3
if errorlevel 1 (
    echo.
    echo WARNING: Failed to pin transformers to 4.57.3. You may see dependency warnings.
)
echo.

echo.
echo.
echo Installing flash-attn
if "%HAS_NVIDIA_GPU%"=="1" (
    echo NVIDIA GPU detected - installing flash-attn wheel for Windows (Python 3.12, Torch 2.9, cu128)
    "%VENV_PATH%\Scripts\python.exe" -m pip install "https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/flash_attn-2.8.2%%2Bcu128torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl"
    if errorlevel 1 (
        echo.
        echo WARNING: flash-attn install failed. The app will still run but may be slower.
    )
) else (
    echo No NVIDIA GPU detected - skipping flash-attn.
)

echo.
 echo Creating models directory
if not exist "%SCRIPT_DIR%models" mkdir "%SCRIPT_DIR%models"
 echo.

echo ========================================
echo Model Download Options
echo ========================================
echo.
echo The following models will be downloaded:
echo 1. Qwen3-TTS-12Hz-1.7B-CustomVoice (~3.5GB - Pre-built voices)
echo 2. Qwen3-TTS-12Hz-1.7B-Base (~3.5GB - For voice cloning)
echo 3. Qwen3-TTS-12Hz-1.7B-VoiceDesign (~3.5GB - For voice design)
echo.
echo Total download size: ~10-12GB
echo Models will be stored in: %SCRIPT_DIR%models
echo.
set /p DOWNLOAD_MODELS="Would you like to download all models from HuggingFace? (y/n): "

if /i "%DOWNLOAD_MODELS%"=="y" (
    echo.
    echo ========================================
    echo Downloading all models
    echo This may take 20-60 minutes depending on your internet connection.
    echo ========================================
    echo.
     
    echo [1/3] Downloading CustomVoice model
    "%VENV_PATH%\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', local_dir='models/Qwen3-TTS-12Hz-1.7B-CustomVoice', local_dir_use_symlinks=False)"
    if errorlevel 1 (
        echo WARNING: CustomVoice model download failed or incomplete
    ) else (
        echo CustomVoice model downloaded successfully!
    )
    
    echo.
    echo [2/3] Downloading Base model for voice cloning
    "%VENV_PATH%\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='models/Qwen3-TTS-12Hz-1.7B-Base', local_dir_use_symlinks=False)"
    if errorlevel 1 (
        echo WARNING: Base model download failed or incomplete
    ) else (
        echo Base model downloaded successfully!
    )
    
    echo.
    echo [3/3] Downloading VoiceDesign model
    "%VENV_PATH%\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', local_dir='models/Qwen3-TTS-12Hz-1.7B-VoiceDesign', local_dir_use_symlinks=False)"
    if errorlevel 1 (
        echo WARNING: VoiceDesign model download failed or incomplete
    ) else (
        echo VoiceDesign model downloaded successfully!
    )
    
    echo.
    echo ========================================
    echo All models downloaded to %SCRIPT_DIR%models!
    echo ========================================
) else (
    echo.
    echo Skipping model download. Models will be downloaded on first use to %SCRIPT_DIR%models.
)

echo.
echo Creating start script
(
 echo @echo off
 echo set "SCRIPT_DIR=%%~dp0"
 echo set "HF_HOME=%%SCRIPT_DIR%%models"
 echo call "%%SCRIPT_DIR%%venv\Scripts\activate.bat"
 echo set "ARGS="
 echo.
 echo echo ========================================
 echo echo Qwen Voice TTS Studio 1.1  - Startup
 echo echo ========================================
 echo echo.
 echo.
 echo python --version
 echo python -c "import torch; print('Torch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>nul
 echo if errorlevel 1 (
 echo     echo Torch: (not available)
 echo )
 echo.
 echo choice /C YN /N /T 10 /D N /M "Use Qwen Voice TTS Studio on other devices in your network (Y/N)?: "
 echo if errorlevel 2 goto :LAN_OFF
 echo if errorlevel 1 goto :LAN_ON
 echo goto :LAN_OFF
 echo.
 echo :LAN_ON
 echo set "LISTEN_IP="
 echo set "LISTEN_PORT="
 echo set /p "LISTEN_IP=Listen IP (Enter for default 0.0.0.0): "
 echo if "%%LISTEN_IP%%"=="" set "LISTEN_IP=0.0.0.0"
 echo set /p "LISTEN_PORT=Port (Enter for default 7860): "
 echo if "%%LISTEN_PORT%%"=="" set "LISTEN_PORT=7860"
 echo set "ARGS=%%ARGS%% --listen %%LISTEN_IP%% --port %%LISTEN_PORT%%"
 echo goto :BUILD
 echo.
 echo :LAN_OFF
 echo set "LISTEN_IP="
 echo set "LISTEN_PORT="
 echo goto :BUILD
 echo.
 echo :BUILD
 echo echo.
 echo echo Launch command:
 echo echo "%%SCRIPT_DIR%%venv\Scripts\python.exe" "%%SCRIPT_DIR%%qwen_voice_gui.py" %%ARGS%%
 echo echo.
 echo.
 echo "%%SCRIPT_DIR%%venv\Scripts\python.exe" "%%SCRIPT_DIR%%qwen_voice_gui.py" %%ARGS%%
 echo pause
 ) > "%SCRIPT_DIR%02Start.bat"

echo.
echo ========================================
echo Voice Sample Generation
echo ========================================
echo.
echo Would you like to generate MP3 samples for all voice personas?
echo This will create preview samples for the 20 built-in voices.
echo Sample text: "Welcome to Qwen TTS Studio. have fun with voices! hehe"
echo.
echo This process may take 10-20 minutes depending on your hardware.
echo You can also run Create_Samples.bat later to generate samples.
echo.
set /p GENERATE_SAMPLES="Generate voice samples now? (y/n): "

if /i "%GENERATE_SAMPLES%"=="y" (
    echo.
    echo Generating voice samples
    "%VENV_PATH%\Scripts\python.exe" -c "from qwen_voice_gui import QwenVoiceGUI; gui = QwenVoiceGUI(); print('Generating samples'); result = gui.generate_all_samples(); print(result)"
    echo.
    echo Voice samples generated in voicesamples folder!
) else (
    echo.
    echo Skipping sample generation. Run Create_Samples.bat later to generate samples.
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo You can now run 02Start.bat to launch the GUI.
echo.
pause
