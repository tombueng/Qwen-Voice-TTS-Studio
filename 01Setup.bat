@echo off
setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
echo ========================================
echo Qwen Voice TTS Studio - Setup Script
echo ========================================
echo.

set "VENV_PATH=%SCRIPT_DIR%venv"

:: ── Find Python 3.12 ──────────────────────────────────────────────────────────
echo Searching for Python 3.12...
set "PYTHON_PATH="

:: Method 1: Python Launcher (py.exe) — installed by the official Python installer
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%p in ('py -3.12 -c "import sys; print(sys.executable)"') do set "PYTHON_PATH=%%p"
    )
)

:: Method 2: python / python3 on PATH
if "%PYTHON_PATH%"=="" (
    for %%c in (python3.12 python3 python) do (
        if "%PYTHON_PATH%"=="" (
            %%c --version >nul 2>&1
            if not errorlevel 1 (
                for /f "tokens=2" %%v in ('%%c --version 2^>^&1') do (
                    echo %%v | findstr /B "3.12" >nul
                    if not errorlevel 1 (
                        for /f "delims=" %%p in ('%%c -c "import sys; print(sys.executable)"') do set "PYTHON_PATH=%%p"
                    )
                )
            )
        )
    )
)

if "%PYTHON_PATH%"=="" (
    echo.
    echo ERROR: Python 3.12 not found!
    echo.
    echo Please install Python 3.12 from:
    echo   https://www.python.org/downloads/
    echo.
    echo During installation, check "Add Python to PATH".
    echo After installation, close this window and re-run 01Setup.bat.
    echo.
    pause
    exit /b 1
)

echo Python 3.12 found: %PYTHON_PATH%
echo.

:: ── Create virtual environment ────────────────────────────────────────────────
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

:: ── PyTorch ───────────────────────────────────────────────────────────────────
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

echo Installing pytz (required by gradio)
"%VENV_PATH%\Scripts\python.exe" -m pip install -U pytz
if errorlevel 1 (
    echo WARNING: pytz install failed. Gradio may fail to import.
)
echo.

echo Installing qwen-asr (for Voice ASR tab)
"%VENV_PATH%\Scripts\python.exe" -m pip install -U qwen-asr
if errorlevel 1 (
    echo WARNING: qwen-asr install failed. Voice ASR tab will not be available.
)
echo.

echo Pinning transformers==4.57.3 (required by qwen-tts)
"%VENV_PATH%\Scripts\python.exe" -m pip install --upgrade --no-deps transformers==4.57.3
if errorlevel 1 (
    echo WARNING: Failed to pin transformers to 4.57.3. You may see dependency warnings.
)
echo.

:: ── flash-attn (NVIDIA only) ──────────────────────────────────────────────────
if "%HAS_NVIDIA_GPU%"=="1" (
    echo Installing flash-attn wheel for Windows (Python 3.12, Torch 2.9, cu128)
    "%VENV_PATH%\Scripts\python.exe" -m pip install "https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/flash_attn-2.8.2%%2Bcu128torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl"
    if errorlevel 1 (
        echo WARNING: flash-attn install failed. The app will still run but may be slower.
    )
) else (
    echo No NVIDIA GPU detected - skipping flash-attn.
)
echo.

:: ── Models ────────────────────────────────────────────────────────────────────
if not exist "%SCRIPT_DIR%models" mkdir "%SCRIPT_DIR%models"

echo ========================================
echo Model Download Options
echo ========================================
echo.
echo The following models will be downloaded:
echo   1. Qwen3-TTS-12Hz-1.7B-CustomVoice  (~3.5 GB)
echo   2. Qwen3-TTS-12Hz-1.7B-Base          (~3.5 GB - voice cloning)
echo   3. Qwen3-TTS-12Hz-1.7B-VoiceDesign   (~3.5 GB - voice design)
echo   4. Qwen3-ASR-1.7B                    (~3.5 GB - speech recognition)
echo.
echo Total download size: ~10-14 GB
echo Models will be stored in: %SCRIPT_DIR%models
echo.
set /p DOWNLOAD_MODELS="Download all models now? (y/n): "

if /i "%DOWNLOAD_MODELS%"=="y" (
    echo.
    echo Downloading models (this may take 20-60 minutes)...
    echo.

    echo [1/4] Downloading CustomVoice model
    "%VENV_PATH%\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', local_dir='models/Qwen3-TTS-12Hz-1.7B-CustomVoice', local_dir_use_symlinks=False)"
    if errorlevel 1 echo WARNING: CustomVoice model download failed or incomplete

    echo.
    echo [2/4] Downloading Base model (voice cloning)
    "%VENV_PATH%\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='models/Qwen3-TTS-12Hz-1.7B-Base', local_dir_use_symlinks=False)"
    if errorlevel 1 echo WARNING: Base model download failed or incomplete

    echo.
    echo [3/4] Downloading VoiceDesign model
    "%VENV_PATH%\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', local_dir='models/Qwen3-TTS-12Hz-1.7B-VoiceDesign', local_dir_use_symlinks=False)"
    if errorlevel 1 echo WARNING: VoiceDesign model download failed or incomplete

    echo.
    echo [4/4] Downloading ASR model
    "%VENV_PATH%\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-1.7B', local_dir='models/Qwen3-ASR-1.7B', local_dir_use_symlinks=False)"
    if errorlevel 1 echo WARNING: ASR model download failed or incomplete

    echo.
    echo ========================================
    echo Models downloaded to %SCRIPT_DIR%models
    echo ========================================
) else (
    echo Skipping model download. Run download_models.py later to fetch models.
)

echo.
echo ========================================
echo Voice Sample Generation
echo ========================================
echo.
echo Would you like to generate audio samples for all built-in voice personas?
echo This may take 10-20 minutes depending on your hardware.
echo You can also run generate_samples.py later.
echo.
set /p GENERATE_SAMPLES="Generate voice samples now? (y/n): "

if /i "%GENERATE_SAMPLES%"=="y" (
    echo.
    echo Generating voice samples...
    "%VENV_PATH%\Scripts\python.exe" "%SCRIPT_DIR%generate_samples.py"
    echo.
    echo Voice samples saved to voicesamples/
) else (
    echo Skipping sample generation.
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Run 02Start.bat to launch the application.
echo.
pause
