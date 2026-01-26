@echo off
set "SCRIPT_DIR=%~dp0"
set "HF_HOME=%SCRIPT_DIR%models"
call "%SCRIPT_DIR%venv\Scripts\activate.bat"
set "ARGS="

echo.
echo ========================================
echo Qwen Voice TTS Studio 1.0  - Startup
echo ========================================
echo.

python --version
python -c "import torch; print('Torch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>nul
if errorlevel 1 (
    echo Torch: (not available)
)

echo.
choice /C YN /N /T 10 /D N /M "Use Qwen Voice TTS Studio on other devices in your network (Y/N)?: "
if errorlevel 2 goto :LAN_OFF
if errorlevel 1 goto :LAN_ON
goto :LAN_OFF

:LAN_ON
set "LISTEN_IP="
set "LISTEN_PORT="
for /f %%a in ('echo prompt $E^| cmd') do set "ESC=%%a"
set "LOCAL_IP="
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /R /C:"IPv4 Address"') do (
    set "TMP_IP=%%A"
    set "TMP_IP=!TMP_IP: =!"
    echo !TMP_IP! | findstr /B /C:"169." >nul
    if errorlevel 1 (
        set "LOCAL_IP=!TMP_IP!"
        goto :GOT_IP
    )
)
:GOT_IP
set "LOCAL_IP=%LOCAL_IP: =%"
if "%LOCAL_IP%"=="" set "LOCAL_IP=127.0.0.1"
echo.
echo LAN address (use this on your phone/PC): %ESC%[35mhttp://%LOCAL_IP%:7860%ESC%[0m
echo.
set /p "LISTEN_IP=Listen IP (Enter for default 0.0.0.0): "
if "%LISTEN_IP%"=="" set "LISTEN_IP=0.0.0.0"
set /p "LISTEN_PORT=Port (Enter for default 7860): "
if "%LISTEN_PORT%"=="" set "LISTEN_PORT=7860"
echo LAN address (use this on your phone/PC): %ESC%[35mhttp://%LOCAL_IP%:%LISTEN_PORT%%ESC%[0m
set "ARGS=%ARGS% --listen %LISTEN_IP% --port %LISTEN_PORT%"
goto :BUILD

:LAN_OFF
set "LISTEN_IP="
set "LISTEN_PORT="
goto :BUILD

:BUILD
echo.
echo Launch command:
echo "%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%qwen_voice_gui.py" %ARGS%
echo.

"%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%qwen_voice_gui.py" %ARGS%
pause
