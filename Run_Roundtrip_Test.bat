@echo off
setlocal EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
set "HF_HOME=%SCRIPT_DIR%models"

echo.
echo ============================================================
echo  Qwen Voice TTS Studio - Round-Trip Test  (TTS ^> ASR)
echo ============================================================
echo.
echo Generates audio for every voice in main_voices.json,
echo transcribes each with Qwen3-ASR, and compares to input text.
echo Results are saved to:  %SCRIPT_DIR%test_roundtrip\
echo.
echo Default test text: "Hello world. My name is Donald."
echo.
echo Options (pass on command line or add manually):
echo   --text   "..."    Override the test text
echo   --voice  "Name"   Test a single voice only
echo   --force           Re-generate TTS audio even if cached
echo   --pass-wer 0.20   Accept up to 20%% WER as PASS (default: 10%%)
echo.

call "%SCRIPT_DIR%venv\Scripts\activate.bat"

echo.
"%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%test_roundtrip.py" %*

echo.
if errorlevel 1 (
    echo Some tests failed or errored — check the output above.
) else (
    echo All tests passed.
)
echo.
pause
