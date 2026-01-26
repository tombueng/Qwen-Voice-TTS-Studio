#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Qwen Voice TTS Studio 0.9 (Beta) - Setup Script (Linux)"
echo "========================================"
echo

echo "NOTE: Linux support is not tested. Please provide a PR if something breaks."
echo

PYTHON_PATH="$SCRIPT_DIR/312/python"
if [[ -x "$PYTHON_PATH" ]]; then
  PY="$PYTHON_PATH"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "ERROR: python3 not found and $PYTHON_PATH not available"
  exit 1
fi

echo "Using Python: $PY"

VENV_PATH="$SCRIPT_DIR/venv"
if [[ -d "$VENV_PATH" ]]; then
  read -r -p "Virtual environment already exists. Reuse (R) or create New (N)? [R/N]: " REUSE
  REUSE="${REUSE:-R}"
  if [[ "${REUSE^^}" == "N" ]]; then
    echo "Removing existing virtual environment"
    rm -rf "$VENV_PATH"
    echo "Creating virtual environment"
    "$PY" -m venv "$VENV_PATH"
  else
    echo "Using existing virtual environment."
  fi
else
  echo "Creating virtual environment"
  "$PY" -m venv "$VENV_PATH"
fi

echo
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip

echo
HAS_NVIDIA_GPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA_GPU=1
  fi
fi

echo "Installing PyTorch"
if [[ "$HAS_NVIDIA_GPU" == "1" ]]; then
  echo "NVIDIA GPU detected - attempting CUDA-enabled PyTorch install (cu128 -> cu121 fallback)"
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 || \
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 || \
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
else
  echo "No NVIDIA GPU detected - installing CPU-only PyTorch."
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
fi

echo
echo "Installing requirements"
python -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo
echo "Installing flash-attn"
if [[ "$HAS_NVIDIA_GPU" == "1" ]]; then
  echo "Attempting to install flash-attn from source (requires build tools + CUDA toolkit on Linux)"
  python -m pip install flash-attn --no-build-isolation || {
    echo
    echo "WARNING: flash-attn install failed. The app will still run but may be slower."
  }
else
  echo "No NVIDIA GPU detected - skipping flash-attn."
fi

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "Run ./02Start.sh to launch."
