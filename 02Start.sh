#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME="$SCRIPT_DIR/models"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/venv/bin/activate"

ARGS=()

echo
echo "========================================"
echo "Qwen Voice TTS Studio 0.9 (Beta) - Startup (Linux)"
echo "========================================"
echo

python --version
python -c "import torch; print('Torch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null || true

echo
read -r -p "Use Qwen Voice TTS Studio on other devices in your network (y/N)?: " LAN
LAN="${LAN:-N}"
if [[ "${LAN^^}" == "Y" ]]; then
  LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -z "${LOCAL_IP}" ]]; then
    LOCAL_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')"
  fi
  if [[ -n "${LOCAL_IP}" ]]; then
    echo -e "LAN address (use this on your phone/PC): \e[35mhttp://${LOCAL_IP}:7860\e[0m"
  fi
  read -r -p "Listen IP (Enter for default 0.0.0.0): " LISTEN_IP
  LISTEN_IP="${LISTEN_IP:-0.0.0.0}"
  read -r -p "Port (Enter for default 7860): " LISTEN_PORT
  LISTEN_PORT="${LISTEN_PORT:-7860}"
  if [[ -n "${LOCAL_IP}" ]]; then
    echo -e "LAN address (use this on your phone/PC): \e[35mhttp://${LOCAL_IP}:${LISTEN_PORT}\e[0m"
  fi
  ARGS+=(--listen "$LISTEN_IP" --port "$LISTEN_PORT")
fi

echo
echo "Launch command:"
echo "python $SCRIPT_DIR/qwen_voice_gui.py ${ARGS[*]}"
echo

python "$SCRIPT_DIR/qwen_voice_gui.py" "${ARGS[@]}"
