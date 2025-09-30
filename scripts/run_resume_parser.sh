#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/akshat/new_rp"
VENV_DIR="${VENV_DIR:-$APP_DIR/.venv}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"
LOG_CONFIG="${LOG_CONFIG:-}"

cd "$APP_DIR"

if [ -n "$CONDA_ENV" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090,SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

UVICORN_CMD=("uvicorn" "app:app" "--host" "0.0.0.0" "--port" "$PORT" "--workers" "$WORKERS")

if [ -n "$LOG_CONFIG" ]; then
  UVICORN_CMD+=("--log-config" "$LOG_CONFIG")
fi

exec "${UVICORN_CMD[@]}"
