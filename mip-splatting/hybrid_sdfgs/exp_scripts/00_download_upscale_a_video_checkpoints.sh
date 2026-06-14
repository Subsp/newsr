#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

UAV_ROOT="${UAV_ROOT:-/Users/ltl/Desktop/codex_playground/video_sr_models/Upscale-A-Video}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-${UAV_ROOT}/pretrained_models}"
MODEL_ROOT="${MODEL_ROOT:-${PRETRAINED_ROOT}/upscale_a_video}"
CHECK_SCRIPT="${CHECK_SCRIPT:-${WORKSPACE_ROOT}/hybrid_sdfgs/tools/check_upscale_a_video_checkpoints.py}"
DOWNLOAD_LLAVA="${DOWNLOAD_LLAVA:-0}"
DRIVE_FOLDER_URL="${DRIVE_FOLDER_URL:-https://drive.google.com/drive/folders/1O8pbeR1hsRlFUU8O4EULe-lOKNGEWZl1?usp=sharing}"
LLAVA_MODEL_ID="${LLAVA_MODEL_ID:-liuhaotian/llava-v1.5-13b}"

echo "[uav-download] UAV_ROOT=${UAV_ROOT}"
echo "[uav-download] MODEL_ROOT=${MODEL_ROOT}"

mkdir -p "${MODEL_ROOT}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[uav-download] missing command: $1" >&2
    return 1
  fi
}

install_gdown() {
  echo "[uav-download] gdown not found, installing into current Python env..."
  if python -m pip install "gdown>=5,<6"; then
    return 0
  fi

  echo "[uav-download] plain pip install failed, retrying with trusted-host..."
  python -m pip install \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    "gdown>=5,<6"
}

echo "[uav-download] current checkpoint status:"
python "${CHECK_SCRIPT}" --root "${MODEL_ROOT}" || true

if ! command -v gdown >/dev/null 2>&1; then
  install_gdown
fi

echo "[uav-download] downloading official pretrained bundle from Google Drive folder..."
gdown --folder "${DRIVE_FOLDER_URL}" -O "${PRETRAINED_ROOT}"

if [[ "${DOWNLOAD_LLAVA}" == "1" ]]; then
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[uav-download] huggingface-cli not found, installing into current Python env..."
    python -m pip install "huggingface_hub>=0.28"
  fi

  echo "[uav-download] downloading optional LLaVA model to ${PRETRAINED_ROOT} ..."
  huggingface-cli download "${LLAVA_MODEL_ID}" \
    --local-dir "${PRETRAINED_ROOT}/liuhaotian-llava-v1.5-13b" \
    --local-dir-use-symlinks False
fi

echo "[uav-download] final checkpoint status:"
python "${CHECK_SCRIPT}" --root "${MODEL_ROOT}"
