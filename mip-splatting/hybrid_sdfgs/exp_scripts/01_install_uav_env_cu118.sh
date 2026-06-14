#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="uav-cu118"
PYTHON_VERSION="3.10"

if ! command -v conda >/dev/null 2>&1; then
  echo "[uav-install] conda not found."
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[uav-install] conda env exists: ${ENV_NAME}"
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip "setuptools<81" wheel

python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

python -m pip install --no-cache-dir \
  accelerate==0.18.0 \
  av==10.0.0 \
  decord==0.6.0 \
  diffusers==0.16.0 \
  "einops>=0.6.1" \
  imageio==2.25.0 \
  imageio-ffmpeg==0.4.8 \
  numpy==1.24.4 \
  timm==0.4.12 \
  transformers==4.28.1 \
  xformers==0.0.20 \
  sentencepiece==0.1.99 \
  rotary-embedding-torch==0.2.3 \
  tqdm \
  pandas \
  omegaconf \
  scipy \
  pyfiglet \
  opencv-python==4.10.0.84

python -m pip check

python - <<'PY'
import torch
import diffusers
import transformers
import cv2
import decord
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_ok:", torch.cuda.is_available())
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("opencv:", cv2.__version__)
print("decord:", decord.__version__)
PY

echo "[uav-install] done env=${ENV_NAME}"
