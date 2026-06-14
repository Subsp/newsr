#!/usr/bin/env bash
set -euo pipefail

UAV_REPO="${UAV_REPO:-/root/autodl-tmp/Upscale-A-Video}"
PYTHON_EXE="${PYTHON_EXE:-python}"
INPUT_PATH="${INPUT_PATH:-/root/autodl-tmp/kitchen/images_8}"
OUTPUT_PATH="${OUTPUT_PATH:-/root/autodl-tmp/priors/kitchen_video_uav_official_x8to2_raw}"

cd "${UAV_REPO}"

"${PYTHON_EXE}" inference_upscale_a_video.py \
  -i "${INPUT_PATH}" \
  -o "${OUTPUT_PATH}" \
  --no_llava \
  --save_image
