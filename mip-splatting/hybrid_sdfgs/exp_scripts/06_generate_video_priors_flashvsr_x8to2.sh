#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/root/autodl-tmp/HBSR"
INPUT_IMAGES="/root/autodl-tmp/kitchen/images_8"
OUTPUT_ROOT="/root/autodl-tmp/priors/kitchen_video_flashvsr_x8to2"
VIDEO_SR_REPO="/root/autodl-tmp/FlashVSR"
FLASHVSR_PYTHON="/root/miniconda3/envs/flashvsr/bin/python"
FLASHVSR_MODEL_DIR="/root/autodl-tmp/hub/FlashVSR-v1.1"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "${VIDEO_SR_REPO}" ]]; then
  echo "[video-prior-flashvsr-x8to2] repo not found: ${VIDEO_SR_REPO}"
  exit 1
fi

if [[ ! -x "${FLASHVSR_PYTHON}" ]]; then
  echo "[video-prior-flashvsr-x8to2] python not found or not executable: ${FLASHVSR_PYTHON}"
  exit 1
fi

if [[ ! -d "${FLASHVSR_MODEL_DIR}" ]]; then
  echo "[video-prior-flashvsr-x8to2] model folder missing: ${FLASHVSR_MODEL_DIR}"
  exit 1
fi

cd "${HBSR_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python hybrid_sdfgs/tools/generate_video_sr_priors.py \
  --model flashvsr \
  --repo_root "${VIDEO_SR_REPO}" \
  --python_exe "${FLASHVSR_PYTHON}" \
  --input_dir "${INPUT_IMAGES}" \
  --output_root "${OUTPUT_ROOT}" \
  --flashvsr_model_dir "${FLASHVSR_MODEL_DIR}" \
  --flashvsr_fps 24 \
  --flashvsr_scale 4.0 \
  --flashvsr_chunk_size 60 \
  --flashvsr_chunk_min 50 \
  --flashvsr_align_mode "ceil" \
  --flashvsr_dtype "bf16" \
  --flashvsr_device "cuda" \
  --flashvsr_kv_ratio 3.0 \
  --flashvsr_local_range 11 \
  --flashvsr_sparse_ratio 2.0 \
  --flashvsr_spatial_tile_w 1600 \
  --flashvsr_spatial_tile_h 1088 \
  --flashvsr_spatial_overlap 128 \
  --flashvsr_view_group_mode "none"

echo "[generate-video-prior-flashvsr-x8to2] done: ${OUTPUT_ROOT}/priors"
