#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/root/autodl-tmp/HBSR"
INPUT_IMAGES="/root/autodl-tmp/kitchen/images_8"
COLMAP_SPARSE_DIR="/root/autodl-tmp/kitchen/sparse/0"
OUTPUT_ROOT="/root/autodl-tmp/priors/kitchen_video_flashvsr_x8to2_officiallike_seqmat"
VIDEO_SR_REPO="/root/autodl-tmp/FlashVSR"
FLASHVSR_PYTHON="/root/miniconda3/envs/flashvsr/bin/python"
FLASHVSR_MODEL_DIR="/root/autodl-tmp/hub/FlashVSR-v1.1"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "${VIDEO_SR_REPO}" ]]; then
  echo "[video-prior-flashvsr-x8to2-officiallike-seqmat] repo not found: ${VIDEO_SR_REPO}"
  exit 1
fi

if [[ ! -x "${FLASHVSR_PYTHON}" ]]; then
  echo "[video-prior-flashvsr-x8to2-officiallike-seqmat] python not found or not executable: ${FLASHVSR_PYTHON}"
  exit 1
fi

if [[ ! -d "${FLASHVSR_MODEL_DIR}" ]]; then
  echo "[video-prior-flashvsr-x8to2-officiallike-seqmat] model folder missing: ${FLASHVSR_MODEL_DIR}"
  exit 1
fi

if [[ ! -d "${COLMAP_SPARSE_DIR}" ]]; then
  echo "[video-prior-flashvsr-x8to2-officiallike-seqmat] COLMAP sparse dir missing: ${COLMAP_SPARSE_DIR}"
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
  --flashvsr_model_type "tiny" \
  --flashvsr_model_dir "${FLASHVSR_MODEL_DIR}" \
  --flashvsr_fps 24 \
  --flashvsr_scale 4.0 \
  --flashvsr_align_mode "floor" \
  --flashvsr_dtype "bf16" \
  --flashvsr_device "cuda" \
  --flashvsr_kv_ratio 3.0 \
  --flashvsr_local_range 11 \
  --flashvsr_sparse_ratio 2.0 \
  --flashvsr_if_buffer \
  --flashvsr_no_match_exact_scale_size \
  --flashvsr_view_group_mode "seqmat_pose_als" \
  --flashvsr_colmap_sparse_dir "${COLMAP_SPARSE_DIR}" \
  --flashvsr_view_group_max_len 8 \
  --flashvsr_view_group_min_len 5 \
  --flashvsr_view_group_thresholds "40,70" \
  --flashvsr_view_dir_weight 0.0

echo "[generate-video-prior-flashvsr-x8to2-officiallike-seqmat] done: ${OUTPUT_ROOT}/priors"
