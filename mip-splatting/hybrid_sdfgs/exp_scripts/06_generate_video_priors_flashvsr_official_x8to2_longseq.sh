#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="${HBSR_ROOT:-/root/autodl-tmp/HBSR}"
INPUT_IMAGES="${INPUT_IMAGES:-/root/autodl-tmp/kitchen/images_8}"
FLASHVSR_MODEL_TYPE="${FLASHVSR_MODEL_TYPE:-tiny}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/priors/kitchen_video_flashvsr_official_${FLASHVSR_MODEL_TYPE}_x8to2_longseq}"
VIDEO_SR_REPO="${VIDEO_SR_REPO:-/root/autodl-tmp/FlashVSR}"
FLASHVSR_PYTHON="${FLASHVSR_PYTHON:-/root/miniconda3/envs/flashvsr/bin/python}"
FLASHVSR_MODEL_DIR="${FLASHVSR_MODEL_DIR:-/root/autodl-tmp/hub/FlashVSR-v1.1}"
FLASHVSR_CHUNK_LEN="${FLASHVSR_CHUNK_LEN:-93}"
FLASHVSR_MIN_CHUNK_LEN="${FLASHVSR_MIN_CHUNK_LEN:-21}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "${FLASHVSR_MODEL_TYPE}" in
  tiny|tiny_long)
    ;;
  *)
    echo "[flashvsr-official-longseq] unsupported FLASHVSR_MODEL_TYPE=${FLASHVSR_MODEL_TYPE}" >&2
    exit 1
    ;;
esac

cd "${HBSR_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${FLASHVSR_PYTHON}" hybrid_sdfgs/tools/run_flashvsr_official_baseline.py \
  --repo_root "${VIDEO_SR_REPO}" \
  --model_dir "${FLASHVSR_MODEL_DIR}" \
  --input_dir "${INPUT_IMAGES}" \
  --output_root "${OUTPUT_ROOT}" \
  --model_type "${FLASHVSR_MODEL_TYPE}" \
  --chunk_len "${FLASHVSR_CHUNK_LEN}" \
  --min_chunk_len "${FLASHVSR_MIN_CHUNK_LEN}"

echo "[generate-video-prior-flashvsr-official-longseq] done: ${OUTPUT_ROOT}/priors"
