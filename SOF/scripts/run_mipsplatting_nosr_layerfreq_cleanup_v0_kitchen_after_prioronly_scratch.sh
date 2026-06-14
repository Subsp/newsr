#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output}"

INPUT_RUN_TAG="${INPUT_RUN_TAG:-mip30k_r1_qwen_prioronly_scratch_v0}"
INPUT_MODEL_DIR="${INPUT_MODEL_DIR:-${OUTPUT_ROOT}/mipsplatting_prior_only_from_scratch_v0/${SCENE_NAME}/${INPUT_RUN_TAG}}"
INPUT_ITERATION="${INPUT_ITERATION:-30000}"
CLEANUP_ITERS="${CLEANUP_ITERS:-2000}"
FINAL_ITER="${FINAL_ITER:-$(( INPUT_ITERATION + CLEANUP_ITERS ))}"

RUN_TAG="${RUN_TAG:-${INPUT_RUN_TAG}_to${FINAL_ITER}_trainr${TRAIN_RESOLUTION:-4}_nosr28_layerfreq_cleanup_v0}"

echo "[nosr-after-prioronly-scratch-v0] input model  : ${INPUT_MODEL_DIR}"
echo "[nosr-after-prioronly-scratch-v0] input iter   : ${INPUT_ITERATION}"
echo "[nosr-after-prioronly-scratch-v0] cleanup iters: ${CLEANUP_ITERS}"
echo "[nosr-after-prioronly-scratch-v0] run tag      : ${RUN_TAG}"

INPUT_MODEL_DIR="${INPUT_MODEL_DIR}" \
INPUT_ITERATION="${INPUT_ITERATION}" \
CLEANUP_ITERS="${CLEANUP_ITERS}" \
FINAL_ITER="${FINAL_ITER}" \
RUN_TAG="${RUN_TAG}" \
bash "${SCRIPT_DIR}/run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen.sh"
