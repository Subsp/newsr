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
TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-1}"

RUN_TAG="${RUN_TAG:-${INPUT_RUN_TAG}_to${FINAL_ITER}_trainr${TRAIN_RESOLUTION}_nosr28_startpreservehf_v1}"

# Preserve only start-model HF whose low-frequency content still agrees with GT.
# This should keep plausible prior texture while letting NoSR suppress wrong-layer
# and semantically shifted detail.
export TRAIN_RESOLUTION
export LAMBDA_SURFACE_START_HF_PRESERVE="${LAMBDA_SURFACE_START_HF_PRESERVE:-0.12}"
export LAYER_FREQUENCY_START_HF_LOWFREQ_KERNEL="${LAYER_FREQUENCY_START_HF_LOWFREQ_KERNEL:-15}"
export LAYER_FREQUENCY_START_HF_LOWFREQ_THRESHOLD="${LAYER_FREQUENCY_START_HF_LOWFREQ_THRESHOLD:-0.060}"
export LAYER_FREQUENCY_START_HF_ENERGY_THRESHOLD="${LAYER_FREQUENCY_START_HF_ENERGY_THRESHOLD:-0.006}"
export LAYER_FREQUENCY_START_HF_MASK_POWER="${LAYER_FREQUENCY_START_HF_MASK_POWER:-1.0}"
export LAYER_FREQUENCY_START_HF_PROTECT_NON_SURFACE="${LAYER_FREQUENCY_START_HF_PROTECT_NON_SURFACE:-1}"
export LAMBDA_NON_SURFACE_HF="${LAMBDA_NON_SURFACE_HF:-0.015}"
export LAMBDA_SURFACE_HF_CLOSURE="${LAMBDA_SURFACE_HF_CLOSURE:-0.010}"
export SURFACE_HF_UPDATE_SCALE="${SURFACE_HF_UPDATE_SCALE:-1.0}"

echo "[nosr-after-prioronly-scratch-preservehf-v0] input model  : ${INPUT_MODEL_DIR}"
echo "[nosr-after-prioronly-scratch-preservehf-v0] input iter   : ${INPUT_ITERATION}"
echo "[nosr-after-prioronly-scratch-preservehf-v0] cleanup iters: ${CLEANUP_ITERS}"
echo "[nosr-after-prioronly-scratch-preservehf-v0] run tag      : ${RUN_TAG}"

INPUT_MODEL_DIR="${INPUT_MODEL_DIR}" \
INPUT_ITERATION="${INPUT_ITERATION}" \
CLEANUP_ITERS="${CLEANUP_ITERS}" \
FINAL_ITER="${FINAL_ITER}" \
RUN_TAG="${RUN_TAG}" \
bash "${SCRIPT_DIR}/run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen.sh"
