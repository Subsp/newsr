#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output}"

# Recipient: the clean 28.5-ish NoSR surface checkpoint.
SURFACE_RUN_TAG="${SURFACE_RUN_TAG:-mip30k_rerun_check_directsrc_r1_v0_to32000_trainr4_nosr28_layerfreq_cleanup_v0}"
INPUT_MODEL_DIR="${INPUT_MODEL_DIR:-${OUTPUT_ROOT}/mipsplatting_nosr_layerfreq_cleanup_v0/${SCENE_NAME}/${SURFACE_RUN_TAG}}"
INPUT_ITERATION="${INPUT_ITERATION:-32000}"

# Donor: the 20dB prior-only scratch model. It contributes only reliable HF.
HF_DONOR_RUN_TAG="${HF_DONOR_RUN_TAG:-mip30k_r1_qwen_prioronly_scratch_v0}"
HF_DONOR_MODEL_DIR="${HF_DONOR_MODEL_DIR:-${OUTPUT_ROOT}/mipsplatting_prior_only_from_scratch_v0/${SCENE_NAME}/${HF_DONOR_RUN_TAG}}"
HF_DONOR_ITERATION="${HF_DONOR_ITERATION:-30000}"
HF_DONOR_CHECKPOINT="${HF_DONOR_CHECKPOINT:-${HF_DONOR_MODEL_DIR}/chkpnt${HF_DONOR_ITERATION}.pth}"

CLEANUP_ITERS="${CLEANUP_ITERS:-1000}"
FINAL_ITER="${FINAL_ITER:-$(( INPUT_ITERATION + CLEANUP_ITERS ))}"
TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-1}"

RUN_TAG="${RUN_TAG:-${SURFACE_RUN_TAG}_plus_${HF_DONOR_RUN_TAG}_hf_to${FINAL_ITER}_trainr${TRAIN_RESOLUTION}_v0}"

# Keep the recipient surface clean while asking it to reproduce donor HF only
# where donor low-frequency content is still close to GT.
export TRAIN_RESOLUTION
export LAYER_FREQUENCY_START_HF_CHECKPOINT="${LAYER_FREQUENCY_START_HF_CHECKPOINT:-${HF_DONOR_CHECKPOINT}}"
export LAMBDA_SURFACE_START_HF_PRESERVE="${LAMBDA_SURFACE_START_HF_PRESERVE:-0.10}"
export LAYER_FREQUENCY_START_HF_LOWFREQ_KERNEL="${LAYER_FREQUENCY_START_HF_LOWFREQ_KERNEL:-15}"
export LAYER_FREQUENCY_START_HF_LOWFREQ_THRESHOLD="${LAYER_FREQUENCY_START_HF_LOWFREQ_THRESHOLD:-0.055}"
export LAYER_FREQUENCY_START_HF_ENERGY_THRESHOLD="${LAYER_FREQUENCY_START_HF_ENERGY_THRESHOLD:-0.006}"
export LAYER_FREQUENCY_START_HF_MASK_POWER="${LAYER_FREQUENCY_START_HF_MASK_POWER:-1.0}"
export LAYER_FREQUENCY_START_HF_PROTECT_NON_SURFACE="${LAYER_FREQUENCY_START_HF_PROTECT_NON_SURFACE:-0}"
export LAMBDA_NON_SURFACE_HF="${LAMBDA_NON_SURFACE_HF:-0.030}"
export LAMBDA_SURFACE_HF_CLOSURE="${LAMBDA_SURFACE_HF_CLOSURE:-0.010}"
export SURFACE_HF_UPDATE_SCALE="${SURFACE_HF_UPDATE_SCALE:-1.25}"

echo "[surface28-from-prior20hf-v0] recipient surface : ${INPUT_MODEL_DIR}/chkpnt${INPUT_ITERATION}.pth"
echo "[surface28-from-prior20hf-v0] HF donor          : ${HF_DONOR_CHECKPOINT}"
echo "[surface28-from-prior20hf-v0] cleanup iters     : ${CLEANUP_ITERS}"
echo "[surface28-from-prior20hf-v0] train resolution  : ${TRAIN_RESOLUTION}"
echo "[surface28-from-prior20hf-v0] run tag           : ${RUN_TAG}"

INPUT_MODEL_DIR="${INPUT_MODEL_DIR}" \
INPUT_ITERATION="${INPUT_ITERATION}" \
CLEANUP_ITERS="${CLEANUP_ITERS}" \
FINAL_ITER="${FINAL_ITER}" \
RUN_TAG="${RUN_TAG}" \
bash "${SCRIPT_DIR}/run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen.sh"
