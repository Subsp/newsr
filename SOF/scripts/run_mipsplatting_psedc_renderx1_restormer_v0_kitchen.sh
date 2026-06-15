#!/usr/bin/env bash
set -euo pipefail

# PSE-DC v0 after the direct Restormer x1 render-restoration prior baseline.
#
# This is intentionally a thin, named wrapper over the preserved NoSR
# sr-prior/lr-mesh cleanup path. It starts from the prior-from-scratch checkpoint
# and routes Restormer high-frequency detail into mesh-derived surface carriers
# while suppressing non-surface high-frequency uptake.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output}"

ENHANCEMENT_BACKEND="${ENHANCEMENT_BACKEND:-restormer}"
REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-images_2}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-render_x1_${ENHANCEMENT_BACKEND}_aligned_${REFERENCE_IMAGES_SUBDIR}_scratch_v0}"
PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${PREPARED_SR_PRIOR_NAME}}"
PRIOR_IMAGE_SUBDIR="${PRIOR_IMAGE_SUBDIR:-fused_priors}"

INPUT_RUN_TAG="${INPUT_RUN_TAG:-mip30k_r1_renderx1_${ENHANCEMENT_BACKEND}_prioronly_scratch_v0}"
INPUT_MODEL_DIR="${INPUT_MODEL_DIR:-${OUTPUT_ROOT}/mipsplatting_prior_only_from_scratch_v0/${SCENE_NAME}/${INPUT_RUN_TAG}}"
INPUT_ITERATION="${INPUT_ITERATION:-30000}"
CLEANUP_ITERS="${CLEANUP_ITERS:-2000}"
FINAL_ITER="${FINAL_ITER:-$(( INPUT_ITERATION + CLEANUP_ITERS ))}"

LOWFREQ_ANCHOR_MODE="${LOWFREQ_ANCHOR_MODE:-directsrc_render}"
HF_RETENTION_PROFILE="${HF_RETENTION_PROFILE:-preserve_v1}"
SURFACE_STATE_PROFILE="${SURFACE_STATE_PROFILE:-relaxed_carrier_v1}"
PSEDC_VERSION_TAG="${PSEDC_VERSION_TAG:-psedc_v0}"
RUN_TAG="${RUN_TAG:-${INPUT_RUN_TAG}_to${FINAL_ITER}_${LOWFREQ_ANCHOR_MODE}_${HF_RETENTION_PROFILE}_${PSEDC_VERSION_TAG}}"

# Keep v0 as a fixed-topology carrier-routing test.
SURFACE_NORMAL_LOCK="${SURFACE_NORMAL_LOCK:-1}"
LAYER_FREQUENCY_SURFACE_TARGET="${LAYER_FREQUENCY_SURFACE_TARGET:-prior}"
EXTERNAL_PRIOR_ROOT="${EXTERNAL_PRIOR_ROOT:-${PREPARED_SR_PRIOR_ROOT}}"
EXTERNAL_PRIOR_SUBDIR="${EXTERNAL_PRIOR_SUBDIR:-${PRIOR_IMAGE_SUBDIR}}"
EXTERNAL_PRIOR_MASK_SUBDIR="${EXTERNAL_PRIOR_MASK_SUBDIR:-}"
PRIOR_CONSISTENCY_THRESHOLD="${PRIOR_CONSISTENCY_THRESHOLD:-1.0}"
PRIOR_MIN_VALID_RATIO="${PRIOR_MIN_VALID_RATIO:-0.0}"
ALLOW_CUSTOM_RUN_TAG="${ALLOW_CUSTOM_RUN_TAG:-1}"

export WORK_ROOT
export SCENE_NAME
export SCENE_ROOT
export SCENE_ASSET_ROOT
export OUTPUT_ROOT
export REFERENCE_IMAGES_SUBDIR
export TARGET_IMAGES_SUBDIR
export PREPARED_SR_PRIOR_NAME
export PREPARED_SR_PRIOR_ROOT
export PRIOR_IMAGE_SUBDIR
export INPUT_RUN_TAG
export INPUT_MODEL_DIR
export INPUT_ITERATION
export CLEANUP_ITERS
export FINAL_ITER
export LOWFREQ_ANCHOR_MODE
export HF_RETENTION_PROFILE
export SURFACE_STATE_PROFILE
export RUN_TAG
export SURFACE_NORMAL_LOCK
export LAYER_FREQUENCY_SURFACE_TARGET
export EXTERNAL_PRIOR_ROOT
export EXTERNAL_PRIOR_SUBDIR
export EXTERNAL_PRIOR_MASK_SUBDIR
export PRIOR_CONSISTENCY_THRESHOLD
export PRIOR_MIN_VALID_RATIO
export ALLOW_CUSTOM_RUN_TAG

echo "[psedc-renderx1-restormer-v0] prepared prior : ${PREPARED_SR_PRIOR_ROOT}/${PRIOR_IMAGE_SUBDIR}"
echo "[psedc-renderx1-restormer-v0] input model    : ${INPUT_MODEL_DIR}/chkpnt${INPUT_ITERATION}.pth"
echo "[psedc-renderx1-restormer-v0] lowfreq anchor : ${LOWFREQ_ANCHOR_MODE}"
echo "[psedc-renderx1-restormer-v0] HF profile     : ${HF_RETENTION_PROFILE}"
echo "[psedc-renderx1-restormer-v0] output tag     : ${RUN_TAG}"

bash "${SCRIPT_DIR}/run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen_srprior_lrmesh.sh"
