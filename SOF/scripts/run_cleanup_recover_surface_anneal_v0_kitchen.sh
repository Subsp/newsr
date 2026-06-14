#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
MIP_ITERATION="${MIP_ITERATION:-30000}"
MESH_PATH="${MESH_PATH:-/root/autodl-tmp/kitchen_MipNerf360_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_15_cleaned_mesh.ply}"

CLEANUP_RUN_NAME="${CLEANUP_RUN_NAME:-mip30k_volume_stress_iterclean_v0}"
CLEANUP_OUTPUT_MODEL="${CLEANUP_OUTPUT_MODEL:-${SOF_ROOT}/output/cleanup_mip_view_aligned_volume_artifacts_v0/${SCENE_NAME}/${CLEANUP_RUN_NAME}/cleaned_mip_model_view_volume_v1}"
CLEANUP_RUN_RENDER="${CLEANUP_RUN_RENDER:-1}"
CLEANUP_RUN_LR_RECOVER="${CLEANUP_RUN_LR_RECOVER:-0}"
CLEANUP_DELETE_QUANTILE="${CLEANUP_DELETE_QUANTILE:-0.930}"
CLEANUP_MAX_PRUNE_FRACTION="${CLEANUP_MAX_PRUNE_FRACTION:-0.120}"
CLEANUP_MAX_PRUNE_COUNT="${CLEANUP_MAX_PRUNE_COUNT:-0}"
CLEANUP_MAX_OPACITY="${CLEANUP_MAX_OPACITY:-1.0}"
CLEANUP_CANDIDATE_MODE="${CLEANUP_CANDIDATE_MODE:-volume_stress}"
CLEANUP_RECOMPUTE_FILTER3D="${CLEANUP_RECOMPUTE_FILTER3D:-1}"
CLEANUP_MAX_VIEWS="${CLEANUP_MAX_VIEWS:-16}"

RECOVER_PROFILE="${RECOVER_PROFILE:-conservative_v0}"
RECOVER_USE_SR_PRIOR="${RECOVER_USE_SR_PRIOR:-0}"
if [[ "${RECOVER_USE_SR_PRIOR}" == "1" ]]; then
  case "${RECOVER_PROFILE}" in
    mip_hr_anchor_v0)
      DEFAULT_RECOVER_RUN_SUFFIX="miphr_v1"
      ;;
    prior_recovery_v1)
      DEFAULT_RECOVER_RUN_SUFFIX="2dsr_v1"
      ;;
    *)
      DEFAULT_RECOVER_RUN_SUFFIX="srprior_v0"
      ;;
  esac
else
  DEFAULT_RECOVER_RUN_SUFFIX="puremip_v0"
fi
RECOVER_RUN_NAME="${RECOVER_RUN_NAME:-${CLEANUP_RUN_NAME}_${RECOVER_PROFILE}_${DEFAULT_RECOVER_RUN_SUFFIX}}"
if [[ "${RECOVER_USE_SR_PRIOR}" == "1" ]]; then
  case "${RECOVER_PROFILE}" in
    mip_hr_anchor_v0)
      DEFAULT_RECOVER_MODEL_NAME="recovered_mip_model_lr_miphr_v1"
      ;;
    prior_recovery_v1)
      DEFAULT_RECOVER_MODEL_NAME="recovered_mip_model_lr_2dsr_v1"
      ;;
    *)
      DEFAULT_RECOVER_MODEL_NAME="recovered_mip_model_lr_v0"
      ;;
  esac
else
  DEFAULT_RECOVER_MODEL_NAME="recovered_mip_model_lr_v0"
fi
RECOVER_OUTPUT_MODEL="${RECOVER_OUTPUT_MODEL:-${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${RECOVER_RUN_NAME}/${DEFAULT_RECOVER_MODEL_NAME}}"
RECOVER_RUN_RENDER="${RECOVER_RUN_RENDER:-1}"

ANNEAL_RUN_NAME="${ANNEAL_RUN_NAME:-${RECOVER_RUN_NAME}_surfaceanneal_v0}"
ANNEAL_OUTPUT_MODEL="${ANNEAL_OUTPUT_MODEL:-${SOF_ROOT}/output/anneal_mip_covariance_to_surface_v0/${SCENE_NAME}/${ANNEAL_RUN_NAME}}"
ANNEAL_MODEL_ITERATION="${ANNEAL_MODEL_ITERATION:--1}"
ANNEAL_IMAGES_SUBDIR="${ANNEAL_IMAGES_SUBDIR:-images_2}"
ANNEAL_SPLIT="${ANNEAL_SPLIT:-test}"
ANNEAL_PREVIEW_VIEWS="${ANNEAL_PREVIEW_VIEWS:-6}"
ANNEAL_RENDER_DELTA_GUARD_VIEWS="${ANNEAL_RENDER_DELTA_GUARD_VIEWS:-6}"
ANNEAL_RENDER_DELTA_GUARD_SPLIT="${ANNEAL_RENDER_DELTA_GUARD_SPLIT:-test}"
ANNEAL_RENDER_DELTA_GUARD_QUANTILE="${ANNEAL_RENDER_DELTA_GUARD_QUANTILE:-0.92}"
ANNEAL_RENDER_DELTA_GUARD_MIN_SCORE="${ANNEAL_RENDER_DELTA_GUARD_MIN_SCORE:-0.03}"
ANNEAL_RENDER_DELTA_GUARD_MAX_FRACTION="${ANNEAL_RENDER_DELTA_GUARD_MAX_FRACTION:-0.06}"

RUN_CLEANUP="${RUN_CLEANUP:-1}"
RUN_RECOVER="${RUN_RECOVER:-1}"
RUN_ANNEAL="${RUN_ANNEAL:-1}"

echo "[iter-clean-recover-anneal-v0] scene       : ${SCENE_ROOT}"
echo "[iter-clean-recover-anneal-v0] mip model   : ${MIP_MODEL_PATH} iter=${MIP_ITERATION}"
echo "[iter-clean-recover-anneal-v0] mesh        : ${MESH_PATH}"
echo "[iter-clean-recover-anneal-v0] cleanup run : ${CLEANUP_RUN_NAME}"
echo "[iter-clean-recover-anneal-v0] recover run : ${RECOVER_RUN_NAME} profile=${RECOVER_PROFILE} use_sr_prior=${RECOVER_USE_SR_PRIOR}"
echo "[iter-clean-recover-anneal-v0] anneal run  : ${ANNEAL_RUN_NAME} iter=${ANNEAL_MODEL_ITERATION}"

if [[ "${RUN_CLEANUP}" == "1" ]]; then
  SCENE_NAME="${SCENE_NAME}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
  MIP_MODEL_PATH="${MIP_MODEL_PATH}" \
  MIP_ITERATION="${MIP_ITERATION}" \
  OUTPUT_ITERATION="${MIP_ITERATION}" \
  RUN_NAME="${CLEANUP_RUN_NAME}" \
  OUTPUT_MODEL="${CLEANUP_OUTPUT_MODEL}" \
  RUN_RENDER="${CLEANUP_RUN_RENDER}" \
  RUN_LR_RECOVER="${CLEANUP_RUN_LR_RECOVER}" \
  DELETE_QUANTILE="${CLEANUP_DELETE_QUANTILE}" \
  MAX_PRUNE_FRACTION="${CLEANUP_MAX_PRUNE_FRACTION}" \
  MAX_PRUNE_COUNT="${CLEANUP_MAX_PRUNE_COUNT}" \
  MAX_OPACITY="${CLEANUP_MAX_OPACITY}" \
  CANDIDATE_MODE="${CLEANUP_CANDIDATE_MODE}" \
  RECOMPUTE_FILTER3D="${CLEANUP_RECOMPUTE_FILTER3D}" \
  MAX_VIEWS="${CLEANUP_MAX_VIEWS}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SOF_ROOT}/scripts/run_cleanup_mip_view_aligned_volume_artifacts_v0_kitchen.sh"
fi

if [[ "${RUN_RECOVER}" == "1" ]]; then
  SCENE_NAME="${SCENE_NAME}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
  START_MODEL_PATH="${CLEANUP_OUTPUT_MODEL}" \
  START_ITERATION="${MIP_ITERATION}" \
  ANCHOR_MODEL_PATH="${MIP_MODEL_PATH}" \
  ANCHOR_ITERATION="${MIP_ITERATION}" \
  MIP_CLOSURE_MODEL_PATH="${MIP_MODEL_PATH}" \
  MIP_CLOSURE_ITERATION="${MIP_ITERATION}" \
  USE_SR_PRIOR="${RECOVER_USE_SR_PRIOR}" \
  REFINE_PROFILE="${RECOVER_PROFILE}" \
  RUN_NAME="${RECOVER_RUN_NAME}" \
  OUTPUT_MODEL="${RECOVER_OUTPUT_MODEL}" \
  RUN_RENDER="${RECOVER_RUN_RENDER}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SOF_ROOT}/scripts/run_recover_cleaned_mip_lr_v0_kitchen.sh"
fi

if [[ "${RUN_ANNEAL}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/anneal_mip_covariance_to_surface_v0.py" \
    --model_path "${RECOVER_OUTPUT_MODEL}" \
    --mesh_path "${MESH_PATH}" \
    --scene_root "${SCENE_ROOT}" \
    --images_subdir "${ANNEAL_IMAGES_SUBDIR}" \
    --split "${ANNEAL_SPLIT}" \
    --iteration "${ANNEAL_MODEL_ITERATION}" \
    --render_delta_guard_views "${ANNEAL_RENDER_DELTA_GUARD_VIEWS}" \
    --render_delta_guard_split "${ANNEAL_RENDER_DELTA_GUARD_SPLIT}" \
    --render_delta_guard_quantile "${ANNEAL_RENDER_DELTA_GUARD_QUANTILE}" \
    --render_delta_guard_min_score "${ANNEAL_RENDER_DELTA_GUARD_MIN_SCORE}" \
    --render_delta_guard_max_fraction "${ANNEAL_RENDER_DELTA_GUARD_MAX_FRACTION}" \
    --output_model_path "${ANNEAL_OUTPUT_MODEL}" \
    --preview_views "${ANNEAL_PREVIEW_VIEWS}"
fi

echo "[done] cleanup model : ${CLEANUP_OUTPUT_MODEL}"
echo "[done] recover model : ${RECOVER_OUTPUT_MODEL}"
echo "[done] anneal model  : ${ANNEAL_OUTPUT_MODEL}"
