#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_TAG="${BASE_TAG:-k22}"

IMAGE22_MIP_MODEL_PATH="${IMAGE22_MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input_init_repair_v0}"
IMAGE22_MIP_ITERATION="${IMAGE22_MIP_ITERATION:-30000}"

START_RUN_NAME="${START_RUN_NAME:-${BASE_TAG}_childsoft}"
START_RUN_ROOT="${START_RUN_ROOT:-${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${START_RUN_NAME}}"
START_MODEL_PATH="${START_MODEL_PATH:-${START_RUN_ROOT}/recovered_mip_model_lr_2dsr_v1}"
START_ITERATION="${START_ITERATION:-31800}"
RUN_START_PIPELINE_IF_MISSING="${RUN_START_PIPELINE_IF_MISSING:-1}"
START_PIPELINE_SCRIPT="${START_PIPELINE_SCRIPT:-${SOF_ROOT}/scripts/run_image22_cleanup_then_post_curve_refit_v1_looser_childsoft_iesrgs_kitchen.sh}"
START_PIPELINE_RUN_BASE_CLEANUP="${START_PIPELINE_RUN_BASE_CLEANUP:-0}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
SR_IMAGES_SUBDIR="${SR_IMAGES_SUBDIR:-images_2}"
RENDER_SPLIT="${RENDER_SPLIT:-train}"

SR_MASK_THRESHOLD="${SR_MASK_THRESHOLD:-0.12}"
SR_MASK_MODE="${SR_MASK_MODE:-soft}"
PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/sof_surface_v0_${IMAGES_SUBDIR}_to_${SR_IMAGES_SUBDIR}_mask${SR_MASK_THRESHOLD}_${SR_MASK_MODE}}"
PREPARED_SR_SUBDIR="${PREPARED_SR_SUBDIR:-fused_priors}"

RUN_RENDER_MIP_SOURCE="${RUN_RENDER_MIP_SOURCE:-1}"
MIP_RENDER_MODEL_PATH="${MIP_RENDER_MODEL_PATH:-${IMAGE22_MIP_MODEL_PATH}}"
MIP_RENDER_ITERATION="${MIP_RENDER_ITERATION:-${IMAGE22_MIP_ITERATION}}"
MIP_RENDER_OUTPUT_ROOT="${MIP_RENDER_OUTPUT_ROOT:-${SOF_ROOT}/output/direct_image_prior_probes_v0/${SCENE_NAME}/${BASE_TAG}_mip_src}"
MIP_RENDER_DIR="${MIP_RENDER_DIR:-${MIP_RENDER_OUTPUT_ROOT}/${RENDER_SPLIT}/ours_${MIP_RENDER_ITERATION}/renders}"
MIP_DIRECT_PRIOR_ROOT="${MIP_DIRECT_PRIOR_ROOT:-${SOF_ROOT}/output/direct_image_prior_probes_v0/${SCENE_NAME}/${BASE_TAG}_mip_prior}"

RUN_SR_DIRECT_PROBE="${RUN_SR_DIRECT_PROBE:-1}"
RUN_MIP_DIRECT_PROBE="${RUN_MIP_DIRECT_PROBE:-1}"

PROBE_PROFILE="${PROBE_PROFILE:-prior_recovery_v1}"
PROBE_ITERATIONS="${PROBE_ITERATIONS:-1400}"
PROBE_XYZ_LR="${PROBE_XYZ_LR:-8e-6}"
PROBE_OPACITY_LR="${PROBE_OPACITY_LR:-8e-4}"
PROBE_SCALE_LR="${PROBE_SCALE_LR:-2e-4}"
PROBE_LAMBDA_XYZ_ANCHOR="${PROBE_LAMBDA_XYZ_ANCHOR:-22.0}"
PROBE_LAMBDA_OPACITY_ANCHOR="${PROBE_LAMBDA_OPACITY_ANCHOR:-0.020}"
PROBE_LAMBDA_SCALE_ANCHOR="${PROBE_LAMBDA_SCALE_ANCHOR:-0.080}"
PROBE_LAMBDA_RISK_OPACITY="${PROBE_LAMBDA_RISK_OPACITY:-0.020}"
PROBE_LAMBDA_RISK_SCALE="${PROBE_LAMBDA_RISK_SCALE:-0.003}"
PROBE_LAMBDA_SR_RISK_OPACITY="${PROBE_LAMBDA_SR_RISK_OPACITY:-0.0}"
PROBE_LAMBDA_SR_RISK_SCALE="${PROBE_LAMBDA_SR_RISK_SCALE:-0.0}"
PROBE_SR_RISK_BOOST="${PROBE_SR_RISK_BOOST:-1.0}"
PROBE_SR_PRIOR_L1_WEIGHT="${PROBE_SR_PRIOR_L1_WEIGHT:-0.10}"
PROBE_SR_PRIOR_HF_WEIGHT="${PROBE_SR_PRIOR_HF_WEIGHT:-0.60}"
PROBE_SR_PRIOR_CONSISTENCY_THRESHOLD="${PROBE_SR_PRIOR_CONSISTENCY_THRESHOLD:-0.0}"
PROBE_SR_PRIOR_MIN_VALID_RATIO="${PROBE_SR_PRIOR_MIN_VALID_RATIO:-0.95}"
PROBE_SR_PRIOR_MIN_PIXELS="${PROBE_SR_PRIOR_MIN_PIXELS:-64}"
PROBE_SR_PRIOR_DELTA_CLIP="${PROBE_SR_PRIOR_DELTA_CLIP:-0.0}"
PROBE_DISABLE_SR_PRIOR_HF_RESIDUAL="${PROBE_DISABLE_SR_PRIOR_HF_RESIDUAL:-1}"
PROBE_MAX_DISPLACEMENT_RATIO="${PROBE_MAX_DISPLACEMENT_RATIO:-0.0024}"
PROBE_RUN_RENDER="${PROBE_RUN_RENDER:-1}"

SR_DIRECT_RUN_NAME="${SR_DIRECT_RUN_NAME:-${BASE_TAG}_probe_sr}"
MIP_DIRECT_RUN_NAME="${MIP_DIRECT_RUN_NAME:-${BASE_TAG}_probe_mip}"

resolve_latest_iteration() {
  local model_path="$1"
  local point_root="${model_path}/point_cloud"
  if [[ ! -d "${point_root}" ]]; then
    return 1
  fi
  local latest
  latest="$(find "${point_root}" -maxdepth 1 -type d -name 'iteration_*' -print | sed 's|.*/iteration_||' | sort -n | tail -n 1)"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  printf '%s\n' "${latest}"
}

resolve_start_iteration() {
  if [[ "${START_ITERATION}" == "latest" || "${START_ITERATION}" == "auto" || "${START_ITERATION}" == "-1" ]]; then
    resolve_latest_iteration "${START_MODEL_PATH}"
  else
    printf '%s\n' "${START_ITERATION}"
  fi
}

rerun_start_pipeline() {
  echo "[direct-supervision-probe] rerun missing start pipeline: ${START_PIPELINE_SCRIPT}"
  SCENE_NAME="${SCENE_NAME}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
  MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP}" \
  MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME}" \
  BASE_TAG="${BASE_TAG}" \
  IMAGE22_MIP_MODEL_PATH="${IMAGE22_MIP_MODEL_PATH}" \
  RUN_BASE_CLEANUP="${START_PIPELINE_RUN_BASE_CLEANUP}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${START_PIPELINE_SCRIPT}"
}

RESOLVED_START_ITERATION="$(resolve_start_iteration || true)"

echo "[direct-supervision-probe] scene            : ${SCENE_ROOT}"
echo "[direct-supervision-probe] start model      : ${START_MODEL_PATH} iter=${RESOLVED_START_ITERATION:-${START_ITERATION}}"
echo "[direct-supervision-probe] sr prior root    : ${PREPARED_SR_PRIOR_ROOT}/${PREPARED_SR_SUBDIR}"
echo "[direct-supervision-probe] mip render model : ${MIP_RENDER_MODEL_PATH} iter=${MIP_RENDER_ITERATION}"
echo "[direct-supervision-probe] mip render dir   : ${MIP_RENDER_DIR}"
echo "[direct-supervision-probe] probe profile    : ${PROBE_PROFILE} iter=${PROBE_ITERATIONS} sr_w=${PROBE_SR_PRIOR_L1_WEIGHT}/${PROBE_SR_PRIOR_HF_WEIGHT}"

if [[ -z "${RESOLVED_START_ITERATION}" || ! -e "${START_MODEL_PATH}/point_cloud/iteration_${RESOLVED_START_ITERATION}/point_cloud.ply" ]]; then
  if [[ "${RUN_START_PIPELINE_IF_MISSING}" == "1" ]]; then
    rerun_start_pipeline
    RESOLVED_START_ITERATION="$(resolve_start_iteration || true)"
  fi
fi

if [[ -z "${RESOLVED_START_ITERATION}" || ! -e "${START_MODEL_PATH}/point_cloud/iteration_${RESOLVED_START_ITERATION}/point_cloud.ply" ]]; then
  echo "[direct-supervision-probe] missing start model: ${START_MODEL_PATH}/point_cloud/iteration_${RESOLVED_START_ITERATION:-${START_ITERATION}}/point_cloud.ply" >&2
  exit 1
fi

if [[ "${RUN_SR_DIRECT_PROBE}" == "1" && ! -d "${PREPARED_SR_PRIOR_ROOT}/${PREPARED_SR_SUBDIR}" ]]; then
  echo "[direct-supervision-probe] missing prepared SR priors: ${PREPARED_SR_PRIOR_ROOT}/${PREPARED_SR_SUBDIR}" >&2
  exit 1
fi

if [[ "${RUN_RENDER_MIP_SOURCE}" == "1" ]]; then
  if [[ ! -e "${MIP_RENDER_MODEL_PATH}/point_cloud/iteration_${MIP_RENDER_ITERATION}/point_cloud.ply" ]]; then
    echo "[direct-supervision-probe] missing mip render source model: ${MIP_RENDER_MODEL_PATH}/point_cloud/iteration_${MIP_RENDER_ITERATION}/point_cloud.ply" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${MIP_RENDER_MODEL_PATH}" \
    --output_dir "${MIP_RENDER_OUTPUT_ROOT}" \
    --images_subdir "${SR_IMAGES_SUBDIR}" \
    --iteration "${MIP_RENDER_ITERATION}" \
    --split "${RENDER_SPLIT}"
fi

if [[ "${RUN_MIP_DIRECT_PROBE}" == "1" ]]; then
  if [[ ! -d "${MIP_RENDER_DIR}" ]]; then
    echo "[direct-supervision-probe] missing mip render dir: ${MIP_RENDER_DIR}" >&2
    exit 1
  fi
  mkdir -p "${MIP_DIRECT_PRIOR_ROOT}"
  rm -rf "${MIP_DIRECT_PRIOR_ROOT}/fused_priors" "${MIP_DIRECT_PRIOR_ROOT}/aligned_references"
  ln -s "${MIP_RENDER_DIR}" "${MIP_DIRECT_PRIOR_ROOT}/fused_priors"
  ln -s "${MIP_RENDER_DIR}" "${MIP_DIRECT_PRIOR_ROOT}/aligned_references"
fi

run_probe() {
  local run_name="$1"
  local sr_prior_root="$2"
  local sr_prior_subdir="$3"
  local sr_anchor_subdir="$4"

  SCENE_NAME="${SCENE_NAME}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
  START_MODEL_PATH="${START_MODEL_PATH}" \
  START_ITERATION="${RESOLVED_START_ITERATION}" \
  ANCHOR_MODEL_PATH="${IMAGE22_MIP_MODEL_PATH}" \
  ANCHOR_ITERATION="${IMAGE22_MIP_ITERATION}" \
  REFINE_PROFILE="${PROBE_PROFILE}" \
  RUN_NAME="${run_name}" \
  RUN_RENDER="${PROBE_RUN_RENDER}" \
  IMAGES_SUBDIR="${IMAGES_SUBDIR}" \
  SR_IMAGES_SUBDIR="${SR_IMAGES_SUBDIR}" \
  SR_PRIOR_ROOT="${sr_prior_root}" \
  SR_PRIOR_SUBDIR="${sr_prior_subdir}" \
  SR_ANCHOR_SUBDIR="${sr_anchor_subdir}" \
  SR_PRIOR_MASK_SUBDIR="unused_masks" \
  SR_PRIOR_L1_WEIGHT="${PROBE_SR_PRIOR_L1_WEIGHT}" \
  SR_PRIOR_HF_WEIGHT="${PROBE_SR_PRIOR_HF_WEIGHT}" \
  SR_PRIOR_CONSISTENCY_THRESHOLD="${PROBE_SR_PRIOR_CONSISTENCY_THRESHOLD}" \
  SR_PRIOR_MIN_VALID_RATIO="${PROBE_SR_PRIOR_MIN_VALID_RATIO}" \
  SR_PRIOR_MIN_PIXELS="${PROBE_SR_PRIOR_MIN_PIXELS}" \
  SR_PRIOR_DELTA_CLIP="${PROBE_SR_PRIOR_DELTA_CLIP}" \
  DISABLE_SR_PRIOR_HF_RESIDUAL="${PROBE_DISABLE_SR_PRIOR_HF_RESIDUAL}" \
  LAMBDA_MIP_HR_LOWFREQ="0.0" \
  ITERATIONS="${PROBE_ITERATIONS}" \
  XYZ_LR="${PROBE_XYZ_LR}" \
  OPACITY_LR="${PROBE_OPACITY_LR}" \
  SCALE_LR="${PROBE_SCALE_LR}" \
  LAMBDA_XYZ_ANCHOR="${PROBE_LAMBDA_XYZ_ANCHOR}" \
  LAMBDA_OPACITY_ANCHOR="${PROBE_LAMBDA_OPACITY_ANCHOR}" \
  LAMBDA_SCALE_ANCHOR="${PROBE_LAMBDA_SCALE_ANCHOR}" \
  LAMBDA_RISK_OPACITY="${PROBE_LAMBDA_RISK_OPACITY}" \
  LAMBDA_RISK_SCALE="${PROBE_LAMBDA_RISK_SCALE}" \
  LAMBDA_SR_RISK_OPACITY="${PROBE_LAMBDA_SR_RISK_OPACITY}" \
  LAMBDA_SR_RISK_SCALE="${PROBE_LAMBDA_SR_RISK_SCALE}" \
  SR_RISK_BOOST="${PROBE_SR_RISK_BOOST}" \
  MAX_DISPLACEMENT_RATIO="${PROBE_MAX_DISPLACEMENT_RATIO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SOF_ROOT}/scripts/run_recover_cleaned_mip_lr_v0_kitchen.sh"
}

if [[ "${RUN_SR_DIRECT_PROBE}" == "1" ]]; then
  run_probe "${SR_DIRECT_RUN_NAME}" "${PREPARED_SR_PRIOR_ROOT}" "${PREPARED_SR_SUBDIR}" "${PREPARED_SR_SUBDIR}"
fi

if [[ "${RUN_MIP_DIRECT_PROBE}" == "1" ]]; then
  run_probe "${MIP_DIRECT_RUN_NAME}" "${MIP_DIRECT_PRIOR_ROOT}" "fused_priors" "aligned_references"
fi

echo "[done] sr direct run  : ${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${SR_DIRECT_RUN_NAME}"
echo "[done] mip render run : ${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${MIP_DIRECT_RUN_NAME}"
if [[ "${RUN_MIP_DIRECT_PROBE}" == "1" ]]; then
  echo "[done] mip render dir : ${MIP_RENDER_DIR}"
  echo "[done] mip prior root : ${MIP_DIRECT_PRIOR_ROOT}"
fi
