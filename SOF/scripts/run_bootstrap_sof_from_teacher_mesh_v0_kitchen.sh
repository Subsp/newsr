#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
RAW_MODEL_NAME="${RAW_MODEL_NAME:-mip30k}"
RAW_MODEL_PATH="${RAW_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${RAW_MODEL_NAME}}"
MIP_ITERATION="${MIP_ITERATION:-30000}"
START_CHECKPOINT="${START_CHECKPOINT:-${RAW_MODEL_PATH}/chkpnt${MIP_ITERATION}.pth}"

TEACHER_MESH_PATH="${TEACHER_MESH_PATH:-}"
MESHRELAX_MODEL_NAME="${MESHRELAX_MODEL_NAME:-${RAW_MODEL_NAME}_meshteacher_v0}"
MESHRELAX_MODEL_PATH="${MESHRELAX_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MESHRELAX_MODEL_NAME}}"
MESHRELAX_CHECKPOINT_PATH="${MESHRELAX_CHECKPOINT_PATH:-${MESHRELAX_MODEL_PATH}/chkpnt${MIP_ITERATION}.pth}"
MESHRELAX_SUMMARY_PATH="${MESHRELAX_SUMMARY_PATH:-${MESHRELAX_MODEL_PATH}/chkpnt${MIP_ITERATION}_meshrelax.summary.json}"
MESHRELAX_PREVIEW_DIR="${MESHRELAX_PREVIEW_DIR:-${MESHRELAX_MODEL_PATH}/chkpnt${MIP_ITERATION}_meshrelax_preview}"

PREPARE_INPUT_PROFILE="${PREPARE_INPUT_PROFILE:-early4ksoft_v1}"
MIP_TO_SOF_PROFILE="${MIP_TO_SOF_PROFILE:-early4ksoft_v1}"
PREPARED_INPUT_NAME="${PREPARED_INPUT_NAME:-${MESHRELAX_MODEL_NAME}_sof_native_input_init_${PREPARE_INPUT_PROFILE}}"
PREPARED_INPUT_MODEL_PATH="${PREPARED_INPUT_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${PREPARED_INPUT_NAME}}"
DETAIL_RUN_NAME="${DETAIL_RUN_NAME:-detail34k_early4ksoft_meshteacher_v0}"
FINAL_ITERATION="${FINAL_ITERATION:-34000}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-34000}"

RUN_MESHRELAX="${RUN_MESHRELAX:-1}"
RUN_MIP_TO_SOF="${RUN_MIP_TO_SOF:-1}"
REUSE_MESHRELAX_IF_PRESENT="${REUSE_MESHRELAX_IF_PRESENT:-1}"
REUSE_DETAIL_IF_PRESENT="${REUSE_DETAIL_IF_PRESENT:-1}"
DETAIL_OUTPUT_MODEL="${DETAIL_OUTPUT_MODEL:-${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/${DETAIL_RUN_NAME}/pulled_mip_model}"

FOCUS_NEAR_FIELD="${FOCUS_NEAR_FIELD:-1}"
NEAR_FIELD_DISTANCE_QUANTILE="${NEAR_FIELD_DISTANCE_QUANTILE:-0.4}"
SUSPICIOUS_SCORE_QUANTILE="${SUSPICIOUS_SCORE_QUANTILE:-0.9}"
MAX_SUSPICIOUS_COUNT="${MAX_SUSPICIOUS_COUNT:-8000}"
CAMERA_STRIDE="${CAMERA_STRIDE:-4}"
MIN_HIT_VIEWS="${MIN_HIT_VIEWS:-2}"
MIN_CONSENSUS_HITS="${MIN_CONSENSUS_HITS:-2}"
ENABLE_ANCHOR_FALLBACK="${ENABLE_ANCHOR_FALLBACK:-1}"
RELOCATION_STRENGTH="${RELOCATION_STRENGTH:-1.2}"
SCALING_SHRINK_STRENGTH="${SCALING_SHRINK_STRENGTH:-0.10}"
DRY_RUN="${DRY_RUN:-0}"

DETAIL_MAX_VIEWS="${DETAIL_MAX_VIEWS:-0}"
DETAIL_SR_MAX_VIEWS="${DETAIL_SR_MAX_VIEWS:-0}"
DETAIL_MIP_OBS_CLOSURE_MAX_VIEWS="${DETAIL_MIP_OBS_CLOSURE_MAX_VIEWS:-0}"
SAVE_EVERY="${SAVE_EVERY:-0}"

if [[ -z "${TEACHER_MESH_PATH}" ]]; then
  echo "[bootstrap-sof-from-teacher-mesh-v0] TEACHER_MESH_PATH is required." >&2
  exit 1
fi
if [[ ! -d "${RAW_MODEL_PATH}" ]]; then
  echo "[bootstrap-sof-from-teacher-mesh-v0] missing raw mip model: ${RAW_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TEACHER_MESH_PATH}" ]]; then
  echo "[bootstrap-sof-from-teacher-mesh-v0] missing teacher mesh: ${TEACHER_MESH_PATH}" >&2
  exit 1
fi
if [[ "${RUN_MESHRELAX}" == "1" && ! -f "${START_CHECKPOINT}" ]]; then
  echo "[bootstrap-sof-from-teacher-mesh-v0] missing start checkpoint: ${START_CHECKPOINT}" >&2
  exit 1
fi

echo "[bootstrap-sof-from-teacher-mesh-v0] scene root      : ${SCENE_ROOT}"
echo "[bootstrap-sof-from-teacher-mesh-v0] raw mip model   : ${RAW_MODEL_PATH} iter=${MIP_ITERATION}"
echo "[bootstrap-sof-from-teacher-mesh-v0] teacher mesh    : ${TEACHER_MESH_PATH}"
echo "[bootstrap-sof-from-teacher-mesh-v0] meshrelax model : ${MESHRELAX_MODEL_PATH}"
echo "[bootstrap-sof-from-teacher-mesh-v0] prepared input  : ${PREPARED_INPUT_MODEL_PATH}"
echo "[bootstrap-sof-from-teacher-mesh-v0] detail run      : ${DETAIL_RUN_NAME}"

if [[ "${RUN_MESHRELAX}" == "1" ]]; then
  if [[ "${REUSE_MESHRELAX_IF_PRESENT}" == "1" && -f "${MESHRELAX_MODEL_PATH}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply" && -f "${MESHRELAX_CHECKPOINT_PATH}" ]]; then
    echo "[bootstrap-sof-from-teacher-mesh-v0] reusing meshrelax output."
  else
    CMD=(
      "${PYTHON_BIN}" -u "${SOF_ROOT}/relax_suspicious_gaussians_to_mesh.py"
      -s "${SCENE_ROOT}"
      -m "${RAW_MODEL_PATH}"
      -i images_8
      --iteration "${MIP_ITERATION}"
      --start_checkpoint "${START_CHECKPOINT}"
      --mesh_path "${TEACHER_MESH_PATH}"
      --near_field_distance_quantile "${NEAR_FIELD_DISTANCE_QUANTILE}"
      --suspicious_score_quantile "${SUSPICIOUS_SCORE_QUANTILE}"
      --max_suspicious_count "${MAX_SUSPICIOUS_COUNT}"
      --camera_stride "${CAMERA_STRIDE}"
      --min_hit_views "${MIN_HIT_VIEWS}"
      --min_consensus_hits "${MIN_CONSENSUS_HITS}"
      --relocation_strength "${RELOCATION_STRENGTH}"
      --scaling_shrink_strength "${SCALING_SHRINK_STRENGTH}"
      --output_checkpoint "${MESHRELAX_CHECKPOINT_PATH}"
      --output_summary "${MESHRELAX_SUMMARY_PATH}"
      --output_preview_dir "${MESHRELAX_PREVIEW_DIR}"
    )
    if [[ "${FOCUS_NEAR_FIELD}" == "1" ]]; then
      CMD+=(--focus_near_field)
    fi
    if [[ "${ENABLE_ANCHOR_FALLBACK}" == "1" ]]; then
      CMD+=(--enable_anchor_fallback)
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      CMD+=(--dry_run)
    fi
    "${CMD[@]}"
  fi
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[bootstrap-sof-from-teacher-mesh-v0] DRY_RUN=1, stop after meshrelax preview."
  exit 0
fi

if [[ "${RUN_MIP_TO_SOF}" == "1" ]]; then
  if [[ "${REUSE_DETAIL_IF_PRESENT}" == "1" && -f "${DETAIL_OUTPUT_MODEL}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply" ]]; then
    echo "[bootstrap-sof-from-teacher-mesh-v0] reusing detail SOF output."
  else
    SCENE_NAME="${SCENE_NAME}" \
    SCENE_ROOT="${SCENE_ROOT}" \
    SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
    MIP_MODEL_PATH="${MESHRELAX_MODEL_PATH}" \
    MIP_INPUT_MODEL_PATH="${PREPARED_INPUT_MODEL_PATH}" \
    MIP_TO_SOF_PROFILE="${MIP_TO_SOF_PROFILE}" \
    PREPARE_INPUT_PROFILE="${PREPARE_INPUT_PROFILE}" \
    RUN_NAME="${DETAIL_RUN_NAME}" \
    FINAL_ITERATION="${FINAL_ITERATION}" \
    OUTPUT_ITERATION="${OUTPUT_ITERATION}" \
    MAX_VIEWS="${DETAIL_MAX_VIEWS}" \
    SR_MAX_VIEWS="${DETAIL_SR_MAX_VIEWS}" \
    MIP_OBS_CLOSURE_MAX_VIEWS="${DETAIL_MIP_OBS_CLOSURE_MAX_VIEWS}" \
    SAVE_EVERY="${SAVE_EVERY}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SOF_ROOT}/scripts/run_mip_to_sof_surface_v0_kitchen.sh"
  fi
fi

echo "[done] meshrelax model : ${MESHRELAX_MODEL_PATH}"
echo "[done] prepared input  : ${PREPARED_INPUT_MODEL_PATH}"
if [[ "${RUN_MIP_TO_SOF}" == "1" ]]; then
  echo "[done] detail model    : ${DETAIL_OUTPUT_MODEL}"
  echo "[done] detail ply      : ${DETAIL_OUTPUT_MODEL}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
fi
