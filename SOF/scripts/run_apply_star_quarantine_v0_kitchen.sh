#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
MIP_ITERATION="${MIP_ITERATION:-30000}"
MIP_RENDER_RESOLUTION="${MIP_RENDER_RESOLUTION:-1}"
MIP_RENDER_DIR="${MIP_RENDER_DIR:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}/test/ours_${MIP_ITERATION}/test_preds_${MIP_RENDER_RESOLUTION}}"

RECOVER_RUN_NAME="${RECOVER_RUN_NAME:-view_aligned_volume_delete_v1_initrepair_v0_prune_more_v1_mip_hr_anchor_v0_miphr_v1}"
MODEL_NAME="${MODEL_NAME:-recovered_mip_model_lr_miphr_v1}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${RECOVER_RUN_NAME}}"
MODEL_PATH="${MODEL_PATH:-${RUN_ROOT}/${MODEL_NAME}}"
ITERATION="${ITERATION:-31600}"

RUN_BASELINE_RECOVERY="${RUN_BASELINE_RECOVERY:-0}"
BASELINE_START_RUN_NAME="${BASELINE_START_RUN_NAME:-view_aligned_volume_delete_v1_initrepair_v0_prune_more_v1}"
BASELINE_START_MODEL_PATH="${BASELINE_START_MODEL_PATH:-${SOF_ROOT}/output/cleanup_mip_view_aligned_volume_artifacts_v0/${SCENE_NAME}/${BASELINE_START_RUN_NAME}/cleaned_mip_model_view_volume_v1}"
BASELINE_ANCHOR_MODEL_PATH="${BASELINE_ANCHOR_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input_init_repair_v0}"
BASELINE_REFINE_PROFILE="${BASELINE_REFINE_PROFILE:-mip_hr_anchor_v0}"
BASELINE_IMAGES_SUBDIR="${BASELINE_IMAGES_SUBDIR:-images_8}"
BASELINE_MAX_VIEWS="${BASELINE_MAX_VIEWS:-20}"
BASELINE_RUN_RENDER="${BASELINE_RUN_RENDER:-0}"

DETECT_RUN_NAME="${DETECT_RUN_NAME:-${RECOVER_RUN_NAME}_starburst_v0}"
SCORE_DIR="${SCORE_DIR:-${SOF_ROOT}/output/starburst_gaussian_scores_v0/${SCENE_NAME}/${DETECT_RUN_NAME}}"
SCORE_PAYLOAD_PATH="${SCORE_PAYLOAD_PATH:-${SCORE_DIR}/starburst_gaussian_scores_v0.pt}"

OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-${RECOVER_RUN_NAME}_starquarantine_v0}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/star_quarantine_v0/${SCENE_NAME}/${OUTPUT_RUN_NAME}}"

INTERACTION_IMAGES_SUBDIR="${INTERACTION_IMAGES_SUBDIR:-images_2}"
REFERENCE_SOURCE="${REFERENCE_SOURCE:-mip_lr_render}"
RUN_DETECT="${RUN_DETECT:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-16}"
VIEW_INDICES="${VIEW_INDICES:-}"

SELECT_QUANTILE="${SELECT_QUANTILE:-0.990}"
MIN_STAR_SCORE="${MIN_STAR_SCORE:-0.08}"
MIN_UNSUPPORTED_SCORE="${MIN_UNSUPPORTED_SCORE:-0.04}"
MIN_STAR_VIEW_COUNT="${MIN_STAR_VIEW_COUNT:-1}"
MAX_CANDIDATE_FRACTION="${MAX_CANDIDATE_FRACTION:-0.015}"
MAX_CANDIDATE_COUNT="${MAX_CANDIDATE_COUNT:-30000}"
DETECT_RENDER_CANDIDATE_AFTER="${DETECT_RENDER_CANDIDATE_AFTER:-0}"

USE_PAYLOAD_CANDIDATE_MASK="${USE_PAYLOAD_CANDIDATE_MASK:-1}"
QUARANTINE_CANDIDATE_MODE="${QUARANTINE_CANDIDATE_MODE:-intersection}"
MIN_QUARANTINE_STARBURST_SCORE="${MIN_QUARANTINE_STARBURST_SCORE:-0.12}"
MIN_QUARANTINE_UNSUPPORTED_SCORE="${MIN_QUARANTINE_UNSUPPORTED_SCORE:-0.04}"
MIN_QUARANTINE_GEOMETRY_RISK="${MIN_QUARANTINE_GEOMETRY_RISK:-0.08}"
MIN_QUARANTINE_VISIBLE_COUNT="${MIN_QUARANTINE_VISIBLE_COUNT:-1}"
MAX_QUARANTINE_FRACTION="${MAX_QUARANTINE_FRACTION:-0.015}"
MAX_QUARANTINE_COUNT="${MAX_QUARANTINE_COUNT:-30000}"
STAR_DC_SCALE="${STAR_DC_SCALE:-1.0}"
STAR_REST_SCALE="${STAR_REST_SCALE:-0.10}"
STAR_TAU_SCALE="${STAR_TAU_SCALE:-0.35}"
STAR_MIN_ALPHA="${STAR_MIN_ALPHA:-1e-6}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-srtest}"
PYTHON_BIN="${PYTHON_BIN:-python}"

resolve_model_path() {
  local candidate_root="$1"
  local iteration="$2"
  if [[ -e "${candidate_root}/point_cloud/iteration_${iteration}/point_cloud.ply" ]]; then
    printf '%s\n' "${candidate_root}"
    return 0
  fi
  if [[ -d "${candidate_root}" ]]; then
    local child
    for child in "${candidate_root}"/*; do
      if [[ -d "${child}" && -e "${child}/point_cloud/iteration_${iteration}/point_cloud.ply" ]]; then
        printf '%s\n' "${child}"
        return 0
      fi
    done
  fi
  return 1
}

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

if RESOLVED_MODEL_PATH="$(resolve_model_path "${MODEL_PATH}" "${ITERATION}")"; then
  MODEL_PATH="${RESOLVED_MODEL_PATH}"
elif RESOLVED_MODEL_PATH="$(resolve_model_path "${RUN_ROOT}" "${ITERATION}")"; then
  MODEL_PATH="${RESOLVED_MODEL_PATH}"
fi

if [[ ! -e "${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" && "${RUN_BASELINE_RECOVERY}" == "1" ]]; then
  echo "[starquarantine-v0] baseline model missing; rerunning 27-style recovery first."
  echo "[starquarantine-v0] baseline start : ${BASELINE_START_MODEL_PATH}"
  echo "[starquarantine-v0] baseline anchor: ${BASELINE_ANCHOR_MODEL_PATH}"
  START_RUN_NAME="${BASELINE_START_RUN_NAME}" \
  START_MODEL_PATH="${BASELINE_START_MODEL_PATH}" \
  ANCHOR_MODEL_PATH="${BASELINE_ANCHOR_MODEL_PATH}" \
  REFINE_PROFILE="${BASELINE_REFINE_PROFILE}" \
  RUN_NAME="${RECOVER_RUN_NAME}" \
  IMAGES_SUBDIR="${BASELINE_IMAGES_SUBDIR}" \
  MAX_VIEWS="${BASELINE_MAX_VIEWS}" \
  RUN_RENDER="${BASELINE_RUN_RENDER}" \
  LAMBDA_MIP_CLOSURE_ALPHA=0 \
  LAMBDA_MIP_CLOSURE_PREMUL=0 \
  LAMBDA_MIP_CLOSURE_DEPTH=0 \
  LAMBDA_MIP_CLOSURE_ALPHA_OVER=0 \
  LAMBDA_MIP_CLOSURE_PREMUL_OVER=0 \
  LAMBDA_PREMUL_HF_EXCESS=0 \
  bash "${SCRIPT_DIR}/run_recover_cleaned_mip_lr_v0_kitchen.sh"

  if RESOLVED_MODEL_PATH="$(resolve_model_path "${MODEL_PATH}" "${ITERATION}")"; then
    MODEL_PATH="${RESOLVED_MODEL_PATH}"
  elif RESOLVED_MODEL_PATH="$(resolve_model_path "${RUN_ROOT}" "${ITERATION}")"; then
    MODEL_PATH="${RESOLVED_MODEL_PATH}"
  fi
fi

if [[ ! -e "${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" ]]; then
  echo "[starquarantine-v0] model point cloud not found: ${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" >&2
  echo "[starquarantine-v0] searched run root      : ${RUN_ROOT}" >&2
  echo "[starquarantine-v0] to rebuild 27 baseline, rerun with RUN_BASELINE_RECOVERY=1" >&2
  exit 1
fi

if [[ "${RUN_DETECT}" == "1" ]]; then
  SCENE_NAME="${SCENE_NAME}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
  MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP}" \
  MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME}" \
  MIP_ITERATION="${MIP_ITERATION}" \
  MIP_RENDER_RESOLUTION="${MIP_RENDER_RESOLUTION}" \
  MIP_RENDER_DIR="${MIP_RENDER_DIR}" \
  RECOVER_RUN_NAME="${RECOVER_RUN_NAME}" \
  MODEL_NAME="${MODEL_NAME}" \
  MODEL_PATH="${MODEL_PATH}" \
  ITERATION="${ITERATION}" \
  RUN_NAME="${DETECT_RUN_NAME}" \
  OUTPUT_DIR="${SCORE_DIR}" \
  INTERACTION_IMAGES_SUBDIR="${INTERACTION_IMAGES_SUBDIR}" \
  REFERENCE_SOURCE="${REFERENCE_SOURCE}" \
  SPLIT="${SPLIT}" \
  MAX_VIEWS="${MAX_VIEWS}" \
  VIEW_INDICES="${VIEW_INDICES}" \
  SELECT_QUANTILE="${SELECT_QUANTILE}" \
  MIN_STAR_SCORE="${MIN_STAR_SCORE}" \
  MIN_UNSUPPORTED_SCORE="${MIN_UNSUPPORTED_SCORE}" \
  MIN_STAR_VIEW_COUNT="${MIN_STAR_VIEW_COUNT}" \
  MAX_CANDIDATE_FRACTION="${MAX_CANDIDATE_FRACTION}" \
  MAX_CANDIDATE_COUNT="${MAX_CANDIDATE_COUNT}" \
  RENDER_CANDIDATE_AFTER="${DETECT_RENDER_CANDIDATE_AFTER}" \
  bash "${SCRIPT_DIR}/run_detect_starburst_gaussian_artifacts_v0_kitchen.sh"
fi

if [[ ! -f "${SCORE_PAYLOAD_PATH}" ]]; then
  echo "[starquarantine-v0] starburst payload not found: ${SCORE_PAYLOAD_PATH}" >&2
  exit 1
fi

echo "[starquarantine-v0] model path  : ${MODEL_PATH}"
echo "[starquarantine-v0] score       : ${SCORE_PAYLOAD_PATH}"
echo "[starquarantine-v0] output model: ${OUTPUT_MODEL_PATH}"
echo "[starquarantine-v0] gates       : dc=${STAR_DC_SCALE} rest=${STAR_REST_SCALE} tau=${STAR_TAU_SCALE}"
echo "[starquarantine-v0] select      : q=${SELECT_QUANTILE} max_frac=${MAX_QUARANTINE_FRACTION} max_count=${MAX_QUARANTINE_COUNT}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/apply_star_quarantine_v0.py"
  --model_path "${MODEL_PATH}"
  --score_payload_path "${SCORE_PAYLOAD_PATH}"
  --output_model_path "${OUTPUT_MODEL_PATH}"
  --iteration "${ITERATION}"
  --score_key starburst_score
  --candidate_key starburst_candidate
  --candidate_mode "${QUARANTINE_CANDIDATE_MODE}"
  --min_starburst_score "${MIN_QUARANTINE_STARBURST_SCORE}"
  --min_unsupported_score "${MIN_QUARANTINE_UNSUPPORTED_SCORE}"
  --min_geometry_risk "${MIN_QUARANTINE_GEOMETRY_RISK}"
  --min_visible_count "${MIN_QUARANTINE_VISIBLE_COUNT}"
  --max_quarantine_fraction "${MAX_QUARANTINE_FRACTION}"
  --max_quarantine_count "${MAX_QUARANTINE_COUNT}"
  --dc_scale "${STAR_DC_SCALE}"
  --rest_scale "${STAR_REST_SCALE}"
  --tau_scale "${STAR_TAU_SCALE}"
  --min_alpha "${STAR_MIN_ALPHA}"
)

if [[ "${USE_PAYLOAD_CANDIDATE_MASK}" == "1" ]]; then
  CMD+=(--use_payload_candidate_mask)
fi

"${CMD[@]}"

echo "[done] quarantined model: ${OUTPUT_MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
echo "[done] payload          : ${OUTPUT_MODEL_PATH}/star_quarantine_payload.pt"
echo "[done] summary          : ${OUTPUT_MODEL_PATH}/star_quarantine_summary.json"

if [[ "${RUN_RENDER}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${OUTPUT_MODEL_PATH}" \
    --images_subdir "${INTERACTION_IMAGES_SUBDIR}" \
    --iteration "${ITERATION}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}"
  echo "[done] preview renders : ${OUTPUT_MODEL_PATH}/${SPLIT}/ours_${ITERATION}/renders"
fi
