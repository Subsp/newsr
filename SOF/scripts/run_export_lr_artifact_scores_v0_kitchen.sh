#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"

DEFAULT_MODEL_PATH="${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/mip_to_soflr_surface_v0/pulled_mip_model"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
ITERATION="${ITERATION:-30000}"

RUN_NAME="${RUN_NAME:-$(basename "${MODEL_PATH}")_lr_artifact_v0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/lr_artifact_scores_v0/${SCENE_NAME}/${RUN_NAME}}"

INTERACTION_IMAGES_SUBDIR="${INTERACTION_IMAGES_SUBDIR:-images_2}"
REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-${IMAGES_SUBDIR:-images_8}}"
SPLIT="${SPLIT:-both}"
MAX_VIEWS="${MAX_VIEWS:-48}"
VISIBILITY_DOWNSAMPLE="${VISIBILITY_DOWNSAMPLE:-8}"
VISIBILITY_TOPK="${VISIBILITY_TOPK:-4}"
VISIBILITY_MAX_VISIBLE="${VISIBILITY_MAX_VISIBLE:-50000}"
VISIBILITY_MAX_PATCH_RADIUS="${VISIBILITY_MAX_PATCH_RADIUS:-1}"

LOWPASS_KERNEL="${LOWPASS_KERNEL:-17}"
RADIUS_RISK_PX="${RADIUS_RISK_PX:-18.0}"
SUPPRESS_QUANTILE="${SUPPRESS_QUANTILE:-0.98}"
PULL_ARTIFACT_QUANTILE="${PULL_ARTIFACT_QUANTILE:-0.60}"
PULL_EDGE_QUANTILE="${PULL_EDGE_QUANTILE:-0.35}"
PULL_MAX_FOOTPRINT_RISK="${PULL_MAX_FOOTPRINT_RISK:-0.75}"
DELETE_QUANTILE="${DELETE_QUANTILE:-0.995}"
DELETE_MIN_LR_FORBIDDEN="${DELETE_MIN_LR_FORBIDDEN:-0.65}"
DELETE_MIN_ARTIFACT="${DELETE_MIN_ARTIFACT:-0.55}"
DELETE_MAX_EDGE_SUPPORT="${DELETE_MAX_EDGE_SUPPORT:-0.45}"
DELETE_MIN_FOOTPRINT_RISK="${DELETE_MIN_FOOTPRINT_RISK:-0.70}"
DELETE_MAX_OPACITY="${DELETE_MAX_OPACITY:-0.08}"
DELETE_MIN_RADIUS_PX="${DELETE_MIN_RADIUS_PX:-12.0}"
DELETE_MAX_RATIO="${DELETE_MAX_RATIO:-0.03}"
NUM_DEBUG_VIEWS="${NUM_DEBUG_VIEWS:-4}"
EXPORT_PLYS="${EXPORT_PLYS:-1}"
EXPORT_PRUNED_MODEL="${EXPORT_PRUNED_MODEL:-1}"
RENDER_PRUNED_AFTER="${RENDER_PRUNED_AFTER:-1}"
PRUNED_RENDER_SPLIT="${PRUNED_RENDER_SPLIT:-test}"
PRUNED_RENDER_MAX_VIEWS="${PRUNED_RENDER_MAX_VIEWS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -e "${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" ]]; then
  echo "[lr-artifact-v0] model point cloud not found: ${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/${INTERACTION_IMAGES_SUBDIR}" ]]; then
  echo "[lr-artifact-v0] interaction image dir not found: ${SCENE_ROOT}/${INTERACTION_IMAGES_SUBDIR}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}" ]]; then
  echo "[lr-artifact-v0] reference image dir not found: ${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}" >&2
  exit 1
fi

echo "[lr-artifact-v0] scene root    : ${SCENE_ROOT}"
echo "[lr-artifact-v0] model path    : ${MODEL_PATH}"
echo "[lr-artifact-v0] output dir    : ${OUTPUT_DIR}"
echo "[lr-artifact-v0] interaction   : ${INTERACTION_IMAGES_SUBDIR}"
echo "[lr-artifact-v0] reference     : ${REFERENCE_IMAGES_SUBDIR}"
echo "[lr-artifact-v0] split/views   : ${SPLIT} max=${MAX_VIEWS}"
echo "[lr-artifact-v0] suppress q    : ${SUPPRESS_QUANTILE}"
echo "[lr-artifact-v0] delete q      : ${DELETE_QUANTILE}"
echo "[lr-artifact-v0] export plys   : ${EXPORT_PLYS}"
echo "[lr-artifact-v0] export pruned : ${EXPORT_PRUNED_MODEL}"
echo "[lr-artifact-v0] render pruned : ${RENDER_PRUNED_AFTER} split=${PRUNED_RENDER_SPLIT} max=${PRUNED_RENDER_MAX_VIEWS}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/export_lr_artifact_gaussian_scores_v0.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --interaction_images_subdir "${INTERACTION_IMAGES_SUBDIR}"
  --reference_images_subdir "${REFERENCE_IMAGES_SUBDIR}"
  --iteration "${ITERATION}"
  --split "${SPLIT}"
  --max_views "${MAX_VIEWS}"
  --visibility_downsample "${VISIBILITY_DOWNSAMPLE}"
  --visibility_topk "${VISIBILITY_TOPK}"
  --visibility_max_visible "${VISIBILITY_MAX_VISIBLE}"
  --visibility_max_patch_radius "${VISIBILITY_MAX_PATCH_RADIUS}"
  --lowpass_kernel "${LOWPASS_KERNEL}"
  --radius_risk_px "${RADIUS_RISK_PX}"
  --suppress_quantile "${SUPPRESS_QUANTILE}"
  --pull_artifact_quantile "${PULL_ARTIFACT_QUANTILE}"
  --pull_edge_quantile "${PULL_EDGE_QUANTILE}"
  --pull_max_footprint_risk "${PULL_MAX_FOOTPRINT_RISK}"
  --delete_quantile "${DELETE_QUANTILE}"
  --delete_min_lr_forbidden "${DELETE_MIN_LR_FORBIDDEN}"
  --delete_min_artifact "${DELETE_MIN_ARTIFACT}"
  --delete_max_edge_support "${DELETE_MAX_EDGE_SUPPORT}"
  --delete_min_footprint_risk "${DELETE_MIN_FOOTPRINT_RISK}"
  --delete_max_opacity "${DELETE_MAX_OPACITY}"
  --delete_min_radius_px "${DELETE_MIN_RADIUS_PX}"
  --delete_max_ratio "${DELETE_MAX_RATIO}"
  --num_debug_views "${NUM_DEBUG_VIEWS}"
)

if [[ "${EXPORT_PLYS}" == "1" ]]; then
  CMD+=(--export_plys)
fi
if [[ "${EXPORT_PRUNED_MODEL}" == "1" ]]; then
  CMD+=(--export_pruned_model)
fi

"${CMD[@]}"

echo "[done] score payload : ${OUTPUT_DIR}/lr_artifact_scores_v0.pt"
echo "[done] summary       : ${OUTPUT_DIR}/summary.json"
echo "[done] debug views   : ${OUTPUT_DIR}/debug_views"
if [[ "${EXPORT_PLYS}" == "1" ]]; then
  echo "[done] suppress ply  : ${OUTPUT_DIR}/suppress_candidate_ply/point_cloud/iteration_${ITERATION}/point_cloud.ply"
  echo "[done] pull ply      : ${OUTPUT_DIR}/pull_allowed_ply/point_cloud/iteration_${ITERATION}/point_cloud.ply"
  echo "[done] delete ply    : ${OUTPUT_DIR}/delete_candidate_ply/point_cloud/iteration_${ITERATION}/point_cloud.ply"
fi
if [[ "${EXPORT_PRUNED_MODEL}" == "1" ]]; then
  echo "[done] pruned model  : ${OUTPUT_DIR}/pruned_delete_candidate_model"
fi

if [[ "${EXPORT_PRUNED_MODEL}" == "1" && "${RENDER_PRUNED_AFTER}" == "1" ]]; then
  PRUNED_MODEL_DIR="${OUTPUT_DIR}/pruned_delete_candidate_model"
  if [[ ! -e "${PRUNED_MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" ]]; then
    echo "[lr-artifact-v0] pruned model point cloud not found: ${PRUNED_MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" >&2
    exit 1
  fi

  case "${PRUNED_RENDER_SPLIT}" in
    train|test|both)
      ;;
    *)
      echo "[lr-artifact-v0] unsupported PRUNED_RENDER_SPLIT=${PRUNED_RENDER_SPLIT}; use train, test, or both" >&2
      exit 1
      ;;
  esac

  echo "[lr-artifact-v0] rendering pruned preview on ${INTERACTION_IMAGES_SUBDIR} (${PRUNED_RENDER_SPLIT}, max=${PRUNED_RENDER_MAX_VIEWS})"
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --model_path "${PRUNED_MODEL_DIR}" \
    --scene_root "${SCENE_ROOT}" \
    --images_subdir "${INTERACTION_IMAGES_SUBDIR}" \
    --iteration "${ITERATION}" \
    --split "${PRUNED_RENDER_SPLIT}" \
    --max_views "${PRUNED_RENDER_MAX_VIEWS}"

  case "${PRUNED_RENDER_SPLIT}" in
    train)
      echo "[done] pruned renders: ${PRUNED_MODEL_DIR}/train/ours_${ITERATION}/renders"
      ;;
    test)
      echo "[done] pruned renders: ${PRUNED_MODEL_DIR}/test/ours_${ITERATION}/renders"
      ;;
    both)
      echo "[done] pruned train renders: ${PRUNED_MODEL_DIR}/train/ours_${ITERATION}/renders"
      echo "[done] pruned test renders : ${PRUNED_MODEL_DIR}/test/ours_${ITERATION}/renders"
      ;;
  esac
fi
