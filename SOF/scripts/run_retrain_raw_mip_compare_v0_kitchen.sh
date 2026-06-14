#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
CURRENT_EXPERIMENT_NAME="${CURRENT_EXPERIMENT_NAME:-mip30k}"
RERUN_EXPERIMENT_NAME="${RERUN_EXPERIMENT_NAME:-mip30k_rerun_v0}"
TRAIN_IMAGES_SUBDIR="${TRAIN_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"

CURRENT_MODEL_PATH="${CURRENT_MODEL_PATH:-${SCENE_ASSET_ROOT}/${EXPERIMENT_GROUP}/${CURRENT_EXPERIMENT_NAME}}"
RERUN_MODEL_DIR="${RERUN_MODEL_DIR:-${SCENE_ASSET_ROOT}/${EXPERIMENT_GROUP}/${RERUN_EXPERIMENT_NAME}}"

ITERATIONS="${ITERATIONS:-30000}"
CURRENT_ITERATION="${CURRENT_ITERATION:-${ITERATIONS}}"
RERUN_ITERATION="${RERUN_ITERATION:-${ITERATIONS}}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_CURRENT_RENDER="${RUN_CURRENT_RENDER:-1}"
RUN_RERUN_RENDER="${RUN_RERUN_RENDER:-1}"
RUN_RERUN_METRICS="${RUN_RERUN_METRICS:-0}"
RUN_RENDER_COMPARE="${RUN_RENDER_COMPARE:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/raw_mip_rerun_compare_v0/${SCENE_NAME}/${RERUN_EXPERIMENT_NAME}}"
COMPARE_JSON="${COMPARE_JSON:-${OUTPUT_ROOT}/model_stats_compare.json}"
RENDER_COMPARE_DIR="${RENDER_COMPARE_DIR:-${OUTPUT_ROOT}/render_compare_current_vs_rerun}"

CURRENT_RENDER_DIR="${CURRENT_RENDER_DIR:-${CURRENT_MODEL_PATH}/test/ours_${CURRENT_ITERATION}/test_preds_${RENDER_RESOLUTION}}"
RERUN_RENDER_DIR="${RERUN_RENDER_DIR:-${RERUN_MODEL_DIR}/test/ours_${RERUN_ITERATION}/test_preds_${RENDER_RESOLUTION}}"

mkdir -p "${OUTPUT_ROOT}"

echo "[raw-mip-rerun-compare-v0] scene          : ${SCENE_ROOT}"
echo "[raw-mip-rerun-compare-v0] current model  : ${CURRENT_MODEL_PATH} iter=${CURRENT_ITERATION}"
echo "[raw-mip-rerun-compare-v0] rerun model    : ${RERUN_MODEL_DIR} iter=${RERUN_ITERATION}"
echo "[raw-mip-rerun-compare-v0] train images   : ${TRAIN_IMAGES_SUBDIR}"
echo "[raw-mip-rerun-compare-v0] render images  : ${TARGET_IMAGES_SUBDIR}"
echo "[raw-mip-rerun-compare-v0] output root    : ${OUTPUT_ROOT}"
echo "[raw-mip-rerun-compare-v0] run train      : ${RUN_TRAIN}"
echo "[raw-mip-rerun-compare-v0] render current : ${RUN_CURRENT_RENDER}"
echo "[raw-mip-rerun-compare-v0] render rerun   : ${RUN_RERUN_RENDER}"

for path in "${SCENE_ROOT}" "${CURRENT_MODEL_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[raw-mip-rerun-compare-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ "${RUN_CURRENT_RENDER}" == "1" ]]; then
  echo
  echo "[0/3] render current raw mip reference"
  CURRENT_EXPERIMENT_NAME="${CURRENT_EXPERIMENT_NAME}" \
  EXPERIMENT_GROUP="${EXPERIMENT_GROUP}" \
  MODEL_DIR="${CURRENT_MODEL_PATH}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  TRAIN_IMAGES_SUBDIR="${TRAIN_IMAGES_SUBDIR}" \
  TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR}" \
  ITERATIONS="${CURRENT_ITERATION}" \
  RENDER_RESOLUTION="${RENDER_RESOLUTION}" \
  RUN_RENDER_AFTER=1 \
  RUN_METRICS_AFTER=0 \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SCRIPT_DIR}/run_train_mipsplatting_kitchen_images8_scene.sh"
else
  echo
  echo "[0/3] skip current render (RUN_CURRENT_RENDER=${RUN_CURRENT_RENDER})"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo
  echo "[1/3] train/rerender raw mip rerun"
  EXPERIMENT_GROUP="${EXPERIMENT_GROUP}" \
  EXPERIMENT_NAME="${RERUN_EXPERIMENT_NAME}" \
  MODEL_DIR="${RERUN_MODEL_DIR}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  TRAIN_IMAGES_SUBDIR="${TRAIN_IMAGES_SUBDIR}" \
  TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR}" \
  ITERATIONS="${RERUN_ITERATION}" \
  RENDER_RESOLUTION="${RENDER_RESOLUTION}" \
  RUN_RENDER_AFTER="${RUN_RERUN_RENDER}" \
  RUN_METRICS_AFTER="${RUN_RERUN_METRICS}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SCRIPT_DIR}/run_train_mipsplatting_kitchen_images8_scene.sh"
else
  echo
  echo "[1/3] skip rerun train (RUN_TRAIN=${RUN_TRAIN})"
fi

echo
echo "[2/3] compare cfg and point-cloud statistics"
"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_raw_mip_runs_v0.py" \
  --current_model_path "${CURRENT_MODEL_PATH}" \
  --current_iteration "${CURRENT_ITERATION}" \
  --rerun_model_path "${RERUN_MODEL_DIR}" \
  --rerun_iteration "${RERUN_ITERATION}" \
  --output_path "${COMPARE_JSON}"

if [[ "${RUN_RENDER_COMPARE}" == "1" ]]; then
  if [[ -d "${CURRENT_RENDER_DIR}" && -d "${RERUN_RENDER_DIR}" ]]; then
    echo
    echo "[3/3] compare current vs rerun renders"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_render_dirs_no_gt.py" \
      --reference_dir "${CURRENT_RENDER_DIR}" \
      --candidate_dir "${RERUN_RENDER_DIR}" \
      --output_dir "${RENDER_COMPARE_DIR}"
  else
    echo
    echo "[3/3] skip render compare; missing dirs:"
    echo "      current: ${CURRENT_RENDER_DIR}"
    echo "      rerun  : ${RERUN_RENDER_DIR}"
    echo "      rerun with RUN_CURRENT_RENDER=1 and RUN_RERUN_RENDER=1 if needed."
  fi
else
  echo
  echo "[3/3] skip render compare (RUN_RENDER_COMPARE=${RUN_RENDER_COMPARE})"
fi

echo
echo "[done] current model : ${CURRENT_MODEL_PATH}"
echo "[done] rerun model   : ${RERUN_MODEL_DIR}"
echo "[done] compare json  : ${COMPARE_JSON}"
echo "[done] render compare: ${RENDER_COMPARE_DIR}"
