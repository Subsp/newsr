#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-${STAGE_NAME}_geometry_only_v0}"
MESH_BOUNDED_RUN_NAME="${MESH_BOUNDED_RUN_NAME:-${SOURCE_RUN_NAME}_mesh_bounded_v0}"
MODEL_PATH="${MODEL_PATH:-${SOF_ROOT}/output/mesh_bounded_gaussians_v0/${SCENE_NAME}/${MESH_BOUNDED_RUN_NAME}}"
ITERATION="${ITERATION:-34000}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-both}"
MAX_VIEWS="${MAX_VIEWS:-16}"
OPACITY_SCALE="${OPACITY_SCALE:-0.06}"
AUTO_OPACITY_CALIBRATE="${AUTO_OPACITY_CALIBRATE:-1}"
TARGET_ALPHA_MEAN="${TARGET_ALPHA_MEAN:-0.25}"
MIN_OPACITY_SCALE="${MIN_OPACITY_SCALE:-1e-8}"
OPACITY_CALIBRATION_STEPS="${OPACITY_CALIBRATION_STEPS:-10}"
RENDER_SCALE_MODIFIER="${RENDER_SCALE_MODIFIER:-0.35}"
VIEW_COLOR_MODE="${VIEW_COLOR_MODE:-camera_sample}"
RENDER_CENTER_DIAGNOSTICS="${RENDER_CENTER_DIAGNOSTICS:-1}"
PNORM_PERCENTILE="${PNORM_PERCENTILE:-99.0}"
LOG_GAIN="${LOG_GAIN:-30.0}"
COLOR_ERROR_ALPHA_MIN="${COLOR_ERROR_ALPHA_MIN:-0.02}"
WRITE_DEBUG_VISIBLE_MODEL="${WRITE_DEBUG_VISIBLE_MODEL:-1}"
DEBUG_VISIBLE_ALPHA="${DEBUG_VISIBLE_ALPHA:-0.75}"
DEBUG_SCALE_MULTIPLIER="${DEBUG_SCALE_MULTIPLIER:-4.0}"
DEBUG_COLOR_RGB="${DEBUG_COLOR_RGB:-1.0,0.05,0.02}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[mesh-bounded-conf-render-v0] missing model path: ${MODEL_PATH}" >&2
  exit 1
fi

echo "[mesh-bounded-conf-render-v0] scene   : ${SCENE_ROOT}"
echo "[mesh-bounded-conf-render-v0] model   : ${MODEL_PATH} iter=${ITERATION}"
echo "[mesh-bounded-conf-render-v0] views   : split=${SPLIT} images=${IMAGES_SUBDIR} max=${MAX_VIEWS}"
echo "[mesh-bounded-conf-render-v0] opacity : scale=${OPACITY_SCALE}"
echo "[mesh-bounded-conf-render-v0] color   : mode=${VIEW_COLOR_MODE} scale_modifier=${RENDER_SCALE_MODIFIER}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_mesh_bounded_confidence_maps_v0.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --images_subdir "${IMAGES_SUBDIR}"
  --split "${SPLIT}"
  --iteration "${ITERATION}"
  --max_views "${MAX_VIEWS}"
  --opacity_scale "${OPACITY_SCALE}"
  --target_alpha_mean "${TARGET_ALPHA_MEAN}"
  --min_opacity_scale "${MIN_OPACITY_SCALE}"
  --opacity_calibration_steps "${OPACITY_CALIBRATION_STEPS}"
  --render_scale_modifier "${RENDER_SCALE_MODIFIER}"
  --view_color_mode "${VIEW_COLOR_MODE}"
  --pnorm_percentile "${PNORM_PERCENTILE}"
  --log_gain "${LOG_GAIN}"
  --color_error_alpha_min "${COLOR_ERROR_ALPHA_MIN}"
  --debug_visible_alpha "${DEBUG_VISIBLE_ALPHA}"
  --debug_scale_multiplier "${DEBUG_SCALE_MULTIPLIER}"
  --debug_color_rgb "${DEBUG_COLOR_RGB}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ "${AUTO_OPACITY_CALIBRATE}" == "1" ]]; then
  CMD+=(--auto_opacity_calibrate)
fi
if [[ "${RENDER_CENTER_DIAGNOSTICS}" == "1" ]]; then
  CMD+=(--render_center_diagnostics)
fi
if [[ "${WRITE_DEBUG_VISIBLE_MODEL}" == "1" ]]; then
  CMD+=(--write_debug_visible_model)
fi
if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
  CMD+=(--white_background)
fi

"${CMD[@]}"
