#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-both}"
MAX_VIEWS="${MAX_VIEWS:-0}"
ITERATION="${ITERATION:--1}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"
LOW_HIT_MAX_VIEWS="${LOW_HIT_MAX_VIEWS:-2}"
ROD_ANISOTROPY_THRESHOLD="${ROD_ANISOTROPY_THRESHOLD:-10.0}"
PREVIEW_MAX_POINTS="${PREVIEW_MAX_POINTS:-50000}"
TOP_RECORDS="${TOP_RECORDS:-256}"
NO_GAUSSIAN_SUBSET_EXPORT="${NO_GAUSSIAN_SUBSET_EXPORT:-0}"
DEBUG_VISIBLE_ALPHA="${DEBUG_VISIBLE_ALPHA:-0.65}"
DEBUG_SCALE_MULTIPLIER="${DEBUG_SCALE_MULTIPLIER:-2.5}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_NAME="${RUN_NAME:-geometry_only_quality_fuse_safe_hf_v1_rerun_v1}"
MODEL_PATH="${MODEL_PATH:-${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${RUN_NAME}/recovered_mip_model_hr_v0}"
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-${RUN_NAME}_global_ray_hit_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/global_ray_hit_gaussians_v0/${SCENE_NAME}/${OUTPUT_RUN_NAME}}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[global-ray-hit-v0] missing model path: ${MODEL_PATH}" >&2
  exit 1
fi

echo "[global-ray-hit-v0] scene       : ${SCENE_ROOT}"
echo "[global-ray-hit-v0] model       : ${MODEL_PATH} iter=${ITERATION}"
echo "[global-ray-hit-v0] views       : split=${SPLIT} images=${IMAGES_SUBDIR} max=${MAX_VIEWS}"
echo "[global-ray-hit-v0] thresholds  : low_hit<=${LOW_HIT_MAX_VIEWS} rod_aniso>=${ROD_ANISOTROPY_THRESHOLD}"
echo "[global-ray-hit-v0] output root : ${OUTPUT_ROOT}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/score_global_ray_hit_gaussians_v0.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --output_root "${OUTPUT_ROOT}"
  --images_subdir "${IMAGES_SUBDIR}"
  --split "${SPLIT}"
  --iteration "${ITERATION}"
  --max_views "${MAX_VIEWS}"
  --low_hit_max_views "${LOW_HIT_MAX_VIEWS}"
  --rod_anisotropy_threshold "${ROD_ANISOTROPY_THRESHOLD}"
  --preview_max_points "${PREVIEW_MAX_POINTS}"
  --top_records "${TOP_RECORDS}"
  --debug_visible_alpha "${DEBUG_VISIBLE_ALPHA}"
  --debug_scale_multiplier "${DEBUG_SCALE_MULTIPLIER}"
)

if [[ "${NO_GAUSSIAN_SUBSET_EXPORT}" == "1" ]]; then
  CMD+=(--no_gaussian_subset_export)
fi
if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
  CMD+=(--white_background)
fi

"${CMD[@]}"

echo "[done] payload  : ${OUTPUT_ROOT}/global_ray_hit_gaussians_v0.pt"
echo "[done] summary  : ${OUTPUT_ROOT}/global_ray_hit_gaussians_v0_summary.json"
echo "[done] previews : ${OUTPUT_ROOT}/never_hit_preview_v0.ply"
echo "[done] previews : ${OUTPUT_ROOT}/never_hit_rodlike_preview_v0.ply"
echo "[done] previews : ${OUTPUT_ROOT}/low_hit_rodlike_preview_v0.ply"
echo "[done] gaussian : ${OUTPUT_ROOT}/never_hit_gaussian_model_v0"
echo "[done] gaussian : ${OUTPUT_ROOT}/never_hit_rodlike_gaussian_model_v0"
echo "[done] gaussian : ${OUTPUT_ROOT}/low_hit_rodlike_gaussian_model_v0"
echo "[done] debug gs : ${OUTPUT_ROOT}/never_hit_rodlike_debug_visible_gaussian_model_v0"
echo "[done] debug gs : ${OUTPUT_ROOT}/low_hit_rodlike_debug_visible_gaussian_model_v0"
