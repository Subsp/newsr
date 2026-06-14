#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
MESH_COMPARE_ROOT="${MESH_COMPARE_ROOT:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}}"
MESH_PATH="${MESH_PATH:-${MESH_COMPARE_ROOT}/${STAGE_NAME}_prepare_stage_sof_export_mesh_v0_${STAGE_NAME}_7.ply}"

CAMERA_MODEL_NAME="${CAMERA_MODEL_NAME:-soflr30k}"
CAMERA_MODEL_PATH="${CAMERA_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_sof_vanilla_images8_v1/${CAMERA_MODEL_NAME}}"

PATCH_OBSERVATION_RUN_NAME="${PATCH_OBSERVATION_RUN_NAME:-mesh_patch_observations_smoke_v0}"
PATCH_OBSERVATION_ROOT="${PATCH_OBSERVATION_ROOT:-${SOF_ROOT}/output/mesh_patch_observations_v0/${SCENE_NAME}/${PATCH_OBSERVATION_RUN_NAME}}"

PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-quality_fuse_v1_maskreparam_geometry_only_safe_hf_v1}"
PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets/prepared_sr_priors/${PREPARED_SR_PRIOR_NAME}}"
PRIOR_DIR="${PRIOR_DIR:-${PREPARED_SR_PRIOR_ROOT}/fused_priors}"
ANCHOR_DIR_DEFAULT="${PREPARED_SR_PRIOR_ROOT}/aligned_references"
if [[ -z "${ANCHOR_DIR+x}" ]]; then
  if [[ -d "${ANCHOR_DIR_DEFAULT}" ]]; then
    ANCHOR_DIR="${ANCHOR_DIR_DEFAULT}"
  else
    ANCHOR_DIR=""
  fi
fi

RUN_NAME="${RUN_NAME:-alternating_prior_surface_v0}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/alternating_prior_surface_v0/${SCENE_NAME}/${RUN_NAME}}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
SPLIT="${SPLIT:-train}"
MAX_VIEWS="${MAX_VIEWS:-16}"
CYCLES="${CYCLES:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-0}"
APPEARANCE_STEPS="${APPEARANCE_STEPS:-200}"
STRUCTURE_STEPS="${STRUCTURE_STEPS:-200}"
STRUCTURE_VIEW_LIMIT="${STRUCTURE_VIEW_LIMIT:-4}"
APPEARANCE_LR="${APPEARANCE_LR:-0.02}"
STRUCTURE_LR="${STRUCTURE_LR:-5e-4}"
APPEARANCE_ANCHOR_LAMBDA="${APPEARANCE_ANCHOR_LAMBDA:-0.05}"
LAMBDA_STRUCTURE_PHOTO="${LAMBDA_STRUCTURE_PHOTO:-1.0}"
LAMBDA_STRUCTURE_MV="${LAMBDA_STRUCTURE_MV:-0.25}"
LAMBDA_STRUCTURE_DELTA="${LAMBDA_STRUCTURE_DELTA:-0.05}"
LAMBDA_STRUCTURE_NORMAL="${LAMBDA_STRUCTURE_NORMAL:-0.02}"
MIN_VIEWS="${MIN_VIEWS:-2}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.05}"
MAX_DISAGREEMENT="${MAX_DISAGREEMENT:-0.10}"
MAX_COUNT="${MAX_COUNT:-0}"
SEED="${SEED:-0}"
DEPTH_MIN="${DEPTH_MIN:-0.02}"
ANCHOR_LOWFREQ_THRESHOLD="${ANCHOR_LOWFREQ_THRESHOLD:-0.08}"
ANCHOR_LOWFREQ_KERNEL="${ANCHOR_LOWFREQ_KERNEL:-15}"
MAX_SURFACE_VERTEX_DISPLACEMENT="${MAX_SURFACE_VERTEX_DISPLACEMENT:-0.02}"
MESHGS_SCALE_MULTIPLIER="${MESHGS_SCALE_MULTIPLIER:-1.0}"
MESHGS_THICKNESS_MULTIPLIER="${MESHGS_THICKNESS_MULTIPLIER:-0.5}"
MESHGS_INIT_OPACITY="${MESHGS_INIT_OPACITY:-0.35}"
INIT_COLOR_SOURCE="${INIT_COLOR_SOURCE:-fused_rgb}"
INIT_COLOR_GRAY_VALUE="${INIT_COLOR_GRAY_VALUE:-0.5}"
SAVE_EVERY_CYCLES="${SAVE_EVERY_CYCLES:-0}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-0}"
SH_DEGREE="${SH_DEGREE:-3}"

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[alternating-prior-surface-v0] missing scene root: ${SCENE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${CAMERA_MODEL_PATH}" ]]; then
  echo "[alternating-prior-surface-v0] missing camera model path: ${CAMERA_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MESH_PATH}" ]]; then
  echo "[alternating-prior-surface-v0] missing mesh: ${MESH_PATH}" >&2
  exit 1
fi
if [[ ! -d "${PATCH_OBSERVATION_ROOT}" ]]; then
  echo "[alternating-prior-surface-v0] missing patch observation root: ${PATCH_OBSERVATION_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${PRIOR_DIR}" ]]; then
  echo "[alternating-prior-surface-v0] missing prior dir: ${PRIOR_DIR}" >&2
  exit 1
fi
if [[ -n "${ANCHOR_DIR}" && ! -d "${ANCHOR_DIR}" ]]; then
  echo "[alternating-prior-surface-v0] missing anchor dir: ${ANCHOR_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_MODEL_PATH}"

echo "[alternating-prior-surface-v0] scene        : ${SCENE_ROOT}"
echo "[alternating-prior-surface-v0] camera model : ${CAMERA_MODEL_PATH}"
echo "[alternating-prior-surface-v0] mesh         : ${MESH_PATH}"
echo "[alternating-prior-surface-v0] patch root   : ${PATCH_OBSERVATION_ROOT}"
echo "[alternating-prior-surface-v0] prior dir    : ${PRIOR_DIR}"
if [[ -n "${ANCHOR_DIR}" ]]; then
  echo "[alternating-prior-surface-v0] anchor dir   : ${ANCHOR_DIR}"
else
  echo "[alternating-prior-surface-v0] anchor dir   : disabled"
fi
echo "[alternating-prior-surface-v0] output       : ${OUTPUT_MODEL_PATH}"
echo "[alternating-prior-surface-v0] schedule     : cycles=${CYCLES} total_steps=${TOTAL_STEPS} A=${APPEARANCE_STEPS} B=${STRUCTURE_STEPS}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/train_alternating_prior_surface_v0.py"
  --scene_root "${SCENE_ROOT}"
  --camera_model_path "${CAMERA_MODEL_PATH}"
  --mesh_path "${MESH_PATH}"
  --patch_observation_root "${PATCH_OBSERVATION_ROOT}"
  --prior_dir "${PRIOR_DIR}"
  --output_model_path "${OUTPUT_MODEL_PATH}"
  --images_subdir "${IMAGES_SUBDIR}"
  --split "${SPLIT}"
  --max_views "${MAX_VIEWS}"
  --cycles "${CYCLES}"
  --total_steps "${TOTAL_STEPS}"
  --appearance_steps "${APPEARANCE_STEPS}"
  --structure_steps "${STRUCTURE_STEPS}"
  --structure_view_limit "${STRUCTURE_VIEW_LIMIT}"
  --appearance_lr "${APPEARANCE_LR}"
  --structure_lr "${STRUCTURE_LR}"
  --appearance_anchor_lambda "${APPEARANCE_ANCHOR_LAMBDA}"
  --lambda_structure_photo "${LAMBDA_STRUCTURE_PHOTO}"
  --lambda_structure_mv "${LAMBDA_STRUCTURE_MV}"
  --lambda_structure_delta "${LAMBDA_STRUCTURE_DELTA}"
  --lambda_structure_normal "${LAMBDA_STRUCTURE_NORMAL}"
  --min_views "${MIN_VIEWS}"
  --min_confidence "${MIN_CONFIDENCE}"
  --max_disagreement "${MAX_DISAGREEMENT}"
  --max_count "${MAX_COUNT}"
  --seed "${SEED}"
  --depth_min "${DEPTH_MIN}"
  --anchor_lowfreq_threshold "${ANCHOR_LOWFREQ_THRESHOLD}"
  --anchor_lowfreq_kernel "${ANCHOR_LOWFREQ_KERNEL}"
  --max_surface_vertex_displacement "${MAX_SURFACE_VERTEX_DISPLACEMENT}"
  --meshgs_scale_multiplier "${MESHGS_SCALE_MULTIPLIER}"
  --meshgs_thickness_multiplier "${MESHGS_THICKNESS_MULTIPLIER}"
  --meshgs_init_opacity "${MESHGS_INIT_OPACITY}"
  --init_color_source "${INIT_COLOR_SOURCE}"
  --init_color_gray_value "${INIT_COLOR_GRAY_VALUE}"
  --save_every_cycles "${SAVE_EVERY_CYCLES}"
  --output_iteration "${OUTPUT_ITERATION}"
  --sh_degree "${SH_DEGREE}"
)

if [[ -n "${ANCHOR_DIR}" ]]; then
  CMD+=(--anchor_dir "${ANCHOR_DIR}")
fi

"${CMD[@]}"

echo "[done] summary : ${OUTPUT_MODEL_PATH}/alternating_prior_surface_v0_summary.json"
echo "[done] output  : ${OUTPUT_MODEL_PATH}"
