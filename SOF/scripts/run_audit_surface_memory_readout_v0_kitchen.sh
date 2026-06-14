#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

BEFORE_RUN_NAME="${BEFORE_RUN_NAME:-debug_stage_00b3_after_scale_canonicalize_geometry_only_v0}"
BEFORE_MODEL_PATH="${BEFORE_MODEL_PATH:-${SOF_ROOT}/output/surface_smooth_filter_v0/${SCENE_NAME}/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_delta_sigma_e3_delta_sigma_bodyaware_stronger_v1}"
BEFORE_ITERATION="${BEFORE_ITERATION:-34000}"

AFTER_RUN_NAME="${AFTER_RUN_NAME:-debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_alternating_mainline5k_memory_v0_latestonly}"
AFTER_MODEL_PATH="${AFTER_MODEL_PATH:-${SOF_ROOT}/output/bounded_surface_alternating_v0/${SCENE_NAME}/${AFTER_RUN_NAME}}"
AFTER_ITERATION="${AFTER_ITERATION:-39000}"

PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-sof_surface_v0_images_8_to_images_2_mask0.12_soft}"
PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${PREPARED_SR_PRIOR_NAME}}"
SR_PRIOR_ROOT="${SR_PRIOR_ROOT:-${PREPARED_SR_PRIOR_ROOT}/priors}"
SR_ANCHOR_ROOT="${SR_ANCHOR_ROOT:-${SR_PRIOR_ROOT}}"
SR_PRIOR_MASK_DIR="${SR_PRIOR_MASK_DIR:-}"
SR_PRIOR_CONSISTENCY_THRESHOLD="${SR_PRIOR_CONSISTENCY_THRESHOLD:-0.0}"
SR_PRIOR_MASK_FLOOR="${SR_PRIOR_MASK_FLOOR:-0.0}"
SR_PRIOR_MASK_SUFFIX="${SR_PRIOR_MASK_SUFFIX:-}"

MEMORY_PATH="${MEMORY_PATH:-${AFTER_MODEL_PATH}/surface_prior_memory_latest.pt}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
MAX_VIEWS="${MAX_VIEWS:-24}"
HAZE_KERNEL="${HAZE_KERNEL:-31}"
APPEARANCE_TARGET_MODE="${APPEARANCE_TARGET_MODE:-}"
APPEARANCE_RESIDUAL_CLIP="${APPEARANCE_RESIDUAL_CLIP:--1}"
APPEARANCE_RESIDUAL_SCALE="${APPEARANCE_RESIDUAL_SCALE:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-${AFTER_MODEL_PATH}/audit_surface_memory_readout_v0}"

if [[ ! -d "${BEFORE_MODEL_PATH}" ]]; then
  echo "[audit-surface-memory-readout-v0] missing before model: ${BEFORE_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${AFTER_MODEL_PATH}" ]]; then
  echo "[audit-surface-memory-readout-v0] missing after model: ${AFTER_MODEL_PATH}" >&2
  exit 1
fi

CMD=(
  python -u "${SOF_ROOT}/scripts/audit_surface_memory_readout_v0.py"
  --scene_root "${SCENE_ROOT}"
  --before_model_path "${BEFORE_MODEL_PATH}"
  --after_model_path "${AFTER_MODEL_PATH}"
  --before_iteration "${BEFORE_ITERATION}"
  --after_iteration "${AFTER_ITERATION}"
  --images_subdir "${IMAGES_SUBDIR}"
  --max_views "${MAX_VIEWS}"
  --haze_kernel "${HAZE_KERNEL}"
  --appearance_target_mode "${APPEARANCE_TARGET_MODE}"
  --appearance_residual_clip "${APPEARANCE_RESIDUAL_CLIP}"
  --appearance_residual_scale "${APPEARANCE_RESIDUAL_SCALE}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ -f "${MEMORY_PATH}" ]]; then
  CMD+=(--memory_path "${MEMORY_PATH}")
else
  echo "[audit-surface-memory-readout-v0] memory not found, running DC/render audit only: ${MEMORY_PATH}" >&2
fi
if [[ -d "${SR_PRIOR_ROOT}" && -d "${SR_ANCHOR_ROOT}" ]]; then
  CMD+=(
    --sr_prior_root "${SR_PRIOR_ROOT}"
    --sr_anchor_root "${SR_ANCHOR_ROOT}"
    --sr_prior_consistency_threshold "${SR_PRIOR_CONSISTENCY_THRESHOLD}"
    --sr_prior_mask_floor "${SR_PRIOR_MASK_FLOOR}"
    --sr_prior_mask_suffix "${SR_PRIOR_MASK_SUFFIX}"
  )
  if [[ -d "${SR_PRIOR_MASK_DIR}" ]]; then
    CMD+=(--sr_prior_mask_dir "${SR_PRIOR_MASK_DIR}")
  fi
else
  echo "[audit-surface-memory-readout-v0] SR prior/anchor dirs unavailable, skipping prior-error audit." >&2
fi

"${CMD[@]}"
