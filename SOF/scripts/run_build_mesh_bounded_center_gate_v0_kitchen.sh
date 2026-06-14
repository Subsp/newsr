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
MESH_BOUNDED_RUN_NAME="${MESH_BOUNDED_RUN_NAME:-${SOURCE_RUN_NAME}_mesh_bounded_color_v0}"
MODEL_PATH="${MODEL_PATH:-${SOF_ROOT}/output/mesh_bounded_gaussians_v0/${SCENE_NAME}/${MESH_BOUNDED_RUN_NAME}}"
ITERATION="${ITERATION:-34000}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-train}"
MAX_VIEWS="${MAX_VIEWS:-0}"

SOURCE_PRIOR_NAME="${SOURCE_PRIOR_NAME:-quality_fuse_v1_maskreparam_geometry_only_safe_hf_v1}"
SOURCE_PRIOR_ROOT="${SOURCE_PRIOR_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets/prepared_sr_priors/${SOURCE_PRIOR_NAME}}"
OUTPUT_PRIOR_NAME="${OUTPUT_PRIOR_NAME:-${SOURCE_PRIOR_NAME}_meshbounded_center_gate_v0}"
OUTPUT_PRIOR_ROOT="${OUTPUT_PRIOR_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets/prepared_sr_priors/${OUTPUT_PRIOR_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/mesh_bounded_center_gate_v0/${SCENE_NAME}/${MESH_BOUNDED_RUN_NAME}_center_gate_v0}"

SOURCE_MASK_SUBDIR="${SOURCE_MASK_SUBDIR:-usable_masks}"
SOURCE_MASK_SUFFIX="${SOURCE_MASK_SUFFIX:-}"
SOURCE_MASK_MULTIPLY="${SOURCE_MASK_MULTIPLY:-1}"
COPY_PRIOR_DIRS="${COPY_PRIOR_DIRS:-0}"

COLOR_SIGMA="${COLOR_SIGMA:-0.16}"
PROXY_CONFIDENCE_POWER="${PROXY_CONFIDENCE_POWER:-0.5}"
PROXY_COLOR_CONFIDENCE_POWER="${PROXY_COLOR_CONFIDENCE_POWER:-0.5}"
PROXY_MIP_SUPPORT_POWER="${PROXY_MIP_SUPPORT_POWER:-0.0}"
DENSITY_SOFT_TARGET="${DENSITY_SOFT_TARGET:-1.0}"
DENSITY_POWER="${DENSITY_POWER:-0.0}"
DILATE_RADIUS="${DILATE_RADIUS:-5}"
BLUR_RADIUS="${BLUR_RADIUS:-1.5}"
GATE_FLOOR="${GATE_FLOOR:-0.0}"
GATE_SCALE="${GATE_SCALE:-1.0}"
DEPTH_MIN="${DEPTH_MIN:-0.01}"
SAVE_DEBUG_MAPS="${SAVE_DEBUG_MAPS:-1}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[mesh-bounded-center-gate-v0] missing model path: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ -n "${SOURCE_PRIOR_ROOT}" && ! -d "${SOURCE_PRIOR_ROOT}" ]]; then
  echo "[mesh-bounded-center-gate-v0] missing source prior root: ${SOURCE_PRIOR_ROOT}" >&2
  exit 1
fi

echo "[mesh-bounded-center-gate-v0] scene      : ${SCENE_ROOT}"
echo "[mesh-bounded-center-gate-v0] model      : ${MODEL_PATH} iter=${ITERATION}"
echo "[mesh-bounded-center-gate-v0] views      : split=${SPLIT} images=${IMAGES_SUBDIR} max=${MAX_VIEWS}"
echo "[mesh-bounded-center-gate-v0] source prior: ${SOURCE_PRIOR_ROOT}"
echo "[mesh-bounded-center-gate-v0] output prior: ${OUTPUT_PRIOR_ROOT}"
echo "[mesh-bounded-center-gate-v0] output root : ${OUTPUT_ROOT}"
echo "[mesh-bounded-center-gate-v0] gate       : color_sigma=${COLOR_SIGMA} dilate=${DILATE_RADIUS} blur=${BLUR_RADIUS}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/build_mesh_bounded_center_gate_v0.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --iteration "${ITERATION}"
  --images_subdir "${IMAGES_SUBDIR}"
  --split "${SPLIT}"
  --max_views "${MAX_VIEWS}"
  --output_root "${OUTPUT_ROOT}"
  --source_prior_root "${SOURCE_PRIOR_ROOT}"
  --output_prior_root "${OUTPUT_PRIOR_ROOT}"
  --source_mask_subdir "${SOURCE_MASK_SUBDIR}"
  --source_mask_suffix "${SOURCE_MASK_SUFFIX}"
  --color_sigma "${COLOR_SIGMA}"
  --proxy_confidence_power "${PROXY_CONFIDENCE_POWER}"
  --proxy_color_confidence_power "${PROXY_COLOR_CONFIDENCE_POWER}"
  --proxy_mip_support_power "${PROXY_MIP_SUPPORT_POWER}"
  --density_soft_target "${DENSITY_SOFT_TARGET}"
  --density_power "${DENSITY_POWER}"
  --dilate_radius "${DILATE_RADIUS}"
  --blur_radius "${BLUR_RADIUS}"
  --gate_floor "${GATE_FLOOR}"
  --gate_scale "${GATE_SCALE}"
  --depth_min "${DEPTH_MIN}"
)

if [[ "${SOURCE_MASK_MULTIPLY}" != "1" ]]; then
  CMD+=(--no_source_mask_multiply)
fi
if [[ "${COPY_PRIOR_DIRS}" == "1" ]]; then
  CMD+=(--copy_prior_dirs)
fi
if [[ "${SAVE_DEBUG_MAPS}" == "1" ]]; then
  CMD+=(--save_debug_maps)
fi

"${CMD[@]}"
