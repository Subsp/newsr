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
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/mesh_bounded_mesh_evidence_v0/${SCENE_NAME}/${MESH_BOUNDED_RUN_NAME}_mesh_evidence_v0}"

COLOR_SIGMA="${COLOR_SIGMA:-0.16}"
PROXY_CONFIDENCE_POWER="${PROXY_CONFIDENCE_POWER:-0.5}"
PROXY_COLOR_CONFIDENCE_POWER="${PROXY_COLOR_CONFIDENCE_POWER:-0.5}"
PROXY_MIP_SUPPORT_POWER="${PROXY_MIP_SUPPORT_POWER:-0.0}"
PROXY_OPACITY_POWER="${PROXY_OPACITY_POWER:-0.0}"
MIN_HIT_VIEWS="${MIN_HIT_VIEWS:-3}"
TRUSTED_COLOR_ERROR="${TRUSTED_COLOR_ERROR:-0.12}"
TRUSTED_GATE="${TRUSTED_GATE:-0.28}"
DEPTH_MIN="${DEPTH_MIN:-0.01}"
SAVE_DEBUG_MAPS="${SAVE_DEBUG_MAPS:-1}"
DEBUG_MAX_VIEWS="${DEBUG_MAX_VIEWS:-16}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[mesh-bounded-mesh-evidence-v0] missing model path: ${MODEL_PATH}" >&2
  exit 1
fi

echo "[mesh-bounded-mesh-evidence-v0] scene  : ${SCENE_ROOT}"
echo "[mesh-bounded-mesh-evidence-v0] model  : ${MODEL_PATH} iter=${ITERATION}"
echo "[mesh-bounded-mesh-evidence-v0] views  : split=${SPLIT} images=${IMAGES_SUBDIR} max=${MAX_VIEWS}"
echo "[mesh-bounded-mesh-evidence-v0] output : ${OUTPUT_ROOT}"
echo "[mesh-bounded-mesh-evidence-v0] trust  : min_views=${MIN_HIT_VIEWS} color<=${TRUSTED_COLOR_ERROR} gate>=${TRUSTED_GATE}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/build_mesh_bounded_mesh_evidence_v0.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --iteration "${ITERATION}"
  --images_subdir "${IMAGES_SUBDIR}"
  --split "${SPLIT}"
  --max_views "${MAX_VIEWS}"
  --output_root "${OUTPUT_ROOT}"
  --color_sigma "${COLOR_SIGMA}"
  --proxy_confidence_power "${PROXY_CONFIDENCE_POWER}"
  --proxy_color_confidence_power "${PROXY_COLOR_CONFIDENCE_POWER}"
  --proxy_mip_support_power "${PROXY_MIP_SUPPORT_POWER}"
  --proxy_opacity_power "${PROXY_OPACITY_POWER}"
  --min_hit_views "${MIN_HIT_VIEWS}"
  --trusted_color_error "${TRUSTED_COLOR_ERROR}"
  --trusted_gate "${TRUSTED_GATE}"
  --depth_min "${DEPTH_MIN}"
  --debug_max_views "${DEBUG_MAX_VIEWS}"
)

if [[ "${SAVE_DEBUG_MAPS}" == "1" ]]; then
  CMD+=(--save_debug_maps)
fi

"${CMD[@]}"
