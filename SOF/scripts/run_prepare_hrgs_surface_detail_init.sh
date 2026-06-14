#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
DETAIL_MODEL_PATH="${DETAIL_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input}"
DETAIL_ITERATION="${DETAIL_ITERATION:-30000}"

CARRIER_PAYLOAD="${CARRIER_PAYLOAD:-}"
ACTION_PAYLOAD="${ACTION_PAYLOAD:-}"
ROUTE_PAYLOAD="${ROUTE_PAYLOAD:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/hrgs_eval}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-surface_detail_init}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${OUTPUT_ROOT}/${SCENE_NAME}_merged_${CHECKPOINT_TAG}}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:--1}"

SURFACE_MIN_CONFIDENCE="${SURFACE_MIN_CONFIDENCE:-0.05}"
SURFACE_MAX_DISAGREEMENT="${SURFACE_MAX_DISAGREEMENT:-1.0}"
SURFACE_MIN_VIEWS="${SURFACE_MIN_VIEWS:-1}"
SURFACE_MAX_COUNT="${SURFACE_MAX_COUNT:-0}"
SURFACE_SEED="${SURFACE_SEED:-0}"
SURFACE_SCALE_MULTIPLIER="${SURFACE_SCALE_MULTIPLIER:-1.0}"
SURFACE_THICKNESS_MULTIPLIER="${SURFACE_THICKNESS_MULTIPLIER:-0.5}"
SURFACE_MIN_SCALE="${SURFACE_MIN_SCALE:-1e-5}"
SURFACE_INIT_OPACITY="${SURFACE_INIT_OPACITY:-0.35}"
DETAIL_ATTACH_SCALE="${DETAIL_ATTACH_SCALE:-0.1}"
DETAIL_UPDATE_SCALE="${DETAIL_UPDATE_SCALE:-1.0}"
DETAIL_DETAIL_SCALE="${DETAIL_DETAIL_SCALE:-1.0}"
DETAIL_PRIOR_COLOR_SCALE="${DETAIL_PRIOR_COLOR_SCALE:-1.0}"
DETAIL_RADIUS_GATE="${DETAIL_RADIUS_GATE:-1}"
DETAIL_RADIUS_IMAGES_SUBDIR="${DETAIL_RADIUS_IMAGES_SUBDIR:-images_2}"
DETAIL_RADIUS_REF_PX="${DETAIL_RADIUS_REF_PX:-96.0}"
DETAIL_RADIUS_MIN_GATE="${DETAIL_RADIUS_MIN_GATE:-0.1}"
DETAIL_RADIUS_MAX_VIEWS="${DETAIL_RADIUS_MAX_VIEWS:-32}"
ROUTE_EXECUTION_MODE="${ROUTE_EXECUTION_MODE:-detail_preserve_v0p8}"
ROUTE_DETAIL_PROTECT_BETA="${ROUTE_DETAIL_PROTECT_BETA:-0.7}"
ROUTE_DETAIL_ATTACH_STRENGTH="${ROUTE_DETAIL_ATTACH_STRENGTH:-0.0}"
ROUTE_DETAIL_GEOMETRY_STRENGTH="${ROUTE_DETAIL_GEOMETRY_STRENGTH:-0.0}"
ROUTE_PRIOR_COLOR_STRENGTH_SCALE="${ROUTE_PRIOR_COLOR_STRENGTH_SCALE:-0.25}"
ROUTE_SUPPRESS_UPDATE_FLOOR="${ROUTE_SUPPRESS_UPDATE_FLOOR:-0.15}"
ROUTE_SUPPRESS_OPACITY_SCALE="${ROUTE_SUPPRESS_OPACITY_SCALE:-0.25}"
ROUTE_MIN_SCALE_GATE="${ROUTE_MIN_SCALE_GATE:-0.3}"
ROUTE_SCALE_GATE_POWER="${ROUTE_SCALE_GATE_POWER:-1.0}"
ROUTE_SCALE_GATE_SUPPRESS_COUPLING="${ROUTE_SCALE_GATE_SUPPRESS_COUPLING:-1.0}"
ROUTE_RISKY_USEFUL_OPACITY_COUPLING="${ROUTE_RISKY_USEFUL_OPACITY_COUPLING:-0.35}"
ROUTE_RISKY_USEFUL_SCALE_COUPLING="${ROUTE_RISKY_USEFUL_SCALE_COUPLING:-0.2}"
ROUTE_HARMFUL_SCALE_COUPLING="${ROUTE_HARMFUL_SCALE_COUPLING:-1.0}"
RUN_RENDER_SANITY="${RUN_RENDER_SANITY:-0}"
RENDER_SANITY_IMAGES_SUBDIR="${RENDER_SANITY_IMAGES_SUBDIR:-images_2}"
RENDER_SANITY_RESOLUTION="${RENDER_SANITY_RESOLUTION:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for path in "${DETAIL_MODEL_PATH}" "${CARRIER_PAYLOAD}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[prepare-surface-detail-init] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ -n "${ACTION_PAYLOAD}" && ! -e "${ACTION_PAYLOAD}" ]]; then
  echo "[prepare-surface-detail-init] action payload not found: ${ACTION_PAYLOAD}" >&2
  exit 1
fi
if [[ -n "${ROUTE_PAYLOAD}" && ! -e "${ROUTE_PAYLOAD}" ]]; then
  echo "[prepare-surface-detail-init] route payload not found: ${ROUTE_PAYLOAD}" >&2
  exit 1
fi

echo "[prepare-surface-detail-init] scene root         : ${SCENE_ROOT}"
echo "[prepare-surface-detail-init] detail model path   : ${DETAIL_MODEL_PATH}"
echo "[prepare-surface-detail-init] detail iteration    : ${DETAIL_ITERATION}"
echo "[prepare-surface-detail-init] carrier payload     : ${CARRIER_PAYLOAD}"
if [[ -n "${ACTION_PAYLOAD}" ]]; then
  echo "[prepare-surface-detail-init] action payload      : ${ACTION_PAYLOAD}"
fi
if [[ -n "${ROUTE_PAYLOAD}" ]]; then
  echo "[prepare-surface-detail-init] route payload       : ${ROUTE_PAYLOAD}"
  echo "[prepare-surface-detail-init] route exec mode     : ${ROUTE_EXECUTION_MODE}"
fi
echo "[prepare-surface-detail-init] detail attach scale : ${DETAIL_ATTACH_SCALE}"
echo "[prepare-surface-detail-init] detail radius gate  : ${DETAIL_RADIUS_GATE}"
echo "[prepare-surface-detail-init] output model path   : ${OUTPUT_MODEL_PATH}"
echo "[prepare-surface-detail-init] checkpoint tag      : ${CHECKPOINT_TAG}"

CMD=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/prepare_hrgs_surface_detail_init.py"
  --detail_model_path "${DETAIL_MODEL_PATH}"
  --carrier_payload "${CARRIER_PAYLOAD}"
  --output_model_path "${OUTPUT_MODEL_PATH}"
  --detail_iteration "${DETAIL_ITERATION}"
  --output_iteration "${OUTPUT_ITERATION}"
  --surface_min_confidence "${SURFACE_MIN_CONFIDENCE}"
  --surface_max_disagreement "${SURFACE_MAX_DISAGREEMENT}"
  --surface_min_views "${SURFACE_MIN_VIEWS}"
  --surface_max_count "${SURFACE_MAX_COUNT}"
  --surface_seed "${SURFACE_SEED}"
  --surface_scale_multiplier "${SURFACE_SCALE_MULTIPLIER}"
  --surface_thickness_multiplier "${SURFACE_THICKNESS_MULTIPLIER}"
  --surface_min_scale "${SURFACE_MIN_SCALE}"
  --surface_init_opacity "${SURFACE_INIT_OPACITY}"
  --detail_attach_scale "${DETAIL_ATTACH_SCALE}"
  --detail_update_scale "${DETAIL_UPDATE_SCALE}"
  --detail_detail_scale "${DETAIL_DETAIL_SCALE}"
  --detail_prior_color_scale "${DETAIL_PRIOR_COLOR_SCALE}"
  --detail_radius_images_subdir "${DETAIL_RADIUS_IMAGES_SUBDIR}"
  --detail_radius_ref_px "${DETAIL_RADIUS_REF_PX}"
  --detail_radius_min_gate "${DETAIL_RADIUS_MIN_GATE}"
  --detail_radius_max_views "${DETAIL_RADIUS_MAX_VIEWS}"
  --route_execution_mode "${ROUTE_EXECUTION_MODE}"
  --route_detail_protect_beta "${ROUTE_DETAIL_PROTECT_BETA}"
  --route_detail_attach_strength "${ROUTE_DETAIL_ATTACH_STRENGTH}"
  --route_detail_geometry_strength "${ROUTE_DETAIL_GEOMETRY_STRENGTH}"
  --route_prior_color_strength_scale "${ROUTE_PRIOR_COLOR_STRENGTH_SCALE}"
  --route_suppress_update_floor "${ROUTE_SUPPRESS_UPDATE_FLOOR}"
  --route_suppress_opacity_scale "${ROUTE_SUPPRESS_OPACITY_SCALE}"
  --route_min_scale_gate "${ROUTE_MIN_SCALE_GATE}"
  --route_scale_gate_power "${ROUTE_SCALE_GATE_POWER}"
  --route_scale_gate_suppress_coupling "${ROUTE_SCALE_GATE_SUPPRESS_COUPLING}"
  --route_risky_useful_opacity_coupling "${ROUTE_RISKY_USEFUL_OPACITY_COUPLING}"
  --route_risky_useful_scale_coupling "${ROUTE_RISKY_USEFUL_SCALE_COUPLING}"
  --route_harmful_scale_coupling "${ROUTE_HARMFUL_SCALE_COUPLING}"
  --python_bin "${PYTHON_BIN}"
)

if [[ -n "${ACTION_PAYLOAD}" ]]; then
  CMD+=(--action_payload "${ACTION_PAYLOAD}")
fi
if [[ -n "${ROUTE_PAYLOAD}" ]]; then
  CMD+=(--route_payload "${ROUTE_PAYLOAD}")
fi
if [[ "${DETAIL_RADIUS_GATE}" == "1" || "${RUN_RENDER_SANITY}" == "1" ]]; then
  CMD+=(--scene_root "${SCENE_ROOT}")
fi
if [[ "${DETAIL_RADIUS_GATE}" == "1" ]]; then
  CMD+=(--detail_radius_gate)
fi
if [[ "${RUN_RENDER_SANITY}" == "1" ]]; then
  CMD+=(
    --render_sanity
    --render_sanity_images_subdir "${RENDER_SANITY_IMAGES_SUBDIR}"
    --render_sanity_resolution "${RENDER_SANITY_RESOLUTION}"
  )
fi

"${CMD[@]}"

echo
echo "[done] merged init root     : ${OUTPUT_MODEL_PATH}"
echo "[done] merged start ply     : ${OUTPUT_MODEL_PATH}/point_cloud/iteration_${DETAIL_ITERATION}/point_cloud.ply"
echo "[done] merged action payload: ${OUTPUT_MODEL_PATH}/merged_action_payload.pt"
