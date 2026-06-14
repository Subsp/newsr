#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MIPSPLATTING_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${DEFAULT_MIPSPLATTING_ROOT}}"

TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k_rerun_v0}"
BASELINE_MODEL_DIR="${BASELINE_MODEL_DIR:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
BASELINE_ITERATION="${BASELINE_ITERATION:-30000}"

SURFACE_STATE_PROFILE="${SURFACE_STATE_PROFILE:-conservative_v0}"
STATE_RUN_NAME_DEFAULT="mip30k_rerun_gs2mesh_surface_state_v0"
if [[ "${SURFACE_STATE_PROFILE}" != "conservative_v0" ]]; then
  STATE_RUN_NAME_DEFAULT="${STATE_RUN_NAME_DEFAULT}_${SURFACE_STATE_PROFILE}"
fi
STATE_RUN_NAME="${STATE_RUN_NAME:-${STATE_RUN_NAME_DEFAULT}}"
STATE_DIR="${STATE_DIR:-${SOF_ROOT}/output/gaussian_surface_state_v0/${SCENE_NAME}/${STATE_RUN_NAME}}"
SURFACE_STATE_PAYLOAD="${SURFACE_STATE_PAYLOAD:-${STATE_DIR}/gaussian_surface_state_v0.pt}"
SURFACE_MASK_KEY="${SURFACE_MASK_KEY:-surface_candidate}"

PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft}"
PRIOR_EDGE_SUBDIR="${PRIOR_EDGE_SUBDIR:-fused_priors}"
PRIOR_MASK_SUBDIR="${PRIOR_MASK_SUBDIR:-usable_masks}"
PRIOR_EDGE_DIR="${PREPARED_SR_PRIOR_ROOT}/${PRIOR_EDGE_SUBDIR}"
PRIOR_MASK_DIR="${PREPARED_SR_PRIOR_ROOT}/${PRIOR_MASK_SUBDIR}"

SIGNAL_MODE="${SIGNAL_MODE:-direct_sr_highfreq}"
if [[ -z "${ANCHOR_MODE+x}" ]]; then
  if [[ "${SIGNAL_MODE}" == "anchor_residual" ]]; then
    ANCHOR_MODE="surface"
  else
    ANCHOR_MODE="full"
  fi
fi

CONSENSUS_TAG_DEFAULT="${MIP_EXPERIMENT_NAME}_${SURFACE_MASK_KEY}_route_consensus_v0"
if [[ "${SIGNAL_MODE}" != "direct_sr_highfreq" ]]; then
  CONSENSUS_TAG_DEFAULT="${CONSENSUS_TAG_DEFAULT}_${SIGNAL_MODE}"
fi
CONSENSUS_TAG="${CONSENSUS_TAG:-${CONSENSUS_TAG_DEFAULT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/surface_route_consensus_v0/${SCENE_NAME}/${CONSENSUS_TAG}}"

MAX_VIEWS="${MAX_VIEWS:-0}"
TOP_K="${TOP_K:-2}"
CELL_GRID="${CELL_GRID:-4}"
TILE_SIZE="${TILE_SIZE:-32}"
MAX_CANDIDATE_PIXELS="${MAX_CANDIDATE_PIXELS:-6000}"
LOCAL_CONSENSUS_VIEWS="${LOCAL_CONSENSUS_VIEWS:-3}"
ROUTE_RADIUS_SCALE="${ROUTE_RADIUS_SCALE:-2.5}"
ROUTE_MIN_RADIUS_PX="${ROUTE_MIN_RADIUS_PX:-1.5}"
MIN_PRIOR_MASK="${MIN_PRIOR_MASK:-0.05}"
MIN_ROUTE_QUALITY="${MIN_ROUTE_QUALITY:-0.08}"
MIN_RESIDUAL_ENERGY="${MIN_RESIDUAL_ENERGY:-0.01}"
ROUTE_MIN_VIEWS="${ROUTE_MIN_VIEWS:-2}"
ROUTE_VAR_TAU="${ROUTE_VAR_TAU:-0.01}"
PRIOR_DELTA_CLIP="${PRIOR_DELTA_CLIP:-0.20}"
LOWFREQ_GATE_KERNEL="${LOWFREQ_GATE_KERNEL:-9}"
LOWFREQ_GATE_TAU="${LOWFREQ_GATE_TAU:-0.08}"
SPARSE_PAYLOAD="${SPARSE_PAYLOAD:-1}"
SAVE_DEBUG_PNG="${SAVE_DEBUG_PNG:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${SURFACE_STATE_PAYLOAD}" ]]; then
  echo "[surface-route-consensus-v0] missing surface-state payload: ${SURFACE_STATE_PAYLOAD}" >&2
  exit 1
fi
if [[ ! -d "${PRIOR_EDGE_DIR}" ]]; then
  echo "[surface-route-consensus-v0] missing prior dir: ${PRIOR_EDGE_DIR}" >&2
  exit 1
fi
if [[ ! -d "${PRIOR_MASK_DIR}" ]]; then
  echo "[surface-route-consensus-v0] missing prior mask dir: ${PRIOR_MASK_DIR}" >&2
  exit 1
fi

echo "[surface-route-consensus-v0] scene            : ${SCENE_ROOT}"
echo "[surface-route-consensus-v0] model            : ${BASELINE_MODEL_DIR} iter=${BASELINE_ITERATION}"
echo "[surface-route-consensus-v0] prepared prior   : ${PREPARED_SR_PRIOR_ROOT}"
echo "[surface-route-consensus-v0] surface payload  : ${SURFACE_STATE_PAYLOAD}"
echo "[surface-route-consensus-v0] surface mask key : ${SURFACE_MASK_KEY}"
echo "[surface-route-consensus-v0] signal mode      : ${SIGNAL_MODE}"
echo "[surface-route-consensus-v0] lowfreq anchor   : ${ANCHOR_MODE}"
echo "[surface-route-consensus-v0] lowfreq gate     : kernel=${LOWFREQ_GATE_KERNEL} tau=${LOWFREQ_GATE_TAU}"
echo "[surface-route-consensus-v0] output root      : ${OUTPUT_ROOT}"

ARGS=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_surface_route_consensus_v0.py"
  --mipsplatting_root "${MIPSPLATTING_ROOT}"
  --scene_root "${SCENE_ROOT}"
  --model_path "${BASELINE_MODEL_DIR}"
  --surface_state_payload "${SURFACE_STATE_PAYLOAD}"
  --surface_mask_key "${SURFACE_MASK_KEY}"
  --prior_dir "${PRIOR_EDGE_DIR}"
  --prior_mask_dir "${PRIOR_MASK_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --images_subdir "${TARGET_IMAGES_SUBDIR}"
  --iteration "${BASELINE_ITERATION}"
  --max_views "${MAX_VIEWS}"
  --top_k "${TOP_K}"
  --cell_grid "${CELL_GRID}"
  --tile_size "${TILE_SIZE}"
  --max_candidate_pixels "${MAX_CANDIDATE_PIXELS}"
  --local_consensus_views "${LOCAL_CONSENSUS_VIEWS}"
  --route_radius_scale "${ROUTE_RADIUS_SCALE}"
  --route_min_radius_px "${ROUTE_MIN_RADIUS_PX}"
  --min_prior_mask "${MIN_PRIOR_MASK}"
  --min_route_quality "${MIN_ROUTE_QUALITY}"
  --min_residual_energy "${MIN_RESIDUAL_ENERGY}"
  --route_min_views "${ROUTE_MIN_VIEWS}"
  --route_var_tau "${ROUTE_VAR_TAU}"
  --prior_delta_clip "${PRIOR_DELTA_CLIP}"
  --anchor_mode "${ANCHOR_MODE}"
  --signal_mode "${SIGNAL_MODE}"
  --lowfreq_gate_kernel "${LOWFREQ_GATE_KERNEL}"
  --lowfreq_gate_tau "${LOWFREQ_GATE_TAU}"
)
if [[ "${SPARSE_PAYLOAD}" == "1" ]]; then
  ARGS+=(--sparse_payload)
fi
if [[ "${SAVE_DEBUG_PNG}" == "1" ]]; then
  ARGS+=(--save_debug_png)
fi

(
  cd "${SOF_ROOT}"
  "${ARGS[@]}"
)
