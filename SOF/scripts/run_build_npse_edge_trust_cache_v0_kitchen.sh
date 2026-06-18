#!/usr/bin/env bash
set -euo pipefail

# Build N-PSE v0 edge/trust/target cache for the render-x1 restoration prior branch.
# This step is intentionally offline: inspect the generated overlays before wiring
# the targets into training.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

ENHANCEMENT_BACKEND="${ENHANCEMENT_BACKEND:-restormer}"
REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-images_2}"
REFERENCE_DIR="${REFERENCE_DIR:-${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}}"

MIP_RENDER_EXPERIMENT_GROUP="${MIP_RENDER_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_RENDER_MODEL_NAME="${MIP_RENDER_MODEL_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
MIP_RENDER_ITERATION="${MIP_RENDER_ITERATION:-30000}"
MIP_RENDER_SPLIT="${MIP_RENDER_SPLIT:-train}"
ANCHOR_DIR="${ANCHOR_DIR:-${SCENE_ASSET_ROOT}/${MIP_RENDER_EXPERIMENT_GROUP}/${MIP_RENDER_MODEL_NAME}/${MIP_RENDER_SPLIT}/ours_${MIP_RENDER_ITERATION}/test_preds_1}"

PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-render_x1_${ENHANCEMENT_BACKEND}_aligned_${REFERENCE_IMAGES_SUBDIR}_scratch_v0}"
SR_DIR="${SR_DIR:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${PREPARED_SR_PRIOR_NAME}/fused_priors}"
DEPTH_PRIOR_DIR="${DEPTH_PRIOR_DIR:?Set DEPTH_PRIOR_DIR to a frame-aligned depth prior directory.}"

OUTPUT_NAME="${OUTPUT_NAME:-render_x1_${ENHANCEMENT_BACKEND}_depthprior_npse_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCENE_ASSET_ROOT}/npse_cache/${OUTPUT_NAME}}"

MATCH_POLICY="${MATCH_POLICY:-llff_train_order}"
PRIOR_LLFFHOLD="${PRIOR_LLFFHOLD:-8}"
ALLOW_EXTRA_INPUTS="${ALLOW_EXTRA_INPUTS:-1}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

HIGHPASS_KERNEL="${HIGHPASS_KERNEL:-15}"
PROP_RADIUS="${PROP_RADIUS:-2}"
PROP_SIGMA="${PROP_SIGMA:-1.25}"
DEPTH_EDGE_PERCENTILE="${DEPTH_EDGE_PERCENTILE:-90}"
SR_EDGE_PERCENTILE="${SR_EDGE_PERCENTILE:-92}"
GEOMETRY_EDGE_THRESHOLD="${GEOMETRY_EDGE_THRESHOLD:-0.55}"
SR_EDGE_THRESHOLD="${SR_EDGE_THRESHOLD:-0.55}"
GEOMETRY_CONFIRM_MODE="${GEOMETRY_CONFIRM_MODE:-sr_confirmed}"
DEPTH_SR_CONFIRM_THRESHOLD="${DEPTH_SR_CONFIRM_THRESHOLD:-0.35}"
DEPTH_ONLY_EDGE_WEIGHT="${DEPTH_ONLY_EDGE_WEIGHT:-0.15}"
DEPTH_ONLY_BARRIER_WEIGHT="${DEPTH_ONLY_BARRIER_WEIGHT:-0.20}"
GEOMETRY_BARRIER_WEIGHT="${GEOMETRY_BARRIER_WEIGHT:-1.0}"
EDGE_POSITION_MODE="${EDGE_POSITION_MODE:-appearance}"
SR_BARRIER_WEIGHT="${SR_BARRIER_WEIGHT:-0.25}"
EDGE_BAND_RADIUS="${EDGE_BAND_RADIUS:-1}"
TRUST_LOWFREQ_TAU="${TRUST_LOWFREQ_TAU:-0.12}"
TRUST_CONSISTENCY_TAU="${TRUST_CONSISTENCY_TAU:-0.08}"
UNCERTAIN_TRUST_THRESHOLD="${UNCERTAIN_TRUST_THRESHOLD:-0.35}"
EDGE_TARGET_MODE="${EDGE_TARGET_MODE:-fidelity}"
EDGE_TARGET_TRUST_POWER="${EDGE_TARGET_TRUST_POWER:-1.5}"
EDGE_TARGET_MIN_WEIGHT="${EDGE_TARGET_MIN_WEIGHT:-0.05}"
EDGE_TARGET_DIRECTION_GATE="${EDGE_TARGET_DIRECTION_GATE:-none}"
EDGE_TARGET_DIRECTION_MIN_COS="${EDGE_TARGET_DIRECTION_MIN_COS:-0.0}"
EDGE_TARGET_DIRECTION_FLOOR="${EDGE_TARGET_DIRECTION_FLOOR:-0.20}"
EDGE_TARGET_DIRECTION_POWER="${EDGE_TARGET_DIRECTION_POWER:-1.0}"
EDGE_TARGET_RESIDUAL_DIRECTION_WEIGHT="${EDGE_TARGET_RESIDUAL_DIRECTION_WEIGHT:-0.50}"
EDGE_TARGET_DIRECTION_BLUR="${EDGE_TARGET_DIRECTION_BLUR:-3}"
EDGE_RESIDUAL_CLIP="${EDGE_RESIDUAL_CLIP:-0.08}"
NPSE_ASSET_PROFILE="${NPSE_ASSET_PROFILE:-full}"

for path in "${ANCHOR_DIR}" "${SR_DIR}" "${DEPTH_PRIOR_DIR}" "${REFERENCE_DIR}"; do
  if [[ ! -d "${path}" ]]; then
    echo "[npse-cache-v0] required dir not found: ${path}" >&2
    exit 1
  fi
done

echo "[npse-cache-v0] anchor dir : ${ANCHOR_DIR}"
echo "[npse-cache-v0] sr dir     : ${SR_DIR}"
echo "[npse-cache-v0] depth dir  : ${DEPTH_PRIOR_DIR}"
echo "[npse-cache-v0] reference  : ${REFERENCE_DIR}"
echo "[npse-cache-v0] output     : ${OUTPUT_ROOT}"

CMD=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_npse_edge_trust_cache_v0.py"
  --anchor_dir "${ANCHOR_DIR}"
  --sr_dir "${SR_DIR}"
  --depth_dir "${DEPTH_PRIOR_DIR}"
  --reference_dir "${REFERENCE_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --match_policy "${MATCH_POLICY}"
  --llffhold "${PRIOR_LLFFHOLD}"
  --highpass_kernel "${HIGHPASS_KERNEL}"
  --prop_radius "${PROP_RADIUS}"
  --prop_sigma "${PROP_SIGMA}"
  --depth_edge_percentile "${DEPTH_EDGE_PERCENTILE}"
  --sr_edge_percentile "${SR_EDGE_PERCENTILE}"
  --geometry_edge_threshold "${GEOMETRY_EDGE_THRESHOLD}"
  --sr_edge_threshold "${SR_EDGE_THRESHOLD}"
  --geometry_confirm_mode "${GEOMETRY_CONFIRM_MODE}"
  --depth_sr_confirm_threshold "${DEPTH_SR_CONFIRM_THRESHOLD}"
  --depth_only_edge_weight "${DEPTH_ONLY_EDGE_WEIGHT}"
  --depth_only_barrier_weight "${DEPTH_ONLY_BARRIER_WEIGHT}"
  --geometry_barrier_weight "${GEOMETRY_BARRIER_WEIGHT}"
  --edge_position_mode "${EDGE_POSITION_MODE}"
  --sr_barrier_weight "${SR_BARRIER_WEIGHT}"
  --edge_band_radius "${EDGE_BAND_RADIUS}"
  --trust_lowfreq_tau "${TRUST_LOWFREQ_TAU}"
  --trust_consistency_tau "${TRUST_CONSISTENCY_TAU}"
  --uncertain_trust_threshold "${UNCERTAIN_TRUST_THRESHOLD}"
  --edge_target_mode "${EDGE_TARGET_MODE}"
  --edge_target_trust_power "${EDGE_TARGET_TRUST_POWER}"
  --edge_target_min_weight "${EDGE_TARGET_MIN_WEIGHT}"
  --edge_target_direction_gate "${EDGE_TARGET_DIRECTION_GATE}"
  --edge_target_direction_min_cos "${EDGE_TARGET_DIRECTION_MIN_COS}"
  --edge_target_direction_floor "${EDGE_TARGET_DIRECTION_FLOOR}"
  --edge_target_direction_power "${EDGE_TARGET_DIRECTION_POWER}"
  --edge_target_residual_direction_weight "${EDGE_TARGET_RESIDUAL_DIRECTION_WEIGHT}"
  --edge_target_direction_blur "${EDGE_TARGET_DIRECTION_BLUR}"
  --edge_residual_clip "${EDGE_RESIDUAL_CLIP}"
  --asset_profile "${NPSE_ASSET_PROFILE}"
  --limit "${LIMIT}"
)

if [[ "${ALLOW_EXTRA_INPUTS}" == "1" ]]; then
  CMD+=(--allow_extra_inputs)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

"${CMD[@]}"

echo "[npse-cache-v0] done: ${OUTPUT_ROOT}"
