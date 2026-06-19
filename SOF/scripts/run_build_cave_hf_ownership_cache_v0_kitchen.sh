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
PYTHON_BIN="${PYTHON_BIN:-python}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
INPUT_EXPERIMENT_NAME="${INPUT_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
INPUT_MODEL_DIR="${INPUT_MODEL_DIR:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${INPUT_EXPERIMENT_NAME}}"
CAVE_MODEL_DIR="${CAVE_MODEL_DIR:-${INPUT_MODEL_DIR}}"
CAVE_ITERATION="${CAVE_ITERATION:-30000}"
CAVE_POINT_CLOUD_PLY="${CAVE_POINT_CLOUD_PLY:-}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SR_DIR="${SR_DIR:-${SCENE_ASSET_ROOT}/prepared_sr_priors/render_x1_restormer_aligned_images_2_scratch_v0/fused_priors}"
ANCHOR_DIR="${ANCHOR_DIR:-${INPUT_MODEL_DIR}/train/ours_30000/test_preds_1}"
NPSE_CACHE="${NPSE_CACHE:-${SCENE_ASSET_ROOT}/npse_cache/render_x1_restormer_depthprior_npse_yellow_fidelity_nogate_full_v0}"
EDGE_MASK_DIR="${EDGE_MASK_DIR:-${NPSE_CACHE}/trust_edge}"

OUTPUT_NAME="${OUTPUT_NAME:-render_x1_restormer_cave_hf_ownership_dense_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCENE_ASSET_ROOT}/cave_hf_ownership/${OUTPUT_NAME}}"

MATCH_POLICY="${MATCH_POLICY:-llff_train_order}"
LLFFHOLD="${LLFFHOLD:-8}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"

HIGHPASS_KERNEL="${HIGHPASS_KERNEL:-9}"
HF_PERCENTILE="${HF_PERCENTILE:-99.0}"
MIN_OPACITY="${MIN_OPACITY:-0.02}"
MIN_RADIUS_PX="${MIN_RADIUS_PX:-0.5}"
MAX_RADIUS_PX="${MAX_RADIUS_PX:-12.0}"
CANDIDATE_RADIUS_SCALE="${CANDIDATE_RADIUS_SCALE:-1.0}"
MAX_GAUSSIANS_PER_VIEW="${MAX_GAUSSIANS_PER_VIEW:-32768}"
MIN_CENTER_HF="${MIN_CENTER_HF:-0.05}"
EDGE_MASK_THRESHOLD="${EDGE_MASK_THRESHOLD:-0.05}"
PROFILE_GRID_RADIUS="${PROFILE_GRID_RADIUS:-2.0}"
PROFILE_GRID_STEPS="${PROFILE_GRID_STEPS:-5}"
PROFILE_MIN_VALID="${PROFILE_MIN_VALID:-0.65}"
NEIGHBOR_RADIUS="${NEIGHBOR_RADIUS:-1}"
CONSISTENCY_FLOOR="${CONSISTENCY_FLOOR:-0.0}"
SCORE_POWER="${SCORE_POWER:-1.0}"
DRAW_MAX_GAUSSIANS="${DRAW_MAX_GAUSSIANS:-32768}"
CARRIER_TOUCH_MAX_GAUSSIANS="${CARRIER_TOUCH_MAX_GAUSSIANS:-65536}"
CARRIER_SCORE_MODE="${CARRIER_SCORE_MODE:-hit}"
OWNERSHIP_SUPPORT_RADIUS="${OWNERSHIP_SUPPORT_RADIUS:-1.0}"
OWNERSHIP_EDGE_THRESHOLD="${OWNERSHIP_EDGE_THRESHOLD:-0.05}"
OWNERSHIP_EDGE_POWER="${OWNERSHIP_EDGE_POWER:-0.0}"

for path in \
  "${SCENE_ROOT}" \
  "${SCENE_ROOT}/sparse/0" \
  "${SCENE_ROOT}/${IMAGES_SUBDIR}" \
  "${CAVE_MODEL_DIR}" \
  "${SR_DIR}" \
  "${ANCHOR_DIR}" \
  "${MIPSPLATTING_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[cave-hf-v1] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ -n "${EDGE_MASK_DIR}" && ! -d "${EDGE_MASK_DIR}" ]]; then
  echo "[cave-hf-v1] edge mask dir not found, continuing without mask: ${EDGE_MASK_DIR}" >&2
  EDGE_MASK_DIR=""
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[cave-hf-v1] scene      : ${SCENE_ROOT}"
echo "[cave-hf-v1] images     : ${IMAGES_SUBDIR}"
echo "[cave-hf-v1] model      : ${CAVE_MODEL_DIR}"
echo "[cave-hf-v1] iteration  : ${CAVE_ITERATION}"
echo "[cave-hf-v1] sr         : ${SR_DIR}"
echo "[cave-hf-v1] anchor     : ${ANCHOR_DIR}"
echo "[cave-hf-v1] edge mask  : ${EDGE_MASK_DIR:-none}"
echo "[cave-hf-v1] output     : ${OUTPUT_ROOT}"
echo "[cave-hf-v1] limit      : ${LIMIT}"
echo "[cave-hf-v1] candidates : max=${MAX_GAUSSIANS_PER_VIEW} min_hf=${MIN_CENTER_HF} opacity=${MIN_OPACITY}"
echo "[cave-hf-v1] carrier    : touch_max=${CARRIER_TOUCH_MAX_GAUSSIANS} score_max=${DRAW_MAX_GAUSSIANS} mode=${CARRIER_SCORE_MODE}"
echo "[cave-hf-v1] renderer   : original GS render radii/visibility + SOFPriorBlock touch"
echo "[cave-hf-v1] profile    : grid=${PROFILE_GRID_STEPS} radius=${PROFILE_GRID_RADIUS} neighbors=${NEIGHBOR_RADIUS}"

ARGS=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_cave_hf_ownership_cache_v0.py"
  --scene_root "${SCENE_ROOT}"
  --images_subdir "${IMAGES_SUBDIR}"
  --model_dir "${CAVE_MODEL_DIR}"
  --iteration "${CAVE_ITERATION}"
  --sr_dir "${SR_DIR}"
  --anchor_dir "${ANCHOR_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --mipsplatting_root "${MIPSPLATTING_ROOT}"
  --match_policy "${MATCH_POLICY}"
  --llffhold "${LLFFHOLD}"
  --limit "${LIMIT}"
  --highpass_kernel "${HIGHPASS_KERNEL}"
  --hf_percentile "${HF_PERCENTILE}"
  --min_opacity "${MIN_OPACITY}"
  --min_radius_px "${MIN_RADIUS_PX}"
  --max_radius_px "${MAX_RADIUS_PX}"
  --candidate_radius_scale "${CANDIDATE_RADIUS_SCALE}"
  --max_gaussians_per_view "${MAX_GAUSSIANS_PER_VIEW}"
  --min_center_hf "${MIN_CENTER_HF}"
  --edge_mask_threshold "${EDGE_MASK_THRESHOLD}"
  --profile_grid_radius "${PROFILE_GRID_RADIUS}"
  --profile_grid_steps "${PROFILE_GRID_STEPS}"
  --profile_min_valid "${PROFILE_MIN_VALID}"
  --neighbor_radius "${NEIGHBOR_RADIUS}"
  --consistency_floor "${CONSISTENCY_FLOOR}"
  --score_power "${SCORE_POWER}"
  --draw_max_gaussians "${DRAW_MAX_GAUSSIANS}"
  --carrier_touch_max_gaussians "${CARRIER_TOUCH_MAX_GAUSSIANS}"
  --carrier_score_mode "${CARRIER_SCORE_MODE}"
  --ownership_support_radius "${OWNERSHIP_SUPPORT_RADIUS}"
  --ownership_edge_threshold "${OWNERSHIP_EDGE_THRESHOLD}"
  --ownership_edge_power "${OWNERSHIP_EDGE_POWER}"
)

if [[ -n "${CAVE_POINT_CLOUD_PLY}" ]]; then
  ARGS+=(--point_cloud_ply "${CAVE_POINT_CLOUD_PLY}")
fi
if [[ -n "${EDGE_MASK_DIR}" ]]; then
  ARGS+=(--edge_mask_dir "${EDGE_MASK_DIR}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"${ARGS[@]}"

echo "[cave-hf-v1] done: ${OUTPUT_ROOT}"
