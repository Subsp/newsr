#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SR_PRIOR_NAME="${SR_PRIOR_NAME:-qwen_steps1_seed42_rcgm_aligned_images2_train244_v0}"
SR_DIR="${SR_DIR:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${SR_PRIOR_NAME}/fused_priors}"
LR_DIR="${LR_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/mip30k_rerun_check_directsrc_r1_v0/train/ours_30000/test_preds_1}"
MASK_DIR="${MASK_DIR:-${SCENE_ASSET_ROOT}/npse_cache/render_x1_restormer_depthprior_npse_yellow_fidelity_nogate_full_v0/trust_edge}"

OUTPUT_NAME="${OUTPUT_NAME:-${SR_PRIOR_NAME}_vs_mip30k_lr_delta_composition_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/sr_lr_delta_composition/${OUTPUT_NAME}}"

MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"
LIMIT="${LIMIT:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-12}"
HIGHPASS_KERNEL="${HIGHPASS_KERNEL:-9}"
LOWPASS_KERNEL="${LOWPASS_KERNEL:-31}"
MASK_POWER="${MASK_POWER:-1.5}"
EDGE_PERCENTILE="${EDGE_PERCENTILE:-90}"
FLAT_PERCENTILE="${FLAT_PERCENTILE:-45}"
VIS_CLIP="${VIS_CLIP:-0.10}"
OVERWRITE="${OVERWRITE:-0}"

for required in "${SR_DIR}" "${LR_DIR}"; do
  if [[ ! -d "${required}" ]]; then
    echo "[sr-lr-delta-v0] required path not found: ${required}" >&2
    exit 1
  fi
done

ARGS=(
  --sr_dir "${SR_DIR}"
  --lr_dir "${LR_DIR}"
  --output_dir "${OUTPUT_ROOT}"
  --match_policy "${MATCH_POLICY}"
  --limit "${LIMIT}"
  --debug_limit "${DEBUG_LIMIT}"
  --highpass_kernel "${HIGHPASS_KERNEL}"
  --lowpass_kernel "${LOWPASS_KERNEL}"
  --mask_power "${MASK_POWER}"
  --edge_percentile "${EDGE_PERCENTILE}"
  --flat_percentile "${FLAT_PERCENTILE}"
  --vis_clip "${VIS_CLIP}"
)

if [[ -d "${MASK_DIR}" ]]; then
  ARGS+=(--mask_dir "${MASK_DIR}")
else
  echo "[sr-lr-delta-v0] mask missing, run without mask: ${MASK_DIR}" >&2
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "[sr-lr-delta-v0] sr     : ${SR_DIR}"
echo "[sr-lr-delta-v0] lr     : ${LR_DIR}"
echo "[sr-lr-delta-v0] mask   : ${MASK_DIR}"
echo "[sr-lr-delta-v0] output : ${OUTPUT_ROOT}"

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/evaluate_sr_lr_delta_composition_v0.py" "${ARGS[@]}"
