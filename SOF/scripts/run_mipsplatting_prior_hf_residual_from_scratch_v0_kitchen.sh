#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

BASE_PREPARED_SR_PRIOR_NAME="${BASE_PREPARED_SR_PRIOR_NAME:-qwen_steps1_seed42_rcgm_aligned_images2_train244_v0}"
BASE_PREPARED_SR_PRIOR_ROOT="${BASE_PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${BASE_PREPARED_SR_PRIOR_NAME}}"
HF_RESIDUAL_PRIOR_NAME="${HF_RESIDUAL_PRIOR_NAME:-${BASE_PREPARED_SR_PRIOR_NAME}_hfresidual_r3_g1_v0}"
HF_RESIDUAL_PRIOR_ROOT="${HF_RESIDUAL_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${HF_RESIDUAL_PRIOR_NAME}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HIGH_PASS_RADIUS="${HIGH_PASS_RADIUS:-3.0}"
HF_RESIDUAL_GAIN="${HF_RESIDUAL_GAIN:-1.0}"
HF_RESIDUAL_DELTA_CLIP="${HF_RESIDUAL_DELTA_CLIP:-0.18}"
HF_RESIDUAL_MASK_POWER="${HF_RESIDUAL_MASK_POWER:-0.75}"
HF_RESIDUAL_MASK_FLOOR="${HF_RESIDUAL_MASK_FLOOR:-0.0}"
FORCE_BUILD_HF_RESIDUAL="${FORCE_BUILD_HF_RESIDUAL:-0}"

RUN_TAG="${RUN_TAG:-mip30k_r1_qwen_priorhfres_scratch_v0}"

if [[ ! -d "${BASE_PREPARED_SR_PRIOR_ROOT}/fused_priors" ]]; then
  echo "[mip-priorhfres-scratch-v0] missing base priors: ${BASE_PREPARED_SR_PRIOR_ROOT}/fused_priors" >&2
  exit 1
fi

if [[ "${FORCE_BUILD_HF_RESIDUAL}" == "1" || ! -f "${HF_RESIDUAL_PRIOR_ROOT}/manifest.json" ]]; then
  echo "[mip-priorhfres-scratch-v0] build HF residual prior cache"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" scripts/build_prior_hf_residual_cache_v0.py \
      --source_root "${BASE_PREPARED_SR_PRIOR_ROOT}" \
      --output_root "${HF_RESIDUAL_PRIOR_ROOT}" \
      --highpass_radius "${HIGH_PASS_RADIUS}" \
      --gain "${HF_RESIDUAL_GAIN}" \
      --delta_clip "${HF_RESIDUAL_DELTA_CLIP}" \
      --mask_power "${HF_RESIDUAL_MASK_POWER}" \
      --mask_floor "${HF_RESIDUAL_MASK_FLOOR}"
  )
else
  echo "[mip-priorhfres-scratch-v0] reuse HF residual prior cache: ${HF_RESIDUAL_PRIOR_ROOT}"
fi

PREPARED_SR_PRIOR_ROOT="${HF_RESIDUAL_PRIOR_ROOT}" \
PRIOR_IMAGE_SUBDIR="fused_priors" \
RUN_TAG="${RUN_TAG}" \
bash "${SCRIPT_DIR}/run_mipsplatting_prior_only_from_scratch_v0_kitchen.sh"
