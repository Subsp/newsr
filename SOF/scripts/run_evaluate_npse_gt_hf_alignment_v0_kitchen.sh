#!/usr/bin/env bash
set -euo pipefail

# Evaluate whether NPSE edge targets inject high frequency aligned with GT.
# This is an offline diagnostic and does not train or mutate any model.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

ENHANCEMENT_BACKEND="${ENHANCEMENT_BACKEND:-restormer}"
REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-images_2}"
GT_DIR="${GT_DIR:-${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}}"

NPSE_CACHE_NAME="${NPSE_CACHE_NAME:-render_x1_${ENHANCEMENT_BACKEND}_depthprior_npse_yellow_fidelity_smoke}"
NPSE_CACHE_ROOT="${NPSE_CACHE_ROOT:-${SCENE_ASSET_ROOT}/npse_cache/${NPSE_CACHE_NAME}}"
EDGE_TARGET_DIR="${EDGE_TARGET_DIR:-${NPSE_CACHE_ROOT}/edge_target}"
TRUST_EDGE_DIR="${TRUST_EDGE_DIR:-${NPSE_CACHE_ROOT}/trust_edge}"

PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-render_x1_${ENHANCEMENT_BACKEND}_aligned_${REFERENCE_IMAGES_SUBDIR}_scratch_v0}"
SR_DIR="${SR_DIR:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${PREPARED_SR_PRIOR_NAME}/fused_priors}"

MIP_RENDER_EXPERIMENT_GROUP="${MIP_RENDER_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_RENDER_MODEL_NAME="${MIP_RENDER_MODEL_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
MIP_RENDER_SPLIT="${MIP_RENDER_SPLIT:-train}"
MIP_RENDER_ITERATION="${MIP_RENDER_ITERATION:-30000}"
ANCHOR_DIR="${ANCHOR_DIR:-${SCENE_ASSET_ROOT}/${MIP_RENDER_EXPERIMENT_GROUP}/${MIP_RENDER_MODEL_NAME}/${MIP_RENDER_SPLIT}/ours_${MIP_RENDER_ITERATION}/test_preds_1}"

OUTPUT_DIR="${OUTPUT_DIR:-${NPSE_CACHE_ROOT}/gt_hf_alignment_v0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
HIGHPASS_KERNEL="${HIGHPASS_KERNEL:-15}"
MASK_POWER="${MASK_POWER:-1.0}"
HARD_MASK_THRESHOLD="${HARD_MASK_THRESHOLD:-0.05}"
ACTIVE_GT_PERCENTILE="${ACTIVE_GT_PERCENTILE:-60}"
LIMIT="${LIMIT:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-12}"
OVERWRITE="${OVERWRITE:-1}"
EVAL_SR_PRIOR="${EVAL_SR_PRIOR:-1}"
EVAL_ANCHOR_ABS="${EVAL_ANCHOR_ABS:-1}"

for path in "${GT_DIR}" "${NPSE_CACHE_ROOT}" "${EDGE_TARGET_DIR}" "${TRUST_EDGE_DIR}"; do
  if [[ ! -d "${path}" ]]; then
    echo "[npse-hf-align-v0] required dir not found: ${path}" >&2
    exit 1
  fi
done

CMD=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/evaluate_npse_gt_hf_alignment_v0.py"
  --gt_dir "${GT_DIR}"
  --candidate "edge_target=${EDGE_TARGET_DIR}"
  --mask_dir "${TRUST_EDGE_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --highpass_kernel "${HIGHPASS_KERNEL}"
  --mask_power "${MASK_POWER}"
  --hard_mask_threshold "${HARD_MASK_THRESHOLD}"
  --active_gt_percentile "${ACTIVE_GT_PERCENTILE}"
  --limit "${LIMIT}"
  --debug_limit "${DEBUG_LIMIT}"
)

if [[ -d "${ANCHOR_DIR}" ]]; then
  CMD+=(--anchor_dir "${ANCHOR_DIR}")
  if [[ "${EVAL_ANCHOR_ABS}" == "1" ]]; then
    CMD+=(--candidate "anchor=${ANCHOR_DIR}")
  fi
else
  echo "[npse-hf-align-v0] anchor dir missing; delta_hf metrics will be unavailable: ${ANCHOR_DIR}" >&2
fi

if [[ "${EVAL_SR_PRIOR}" == "1" ]]; then
  if [[ -d "${SR_DIR}" ]]; then
    CMD+=(--candidate "sr_prior=${SR_DIR}")
  else
    echo "[npse-hf-align-v0] SR prior dir missing; skip sr_prior candidate: ${SR_DIR}" >&2
  fi
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "[npse-hf-align-v0] gt          : ${GT_DIR}"
echo "[npse-hf-align-v0] npse cache  : ${NPSE_CACHE_ROOT}"
echo "[npse-hf-align-v0] edge target : ${EDGE_TARGET_DIR}"
echo "[npse-hf-align-v0] trust edge  : ${TRUST_EDGE_DIR}"
echo "[npse-hf-align-v0] anchor      : ${ANCHOR_DIR}"
echo "[npse-hf-align-v0] output      : ${OUTPUT_DIR}"

"${CMD[@]}"
