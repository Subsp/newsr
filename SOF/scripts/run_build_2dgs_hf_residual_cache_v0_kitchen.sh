#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXTERNAL_REPO_ROOT="${EXTERNAL_REPO_ROOT:-${WORK_ROOT}/external/GaussianImage}"

NPSE_ROOT="${NPSE_ROOT:-${SCENE_ASSET_ROOT}/npse_cache/render_x1_restormer_depthprior_npse_yellow_fidelity_nogate_full_v0}"
TARGET_DIR="${TARGET_DIR:-${NPSE_ROOT}/edge_target}"
MASK_DIR="${MASK_DIR:-${NPSE_ROOT}/trust_edge}"
ANCHOR_DIR="${ANCHOR_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/mip30k_rerun_check_directsrc_r1_v0/train/ours_30000/test_preds_1}"

OUTPUT_NAME="${OUTPUT_NAME:-render_x1_restormer_gaussianimage_hf_residual_v0_smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCENE_ASSET_ROOT}/2dgs_hf_residual/${OUTPUT_NAME}}"

MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"
LIMIT="${LIMIT:-8}"
DEBUG_LIMIT="${DEBUG_LIMIT:-24}"
OVERWRITE="${OVERWRITE:-0}"

HIGHPASS_KERNEL="${HIGHPASS_KERNEL:-9}"
DETAIL_ALPHA="${DETAIL_ALPHA:-0.8}"
RESIDUAL_CLIP="${RESIDUAL_CLIP:-0.08}"
CONFIDENCE_POWER="${CONFIDENCE_POWER:-1.5}"
MASK_POWER="${MASK_POWER:-1.0}"

NUM_GAUSSIANS="${NUM_GAUSSIANS:-4096}"
GAUSSIANIMAGE_MODEL="${GAUSSIANIMAGE_MODEL:-cholesky}"
GAUSSIANIMAGE_OPTIMIZER="${GAUSSIANIMAGE_OPTIMIZER:-adam}"
ITERATIONS="${ITERATIONS:-700}"
LR="${LR:-0.001}"
LOSS="${LOSS:-l1_l2}"
LAMBDA_L1="${LAMBDA_L1:-0.5}"
LAMBDA_L2="${LAMBDA_L2:-0.5}"
BACKGROUND_WEIGHT="${BACKGROUND_WEIGHT:-0.005}"
NEUTRAL_OUTSIDE_MASK="${NEUTRAL_OUTSIDE_MASK:-0}"
SAVE_PT="${SAVE_PT:-0}"

for required in "${EXTERNAL_REPO_ROOT}" "${TARGET_DIR}" "${MASK_DIR}" "${ANCHOR_DIR}"; do
  if [[ ! -d "${required}" ]]; then
    echo "[2dgs-hf-v0] required path not found: ${required}" >&2
    if [[ "${required}" == "${EXTERNAL_REPO_ROOT}" ]]; then
      echo "[2dgs-hf-v0] clone official GaussianImage first:" >&2
      echo "  mkdir -p ${WORK_ROOT}/external && git clone https://github.com/Xinjie-Q/GaussianImage.git ${EXTERNAL_REPO_ROOT}" >&2
    fi
    exit 1
  fi
done

ARGS=(
  --external_repo_root "${EXTERNAL_REPO_ROOT}"
  --target_dir "${TARGET_DIR}"
  --anchor_dir "${ANCHOR_DIR}"
  --mask_dir "${MASK_DIR}"
  --output_dir "${OUTPUT_ROOT}"
  --match_policy "${MATCH_POLICY}"
  --highpass_kernel "${HIGHPASS_KERNEL}"
  --detail_alpha "${DETAIL_ALPHA}"
  --residual_clip "${RESIDUAL_CLIP}"
  --confidence_power "${CONFIDENCE_POWER}"
  --mask_power "${MASK_POWER}"
  --num_gaussians "${NUM_GAUSSIANS}"
  --model "${GAUSSIANIMAGE_MODEL}"
  --optimizer "${GAUSSIANIMAGE_OPTIMIZER}"
  --iterations "${ITERATIONS}"
  --lr "${LR}"
  --loss "${LOSS}"
  --lambda_l1 "${LAMBDA_L1}"
  --lambda_l2 "${LAMBDA_L2}"
  --background_weight "${BACKGROUND_WEIGHT}"
  --limit "${LIMIT}"
  --debug_limit "${DEBUG_LIMIT}"
)

if [[ "${NEUTRAL_OUTSIDE_MASK}" == "1" ]]; then
  ARGS+=(--neutral_outside_mask)
else
  ARGS+=(--no_neutral_outside_mask)
fi

if [[ "${SAVE_PT}" == "1" ]]; then
  ARGS+=(--save_pt)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "[gaussianimage-hf-v0] external: ${EXTERNAL_REPO_ROOT}"
echo "[gaussianimage-hf-v0] target  : ${TARGET_DIR}"
echo "[gaussianimage-hf-v0] anchor  : ${ANCHOR_DIR}"
echo "[gaussianimage-hf-v0] mask    : ${MASK_DIR}"
echo "[gaussianimage-hf-v0] output  : ${OUTPUT_ROOT}"

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_gaussianimage_hf_residual_cache_v0.py" "${ARGS[@]}"

echo "[gaussianimage-hf-v0] inspect dirs:"
echo "  ${OUTPUT_ROOT}/sheet"
echo "  ${OUTPUT_ROOT}/overlay"
echo "  ${OUTPUT_ROOT}/primitive_overlay"
echo "  ${OUTPUT_ROOT}/recon_hf"
echo "  ${OUTPUT_ROOT}/target_hf"
