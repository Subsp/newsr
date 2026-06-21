#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

NPSE_ROOT="${NPSE_ROOT:-${SCENE_ASSET_ROOT}/npse_cache/render_x1_restormer_depthprior_npse_yellow_fidelity_nogate_full_v0}"
SR_DIR="${SR_DIR:-${SCENE_ASSET_ROOT}/prepared_sr_priors/render_x1_restormer_aligned_images_2_scratch_v0/fused_priors}"
LR_RENDER_DIR="${LR_RENDER_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/mip30k_rerun_check_directsrc_r1_v0/train/ours_30000/test_preds_1}"
MASK_DIR="${MASK_DIR:-${NPSE_ROOT}/trust_edge}"

OUTPUT_NAME="${OUTPUT_NAME:-render_x1_restormer_sr_lr_rgb_delta_carrier_v0_smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/2dgs_sr_lr_delta_carrier/${OUTPUT_NAME}}"

TARGET_DIR="${SR_DIR}" \
ANCHOR_DIR="${LR_RENDER_DIR}" \
MASK_DIR="${MASK_DIR}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
FIT_TARGET_MODE="${FIT_TARGET_MODE:-rgb_delta}" \
RGB_LOSS_WEIGHT_MODE="${RGB_LOSS_WEIGHT_MODE:-full}" \
DETAIL_ALPHA="${DETAIL_ALPHA:-1.0}" \
RESIDUAL_CLIP="${RESIDUAL_CLIP:-0.35}" \
BACKGROUND_WEIGHT="${BACKGROUND_WEIGHT:-0.0015}" \
"${SCRIPT_DIR}/run_build_2dgs_hf_residual_cache_v0_kitchen.sh"
