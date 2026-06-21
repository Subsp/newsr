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
NPSE_ROOT="${NPSE_ROOT:-${SCENE_ASSET_ROOT}/npse_cache/render_x1_restormer_depthprior_npse_yellow_fidelity_nogate_full_v0}"
MASK_DIR="${MASK_DIR:-${NPSE_ROOT}/trust_edge}"

OUTPUT_NAME="${OUTPUT_NAME:-${SR_PRIOR_NAME}_sr_hf_evidence_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCENE_ASSET_ROOT}/sr_hf_evidence/${OUTPUT_NAME}}"

MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"
LIMIT="${LIMIT:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-12}"
OVERWRITE="${OVERWRITE:-0}"

HIGHPASS_KERNEL="${HIGHPASS_KERNEL:-9}"
TENSOR_RADIUS="${TENSOR_RADIUS:-4}"
MASK_POWER="${MASK_POWER:-1.5}"
HF_PERCENTILE="${HF_PERCENTILE:-82}"
EDGE_PERCENTILE="${EDGE_PERCENTILE:-88}"
FLAT_PERCENTILE="${FLAT_PERCENTILE:-45}"
GEOMETRY_SCORE_THRESHOLD="${GEOMETRY_SCORE_THRESHOLD:-0.18}"
TEXTURE_SCORE_THRESHOLD="${TEXTURE_SCORE_THRESHOLD:-0.15}"
NOISE_SCORE_THRESHOLD="${NOISE_SCORE_THRESHOLD:-0.12}"
TEXTURE_COHERENCE_MIN="${TEXTURE_COHERENCE_MIN:-0.35}"
NOISE_COHERENCE_MAX="${NOISE_COHERENCE_MAX:-0.28}"
GEOMETRY_TEXTURE_SUPPRESSION="${GEOMETRY_TEXTURE_SUPPRESSION:-0.50}"
CARRIER_TEXTURE_WEIGHT="${CARRIER_TEXTURE_WEIGHT:-0.35}"
CARRIER_NOISE_WEIGHT="${CARRIER_NOISE_WEIGHT:-0.0}"
MAX_PRIMITIVES_PER_FRAME="${MAX_PRIMITIVES_PER_FRAME:-32768}"
PRIMITIVE_NMS_RADIUS_PX="${PRIMITIVE_NMS_RADIUS_PX:-2}"
SIGMA_LONG_PX="${SIGMA_LONG_PX:-2.4}"
SIGMA_SHORT_PX="${SIGMA_SHORT_PX:-0.35}"
VIS_CLIP="${VIS_CLIP:-0.10}"

for required in "${SR_DIR}" "${LR_DIR}" "${MASK_DIR}"; do
  if [[ ! -d "${required}" ]]; then
    echo "[sr-hf-evidence-v0] required path not found: ${required}" >&2
    exit 1
  fi
done

ARGS=(
  --sr_dir "${SR_DIR}"
  --lr_dir "${LR_DIR}"
  --mask_dir "${MASK_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --match_policy "${MATCH_POLICY}"
  --limit "${LIMIT}"
  --debug_limit "${DEBUG_LIMIT}"
  --highpass_kernel "${HIGHPASS_KERNEL}"
  --tensor_radius "${TENSOR_RADIUS}"
  --mask_power "${MASK_POWER}"
  --hf_percentile "${HF_PERCENTILE}"
  --edge_percentile "${EDGE_PERCENTILE}"
  --flat_percentile "${FLAT_PERCENTILE}"
  --geometry_score_threshold "${GEOMETRY_SCORE_THRESHOLD}"
  --texture_score_threshold "${TEXTURE_SCORE_THRESHOLD}"
  --noise_score_threshold "${NOISE_SCORE_THRESHOLD}"
  --texture_coherence_min "${TEXTURE_COHERENCE_MIN}"
  --noise_coherence_max "${NOISE_COHERENCE_MAX}"
  --geometry_texture_suppression "${GEOMETRY_TEXTURE_SUPPRESSION}"
  --carrier_texture_weight "${CARRIER_TEXTURE_WEIGHT}"
  --carrier_noise_weight "${CARRIER_NOISE_WEIGHT}"
  --max_primitives_per_frame "${MAX_PRIMITIVES_PER_FRAME}"
  --primitive_nms_radius_px "${PRIMITIVE_NMS_RADIUS_PX}"
  --sigma_long_px "${SIGMA_LONG_PX}"
  --sigma_short_px "${SIGMA_SHORT_PX}"
  --vis_clip "${VIS_CLIP}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "[sr-hf-evidence-v0] sr     : ${SR_DIR}"
echo "[sr-hf-evidence-v0] lr     : ${LR_DIR}"
echo "[sr-hf-evidence-v0] mask   : ${MASK_DIR}"
echo "[sr-hf-evidence-v0] output : ${OUTPUT_ROOT}"

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_sr_hf_evidence_cache_v0.py" "${ARGS[@]}"

echo "[sr-hf-evidence-v0] inspect dirs:"
echo "  ${OUTPUT_ROOT}/sheet"
echo "  ${OUTPUT_ROOT}/evidence_type"
echo "  ${OUTPUT_ROOT}/geometry_carrier_rgb"
echo "  ${OUTPUT_ROOT}/texture_carrier_rgb"
echo "  ${OUTPUT_ROOT}/structure_carrier_rgb"
echo "  ${OUTPUT_ROOT}/geometry_weight"
echo "  ${OUTPUT_ROOT}/texture_weight"
echo "  ${OUTPUT_ROOT}/noise_weight"
echo "  ${OUTPUT_ROOT}/primitive_overlay"
echo "  ${OUTPUT_ROOT}/primitives"
