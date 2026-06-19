#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_EXPERIMENT_NAME="${INPUT_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
RUN_TAG="${RUN_TAG:-${INPUT_EXPERIMENT_NAME}_cave_hf_transfer_v0}"
MODEL_DIR="${MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_nosr_layerfreq_cleanup_v0/${SCENE_NAME}/${RUN_TAG}}"

SUBSET_NAME="${SUBSET_NAME:-cave_hf_subset_render_v0}"
SUBSET_ROOT="${SUBSET_ROOT:-${MODEL_DIR}/${SUBSET_NAME}}"
SPLIT="${SPLIT:-train}"

TARGET_DIR="${TARGET_DIR:-${SCENE_ASSET_ROOT}/prepared_sr_priors/render_x1_restormer_aligned_images_2_scratch_v0/fused_priors}"
ANCHOR_DIR="${ANCHOR_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/${INPUT_EXPERIMENT_NAME}/train/ours_30000/test_preds_1}"
TARGET_MODE="${TARGET_MODE:-residual_hf}"
MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"

OUTPUT_NAME="${OUTPUT_NAME:-hf_subset_alignment_v0_${SPLIT}_${TARGET_MODE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SUBSET_ROOT}/${OUTPUT_NAME}}"

HIGHPASS_KERNEL="${HIGHPASS_KERNEL:-9}"
NORM_PERCENTILE="${NORM_PERCENTILE:-99}"
TARGET_PERCENTILES="${TARGET_PERCENTILES:-90,95,97}"
HF_PERCENTILE="${HF_PERCENTILE:-90}"
MIN_TARGET="${MIN_TARGET:-0.05}"
MIN_HF="${MIN_HF:-0.05}"
LIMIT="${LIMIT:-0}"
WRITE_IMAGES="${WRITE_IMAGES:-1}"

for path in "${SUBSET_ROOT}/${SPLIT}/hf_rgb" "${TARGET_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[eval-cave-hf-align-v0] required path not found: ${path}" >&2
    exit 1
  fi
done
if [[ "${TARGET_MODE}" == "residual_hf" && ! -d "${ANCHOR_DIR}" ]]; then
  echo "[eval-cave-hf-align-v0] residual_hf requires anchor dir: ${ANCHOR_DIR}" >&2
  exit 1
fi

echo "[eval-cave-hf-align-v0] subset : ${SUBSET_ROOT}"
echo "[eval-cave-hf-align-v0] split  : ${SPLIT}"
echo "[eval-cave-hf-align-v0] target : ${TARGET_DIR}"
echo "[eval-cave-hf-align-v0] anchor : ${ANCHOR_DIR}"
echo "[eval-cave-hf-align-v0] output : ${OUTPUT_ROOT}"

ARGS=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/evaluate_hf_subset_image_alignment_v0.py"
  --subset_root "${SUBSET_ROOT}"
  --split "${SPLIT}"
  --target_dir "${TARGET_DIR}"
  --anchor_dir "${ANCHOR_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --target_mode "${TARGET_MODE}"
  --match_policy "${MATCH_POLICY}"
  --highpass_kernel "${HIGHPASS_KERNEL}"
  --norm_percentile "${NORM_PERCENTILE}"
  --target_percentiles "${TARGET_PERCENTILES}"
  --hf_percentile "${HF_PERCENTILE}"
  --min_target "${MIN_TARGET}"
  --min_hf "${MIN_HF}"
  --limit "${LIMIT}"
)
if [[ "${WRITE_IMAGES}" == "1" ]]; then
  ARGS+=(--write_images)
fi

"${ARGS[@]}"

echo "[eval-cave-hf-align-v0] inspect:"
echo "  ${OUTPUT_ROOT}/overlay"
echo "  ${OUTPUT_ROOT}/target_hf"
echo "  ${OUTPUT_ROOT}/hf_hf"
