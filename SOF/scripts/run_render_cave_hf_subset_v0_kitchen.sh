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

INPUT_EXPERIMENT_NAME="${INPUT_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
RUN_TAG="${RUN_TAG:-${INPUT_EXPERIMENT_NAME}_cave_hf_transfer_v0}"
MODEL_DIR="${MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_nosr_layerfreq_cleanup_v0/${SCENE_NAME}/${RUN_TAG}}"
ITERATION="${ITERATION:-32000}"

CAVE_REASSIGN_NAME="${CAVE_REASSIGN_NAME:-render_x1_restormer_cave_hf_reassignment_dense_v1}"
CAVE_REASSIGN_PAYLOAD="${CAVE_REASSIGN_PAYLOAD:-${SCENE_ASSET_ROOT}/cave_hf_reassignment/${CAVE_REASSIGN_NAME}/cave_hf_reassignment_v0.pt}"

OUTPUT_NAME="${OUTPUT_NAME:-cave_hf_subset_render_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MODEL_DIR}/${OUTPUT_NAME}}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
RESOLUTION="${RESOLUTION:-1}"
LIMIT="${LIMIT:-0}"

MASK_KEY="${MASK_KEY:-}"
SCORE_KEY="${SCORE_KEY:-hf_score}"
CANDIDATE_KEY="${CANDIDATE_KEY:-hf_candidate}"
INCLUDE_MASK_KEY="${INCLUDE_MASK_KEY:-hf_owned}"
KEEP_RATIO="${KEEP_RATIO:-0.85}"
MIN_SCORE="${MIN_SCORE:-0.03}"
HF_PERCENTILE="${HF_PERCENTILE:-99.0}"
RENDER_LF="${RENDER_LF:-1}"
RENDER_ALL="${RENDER_ALL:-1}"

for path in "${SCENE_ROOT}" "${SCENE_ROOT}/sparse/0" "${SCENE_ROOT}/${IMAGES_SUBDIR}" "${MODEL_DIR}" "${CAVE_REASSIGN_PAYLOAD}" "${MIPSPLATTING_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[render-cave-hf-subset-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

echo "[render-cave-hf-subset-v0] model   : ${MODEL_DIR}"
echo "[render-cave-hf-subset-v0] payload : ${CAVE_REASSIGN_PAYLOAD}"
echo "[render-cave-hf-subset-v0] output  : ${OUTPUT_ROOT}"
echo "[render-cave-hf-subset-v0] split   : ${SPLIT} limit=${LIMIT}"
echo "[render-cave-hf-subset-v0] select  : score=${SCORE_KEY:-none} candidate=${CANDIDATE_KEY:-none} keep=${KEEP_RATIO} min=${MIN_SCORE} include=${INCLUDE_MASK_KEY:-none}"

ARGS=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/render_gaussian_subset_from_payload_v0.py"
  --scene_root "${SCENE_ROOT}"
  --images_subdir "${IMAGES_SUBDIR}"
  --model_dir "${MODEL_DIR}"
  --iteration "${ITERATION}"
  --mask_payload "${CAVE_REASSIGN_PAYLOAD}"
  --mask_key "${MASK_KEY}"
  --score_key "${SCORE_KEY}"
  --candidate_key "${CANDIDATE_KEY}"
  --include_mask_key "${INCLUDE_MASK_KEY}"
  --keep_ratio "${KEEP_RATIO}"
  --min_score "${MIN_SCORE}"
  --output_root "${OUTPUT_ROOT}"
  --mipsplatting_root "${MIPSPLATTING_ROOT}"
  --split "${SPLIT}"
  --resolution "${RESOLUTION}"
  --limit "${LIMIT}"
  --hf_percentile "${HF_PERCENTILE}"
)

if [[ "${RENDER_LF}" == "1" ]]; then
  ARGS+=(--render_lf)
fi
if [[ "${RENDER_ALL}" == "1" ]]; then
  ARGS+=(--render_all)
fi

"${ARGS[@]}"

echo "[render-cave-hf-subset-v0] inspect:"
echo "  ${OUTPUT_ROOT}/${SPLIT}/hf_rgb"
echo "  ${OUTPUT_ROOT}/${SPLIT}/hf_hf_abs"
echo "  ${OUTPUT_ROOT}/${SPLIT}/lf_hf_abs"
echo "  ${OUTPUT_ROOT}/${SPLIT}/all_rgb"
