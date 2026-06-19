#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CAVE_CACHE_NAME="${CAVE_CACHE_NAME:-render_x1_restormer_cave_hf_ownership_dense_v2_smoke}"
CAVE_CACHE_ROOT="${CAVE_CACHE_ROOT:-${SCENE_ASSET_ROOT}/cave_hf_ownership/${CAVE_CACHE_NAME}}"

OUTPUT_NAME="${OUTPUT_NAME:-render_x1_restormer_cave_hf_reassignment_dense_v0_smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCENE_ASSET_ROOT}/cave_hf_reassignment/${OUTPUT_NAME}}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/cave_hf_reassignment_v0.pt}"

NUM_GAUSSIANS="${NUM_GAUSSIANS:-0}"
SCORE_MODE="${SCORE_MODE:-image_hit_consistency}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.25}"
VIEW_SUPPORT_WEIGHT="${VIEW_SUPPORT_WEIGHT:-0.15}"
VIEW_SUPPORT_CAP="${VIEW_SUPPORT_CAP:-3.0}"
HF_KEEP_RATIO="${HF_KEEP_RATIO:-0.50}"
MIN_HF_SCORE="${MIN_HF_SCORE:-0.05}"
MIN_TOUCH_VIEWS="${MIN_TOUCH_VIEWS:-1}"
MIN_VALIDATED_SCORE="${MIN_VALIDATED_SCORE:-0.0}"
UNCERTAIN_SCORE_MARGIN="${UNCERTAIN_SCORE_MARGIN:-0.5}"

if [[ ! -d "${CAVE_CACHE_ROOT}/per_view" ]]; then
  echo "[cave-reassign-v0] required CAVE per_view dir not found: ${CAVE_CACHE_ROOT}/per_view" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[cave-reassign-v0] cave cache : ${CAVE_CACHE_ROOT}"
echo "[cave-reassign-v0] output     : ${OUTPUT_PATH}"
echo "[cave-reassign-v0] score mode : ${SCORE_MODE}"
echo "[cave-reassign-v0] keep ratio : ${HF_KEEP_RATIO}"
echo "[cave-reassign-v0] min score  : ${MIN_HF_SCORE}"
echo "[cave-reassign-v0] min views  : ${MIN_TOUCH_VIEWS}"

ARGS=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_cave_hf_reassignment_payload_v0.py"
  --cave_cache_root "${CAVE_CACHE_ROOT}"
  --output_path "${OUTPUT_PATH}"
  --num_gaussians "${NUM_GAUSSIANS}"
  --score_mode "${SCORE_MODE}"
  --consistency_weight "${CONSISTENCY_WEIGHT}"
  --view_support_weight "${VIEW_SUPPORT_WEIGHT}"
  --view_support_cap "${VIEW_SUPPORT_CAP}"
  --hf_keep_ratio "${HF_KEEP_RATIO}"
  --min_hf_score "${MIN_HF_SCORE}"
  --min_touch_views "${MIN_TOUCH_VIEWS}"
  --min_validated_score "${MIN_VALIDATED_SCORE}"
  --uncertain_score_margin "${UNCERTAIN_SCORE_MARGIN}"
)

"${ARGS[@]}"

echo "[cave-reassign-v0] payload key suggestions:"
echo "  HF mask strict : hf_owned"
echo "  LF mask strict : lf_safe"
echo "  LF mask loose  : lf_owned"
echo "  uncertain      : hf_uncertain"
echo "[cave-reassign-v0] done: ${OUTPUT_PATH}"
