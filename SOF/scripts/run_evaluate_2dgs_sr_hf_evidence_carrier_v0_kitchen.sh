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
EVIDENCE_NAME="${EVIDENCE_NAME:-${SR_PRIOR_NAME}_sr_hf_evidence_v0}"
MASK_SUBDIR="${MASK_SUBDIR:-geometry_weight}"
CARRIER_OUTPUT_NAME="${CARRIER_OUTPUT_NAME:-${EVIDENCE_NAME}_geometry_carrier_rgb_2dgs_v0}"

CARRIER_ROOT="${CARRIER_ROOT:-${SOF_ROOT}/output/2dgs_sr_hf_evidence_carrier/${CARRIER_OUTPUT_NAME}}"
TARGET_DIR="${TARGET_DIR:-${CARRIER_ROOT}/evidence_target}"
RENDER_DIR="${RENDER_DIR:-${CARRIER_ROOT}/evidence_render}"
MASK_DIR="${MASK_DIR:-${SCENE_ASSET_ROOT}/sr_hf_evidence/${EVIDENCE_NAME}/${MASK_SUBDIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CARRIER_ROOT}/expression_metrics_v0}"

MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"
LIMIT="${LIMIT:-0}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.05}"
PEAK="${PEAK:-1.0}"
OVERWRITE="${OVERWRITE:-1}"

for required in "${TARGET_DIR}" "${RENDER_DIR}"; do
  if [[ ! -d "${required}" ]]; then
    echo "[2dgs-carrier-psnr-v0] required path not found: ${required}" >&2
    exit 1
  fi
done

ARGS=(
  --target_dir "${TARGET_DIR}"
  --render_dir "${RENDER_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --match_policy "${MATCH_POLICY}"
  --limit "${LIMIT}"
  --active_threshold "${ACTIVE_THRESHOLD}"
  --peak "${PEAK}"
)
if [[ -d "${MASK_DIR}" ]]; then
  ARGS+=(--mask_dir "${MASK_DIR}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "[2dgs-carrier-psnr-v0] target : ${TARGET_DIR}"
echo "[2dgs-carrier-psnr-v0] render : ${RENDER_DIR}"
echo "[2dgs-carrier-psnr-v0] mask   : ${MASK_DIR}"
echo "[2dgs-carrier-psnr-v0] output : ${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/evaluate_2dgs_carrier_expression_v0.py" "${ARGS[@]}"
