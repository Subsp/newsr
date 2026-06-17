#!/usr/bin/env bash
set -euo pipefail

# Sweep non-oracle NPSE edge-target direction-gate settings, then evaluate each
# cache against GT high-frequency diagnostics. GT is used only by the diagnostic
# step, never by cache generation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
ENHANCEMENT_BACKEND="${ENHANCEMENT_BACKEND:-restormer}"

DEPTH_PRIOR_DIR="${DEPTH_PRIOR_DIR:?Set DEPTH_PRIOR_DIR to the aligned depth prior directory.}"
LIMIT="${LIMIT:-8}"
OVERWRITE="${OVERWRITE:-1}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-render_x1_${ENHANCEMENT_BACKEND}_depthprior_npse_yellow_fidelity_sweep}"
SWEEP_VARIANTS="${SWEEP_VARIANTS:-nogate grad_f03 grad_f05 grad_f05_p05 resgrad_f05_rw025}"

echo "[npse-dirgate-sweep-v0] depth prior : ${DEPTH_PRIOR_DIR}"
echo "[npse-dirgate-sweep-v0] limit       : ${LIMIT}"
echo "[npse-dirgate-sweep-v0] prefix      : ${OUTPUT_PREFIX}"
echo "[npse-dirgate-sweep-v0] variants    : ${SWEEP_VARIANTS}"

for variant in ${SWEEP_VARIANTS}; do
  EDGE_TARGET_DIRECTION_GATE_VALUE="none"
  EDGE_TARGET_DIRECTION_FLOOR_VALUE="0.20"
  EDGE_TARGET_DIRECTION_POWER_VALUE="1.0"
  EDGE_TARGET_RESIDUAL_DIRECTION_WEIGHT_VALUE="0.50"
  EDGE_TARGET_DIRECTION_MIN_COS_VALUE="0.0"
  EDGE_TARGET_DIRECTION_BLUR_VALUE="3"

  case "${variant}" in
    nogate)
      EDGE_TARGET_DIRECTION_GATE_VALUE="none"
      ;;
    grad_f03)
      EDGE_TARGET_DIRECTION_GATE_VALUE="anchor_sr_gradient"
      EDGE_TARGET_DIRECTION_FLOOR_VALUE="0.30"
      ;;
    grad_f05)
      EDGE_TARGET_DIRECTION_GATE_VALUE="anchor_sr_gradient"
      EDGE_TARGET_DIRECTION_FLOOR_VALUE="0.50"
      ;;
    grad_f05_p05)
      EDGE_TARGET_DIRECTION_GATE_VALUE="anchor_sr_gradient"
      EDGE_TARGET_DIRECTION_FLOOR_VALUE="0.50"
      EDGE_TARGET_DIRECTION_POWER_VALUE="0.50"
      ;;
    resgrad_f05_rw025)
      EDGE_TARGET_DIRECTION_GATE_VALUE="anchor_sr_residual_gradient"
      EDGE_TARGET_DIRECTION_FLOOR_VALUE="0.50"
      EDGE_TARGET_RESIDUAL_DIRECTION_WEIGHT_VALUE="0.25"
      ;;
    *)
      echo "[npse-dirgate-sweep-v0] unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac

  OUTPUT_NAME="${OUTPUT_PREFIX}_${variant}"
  echo
  echo "[npse-dirgate-sweep-v0] === ${variant} -> ${OUTPUT_NAME} ==="
  DEPTH_PRIOR_DIR="${DEPTH_PRIOR_DIR}" \
  OUTPUT_NAME="${OUTPUT_NAME}" \
  EDGE_TARGET_DIRECTION_GATE="${EDGE_TARGET_DIRECTION_GATE_VALUE}" \
  EDGE_TARGET_DIRECTION_FLOOR="${EDGE_TARGET_DIRECTION_FLOOR_VALUE}" \
  EDGE_TARGET_DIRECTION_POWER="${EDGE_TARGET_DIRECTION_POWER_VALUE}" \
  EDGE_TARGET_RESIDUAL_DIRECTION_WEIGHT="${EDGE_TARGET_RESIDUAL_DIRECTION_WEIGHT_VALUE}" \
  EDGE_TARGET_DIRECTION_MIN_COS="${EDGE_TARGET_DIRECTION_MIN_COS_VALUE}" \
  EDGE_TARGET_DIRECTION_BLUR="${EDGE_TARGET_DIRECTION_BLUR_VALUE}" \
  LIMIT="${LIMIT}" \
  OVERWRITE="${OVERWRITE}" \
  bash "${SCRIPT_DIR}/run_build_npse_edge_trust_cache_v0_kitchen.sh"

  NPSE_CACHE_NAME="${OUTPUT_NAME}" \
  LIMIT="${LIMIT}" \
  OVERWRITE=1 \
  bash "${SCRIPT_DIR}/run_evaluate_npse_gt_hf_alignment_v0_kitchen.sh"
done

echo
echo "[npse-dirgate-sweep-v0] edge_target summary"
for variant in ${SWEEP_VARIANTS}; do
  OUTPUT_NAME="${OUTPUT_PREFIX}_${variant}"
  SUMMARY_JSON="${SCENE_ASSET_ROOT}/npse_cache/${OUTPUT_NAME}/gt_hf_alignment_v0/summary.json"
  if [[ ! -f "${SUMMARY_JSON}" ]]; then
    echo "  ${variant}: missing ${SUMMARY_JSON}"
    continue
  fi
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); s=d["summary"].get("edge_target",{}); print("  {}: frames={} abs_l1={:.6f} abs_corr={:.4f} abs_energy={:.3f} delta_corr={:.4f} delta_energy={:.3f} over={:.3f}".format(sys.argv[2], int(s.get("frames", 0)), s.get("abs_hf_l1_rgb", float("nan")), s.get("abs_hf_corr_luma", float("nan")), s.get("abs_hf_energy_ratio", float("nan")), s.get("delta_hf_corr_luma", float("nan")), s.get("delta_hf_energy_ratio", float("nan")), s.get("delta_hf_over_injection", float("nan"))))' "${SUMMARY_JSON}" "${variant}"
done
