#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
MESH_COMPARE_ROOT="${MESH_COMPARE_ROOT:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}}"
MESH_PATH="${MESH_PATH:-${MESH_COMPARE_ROOT}/${STAGE_NAME}_prepare_stage_sof_export_mesh_v0_${STAGE_NAME}_7.ply}"

SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-${STAGE_NAME}_geometry_only_v0}"
MESH_BOUNDED_RUN_NAME="${MESH_BOUNDED_RUN_NAME:-${SOURCE_RUN_NAME}_mesh_bounded_color_v0}"
EVIDENCE_RUN_NAME="${EVIDENCE_RUN_NAME:-${MESH_BOUNDED_RUN_NAME}_mesh_evidence_v0}"
EVIDENCE_PATH="${EVIDENCE_PATH:-${SOF_ROOT}/output/mesh_bounded_mesh_evidence_v0/${SCENE_NAME}/${EVIDENCE_RUN_NAME}/mesh_bounded_mesh_evidence_v0.pt}"
REFINE_RUN_NAME="${REFINE_RUN_NAME:-${EVIDENCE_RUN_NAME}_normal_offset_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/mesh_surface_refine_v0/${SCENE_NAME}/${REFINE_RUN_NAME}}"

DEVICE="${DEVICE:-auto}"
MAX_CORRESPONDENCES="${MAX_CORRESPONDENCES:-120000}"
MIN_EVIDENCE_WEIGHT="${MIN_EVIDENCE_WEIGHT:-0.28}"
MAX_D_NORM="${MAX_D_NORM:-1.0}"
MAX_ABS_OFFSET="${MAX_ABS_OFFSET:-0.03}"
ITERATIONS="${ITERATIONS:-800}"
LR="${LR:-8e-4}"
LAMBDA_DELTA="${LAMBDA_DELTA:-0.02}"
LAMBDA_LAP="${LAMBDA_LAP:-0.20}"
LAMBDA_CLIP="${LAMBDA_CLIP:-1.0}"
OFFSET_SCALE="${OFFSET_SCALE:-0.01}"
OFFSET_CLIP="${OFFSET_CLIP:-0.02}"
ROBUST_EPS="${ROBUST_EPS:-1e-3}"
DEBUG_POINT_CAP="${DEBUG_POINT_CAP:-100000}"
SEED="${SEED:-0}"
ALLOW_UNTRUSTED="${ALLOW_UNTRUSTED:-0}"
SKIP_MESH_EXPORT="${SKIP_MESH_EXPORT:-0}"

if [[ ! -f "${MESH_PATH}" ]]; then
  echo "[mesh-normal-offset-refine-v0] missing mesh: ${MESH_PATH}" >&2
  exit 1
fi
if [[ ! -f "${EVIDENCE_PATH}" ]]; then
  echo "[mesh-normal-offset-refine-v0] missing evidence: ${EVIDENCE_PATH}" >&2
  exit 1
fi

echo "[mesh-normal-offset-refine-v0] mesh     : ${MESH_PATH}"
echo "[mesh-normal-offset-refine-v0] evidence : ${EVIDENCE_PATH}"
echo "[mesh-normal-offset-refine-v0] output   : ${OUTPUT_ROOT}"
echo "[mesh-normal-offset-refine-v0] filter   : weight>=${MIN_EVIDENCE_WEIGHT} d_norm<=${MAX_D_NORM} abs_offset<=${MAX_ABS_OFFSET} max=${MAX_CORRESPONDENCES}"
echo "[mesh-normal-offset-refine-v0] optimize : iter=${ITERATIONS} lr=${LR} delta=${LAMBDA_DELTA} lap=${LAMBDA_LAP} clip=${OFFSET_CLIP}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/refine_mesh_from_bounded_evidence_v0.py"
  --mesh_path "${MESH_PATH}"
  --evidence_path "${EVIDENCE_PATH}"
  --output_root "${OUTPUT_ROOT}"
  --device "${DEVICE}"
  --max_correspondences "${MAX_CORRESPONDENCES}"
  --min_evidence_weight "${MIN_EVIDENCE_WEIGHT}"
  --max_d_norm "${MAX_D_NORM}"
  --max_abs_offset "${MAX_ABS_OFFSET}"
  --iterations "${ITERATIONS}"
  --lr "${LR}"
  --lambda_delta "${LAMBDA_DELTA}"
  --lambda_lap "${LAMBDA_LAP}"
  --lambda_clip "${LAMBDA_CLIP}"
  --offset_scale "${OFFSET_SCALE}"
  --offset_clip "${OFFSET_CLIP}"
  --robust_eps "${ROBUST_EPS}"
  --debug_point_cap "${DEBUG_POINT_CAP}"
  --seed "${SEED}"
)

if [[ "${ALLOW_UNTRUSTED}" == "1" ]]; then
  CMD+=(--allow_untrusted)
fi
if [[ "${SKIP_MESH_EXPORT}" == "1" ]]; then
  CMD+=(--skip_mesh_export)
fi

"${CMD[@]}"
