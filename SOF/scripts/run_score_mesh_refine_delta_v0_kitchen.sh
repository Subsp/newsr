#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-${STAGE_NAME}_geometry_only_v0}"
MESH_BOUNDED_RUN_NAME="${MESH_BOUNDED_RUN_NAME:-${SOURCE_RUN_NAME}_mesh_bounded_color_v0}"
EVIDENCE_RUN_NAME="${EVIDENCE_RUN_NAME:-${MESH_BOUNDED_RUN_NAME}_mesh_evidence_v0}"
REFINE_RUN_NAME="${REFINE_RUN_NAME:-${EVIDENCE_RUN_NAME}_normal_offset_v0}"

MESH_COMPARE_ROOT="${MESH_COMPARE_ROOT:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}}"
SOURCE_MESH_PATH="${SOURCE_MESH_PATH:-${MESH_COMPARE_ROOT}/${STAGE_NAME}_prepare_stage_sof_export_mesh_v0_${STAGE_NAME}_7.ply}"
REFINE_ROOT="${REFINE_ROOT:-${SOF_ROOT}/output/mesh_surface_refine_v0/${SCENE_NAME}/${REFINE_RUN_NAME}}"
REFINED_MESH_PATH="${REFINED_MESH_PATH:-${REFINE_ROOT}/refined_mesh_v0.ply}"
OFFSET_PAYLOAD_PATH="${OFFSET_PAYLOAD_PATH:-${REFINE_ROOT}/mesh_normal_offset_refine_v0.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/mesh_refine_delta_v0/${SCENE_NAME}/${REFINE_RUN_NAME}_delta_score_v0}"

EDGE_SAMPLE_FACES="${EDGE_SAMPLE_FACES:-500000}"
DEBUG_POINT_CAP="${DEBUG_POINT_CAP:-200000}"
DEBUG_VMAX="${DEBUG_VMAX:-0.02}"
COMPUTE_FULL_VERTEX_NORMALS="${COMPUTE_FULL_VERTEX_NORMALS:-0}"
SEED="${SEED:-0}"

if [[ ! -f "${SOURCE_MESH_PATH}" ]]; then
  echo "[mesh-refine-delta-v0] missing source mesh: ${SOURCE_MESH_PATH}" >&2
  exit 1
fi
if [[ ! -f "${REFINED_MESH_PATH}" ]]; then
  echo "[mesh-refine-delta-v0] missing refined mesh: ${REFINED_MESH_PATH}" >&2
  exit 1
fi

echo "[mesh-refine-delta-v0] source  : ${SOURCE_MESH_PATH}"
echo "[mesh-refine-delta-v0] refined : ${REFINED_MESH_PATH}"
echo "[mesh-refine-delta-v0] payload : ${OFFSET_PAYLOAD_PATH}"
echo "[mesh-refine-delta-v0] output  : ${OUTPUT_ROOT}"
echo "[mesh-refine-delta-v0] normals : full_vertex=${COMPUTE_FULL_VERTEX_NORMALS}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/score_mesh_refine_delta_v0.py"
  --source_mesh_path "${SOURCE_MESH_PATH}" \
  --refined_mesh_path "${REFINED_MESH_PATH}" \
  --offset_payload_path "${OFFSET_PAYLOAD_PATH}" \
  --output_root "${OUTPUT_ROOT}" \
  --edge_sample_faces "${EDGE_SAMPLE_FACES}" \
  --debug_point_cap "${DEBUG_POINT_CAP}" \
  --debug_vmax "${DEBUG_VMAX}" \
  --seed "${SEED}"
)

if [[ "${COMPUTE_FULL_VERTEX_NORMALS}" == "1" ]]; then
  CMD+=(--compute_full_vertex_normals)
fi

"${CMD[@]}"
