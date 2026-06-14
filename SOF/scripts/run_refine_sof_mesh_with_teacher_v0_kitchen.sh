#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-detail34k_early4ksoft_meshteacher_v0}"
SOURCE_MODEL_PATH="${SOURCE_MODEL_PATH:-${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/${SOURCE_RUN_NAME}/pulled_mip_model}"
SOURCE_ITERATION="${SOURCE_ITERATION:-34000}"
SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_2}"

TEACHER_MESH_PATH="${TEACHER_MESH_PATH:-}"
BASE_MESH_NAME="${BASE_MESH_NAME:-sof_intrinsic_mesh_v0}"
BASE_MESH_PATH="${BASE_MESH_PATH:-${SOURCE_MODEL_PATH}/test/ours_${SOURCE_ITERATION}/${BASE_MESH_NAME}_7.ply}"
EXTRACT_MESH_IF_MISSING="${EXTRACT_MESH_IF_MISSING:-1}"

RUN_NAME="${RUN_NAME:-${SOURCE_RUN_NAME}_teacher_guided_mesh_refine_v0}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/teacher_guided_sof_mesh_refine_v0/${SCENE_NAME}/${RUN_NAME}}"
EVIDENCE_PATH="${EVIDENCE_PATH:-${RUN_ROOT}/teacher_guided_sof_mesh_evidence_v0.pt}"
SUMMARY_PATH="${SUMMARY_PATH:-${RUN_ROOT}/teacher_guided_sof_mesh_evidence_v0_summary.json}"
DEBUG_ROOT="${DEBUG_ROOT:-${RUN_ROOT}/debug_teacher_guided_mesh_evidence_v0}"
REFINE_ROOT="${REFINE_ROOT:-${RUN_ROOT}/mesh_refine}"

FACE_STRIDE="${FACE_STRIDE:-1}"
MAX_FACES="${MAX_FACES:-0}"
BARY_LAYOUT="${BARY_LAYOUT:-3}"
SURFACE_QUERY_MODE="${SURFACE_QUERY_MODE:-auto}"
MESH_SURFACE_SAMPLE_COUNT="${MESH_SURFACE_SAMPLE_COUNT:-500000}"
SURFACE_QUERY_CHUNK_SIZE="${SURFACE_QUERY_CHUNK_SIZE:-131072}"
TAU_EDGE_SCALE="${TAU_EDGE_SCALE:-1.0}"
TAU_FLOOR="${TAU_FLOOR:-0.0001}"
SIGNED_OFFSET_SCALE="${SIGNED_OFFSET_SCALE:-1.0}"
NORMAL_AGREEMENT_POWER="${NORMAL_AGREEMENT_POWER:-1.0}"
D_NORM_SIGMA="${D_NORM_SIGMA:-1.0}"
TANGENT_NORM_SIGMA="${TANGENT_NORM_SIGMA:-0.75}"
MIN_TRUSTED_NORMAL_AGREEMENT="${MIN_TRUSTED_NORMAL_AGREEMENT:-0.60}"
MAX_TRUSTED_D_NORM="${MAX_TRUSTED_D_NORM:-1.5}"
MAX_TRUSTED_TANGENT_NORM="${MAX_TRUSTED_TANGENT_NORM:-0.75}"
MAX_TRUSTED_ABS_OFFSET="${MAX_TRUSTED_ABS_OFFSET:-0.0}"

REFINE_DEVICE="${REFINE_DEVICE:-auto}"
MAX_CORRESPONDENCES="${MAX_CORRESPONDENCES:-120000}"
MIN_EVIDENCE_WEIGHT="${MIN_EVIDENCE_WEIGHT:-0.05}"
MAX_D_NORM="${MAX_D_NORM:-1.5}"
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

if [[ -z "${TEACHER_MESH_PATH}" ]]; then
  echo "[teacher-guided-sof-mesh-refine-v0] TEACHER_MESH_PATH is required." >&2
  exit 1
fi
if [[ ! -d "${SOURCE_MODEL_PATH}" ]]; then
  echo "[teacher-guided-sof-mesh-refine-v0] missing source model: ${SOURCE_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TEACHER_MESH_PATH}" ]]; then
  echo "[teacher-guided-sof-mesh-refine-v0] missing teacher mesh: ${TEACHER_MESH_PATH}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}" "${REFINE_ROOT}" "${DEBUG_ROOT}"

if [[ ! -f "${BASE_MESH_PATH}" ]]; then
  if [[ "${EXTRACT_MESH_IF_MISSING}" != "1" ]]; then
    echo "[teacher-guided-sof-mesh-refine-v0] base SOF mesh missing and EXTRACT_MESH_IF_MISSING=0: ${BASE_MESH_PATH}" >&2
    exit 1
  fi
  echo "[teacher-guided-sof-mesh-refine-v0] extracting base SOF mesh ..."
  "${PYTHON_BIN}" -u "${SOF_ROOT}/extract_mesh_tets.py" \
    -s "${SCENE_ROOT}" \
    -m "${SOURCE_MODEL_PATH}" \
    -i "${SOURCE_IMAGES_SUBDIR}" \
    --iteration "${SOURCE_ITERATION}" \
    --mesh_name "${BASE_MESH_NAME}" \
    --eval \
    --data_device cpu \
    --filter_mesh
fi

if [[ ! -f "${BASE_MESH_PATH}" ]]; then
  echo "[teacher-guided-sof-mesh-refine-v0] base SOF mesh missing after extraction: ${BASE_MESH_PATH}" >&2
  exit 1
fi

echo "[teacher-guided-sof-mesh-refine-v0] source model    : ${SOURCE_MODEL_PATH} iter=${SOURCE_ITERATION}"
echo "[teacher-guided-sof-mesh-refine-v0] base SOF mesh   : ${BASE_MESH_PATH}"
echo "[teacher-guided-sof-mesh-refine-v0] teacher mesh    : ${TEACHER_MESH_PATH}"
echo "[teacher-guided-sof-mesh-refine-v0] evidence output : ${EVIDENCE_PATH}"
echo "[teacher-guided-sof-mesh-refine-v0] refine root     : ${REFINE_ROOT}"

"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/build_teacher_guided_sof_mesh_evidence_v0.py" \
  --base_mesh_path "${BASE_MESH_PATH}" \
  --teacher_mesh_path "${TEACHER_MESH_PATH}" \
  --output_path "${EVIDENCE_PATH}" \
  --summary_path "${SUMMARY_PATH}" \
  --debug_root "${DEBUG_ROOT}" \
  --face_stride "${FACE_STRIDE}" \
  --max_faces "${MAX_FACES}" \
  --bary_layout "${BARY_LAYOUT}" \
  --surface_query_mode "${SURFACE_QUERY_MODE}" \
  --mesh_surface_sample_count "${MESH_SURFACE_SAMPLE_COUNT}" \
  --surface_query_chunk_size "${SURFACE_QUERY_CHUNK_SIZE}" \
  --tau_edge_scale "${TAU_EDGE_SCALE}" \
  --tau_floor "${TAU_FLOOR}" \
  --signed_offset_scale "${SIGNED_OFFSET_SCALE}" \
  --normal_agreement_power "${NORMAL_AGREEMENT_POWER}" \
  --d_norm_sigma "${D_NORM_SIGMA}" \
  --tangent_norm_sigma "${TANGENT_NORM_SIGMA}" \
  --min_trusted_normal_agreement "${MIN_TRUSTED_NORMAL_AGREEMENT}" \
  --max_trusted_d_norm "${MAX_TRUSTED_D_NORM}" \
  --max_trusted_tangent_norm "${MAX_TRUSTED_TANGENT_NORM}" \
  --max_abs_offset "${MAX_TRUSTED_ABS_OFFSET}" \
  --debug_point_cap "${DEBUG_POINT_CAP}" \
  --seed "${SEED}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/refine_mesh_from_bounded_evidence_v0.py"
  --mesh_path "${BASE_MESH_PATH}"
  --evidence_path "${EVIDENCE_PATH}"
  --output_root "${REFINE_ROOT}"
  --device "${REFINE_DEVICE}"
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

"${CMD[@]}"

echo "[done] base mesh     : ${BASE_MESH_PATH}"
echo "[done] evidence      : ${EVIDENCE_PATH}"
echo "[done] refine mesh   : ${REFINE_ROOT}/refined_mesh_v0.ply"
echo "[done] refine payload: ${REFINE_ROOT}/mesh_normal_offset_refine_v0.pt"
echo "[done] refine summary: ${REFINE_ROOT}/mesh_normal_offset_refine_v0_summary.json"
