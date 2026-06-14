#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
VGGT_ROOT="${VGGT_ROOT:-${WORK_ROOT}/vggt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LR_SOF_MODEL="${LR_SOF_MODEL:-${SCENE_ROOT}/_hrgsrefiner_assets/kitchen_sof_vanilla_images8_v1/soflr30k}"
LR_MESH_PATH="${LR_MESH_PATH:-${LR_SOF_MODEL}/test/ours_30000/lr_sof_mesh_v0_7.ply}"

RUN_NAME="${RUN_NAME:-soflr_vggt_depth_mesh_refine_v0}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/vggt_depth_mesh_refine_v0/${SCENE_NAME}/${RUN_NAME}}"
CORRECTION_DIR="${CORRECTION_DIR:-${RUN_ROOT}/correction}"
EVIDENCE_PATH="${EVIDENCE_PATH:-${RUN_ROOT}/soflr_vggt_mesh_evidence_v0.pt}"
REFINE_ROOT="${REFINE_ROOT:-${RUN_ROOT}/mesh_refine}"
DELTA_SCORE_ROOT="${DELTA_SCORE_ROOT:-${RUN_ROOT}/mesh_delta_score}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
LOAD_ITERATION="${LOAD_ITERATION:-30000}"
CORRECTION_DEVICE="${CORRECTION_DEVICE:-cuda}"
REFINE_DEVICE="${REFINE_DEVICE:-auto}"
MAX_VIEWS="${MAX_VIEWS:-8}"
FACE_K="${FACE_K:-8}"
BINDING_CHUNK_SIZE="${BINDING_CHUNK_SIZE:-50000}"
DEPTH_MIN="${DEPTH_MIN:-0.02}"
MIN_ALPHA="${MIN_ALPHA:-0.05}"
MIN_VGGT_CONFIDENCE="${MIN_VGGT_CONFIDENCE:-0.05}"
DEPTH_ALIGN_MIN_PIXELS="${DEPTH_ALIGN_MIN_PIXELS:-2048}"
NORMAL_DENOMINATOR_MIN="${NORMAL_DENOMINATOR_MIN:-0.15}"
MAX_CORRECTION_RATIO="${MAX_CORRECTION_RATIO:-0.003}"
MAX_CORRECTION_ABS="${MAX_CORRECTION_ABS:-0.0}"

TAU_EDGE_SCALE="${TAU_EDGE_SCALE:-1.0}"
TAU_FLOOR="${TAU_FLOOR:-0.0001}"
SIGNED_OFFSET_SCALE="${SIGNED_OFFSET_SCALE:-1.0}"
CONFIDENCE_POWER="${CONFIDENCE_POWER:-1.0}"
WEIGHT_SUM_POWER="${WEIGHT_SUM_POWER:-0.0}"
WEIGHT_SUM_REF="${WEIGHT_SUM_REF:-0.0}"
VIEW_COUNT_POWER="${VIEW_COUNT_POWER:-0.0}"
VIEW_COUNT_REF="${VIEW_COUNT_REF:-0.0}"
MIN_TRUSTED_CONFIDENCE="${MIN_TRUSTED_CONFIDENCE:-0.05}"
MIN_TRUSTED_VIEW_COUNT="${MIN_TRUSTED_VIEW_COUNT:-2}"
MIN_TRUSTED_WEIGHT_SUM="${MIN_TRUSTED_WEIGHT_SUM:-0.0}"
TRUSTED_MAX_D_NORM="${TRUSTED_MAX_D_NORM:-1.0}"
TRUSTED_MAX_ABS_OFFSET="${TRUSTED_MAX_ABS_OFFSET:-0.0}"

MAX_CORRESPONDENCES="${MAX_CORRESPONDENCES:-120000}"
MIN_EVIDENCE_WEIGHT="${MIN_EVIDENCE_WEIGHT:-0.05}"
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
RUN_SCORE="${RUN_SCORE:-1}"
SCORE_DEBUG_VMAX="${SCORE_DEBUG_VMAX:-0.02}"
SCORE_EDGE_SAMPLE_FACES="${SCORE_EDGE_SAMPLE_FACES:-500000}"

if [[ ! -d "${LR_SOF_MODEL}" ]]; then
  echo "[depth-mesh-refine-v0] missing SOFLR model: ${LR_SOF_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${LR_MESH_PATH}" ]]; then
  echo "[depth-mesh-refine-v0] missing LR mesh: ${LR_MESH_PATH}" >&2
  exit 1
fi

mkdir -p "${CORRECTION_DIR}" "${REFINE_ROOT}"
if [[ "${RUN_SCORE}" == "1" ]]; then
  mkdir -p "${DELTA_SCORE_ROOT}"
fi

echo "[depth-mesh-refine-v0] scene          : ${SCENE_ROOT}"
echo "[depth-mesh-refine-v0] soflr model    : ${LR_SOF_MODEL}"
echo "[depth-mesh-refine-v0] base mesh      : ${LR_MESH_PATH}"
echo "[depth-mesh-refine-v0] run root       : ${RUN_ROOT}"
echo "[depth-mesh-refine-v0] devices        : correction=${CORRECTION_DEVICE} refine=${REFINE_DEVICE}"
echo "[depth-mesh-refine-v0] correction     : views=${MAX_VIEWS} ratio=${MAX_CORRECTION_RATIO} abs=${MAX_CORRECTION_ABS}"
echo "[depth-mesh-refine-v0] trust gates    : conf>=${MIN_TRUSTED_CONFIDENCE} views>=${MIN_TRUSTED_VIEW_COUNT} d_norm<=${TRUSTED_MAX_D_NORM}"
echo "[depth-mesh-refine-v0] refine filter  : weight>=${MIN_EVIDENCE_WEIGHT} d_norm<=${MAX_D_NORM} abs_offset<=${MAX_ABS_OFFSET}"
echo "[depth-mesh-refine-v0] refine optimize: iter=${ITERATIONS} lr=${LR} delta=${LAMBDA_DELTA} lap=${LAMBDA_LAP} clip=${OFFSET_CLIP}"

"${PYTHON_BIN}" -u "${SOF_ROOT}/build_soflr_vggt_bound_gs_correction_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --soflr_model_path "${LR_SOF_MODEL}" \
  --lr_mesh_path "${LR_MESH_PATH}" \
  --output_dir "${CORRECTION_DIR}" \
  --vggt_root "${VGGT_ROOT}" \
  --images_subdir "${IMAGES_SUBDIR}" \
  --load_iteration "${LOAD_ITERATION}" \
  --max_views "${MAX_VIEWS}" \
  --device "${CORRECTION_DEVICE}" \
  --face_k "${FACE_K}" \
  --binding_chunk_size "${BINDING_CHUNK_SIZE}" \
  --depth_min "${DEPTH_MIN}" \
  --min_alpha "${MIN_ALPHA}" \
  --min_vggt_confidence "${MIN_VGGT_CONFIDENCE}" \
  --depth_align_min_pixels "${DEPTH_ALIGN_MIN_PIXELS}" \
  --normal_denominator_min "${NORMAL_DENOMINATOR_MIN}" \
  --max_correction_ratio "${MAX_CORRECTION_RATIO}" \
  --max_correction_abs "${MAX_CORRECTION_ABS}"

"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/convert_soflr_vggt_correction_to_mesh_evidence_v0.py" \
  --mesh_path "${LR_MESH_PATH}" \
  --correction_payload "${CORRECTION_DIR}/correction_payload_v0.npz" \
  --output_path "${EVIDENCE_PATH}" \
  --tau_edge_scale "${TAU_EDGE_SCALE}" \
  --tau_floor "${TAU_FLOOR}" \
  --signed_offset_scale "${SIGNED_OFFSET_SCALE}" \
  --confidence_power "${CONFIDENCE_POWER}" \
  --weight_sum_power "${WEIGHT_SUM_POWER}" \
  --weight_sum_ref "${WEIGHT_SUM_REF}" \
  --view_count_power "${VIEW_COUNT_POWER}" \
  --view_count_ref "${VIEW_COUNT_REF}" \
  --min_trusted_confidence "${MIN_TRUSTED_CONFIDENCE}" \
  --min_trusted_view_count "${MIN_TRUSTED_VIEW_COUNT}" \
  --min_trusted_weight_sum "${MIN_TRUSTED_WEIGHT_SUM}" \
  --trusted_max_d_norm "${TRUSTED_MAX_D_NORM}" \
  --trusted_max_abs_offset "${TRUSTED_MAX_ABS_OFFSET}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/refine_mesh_from_bounded_evidence_v0.py"
  --mesh_path "${LR_MESH_PATH}"
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

if [[ "${RUN_SCORE}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/score_mesh_refine_delta_v0.py" \
    --source_mesh_path "${LR_MESH_PATH}" \
    --refined_mesh_path "${REFINE_ROOT}/refined_mesh_v0.ply" \
    --offset_payload_path "${REFINE_ROOT}/mesh_normal_offset_refine_v0.pt" \
    --output_root "${DELTA_SCORE_ROOT}" \
    --edge_sample_faces "${SCORE_EDGE_SAMPLE_FACES}" \
    --debug_point_cap "${DEBUG_POINT_CAP}" \
    --debug_vmax "${SCORE_DEBUG_VMAX}" \
    --seed "${SEED}"
fi

echo "[done] correction payload : ${CORRECTION_DIR}/correction_payload_v0.npz"
echo "[done] mesh evidence      : ${EVIDENCE_PATH}"
echo "[done] refined mesh       : ${REFINE_ROOT}/refined_mesh_v0.ply"
echo "[done] refine payload     : ${REFINE_ROOT}/mesh_normal_offset_refine_v0.pt"
echo "[done] refine summary     : ${REFINE_ROOT}/mesh_normal_offset_refine_v0_summary.json"
if [[ "${RUN_SCORE}" == "1" ]]; then
  echo "[done] delta summary      : ${DELTA_SCORE_ROOT}/mesh_refine_delta_v0_summary.json"
fi
