#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting}"

BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/${BASE_EXPERIMENT_NAME}}"
BASE_ITERATION="${BASE_ITERATION:-30000}"
BASE_PLY="${BASE_MODEL_DIR}/point_cloud/iteration_${BASE_ITERATION}/point_cloud.ply"

DEPTH_DIR="${DEPTH_DIR:-${SCENE_ASSET_ROOT}/depth_prior_aligned_gs2mesh/render_x1_depthprior_images_2_train_gs2mesh_aligned_full_v0/aligned_depth}"

EVIDENCE_NAME="${EVIDENCE_NAME:-qwen_vosr_sr_hf_effective_verywide_8view_v0}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${SCENE_ASSET_ROOT}/sr_hf_evidence/${EVIDENCE_NAME}}"
EVIDENCE_RGB_DIR="${EVIDENCE_RGB_DIR:-${EVIDENCE_ROOT}/effective_hf_carrier_rgb}"
EVIDENCE_WEIGHT_DIR="${EVIDENCE_WEIGHT_DIR:-${EVIDENCE_ROOT}/effective_hf_weight}"

CARRIER_NAME="${CARRIER_NAME:-qwen_vosr_effective_verywide_2dgs_one_v0}"
CARRIER_ROOT="${CARRIER_ROOT:-${SOF_ROOT}/output/2dgs_sr_hf_evidence_carrier/${CARRIER_NAME}}"
PRIMITIVE_DIR="${PRIMITIVE_DIR:-${CARRIER_ROOT}/primitives}"

OUTPUT_NAME="${OUTPUT_NAME:-${BASE_EXPERIMENT_NAME}_spray_2dgs_effective_hf_v0}"
OUTPUT_MODEL_DIR="${OUTPUT_MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_2dgs_hf_spray_v0/${SCENE_NAME}/${OUTPUT_NAME}}"
NEWBORN_MODEL_DIR="${NEWBORN_MODEL_DIR:-${OUTPUT_MODEL_DIR}_newborn_only}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-${BASE_ITERATION}}"
MERGE_SCRIPT="${MERGE_SCRIPT:-}"

MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
MAX_PRIMITIVES_PER_VIEW="${MAX_PRIMITIVES_PER_VIEW:-65536}"
MAX_TOTAL_NEWBORN="${MAX_TOTAL_NEWBORN:-0}"
MIN_WEIGHT="${MIN_WEIGHT:-0.015}"
MIN_PRIMITIVE_OPACITY="${MIN_PRIMITIVE_OPACITY:-0.0}"
DEPTH_MIN="${DEPTH_MIN:-0.02}"
SCALE_MULTIPLIER="${SCALE_MULTIPLIER:-1.0}"
SCALE_MIN="${SCALE_MIN:-0.0005}"
SCALE_MAX="${SCALE_MAX:-0.012}"
NORMAL_SCALE_RATIO="${NORMAL_SCALE_RATIO:-0.35}"
NORMAL_SCALE_MIN="${NORMAL_SCALE_MIN:-0.0004}"
NORMAL_SCALE_MAX="${NORMAL_SCALE_MAX:-0.003}"
OPACITY_FLOOR="${OPACITY_FLOOR:-0.015}"
OPACITY_SCALE="${OPACITY_SCALE:-0.10}"
OPACITY_POWER="${OPACITY_POWER:-0.75}"
OPACITY_MIN="${OPACITY_MIN:-0.01}"
OPACITY_MAX="${OPACITY_MAX:-0.12}"
WRITE_CPU_MERGED_PREVIEW="${WRITE_CPU_MERGED_PREVIEW:-0}"

for required in "${BASE_MODEL_DIR}" "${BASE_PLY}" "${DEPTH_DIR}" "${PRIMITIVE_DIR}" "${EVIDENCE_RGB_DIR}" "${EVIDENCE_WEIGHT_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[spray-2dgs-to-3d-v0] required path not found: ${required}" >&2
    exit 1
  fi
done

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_MODEL_DIR}" "${NEWBORN_MODEL_DIR}"
fi

echo "[spray-2dgs-to-3d-v0] base model : ${BASE_MODEL_DIR}"
echo "[spray-2dgs-to-3d-v0] base ply   : ${BASE_PLY}"
echo "[spray-2dgs-to-3d-v0] depth      : ${DEPTH_DIR}"
echo "[spray-2dgs-to-3d-v0] primitives : ${PRIMITIVE_DIR}"
echo "[spray-2dgs-to-3d-v0] evidence   : ${EVIDENCE_RGB_DIR}"
echo "[spray-2dgs-to-3d-v0] weight     : ${EVIDENCE_WEIGHT_DIR}"
echo "[spray-2dgs-to-3d-v0] newborn   : ${NEWBORN_MODEL_DIR}"
echo "[spray-2dgs-to-3d-v0] output    : ${OUTPUT_MODEL_DIR}"

SPRAY_ARGS=(
  --base_model_dir "${BASE_MODEL_DIR}"
  --base_iteration "${BASE_ITERATION}"
  --depth_dir "${DEPTH_DIR}"
  --primitive_dir "${PRIMITIVE_DIR}"
  --carrier_rgb_dir "${EVIDENCE_RGB_DIR}"
  --carrier_weight_dir "${EVIDENCE_WEIGHT_DIR}"
  --output_model_dir "${OUTPUT_MODEL_DIR}"
  --newborn_model_dir "${NEWBORN_MODEL_DIR}"
  --match_policy "${MATCH_POLICY}"
  --limit "${LIMIT}"
  --max_primitives_per_view "${MAX_PRIMITIVES_PER_VIEW}"
  --max_total_newborn "${MAX_TOTAL_NEWBORN}"
  --min_weight "${MIN_WEIGHT}"
  --min_primitive_opacity "${MIN_PRIMITIVE_OPACITY}"
  --depth_min "${DEPTH_MIN}"
  --scale_multiplier "${SCALE_MULTIPLIER}"
  --scale_min "${SCALE_MIN}"
  --scale_max "${SCALE_MAX}"
  --normal_scale_ratio "${NORMAL_SCALE_RATIO}"
  --normal_scale_min "${NORMAL_SCALE_MIN}"
  --normal_scale_max "${NORMAL_SCALE_MAX}"
  --opacity_floor "${OPACITY_FLOOR}"
  --opacity_scale "${OPACITY_SCALE}"
  --opacity_power "${OPACITY_POWER}"
  --opacity_min "${OPACITY_MIN}"
  --opacity_max "${OPACITY_MAX}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  SPRAY_ARGS+=(--overwrite)
fi
if [[ "${WRITE_CPU_MERGED_PREVIEW}" == "1" ]]; then
  SPRAY_ARGS+=(--write_cpu_merged_preview)
fi

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/spray_2dgs_hf_carrier_to_3d_v0.py" "${SPRAY_ARGS[@]}"

NEWBORN_PLY="${NEWBORN_MODEL_DIR}/point_cloud/iteration_${BASE_ITERATION}/point_cloud.ply"
if [[ ! -f "${NEWBORN_PLY}" ]]; then
  echo "[spray-2dgs-to-3d-v0] newborn PLY not found after lift: ${NEWBORN_PLY}" >&2
  exit 1
fi

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}:${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${MERGE_SCRIPT}" ]]; then
  if [[ -f "${SOF_ROOT}/merge_gaussian_plys_v0.py" ]]; then
    MERGE_SCRIPT="${SOF_ROOT}/merge_gaussian_plys_v0.py"
  else
    MERGE_SCRIPT="${SOF_ROOT}/scripts/merge_gaussian_plys_v0.py"
  fi
fi
if [[ ! -f "${MERGE_SCRIPT}" ]]; then
  echo "[spray-2dgs-to-3d-v0] merge script not found: ${MERGE_SCRIPT}" >&2
  exit 1
fi
echo "[spray-2dgs-to-3d-v0] merge     : ${MERGE_SCRIPT}"
"${PYTHON_BIN}" "${MERGE_SCRIPT}" \
  --source_path "${SCENE_ROOT}" \
  --model_path "${OUTPUT_MODEL_DIR}" \
  --images images_2 \
  --resolution 1 \
  --eval \
  --base_ply "${BASE_PLY}" \
  --extra_ply "${NEWBORN_PLY}" \
  --copy_config_from "${BASE_MODEL_DIR}" \
  --output_model_path "${OUTPUT_MODEL_DIR}" \
  --output_iteration "${OUTPUT_ITERATION}"

echo "[spray-2dgs-to-3d-v0] done model:"
echo "  ${OUTPUT_MODEL_DIR}"
echo "[spray-2dgs-to-3d-v0] merged ply:"
echo "  ${OUTPUT_MODEL_DIR}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
echo "[spray-2dgs-to-3d-v0] tags:"
echo "  ${OUTPUT_MODEL_DIR}/point_cloud/iteration_${OUTPUT_ITERATION}/gaussian_tags.pt"
echo "[spray-2dgs-to-3d-v0] summary:"
echo "  ${OUTPUT_MODEL_DIR}/spray_2dgs_hf_carrier_to_3d_v0_summary.json"
