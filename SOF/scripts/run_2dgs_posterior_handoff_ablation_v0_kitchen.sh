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
INPUT_NAME="${INPUT_NAME:-mip30k_lr30000_2dgs_posterior_v0}"
INPUT_MODEL_DIR="${INPUT_MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_2dgs_posterior_spray_v0/${SCENE_NAME}/${INPUT_NAME}}"
ITERATION="${ITERATION:-30000}"
SPLIT="${SPLIT:-train}"
MAX_VIEWS="${MAX_VIEWS:-8}"
VIEW_SELECT_MODE="${VIEW_SELECT_MODE:-first}"
OVERWRITE="${OVERWRITE:-0}"

FRONT_NAME="${FRONT_NAME:-${INPUT_NAME}_front_offset_v0}"
HANDOFF_NAME="${HANDOFF_NAME:-${INPUT_NAME}_handoff_v0}"
FRONT_MODEL_DIR="${FRONT_MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_2dgs_posterior_integrated_v0/${SCENE_NAME}/${FRONT_NAME}}"
HANDOFF_MODEL_DIR="${HANDOFF_MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_2dgs_posterior_integrated_v0/${SCENE_NAME}/${HANDOFF_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/2dgs_posterior_handoff_ablation_v0/${INPUT_NAME}}"
CHECK_ROOT="${CHECK_ROOT:-${WORK_ROOT}/check/2dgs_posterior_handoff/${INPUT_NAME}}"

METADATA_PATH="${METADATA_PATH:-${INPUT_MODEL_DIR}/point_cloud/iteration_${ITERATION}/sprayed_2dgs_posterior_metadata_v0.npz}"
GT_DIR="${GT_DIR:-${BASE_MODEL_DIR}/train/ours_${ITERATION}/gt_1}"
RUN_QUALITY_METRICS="${RUN_QUALITY_METRICS:-1}"

FRONT_OFFSET="${FRONT_OFFSET:-0.0015}"
PARENT_TAU_FRACTION_MAX="${PARENT_TAU_FRACTION_MAX:-0.12}"
PARENT_BUDGET_SCALE="${PARENT_BUDGET_SCALE:-0.18}"
NEWBORN_TAU_FROM_BUDGET="${NEWBORN_TAU_FROM_BUDGET:-1.0}"
NEWBORN_TAU_SCALE="${NEWBORN_TAU_SCALE:-1.0}"
NEWBORN_ALPHA_FLOOR="${NEWBORN_ALPHA_FLOOR:-0.004}"
NEWBORN_ALPHA_MAX="${NEWBORN_ALPHA_MAX:-0.10}"
NEWBORN_SCALE_MULTIPLIER="${NEWBORN_SCALE_MULTIPLIER:-1.0}"
NEWBORN_SCALE_MAX="${NEWBORN_SCALE_MAX:-0.012}"

PRIOR_BOOST_TAU_SCALE="${PRIOR_BOOST_TAU_SCALE:-20.0}"
PRIOR_BOOST_SCALE_MULTIPLIER="${PRIOR_BOOST_SCALE_MULTIPLIER:-2.0}"
DELTA_VIS_SCALE="${DELTA_VIS_SCALE:-80.0}"
TARGET_THRESHOLD="${TARGET_THRESHOLD:-0.18}"

BASE_PLY="${BASE_MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
INPUT_PLY="${INPUT_MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply"

for required in "${BASE_MODEL_DIR}" "${BASE_PLY}" "${INPUT_MODEL_DIR}" "${INPUT_PLY}" "${METADATA_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[2dgs-handoff-ablation-v0] required path not found: ${required}" >&2
    exit 1
  fi
done

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${FRONT_MODEL_DIR}" "${HANDOFF_MODEL_DIR}" "${OUTPUT_ROOT}" "${CHECK_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}" "${CHECK_ROOT}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}:${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[2dgs-handoff-ablation-v0] base     : ${BASE_MODEL_DIR}"
echo "[2dgs-handoff-ablation-v0] append   : ${INPUT_MODEL_DIR}"
echo "[2dgs-handoff-ablation-v0] front    : ${FRONT_MODEL_DIR}"
echo "[2dgs-handoff-ablation-v0] handoff  : ${HANDOFF_MODEL_DIR}"
echo "[2dgs-handoff-ablation-v0] output   : ${OUTPUT_ROOT}"
echo "[2dgs-handoff-ablation-v0] check    : ${CHECK_ROOT}"
echo "[2dgs-handoff-ablation-v0] views    : ${SPLIT}/${MAX_VIEWS} mode=${VIEW_SELECT_MODE}"

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/integrate_2dgs_posterior_handoff_v0.py" \
  --input_model_dir "${INPUT_MODEL_DIR}" \
  --output_model_dir "${FRONT_MODEL_DIR}" \
  --iteration "${ITERATION}" \
  --metadata_path "${METADATA_PATH}" \
  --mode front_offset \
  --front_offset "${FRONT_OFFSET}" \
  --newborn_scale_multiplier "${NEWBORN_SCALE_MULTIPLIER}" \
  --newborn_scale_max "${NEWBORN_SCALE_MAX}" \
  --overwrite

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/integrate_2dgs_posterior_handoff_v0.py" \
  --input_model_dir "${INPUT_MODEL_DIR}" \
  --output_model_dir "${HANDOFF_MODEL_DIR}" \
  --iteration "${ITERATION}" \
  --metadata_path "${METADATA_PATH}" \
  --mode handoff \
  --front_offset "${FRONT_OFFSET}" \
  --parent_tau_fraction_max "${PARENT_TAU_FRACTION_MAX}" \
  --parent_budget_scale "${PARENT_BUDGET_SCALE}" \
  --newborn_tau_from_budget "${NEWBORN_TAU_FROM_BUDGET}" \
  --newborn_tau_scale "${NEWBORN_TAU_SCALE}" \
  --newborn_alpha_floor "${NEWBORN_ALPHA_FLOOR}" \
  --newborn_alpha_max "${NEWBORN_ALPHA_MAX}" \
  --newborn_scale_multiplier "${NEWBORN_SCALE_MULTIPLIER}" \
  --newborn_scale_max "${NEWBORN_SCALE_MAX}" \
  --overwrite

export_variant() {
  local output_root="$1"
  local model_path="$2"
  local source="$3"
  local key="$4"
  local mode="$5"
  local tau="$6"
  local scale="$7"
  shift 7
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${model_path}" \
    --output_root "${output_root}" \
    --images_subdir images_2 \
    --iteration "${ITERATION}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}" \
    --view_select_mode "${VIEW_SELECT_MODE}" \
    --selection_source "${source}" \
    --selection_key "${key}" \
    --selection_mode "${mode}" \
    --tau_scale "${tau}" \
    --scale_multiplier "${scale}" \
    "$@"
}

R00_EXPORT="${OUTPUT_ROOT}/_R00_base"
R10_EXPORT="${OUTPUT_ROOT}/_R10_append"
FRONT_EXPORT="${OUTPUT_ROOT}/_front_offset"
R01_EXPORT="${OUTPUT_ROOT}/_R01_parent_only"
R11_EXPORT="${OUTPUT_ROOT}/_R11_handoff"
PRIOR_BOOST_EXPORT="${OUTPUT_ROOT}/_handoff_prior_boost"

echo "[2dgs-handoff-ablation-v0] render R00 base"
export_variant "${R00_EXPORT}" "${BASE_MODEL_DIR}" lineage full full 1.0 1.0

echo "[2dgs-handoff-ablation-v0] render R10 append"
export_variant "${R10_EXPORT}" "${INPUT_MODEL_DIR}" tracking full full 1.0 1.0

echo "[2dgs-handoff-ablation-v0] render front_offset"
export_variant "${FRONT_EXPORT}" "${FRONT_MODEL_DIR}" tracking full full 1.0 1.0

echo "[2dgs-handoff-ablation-v0] render R01 modified_base_only"
export_variant "${R01_EXPORT}" "${HANDOFF_MODEL_DIR}" tracking prior_injected selected_removed 1.0 1.0

echo "[2dgs-handoff-ablation-v0] render R11 handoff"
export_variant "${R11_EXPORT}" "${HANDOFF_MODEL_DIR}" tracking full full 1.0 1.0

echo "[2dgs-handoff-ablation-v0] render handoff prior boost"
export_variant \
  "${PRIOR_BOOST_EXPORT}" \
  "${HANDOFF_MODEL_DIR}" \
  tracking \
  prior_injected \
  selected_only \
  "${PRIOR_BOOST_TAU_SCALE}" \
  "${PRIOR_BOOST_SCALE_MULTIPLIER}" \
  --save_alpha

R00_DIR="${R00_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
R10_DIR="${R10_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
FRONT_DIR="${FRONT_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
R01_DIR="${R01_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
R11_DIR="${R11_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
PRIOR_BOOST_DIR="${PRIOR_BOOST_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
PRIOR_ALPHA_DIR="${PRIOR_BOOST_EXPORT}/${SPLIT}/ours_${ITERATION}/alpha"

mkdir -p \
  "${CHECK_ROOT}/R00_base" \
  "${CHECK_ROOT}/R10_append" \
  "${CHECK_ROOT}/front_offset" \
  "${CHECK_ROOT}/R01_parent_only" \
  "${CHECK_ROOT}/R11_handoff" \
  "${CHECK_ROOT}/handoff_prior_boost" \
  "${CHECK_ROOT}/handoff_prior_alpha"

shopt -s nullglob
for image_path in "${R00_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/R00_base/"; done
for image_path in "${R10_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/R10_append/"; done
for image_path in "${FRONT_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/front_offset/"; done
for image_path in "${R01_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/R01_parent_only/"; done
for image_path in "${R11_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/R11_handoff/"; done
for image_path in "${PRIOR_BOOST_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/handoff_prior_boost/"; done
for image_path in "${PRIOR_ALPHA_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/handoff_prior_alpha/"; done
shopt -u nullglob

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/evaluate_2dgs_handoff_causal_v0.py" \
  --r00_dir "${CHECK_ROOT}/R00_base" \
  --r10_dir "${CHECK_ROOT}/R10_append" \
  --r01_dir "${CHECK_ROOT}/R01_parent_only" \
  --r11_dir "${CHECK_ROOT}/R11_handoff" \
  --gt_dir "${GT_DIR}" \
  --output_dir "${CHECK_ROOT}/causal_metrics" \
  --match_policy stem \
  --vis_scale "${DELTA_VIS_SCALE}" \
  --target_threshold "${TARGET_THRESHOLD}" \
  --overwrite

if [[ "${RUN_QUALITY_METRICS}" == "1" && -d "${GT_DIR}" ]]; then
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/evaluate_spray_preview_quality_v0.py" \
    --base_dir "${CHECK_ROOT}/R00_base" \
    --gt_dir "${GT_DIR}" \
    --variant append "${CHECK_ROOT}/R10_append" \
    --variant front_offset "${CHECK_ROOT}/front_offset" \
    --variant parent_only "${CHECK_ROOT}/R01_parent_only" \
    --variant handoff "${CHECK_ROOT}/R11_handoff" \
    --output_dir "${CHECK_ROOT}/quality_metrics" \
    --match_policy order \
    --target_threshold "${TARGET_THRESHOLD}" \
    --overwrite
fi

cat > "${CHECK_ROOT}/README.txt" <<EOF
2DGS posterior handoff ablation.

R00 = base only:
  ${CHECK_ROOT}/R00_base
R10 = append-only merged model:
  ${CHECK_ROOT}/R10_append
front_offset = newborn moved along recovered normal, no handoff:
  ${CHECK_ROOT}/front_offset
R01 = modified base only, newborn muted:
  ${CHECK_ROOT}/R01_parent_only
R11 = modified base + newborn:
  ${CHECK_ROOT}/R11_handoff
handoff newborn boosted for localization only:
  ${CHECK_ROOT}/handoff_prior_boost
  ${CHECK_ROOT}/handoff_prior_alpha

Main causal metrics:
  ${CHECK_ROOT}/causal_metrics/summary.json
  ${CHECK_ROOT}/causal_metrics/append_R10_minus_R00_x
  ${CHECK_ROOT}/causal_metrics/parent_R01_minus_R00_x
  ${CHECK_ROOT}/causal_metrics/newborn_R11_minus_R01_x
  ${CHECK_ROOT}/causal_metrics/total_R11_minus_R00_x
  ${CHECK_ROOT}/causal_metrics/interaction_x

Quality metrics:
  ${CHECK_ROOT}/quality_metrics/summary.json

Generated models:
  front : ${FRONT_MODEL_DIR}
  handoff: ${HANDOFF_MODEL_DIR}
EOF

echo "[2dgs-handoff-ablation-v0] shallow outputs:"
echo "  ${CHECK_ROOT}/README.txt"
echo "  ${CHECK_ROOT}/R00_base"
echo "  ${CHECK_ROOT}/R10_append"
echo "  ${CHECK_ROOT}/front_offset"
echo "  ${CHECK_ROOT}/R01_parent_only"
echo "  ${CHECK_ROOT}/R11_handoff"
echo "  ${CHECK_ROOT}/handoff_prior_boost"
echo "  ${CHECK_ROOT}/causal_metrics/summary.json"
echo "  ${CHECK_ROOT}/quality_metrics/summary.json"
