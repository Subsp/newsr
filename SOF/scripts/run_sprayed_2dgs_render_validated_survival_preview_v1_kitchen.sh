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

RUN_TAG="${RUN_TAG:-mip30k_rerun_check_directsrc_r1_v0_spray_2dgs_effective_hf_gaulayer_anchoradd_8view_v0}"
MODEL_DIR="${MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_2dgs_hf_spray_v0/${SCENE_NAME}/${RUN_TAG}}"
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/${BASE_EXPERIMENT_NAME}}"
GT_DIR="${GT_DIR:-${SCENE_ROOT}/images_2}"
ITERATION="${ITERATION:-30000}"
METRIC_GT_DIR="${METRIC_GT_DIR:-${BASE_MODEL_DIR}/train/ours_${ITERATION}/gt_1}"
if [[ ! -d "${METRIC_GT_DIR}" ]]; then
  METRIC_GT_DIR="${GT_DIR}"
fi
SPLIT="${SPLIT:-train}"
MAX_VIEWS="${MAX_VIEWS:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/spray_survival_preview/${RUN_TAG}_render_validated_v3}"
OVERWRITE="${OVERWRITE:-1}"

PRIOR_TAU_SCALE="${PRIOR_TAU_SCALE:-20.0}"
PRIOR_SCALE_MULTIPLIER="${PRIOR_SCALE_MULTIPLIER:-2.0}"
BOOST_TAU_SCALE="${BOOST_TAU_SCALE:-20.0}"
BOOST_SCALE_MULTIPLIER="${BOOST_SCALE_MULTIPLIER:-2.0}"
SOFT_PROBATION_TAU_SCALE="${SOFT_PROBATION_TAU_SCALE:-0.25}"
SOFT_PROBATION_DC_SCALE="${SOFT_PROBATION_DC_SCALE:-0.70}"
ENERGY_TAU_SCALE="${ENERGY_TAU_SCALE:-0.70}"
ENERGY_DC_SCALE="${ENERGY_DC_SCALE:-0.88}"

TARGET_COVERAGE_MULTIPLIER="${TARGET_COVERAGE_MULTIPLIER:-1.0}"
PROBATION_COVERAGE_MULTIPLIER="${PROBATION_COVERAGE_MULTIPLIER:-1.8}"
MIN_KEEP_FRACTION="${MIN_KEEP_FRACTION:-0.01}"
MAX_KEEP_FRACTION="${MAX_KEEP_FRACTION:-0.45}"
MIN_SCORE_FLOOR="${MIN_SCORE_FLOOR:-0.05}"
HIGHPASS_KERNEL="${HIGHPASS_KERNEL:-9}"
LOWPASS_KERNEL="${LOWPASS_KERNEL:-21}"
ORIENTATION_KERNEL="${ORIENTATION_KERNEL:-9}"
FOOTPRINT_LONG_SCALE="${FOOTPRINT_LONG_SCALE:-0.85}"
FOOTPRINT_SHORT_SCALE="${FOOTPRINT_SHORT_SCALE:-0.85}"
FOOTPRINT_MAX_RADIUS_PX="${FOOTPRINT_MAX_RADIUS_PX:-12.0}"
MIN_DIRECTION_ALIGN="${MIN_DIRECTION_ALIGN:-0.52}"
DIRECTION_PENALTY_WEIGHT="${DIRECTION_PENALTY_WEIGHT:-0.28}"
FOOTPRINT_LEAK_PENALTY_WEIGHT="${FOOTPRINT_LEAK_PENALTY_WEIGHT:-0.34}"
OWNER_DIRECTION_BINS="${OWNER_DIRECTION_BINS:-12}"
OWNER_TOP_BINS="${OWNER_TOP_BINS:-2}"
OWNER_MIN_GROUP_SIZE="${OWNER_MIN_GROUP_SIZE:-4}"
OWNER_PENALTY_WEIGHT="${OWNER_PENALTY_WEIGHT:-0.18}"

POINT_DIR="${MODEL_DIR}/point_cloud/iteration_${ITERATION}"
if [[ ! -f "${POINT_DIR}/point_cloud.ply" ]]; then
  echo "[render-validated-survival-v1] point cloud not found: ${POINT_DIR}/point_cloud.ply" >&2
  exit 1
fi
for required in "${BASE_MODEL_DIR}" "${GT_DIR}"; do
  if [[ ! -d "${required}" ]]; then
    echo "[render-validated-survival-v1] required path not found: ${required}" >&2
    exit 1
  fi
done
if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}:${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[render-validated-survival-v1] model  : ${MODEL_DIR}"
echo "[render-validated-survival-v1] base   : ${BASE_MODEL_DIR}"
echo "[render-validated-survival-v1] gt     : ${GT_DIR}"
echo "[render-validated-survival-v1] metric gt: ${METRIC_GT_DIR}"
echo "[render-validated-survival-v1] output : ${OUTPUT_ROOT}"
echo "[render-validated-survival-v1] source views: first ${MAX_VIEWS} ${SPLIT} views"

export_variant() {
  local model_path="$1"
  local export_root="$2"
  shift 2
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${model_path}" \
    --output_root "${export_root}" \
    --images_subdir images_2 \
    --iteration "${ITERATION}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}" \
    --view_select_mode first \
    "$@"
}

BASE_EXPORT="${OUTPUT_ROOT}/_base_source_variant"
MERGED_EXPORT="${OUTPUT_ROOT}/_merged_source_variant"
export_variant "${BASE_MODEL_DIR}" "${BASE_EXPORT}" \
  --selection_source lineage \
  --selection_key full \
  --selection_mode full
export_variant "${MODEL_DIR}" "${MERGED_EXPORT}" \
  --selection_source lineage \
  --selection_key full \
  --selection_mode full

BASE_RENDER_DIR="${BASE_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
MERGED_RENDER_DIR="${MERGED_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
PAYLOAD_DIR="${OUTPUT_ROOT}/survival_payload_v1"
PAYLOAD_PATH="${PAYLOAD_DIR}/sprayed_2dgs_render_validated_survival_payload_v1.pt"

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_sprayed_2dgs_render_validated_survival_payload_v1.py" \
  --model_dir "${MODEL_DIR}" \
  --iteration "${ITERATION}" \
  --base_render_dir "${BASE_RENDER_DIR}" \
  --merged_render_dir "${MERGED_RENDER_DIR}" \
  --gt_dir "${GT_DIR}" \
  --output_dir "${PAYLOAD_DIR}" \
  --limit_views "${MAX_VIEWS}" \
  --highpass_kernel "${HIGHPASS_KERNEL}" \
  --lowpass_kernel "${LOWPASS_KERNEL}" \
  --orientation_kernel "${ORIENTATION_KERNEL}" \
  --footprint_long_scale "${FOOTPRINT_LONG_SCALE}" \
  --footprint_short_scale "${FOOTPRINT_SHORT_SCALE}" \
  --footprint_max_radius_px "${FOOTPRINT_MAX_RADIUS_PX}" \
  --min_direction_align "${MIN_DIRECTION_ALIGN}" \
  --direction_penalty_weight "${DIRECTION_PENALTY_WEIGHT}" \
  --footprint_leak_penalty_weight "${FOOTPRINT_LEAK_PENALTY_WEIGHT}" \
  --owner_direction_bins "${OWNER_DIRECTION_BINS}" \
  --owner_top_bins "${OWNER_TOP_BINS}" \
  --owner_min_group_size "${OWNER_MIN_GROUP_SIZE}" \
  --owner_penalty_weight "${OWNER_PENALTY_WEIGHT}" \
  --target_coverage_multiplier "${TARGET_COVERAGE_MULTIPLIER}" \
  --probation_coverage_multiplier "${PROBATION_COVERAGE_MULTIPLIER}" \
  --min_keep_fraction "${MIN_KEEP_FRACTION}" \
  --max_keep_fraction "${MAX_KEEP_FRACTION}" \
  --min_score_floor "${MIN_SCORE_FLOOR}"

render_payload_variant() {
  local key="$1"
  local mode="$2"
  local out_name="$3"
  local tau="$4"
  local scale="$5"
  local post_mute_key="${6:-}"
  local weak_key="${7:-}"
  local weak_tau="${8:-1.0}"
  local weak_dc="${9:-1.0}"
  local export_root="${OUTPUT_ROOT}/_${out_name}_variant"
  local selected_count
  selected_count="$("${PYTHON_BIN}" -c 'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu"); print(int(p[sys.argv[2]].reshape(-1).sum().item()))' "${PAYLOAD_PATH}" "${key}")"
  if [[ "${mode}" == "selected_only" && "${selected_count}" == "0" ]]; then
    mkdir -p "${OUTPUT_ROOT}/${out_name}_${SPLIT}"
    echo "[render-validated-survival-v1] skip ${out_name}: ${key} selected zero gaussians"
    return
  fi
  local args=(
    --selection_source payload
    --mask_payload_path "${PAYLOAD_PATH}"
    --selection_key "${key}"
    --selection_mode "${mode}"
    --tau_scale "${tau}"
    --scale_multiplier "${scale}"
  )
  if [[ -n "${post_mute_key}" ]]; then
    args+=(
      --post_mute_selection_source payload
      --post_mute_mask_payload_path "${PAYLOAD_PATH}"
      --post_mute_selection_key "${post_mute_key}"
    )
  fi
  if [[ -n "${weak_key}" ]]; then
    args+=(
      --weak_selection_source payload
      --weak_mask_payload_path "${PAYLOAD_PATH}"
      --weak_selection_key "${weak_key}"
      --weak_tau_scale "${weak_tau}"
      --weak_dc_scale "${weak_dc}"
    )
  fi
  export_variant "${MODEL_DIR}" "${export_root}" "${args[@]}"
}

copy_renders() {
  local export_name="$1"
  local flat_name="$2"
  local render_dir="${OUTPUT_ROOT}/_${export_name}_variant/${SPLIT}/ours_${ITERATION}/renders"
  local flat_dir="${OUTPUT_ROOT}/${flat_name}"
  mkdir -p "${flat_dir}"
  shopt -s nullglob
  for image_path in "${render_dir}"/*.png; do
    cp "${image_path}" "${flat_dir}/"
  done
  shopt -u nullglob
}

mkdir -p "${OUTPUT_ROOT}/base_source_${SPLIT}" "${OUTPUT_ROOT}/merged_source_${SPLIT}"
cp "${BASE_RENDER_DIR}"/*.png "${OUTPUT_ROOT}/base_source_${SPLIT}/"
cp "${MERGED_RENDER_DIR}"/*.png "${OUTPUT_ROOT}/merged_source_${SPLIT}/"

render_payload_variant "survive_prior" "selected_only" "survive_prior" "${PRIOR_TAU_SCALE}" "${PRIOR_SCALE_MULTIPLIER}"
render_payload_variant "probation_prior" "selected_only" "probation_prior" "${PRIOR_TAU_SCALE}" "${PRIOR_SCALE_MULTIPLIER}"
render_payload_variant "suppress_prior" "selected_only" "suppress_prior" "${PRIOR_TAU_SCALE}" "${PRIOR_SCALE_MULTIPLIER}"
render_payload_variant "keep_prior" "full" "merged_survive" "1.0" "1.0" "drop_prior"
render_payload_variant "keep_prior" "full" "merged_survive_boost" "${BOOST_TAU_SCALE}" "${BOOST_SCALE_MULTIPLIER}" "drop_prior"
render_payload_variant "candidate_prior" "full" "merged_candidate_boost" "${BOOST_TAU_SCALE}" "${BOOST_SCALE_MULTIPLIER}" "drop_candidate_prior"
render_payload_variant "candidate_prior" "full" "merged_soft" "1.0" "1.0" "drop_candidate_prior" "probation_prior" "${SOFT_PROBATION_TAU_SCALE}" "${SOFT_PROBATION_DC_SCALE}"
render_payload_variant "keep_prior" "full" "merged_survive_energy" "1.0" "1.0" "drop_prior" "survive_prior" "${ENERGY_TAU_SCALE}" "${ENERGY_DC_SCALE}"
render_payload_variant "candidate_prior" "full" "merged_soft_energy" "${ENERGY_TAU_SCALE}" "1.0" "drop_candidate_prior" "probation_prior" "${SOFT_PROBATION_TAU_SCALE}" "${SOFT_PROBATION_DC_SCALE}"

copy_renders "survive_prior" "survive_prior_${SPLIT}"
copy_renders "probation_prior" "probation_prior_${SPLIT}"
copy_renders "suppress_prior" "suppress_prior_${SPLIT}"
copy_renders "merged_survive" "merged_survive_${SPLIT}"
copy_renders "merged_survive_boost" "merged_survive_boost_${SPLIT}"
copy_renders "merged_candidate_boost" "merged_candidate_boost_${SPLIT}"
copy_renders "merged_soft" "merged_soft_${SPLIT}"
copy_renders "merged_survive_energy" "merged_survive_energy_${SPLIT}"
copy_renders "merged_soft_energy" "merged_soft_energy_${SPLIT}"

METRICS_DIR="${OUTPUT_ROOT}/quality_metrics_v0"
"${PYTHON_BIN}" "${SOF_ROOT}/scripts/evaluate_spray_preview_quality_v0.py" \
  --base_dir "${OUTPUT_ROOT}/base_source_${SPLIT}" \
  --gt_dir "${METRIC_GT_DIR}" \
  --output_dir "${METRICS_DIR}" \
  --match_policy order \
  --limit "${MAX_VIEWS}" \
  --highpass_kernel "${HIGHPASS_KERNEL}" \
  --lowpass_kernel "${LOWPASS_KERNEL}" \
  --variant merged_source "${OUTPUT_ROOT}/merged_source_${SPLIT}" \
  --variant merged_survive "${OUTPUT_ROOT}/merged_survive_${SPLIT}" \
  --variant merged_soft "${OUTPUT_ROOT}/merged_soft_${SPLIT}" \
  --variant merged_survive_energy "${OUTPUT_ROOT}/merged_survive_energy_${SPLIT}" \
  --variant merged_soft_energy "${OUTPUT_ROOT}/merged_soft_energy_${SPLIT}" \
  --overwrite

cat > "${OUTPUT_ROOT}/README.txt" <<EOF
Render-validated sprayed 2DGS HF survival preview v3.

This variant adds footprint hit/leak, image tangent direction agreement,
per-owner direction-mode arbitration, soft probation rendering, and quality metrics.

model: ${MODEL_DIR}
base: ${BASE_MODEL_DIR}
gt: ${GT_DIR}
metric gt: ${METRIC_GT_DIR}
iteration: ${ITERATION}
split: ${SPLIT}
source views: first ${MAX_VIEWS}
payload: ${PAYLOAD_PATH}

Inspect:
  ${OUTPUT_ROOT}/survival_payload_v1/summary.json
  ${OUTPUT_ROOT}/base_source_${SPLIT}
  ${OUTPUT_ROOT}/merged_source_${SPLIT}
  ${OUTPUT_ROOT}/survive_prior_${SPLIT}
  ${OUTPUT_ROOT}/probation_prior_${SPLIT}
  ${OUTPUT_ROOT}/suppress_prior_${SPLIT}
  ${OUTPUT_ROOT}/merged_survive_${SPLIT}
  ${OUTPUT_ROOT}/merged_survive_boost_${SPLIT}
  ${OUTPUT_ROOT}/merged_candidate_boost_${SPLIT}
  ${OUTPUT_ROOT}/merged_soft_${SPLIT}
  ${OUTPUT_ROOT}/merged_survive_energy_${SPLIT}
  ${OUTPUT_ROOT}/merged_soft_energy_${SPLIT}
  ${METRICS_DIR}/summary.json
EOF

echo "[render-validated-survival-v1] shallow outputs:"
echo "  ${OUTPUT_ROOT}/survival_payload_v1/summary.json"
echo "  ${OUTPUT_ROOT}/base_source_${SPLIT}"
echo "  ${OUTPUT_ROOT}/merged_source_${SPLIT}"
echo "  ${OUTPUT_ROOT}/survive_prior_${SPLIT}"
echo "  ${OUTPUT_ROOT}/probation_prior_${SPLIT}"
echo "  ${OUTPUT_ROOT}/suppress_prior_${SPLIT}"
echo "  ${OUTPUT_ROOT}/merged_survive_${SPLIT}"
echo "  ${OUTPUT_ROOT}/merged_survive_boost_${SPLIT}"
echo "  ${OUTPUT_ROOT}/merged_candidate_boost_${SPLIT}"
echo "  ${OUTPUT_ROOT}/merged_soft_${SPLIT}"
echo "  ${OUTPUT_ROOT}/merged_survive_energy_${SPLIT}"
echo "  ${OUTPUT_ROOT}/merged_soft_energy_${SPLIT}"
echo "  ${METRICS_DIR}/summary.json"
echo "  ${OUTPUT_ROOT}/README.txt"
