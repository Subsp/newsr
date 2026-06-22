#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting}"

RUN_TAG="${RUN_TAG:-mip30k_rerun_check_directsrc_r1_v0_spray_2dgs_effective_hf_gaulayer_anchoradd_8view_v0}"
MODEL_DIR="${MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_2dgs_hf_spray_v0/${SCENE_NAME}/${RUN_TAG}}"
ITERATION="${ITERATION:-30000}"
SPLIT="${SPLIT:-train}"
MAX_VIEWS="${MAX_VIEWS:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/spray_survival_preview/${RUN_TAG}}"
OVERWRITE="${OVERWRITE:-1}"

PRIOR_TAU_SCALE="${PRIOR_TAU_SCALE:-20.0}"
PRIOR_SCALE_MULTIPLIER="${PRIOR_SCALE_MULTIPLIER:-2.0}"
BOOST_TAU_SCALE="${BOOST_TAU_SCALE:-20.0}"
BOOST_SCALE_MULTIPLIER="${BOOST_SCALE_MULTIPLIER:-2.0}"

SURVIVE_MIN_SCORE="${SURVIVE_MIN_SCORE:-0.46}"
PROBATION_MIN_SCORE="${PROBATION_MIN_SCORE:-0.26}"
SUPPRESS_MIN_SCORE="${SUPPRESS_MIN_SCORE:-0.11}"
SOURCE_MIN_SURVIVE="${SOURCE_MIN_SURVIVE:-0.11}"
SOURCE_MIN_PROBATION="${SOURCE_MIN_PROBATION:-0.055}"
MIN_GROUP_VIEWS_SURVIVE="${MIN_GROUP_VIEWS_SURVIVE:-2}"
MIN_GROUP_MEMBERS_SURVIVE="${MIN_GROUP_MEMBERS_SURVIVE:-4}"
BAD_DISTANCE_PX="${BAD_DISTANCE_PX:-4.0}"
DISTANCE_SIGMA_PX="${DISTANCE_SIGMA_PX:-1.75}"
RISK_OPACITY="${RISK_OPACITY:-0.12}"
RISK_SCALE_LONG="${RISK_SCALE_LONG:-0.010}"

POINT_DIR="${MODEL_DIR}/point_cloud/iteration_${ITERATION}"
if [[ ! -f "${POINT_DIR}/point_cloud.ply" ]]; then
  echo "[spray-survival-preview-v0] point cloud not found: ${POINT_DIR}/point_cloud.ply" >&2
  exit 1
fi
if [[ ! -f "${POINT_DIR}/gaussian_tags.pt" ]]; then
  echo "[spray-survival-preview-v0] tags not found: ${POINT_DIR}/gaussian_tags.pt" >&2
  exit 1
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}:${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PAYLOAD_DIR="${OUTPUT_ROOT}/survival_payload_v0"
PAYLOAD_PATH="${PAYLOAD_DIR}/sprayed_2dgs_survival_payload_v0.pt"

echo "[spray-survival-preview-v0] model  : ${MODEL_DIR}"
echo "[spray-survival-preview-v0] output : ${OUTPUT_ROOT}"
echo "[spray-survival-preview-v0] split  : ${SPLIT} views=${MAX_VIEWS}"

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_sprayed_2dgs_survival_payload_v0.py" \
  --model_dir "${MODEL_DIR}" \
  --iteration "${ITERATION}" \
  --output_dir "${PAYLOAD_DIR}" \
  --distance_sigma_px "${DISTANCE_SIGMA_PX}" \
  --bad_distance_px "${BAD_DISTANCE_PX}" \
  --min_group_views_survive "${MIN_GROUP_VIEWS_SURVIVE}" \
  --min_group_members_survive "${MIN_GROUP_MEMBERS_SURVIVE}" \
  --source_min_survive "${SOURCE_MIN_SURVIVE}" \
  --source_min_probation "${SOURCE_MIN_PROBATION}" \
  --survive_min_score "${SURVIVE_MIN_SCORE}" \
  --probation_min_score "${PROBATION_MIN_SCORE}" \
  --suppress_min_score "${SUPPRESS_MIN_SCORE}" \
  --risk_opacity "${RISK_OPACITY}" \
  --risk_scale_long "${RISK_SCALE_LONG}"

render_variant() {
  local key="$1"
  local mode="$2"
  local out_name="$3"
  local tau="$4"
  local scale="$5"
  local post_mute_key="${6:-}"
  local export_root="${OUTPUT_ROOT}/_${out_name}_variant"
  local selected_count
  selected_count="$("${PYTHON_BIN}" -c 'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu"); print(int(p[sys.argv[2]].reshape(-1).sum().item()))' "${PAYLOAD_PATH}" "${key}")"
  if [[ "${mode}" == "selected_only" && "${selected_count}" == "0" ]]; then
    mkdir -p "${OUTPUT_ROOT}/${out_name}_${SPLIT}"
    echo "[spray-survival-preview-v0] skip ${out_name}: ${key} selected zero gaussians"
    return
  fi

  local args=(
    --scene_root "${SCENE_ROOT}"
    --model_path "${MODEL_DIR}"
    --output_root "${export_root}"
    --images_subdir images_2
    --iteration "${ITERATION}"
    --split "${SPLIT}"
    --max_views "${MAX_VIEWS}"
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
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py" "${args[@]}"
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

render_variant "survive_prior" "selected_only" "survive_prior" "${PRIOR_TAU_SCALE}" "${PRIOR_SCALE_MULTIPLIER}"
render_variant "probation_prior" "selected_only" "probation_prior" "${PRIOR_TAU_SCALE}" "${PRIOR_SCALE_MULTIPLIER}"
render_variant "suppress_prior" "selected_only" "suppress_prior" "${PRIOR_TAU_SCALE}" "${PRIOR_SCALE_MULTIPLIER}"
render_variant "keep_prior" "full" "merged_keep" "1.0" "1.0" "drop_prior"
render_variant "keep_prior" "full" "merged_keep_boost" "${BOOST_TAU_SCALE}" "${BOOST_SCALE_MULTIPLIER}" "drop_prior"

copy_renders "survive_prior" "survive_prior_${SPLIT}"
copy_renders "probation_prior" "probation_prior_${SPLIT}"
copy_renders "suppress_prior" "suppress_prior_${SPLIT}"
copy_renders "merged_keep" "merged_keep_${SPLIT}"
copy_renders "merged_keep_boost" "merged_keep_boost_${SPLIT}"

cat > "${OUTPUT_ROOT}/README.txt" <<EOF
Sprayed 2DGS HF survival preview.

model: ${MODEL_DIR}
iteration: ${ITERATION}
split: ${SPLIT}
max_views: ${MAX_VIEWS}
payload: ${PAYLOAD_PATH}

Inspect:
  ${OUTPUT_ROOT}/survival_payload_v0/summary.json
  ${OUTPUT_ROOT}/survive_prior_${SPLIT}
  ${OUTPUT_ROOT}/probation_prior_${SPLIT}
  ${OUTPUT_ROOT}/suppress_prior_${SPLIT}
  ${OUTPUT_ROOT}/merged_keep_${SPLIT}
  ${OUTPUT_ROOT}/merged_keep_boost_${SPLIT}
EOF

echo "[spray-survival-preview-v0] shallow outputs:"
echo "  ${OUTPUT_ROOT}/survival_payload_v0/summary.json"
echo "  ${OUTPUT_ROOT}/survive_prior_${SPLIT}"
echo "  ${OUTPUT_ROOT}/probation_prior_${SPLIT}"
echo "  ${OUTPUT_ROOT}/suppress_prior_${SPLIT}"
echo "  ${OUTPUT_ROOT}/merged_keep_${SPLIT}"
echo "  ${OUTPUT_ROOT}/merged_keep_boost_${SPLIT}"
echo "  ${OUTPUT_ROOT}/README.txt"
