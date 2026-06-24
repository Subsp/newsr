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

OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/2dgs_posterior_expressibility_oracle_v0/${INPUT_NAME}}"
CHECK_ROOT="${CHECK_ROOT:-${WORK_ROOT}/check/2dgs_posterior_oracle/${INPUT_NAME}}"
TARGET_DIR="${TARGET_DIR:-${BASE_MODEL_DIR}/train/ours_${ITERATION}/gt_1}"
MATCH_POLICY="${MATCH_POLICY:-stem}"

ORACLE_VARIANTS="${ORACLE_VARIANTS:-tau1_s1:1:1 tau4_s1:4:1 tau8_s1:8:1 tau4_s2:4:2 tau8_s2:8:2}"
TARGET_THRESHOLD="${TARGET_THRESHOLD:-0.18}"
GEOM_THRESHOLD="${GEOM_THRESHOLD:-0.08}"
REACHABLE_DILATE="${REACHABLE_DILATE:-2}"
LEAK_WEIGHT="${LEAK_WEIGHT:-0.15}"
DEBUG_LIMIT="${DEBUG_LIMIT:-8}"
SAVE_VARIANT_MODEL="${SAVE_VARIANT_MODEL:-0}"

BASE_PLY="${BASE_MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
INPUT_PLY="${INPUT_MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply"

for required in "${BASE_MODEL_DIR}" "${BASE_PLY}" "${INPUT_MODEL_DIR}" "${INPUT_PLY}" "${TARGET_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[2dgs-express-oracle-v0] required path not found: ${required}" >&2
    exit 1
  fi
done

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_ROOT}" "${CHECK_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}" "${CHECK_ROOT}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}:${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[2dgs-express-oracle-v0] base    : ${BASE_MODEL_DIR}"
echo "[2dgs-express-oracle-v0] append  : ${INPUT_MODEL_DIR}"
echo "[2dgs-express-oracle-v0] target  : ${TARGET_DIR}"
echo "[2dgs-express-oracle-v0] output  : ${OUTPUT_ROOT}"
echo "[2dgs-express-oracle-v0] check   : ${CHECK_ROOT}"
echo "[2dgs-express-oracle-v0] views   : ${SPLIT}/${MAX_VIEWS} mode=${VIEW_SELECT_MODE}"
echo "[2dgs-express-oracle-v0] variants: ${ORACLE_VARIANTS}"

export_variant() {
  local output_root="$1"
  local model_path="$2"
  local source="$3"
  local key="$4"
  local mode="$5"
  local tau="$6"
  local scale="$7"
  shift 7
  local export_extra_args=()
  if [[ "${SAVE_VARIANT_MODEL}" != "1" ]]; then
    export_extra_args+=(--no_save_variant_model)
  fi
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
    "${export_extra_args[@]}" \
    "$@"
}

copy_renders() {
  local src="$1"
  local dst="$2"
  mkdir -p "${dst}"
  shopt -s nullglob
  for image_path in "${src}"/*.png; do
    cp "${image_path}" "${dst}/"
  done
  shopt -u nullglob
}

BASE_EXPORT="${OUTPUT_ROOT}/_R00_base"
APPEND_EXPORT="${OUTPUT_ROOT}/_R10_append"

echo "[2dgs-express-oracle-v0] render R00 base"
export_variant "${BASE_EXPORT}" "${BASE_MODEL_DIR}" lineage full full 1.0 1.0
BASE_RENDER_DIR="${BASE_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
copy_renders "${BASE_RENDER_DIR}" "${CHECK_ROOT}/R00_base"

echo "[2dgs-express-oracle-v0] render R10 append"
export_variant "${APPEND_EXPORT}" "${INPUT_MODEL_DIR}" tracking full full 1.0 1.0
APPEND_RENDER_DIR="${APPEND_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
copy_renders "${APPEND_RENDER_DIR}" "${CHECK_ROOT}/R10_append"

read -r -a VARIANT_SPECS <<< "${ORACLE_VARIANTS}"
EVAL_VARIANT_ARGS=()
for spec in "${VARIANT_SPECS[@]}"; do
  IFS=":" read -r name tau scale <<< "${spec}"
  if [[ -z "${name:-}" || -z "${tau:-}" || -z "${scale:-}" ]]; then
    echo "[2dgs-express-oracle-v0] invalid variant spec: ${spec}" >&2
    exit 1
  fi
  export_dir="${OUTPUT_ROOT}/_prior_${name}"
  echo "[2dgs-express-oracle-v0] render prior ${name} tau=${tau} scale=${scale}"
  export_variant \
    "${export_dir}" \
    "${INPUT_MODEL_DIR}" \
    tracking \
    prior_injected \
    selected_only \
    "${tau}" \
    "${scale}" \
    --save_alpha
  render_dir="${export_dir}/${SPLIT}/ours_${ITERATION}/renders"
  alpha_dir="${export_dir}/${SPLIT}/ours_${ITERATION}/alpha"
  check_render="${CHECK_ROOT}/prior_${name}"
  check_alpha="${CHECK_ROOT}/alpha_${name}"
  copy_renders "${render_dir}" "${check_render}"
  copy_renders "${alpha_dir}" "${check_alpha}"
  EVAL_VARIANT_ARGS+=(--variant "${name}" "${check_render}" "${check_alpha}")
done

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/evaluate_2dgs_expressibility_oracle_v0.py" \
  --base_dir "${CHECK_ROOT}/R00_base" \
  --append_dir "${CHECK_ROOT}/R10_append" \
  --target_dir "${TARGET_DIR}" \
  --output_dir "${CHECK_ROOT}/expressibility_metrics_v0" \
  --match_policy "${MATCH_POLICY}" \
  --target_threshold "${TARGET_THRESHOLD}" \
  --geom_threshold "${GEOM_THRESHOLD}" \
  --reachable_dilate "${REACHABLE_DILATE}" \
  --leak_weight "${LEAK_WEIGHT}" \
  --debug_limit "${DEBUG_LIMIT}" \
  --overwrite \
  "${EVAL_VARIANT_ARGS[@]}"

cat > "${CHECK_ROOT}/README.txt" <<EOF
2DGS posterior expressibility oracle.

Base render:
  ${CHECK_ROOT}/R00_base
Append render:
  ${CHECK_ROOT}/R10_append
Newborn-only prior variants:
  ${ORACLE_VARIANTS}

Main summary:
  ${CHECK_ROOT}/expressibility_metrics_v0/summary.json
Debug images:
  ${CHECK_ROOT}/expressibility_metrics_v0/debug

Interpretation:
  geom_support_recall tells whether newborn projected support reaches target HF.
  append_* tells whether the actually appended newborn is visible.
  alpha_rgb_oracle_* asks whether the alpha footprint alone could fit signed HF with optimal RGB.
  prior_hf_rgb_oracle_* asks whether the rendered newborn HF pattern could fit signed HF after optimal channel scaling.
EOF

echo "[2dgs-express-oracle-v0] shallow outputs:"
echo "  ${CHECK_ROOT}/README.txt"
echo "  ${CHECK_ROOT}/R00_base"
echo "  ${CHECK_ROOT}/R10_append"
echo "  ${CHECK_ROOT}/prior_*"
echo "  ${CHECK_ROOT}/alpha_*"
echo "  ${CHECK_ROOT}/expressibility_metrics_v0/summary.json"
echo "  ${CHECK_ROOT}/expressibility_metrics_v0/debug"
