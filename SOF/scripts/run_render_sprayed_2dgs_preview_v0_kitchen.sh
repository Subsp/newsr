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

RUN_TAG="${RUN_TAG:-mip30k_rerun_check_directsrc_r1_v0_spray_2dgs_effective_hf_one_v0}"
MODEL_DIR="${MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_2dgs_hf_spray_v0/${SCENE_NAME}/${RUN_TAG}}"
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/${BASE_EXPERIMENT_NAME}}"
ITERATION="${ITERATION:-30000}"
SPLIT="${SPLIT:-train}"
MAX_VIEWS="${MAX_VIEWS:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/spray_preview/${RUN_TAG}}"
OVERWRITE="${OVERWRITE:-0}"
RUN_BASE_CONTROL="${RUN_BASE_CONTROL:-1}"
RUN_MERGED_PRIOR_BOOST="${RUN_MERGED_PRIOR_BOOST:-1}"
RUN_MERGED_BASE_DELTA="${RUN_MERGED_BASE_DELTA:-1}"

PRIOR_TAU_SCALE="${PRIOR_TAU_SCALE:-20.0}"
PRIOR_SCALE_MULTIPLIER="${PRIOR_SCALE_MULTIPLIER:-2.0}"
PRIOR_FILTER_MULTIPLIER="${PRIOR_FILTER_MULTIPLIER:-1.0}"
BOOST_TAU_SCALE="${BOOST_TAU_SCALE:-20.0}"
BOOST_SCALE_MULTIPLIER="${BOOST_SCALE_MULTIPLIER:-2.0}"
BOOST_FILTER_MULTIPLIER="${BOOST_FILTER_MULTIPLIER:-1.0}"
DELTA_VIS_SCALE="${DELTA_VIS_SCALE:-30.0}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[spray-preview-v0] model dir not found: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" ]]; then
  echo "[spray-preview-v0] point cloud not found: ${MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" >&2
  exit 1
fi
if [[ "${RUN_BASE_CONTROL}" == "1" && ! -f "${BASE_MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" ]]; then
  echo "[spray-preview-v0] base point cloud not found: ${BASE_MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" >&2
  exit 1
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}:${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PRIOR_EXPORT="${OUTPUT_ROOT}/_prior_injected_variant"
MERGED_EXPORT="${OUTPUT_ROOT}/_merged_full_variant"
BASE_EXPORT="${OUTPUT_ROOT}/_base_full_variant"
BOOST_EXPORT="${OUTPUT_ROOT}/_merged_prior_boost_variant"

echo "[spray-preview-v0] model      : ${MODEL_DIR}"
echo "[spray-preview-v0] base model : ${BASE_MODEL_DIR}"
echo "[spray-preview-v0] output     : ${OUTPUT_ROOT}"
echo "[spray-preview-v0] split/views : ${SPLIT}/${MAX_VIEWS}"
echo "[spray-preview-v0] prior vis  : tau=${PRIOR_TAU_SCALE} scale=${PRIOR_SCALE_MULTIPLIER}"
echo "[spray-preview-v0] boost vis  : tau=${BOOST_TAU_SCALE} scale=${BOOST_SCALE_MULTIPLIER}"

if [[ "${RUN_BASE_CONTROL}" == "1" ]]; then
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${BASE_MODEL_DIR}" \
    --output_root "${BASE_EXPORT}" \
    --images_subdir images_2 \
    --iteration "${ITERATION}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}" \
    --selection_source lineage \
    --selection_key full \
    --selection_mode full
fi

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --model_path "${MODEL_DIR}" \
  --output_root "${PRIOR_EXPORT}" \
  --images_subdir images_2 \
  --iteration "${ITERATION}" \
  --split "${SPLIT}" \
  --max_views "${MAX_VIEWS}" \
  --selection_source tracking \
  --selection_key prior_injected \
  --selection_mode selected_only \
  --tau_scale "${PRIOR_TAU_SCALE}" \
  --scale_multiplier "${PRIOR_SCALE_MULTIPLIER}" \
  --filter_multiplier "${PRIOR_FILTER_MULTIPLIER}" \
  --save_alpha

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --model_path "${MODEL_DIR}" \
  --output_root "${MERGED_EXPORT}" \
  --images_subdir images_2 \
  --iteration "${ITERATION}" \
  --split "${SPLIT}" \
  --max_views "${MAX_VIEWS}" \
  --selection_source tracking \
  --selection_key full \
  --selection_mode full

if [[ "${RUN_MERGED_PRIOR_BOOST}" == "1" ]]; then
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${MODEL_DIR}" \
    --output_root "${BOOST_EXPORT}" \
    --images_subdir images_2 \
    --iteration "${ITERATION}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}" \
    --selection_source tracking \
    --selection_key prior_injected \
    --selection_mode full \
    --tau_scale "${BOOST_TAU_SCALE}" \
    --scale_multiplier "${BOOST_SCALE_MULTIPLIER}" \
    --filter_multiplier "${BOOST_FILTER_MULTIPLIER}"
fi

PRIOR_RENDER_DIR="${PRIOR_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
PRIOR_ALPHA_DIR="${PRIOR_EXPORT}/${SPLIT}/ours_${ITERATION}/alpha"
MERGED_RENDER_DIR="${MERGED_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
BASE_RENDER_DIR="${BASE_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
BOOST_RENDER_DIR="${BOOST_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"

mkdir -p \
  "${OUTPUT_ROOT}/base_train" \
  "${OUTPUT_ROOT}/prior_train_vis" \
  "${OUTPUT_ROOT}/prior_alpha_train" \
  "${OUTPUT_ROOT}/merged_train" \
  "${OUTPUT_ROOT}/merged_prior_boost_train" \
  "${OUTPUT_ROOT}/delta_merged_base"

if [[ "${SPLIT}" != "train" ]]; then
  mkdir -p \
    "${OUTPUT_ROOT}/base_${SPLIT}" \
    "${OUTPUT_ROOT}/prior_${SPLIT}_vis" \
    "${OUTPUT_ROOT}/prior_alpha_${SPLIT}" \
    "${OUTPUT_ROOT}/merged_${SPLIT}" \
    "${OUTPUT_ROOT}/merged_prior_boost_${SPLIT}" \
    "${OUTPUT_ROOT}/delta_merged_base_${SPLIT}"
fi

BASE_FLAT="${OUTPUT_ROOT}/base_${SPLIT}"
PRIOR_FLAT="${OUTPUT_ROOT}/prior_${SPLIT}_vis"
ALPHA_FLAT="${OUTPUT_ROOT}/prior_alpha_${SPLIT}"
MERGED_FLAT="${OUTPUT_ROOT}/merged_${SPLIT}"
BOOST_FLAT="${OUTPUT_ROOT}/merged_prior_boost_${SPLIT}"
DELTA_FLAT="${OUTPUT_ROOT}/delta_merged_base_${SPLIT}"
if [[ "${SPLIT}" == "train" ]]; then
  BASE_FLAT="${OUTPUT_ROOT}/base_train"
  PRIOR_FLAT="${OUTPUT_ROOT}/prior_train_vis"
  ALPHA_FLAT="${OUTPUT_ROOT}/prior_alpha_train"
  MERGED_FLAT="${OUTPUT_ROOT}/merged_train"
  BOOST_FLAT="${OUTPUT_ROOT}/merged_prior_boost_train"
  DELTA_FLAT="${OUTPUT_ROOT}/delta_merged_base"
fi
mkdir -p "${BASE_FLAT}" "${PRIOR_FLAT}" "${ALPHA_FLAT}" "${MERGED_FLAT}" "${BOOST_FLAT}" "${DELTA_FLAT}"

shopt -s nullglob
if [[ "${RUN_BASE_CONTROL}" == "1" ]]; then
  for image_path in "${BASE_RENDER_DIR}"/*.png; do
    cp "${image_path}" "${BASE_FLAT}/"
  done
fi
for image_path in "${PRIOR_RENDER_DIR}"/*.png; do
  cp "${image_path}" "${PRIOR_FLAT}/"
done
for image_path in "${PRIOR_ALPHA_DIR}"/*.png; do
  cp "${image_path}" "${ALPHA_FLAT}/"
done
for image_path in "${MERGED_RENDER_DIR}"/*.png; do
  cp "${image_path}" "${MERGED_FLAT}/"
done
if [[ "${RUN_MERGED_PRIOR_BOOST}" == "1" ]]; then
  for image_path in "${BOOST_RENDER_DIR}"/*.png; do
    cp "${image_path}" "${BOOST_FLAT}/"
  done
fi
shopt -u nullglob

if [[ "${RUN_BASE_CONTROL}" == "1" && "${RUN_MERGED_BASE_DELTA}" == "1" ]]; then
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/compare_render_dirs_v0.py" \
    --base_dir "${BASE_FLAT}" \
    --current_dir "${MERGED_FLAT}" \
    --output_dir "${DELTA_FLAT}" \
    --match_policy stem \
    --vis_scale "${DELTA_VIS_SCALE}" \
    --overwrite
fi

cat > "${OUTPUT_ROOT}/README.txt" <<EOF
Sprayed 2DGS HF preview.

model: ${MODEL_DIR}
base model: ${BASE_MODEL_DIR}
iteration: ${ITERATION}
split: ${SPLIT}
max_views: ${MAX_VIEWS}

Inspect first:
  ${BASE_FLAT}/00000.png        # original base field control with the same camera sampling
  ${PRIOR_FLAT}/00000.png       # only sprayed/prior_injected HF carrier, brightened for visibility
  ${ALPHA_FLAT}/00000.png       # sprayed carrier alpha
  ${MERGED_FLAT}/00000.png      # full merged model render
  ${BOOST_FLAT}/00000.png       # full field, but only sprayed/prior_injected is boosted for localization
  ${DELTA_FLAT}/00000.png       # amplified |merged - base|, real-strength contribution check
  ${DELTA_FLAT}/summary.json

Nested export roots:
  ${BASE_EXPORT}
  ${PRIOR_EXPORT}
  ${MERGED_EXPORT}
  ${BOOST_EXPORT}
EOF

echo "[spray-preview-v0] shallow outputs:"
echo "  ${BASE_FLAT}"
echo "  ${PRIOR_FLAT}"
echo "  ${ALPHA_FLAT}"
echo "  ${MERGED_FLAT}"
echo "  ${BOOST_FLAT}"
echo "  ${DELTA_FLAT}"
echo "  ${OUTPUT_ROOT}/README.txt"
