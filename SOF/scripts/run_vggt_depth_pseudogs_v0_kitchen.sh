#!/usr/bin/env bash
set -euo pipefail

SOF_ROOT="${SOF_ROOT:-/root/autodl-tmp/SOFSR}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
VGGT_ROOT="${VGGT_ROOT:-${WORK_ROOT}/vggt}"

BASE_MODEL="${BASE_MODEL:-${SCENE_ROOT}/_hrgsrefiner_assets/kitchen_sof_vanilla_images8_v1/soflr30k}"
RUN_NAME="${RUN_NAME:-vggt_depth_pseudogs_v0}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/vggt_depth_pseudogs_v0/${SCENE_NAME}/${RUN_NAME}}"
PSEUDOGS_MODEL="${PSEUDOGS_MODEL:-${RUN_ROOT}/pseudogs_model}"
BUILD_DIR="${BUILD_DIR:-${RUN_ROOT}/build}"
BASE_RENDER_DIR="${BASE_RENDER_DIR:-${RUN_ROOT}/base_renders_no_gt}"
RENDER_DIR="${RENDER_DIR:-${RUN_ROOT}/renders_no_gt}"
RENDER_COMPARE_DIR="${RENDER_COMPARE_DIR:-${RUN_ROOT}/render_diff_vs_base}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
RENDER_IMAGES_SUBDIR="${RENDER_IMAGES_SUBDIR:-images_2}"
LOAD_ITERATION="${LOAD_ITERATION:-30000}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-30000}"
MAX_VIEWS="${MAX_VIEWS:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-50000}"

MIN_ALPHA="${MIN_ALPHA:-0.08}"
MIN_VGGT_CONFIDENCE="${MIN_VGGT_CONFIDENCE:-0.05}"
MIN_VIEWS="${MIN_VIEWS:-2}"
MIN_WEIGHT="${MIN_WEIGHT:-0.02}"
ZBUFFER_TOLERANCE_ABS="${ZBUFFER_TOLERANCE_ABS:-0.03}"
ZBUFFER_TOLERANCE_REL="${ZBUFFER_TOLERANCE_REL:-0.015}"
MAX_DEPTH_RESIDUAL_ABS="${MAX_DEPTH_RESIDUAL_ABS:-0.0}"
MAX_DEPTH_RESIDUAL_RATIO="${MAX_DEPTH_RESIDUAL_RATIO:-0.05}"
BLEND="${BLEND:-1.0}"

RUN_RENDER="${RUN_RENDER:-1}"
RUN_BASE_RENDER="${RUN_BASE_RENDER:-1}"
RUN_RENDER_COMPARE="${RUN_RENDER_COMPARE:-1}"
RENDER_SPLIT="${RENDER_SPLIT:-test}"

echo "[vggt-depth-pseudogs-v0] scene root   : ${SCENE_ROOT}"
echo "[vggt-depth-pseudogs-v0] base model   : ${BASE_MODEL}"
echo "[vggt-depth-pseudogs-v0] run root     : ${RUN_ROOT}"
echo "[vggt-depth-pseudogs-v0] max views    : ${MAX_VIEWS}"
echo "[vggt-depth-pseudogs-v0] gates        : alpha>=${MIN_ALPHA} conf>=${MIN_VGGT_CONFIDENCE} views>=${MIN_VIEWS} weight>=${MIN_WEIGHT}"
echo "[vggt-depth-pseudogs-v0] zbuffer      : abs=${ZBUFFER_TOLERANCE_ABS} rel=${ZBUFFER_TOLERANCE_REL}"
echo "[vggt-depth-pseudogs-v0] residual     : abs=${MAX_DEPTH_RESIDUAL_ABS} ratio=${MAX_DEPTH_RESIDUAL_RATIO} blend=${BLEND}"

mkdir -p "${BUILD_DIR}" "${PSEUDOGS_MODEL}" "${RENDER_DIR}"

python -u "${SOF_ROOT}/build_vggt_depth_pseudogs_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --base_model_path "${BASE_MODEL}" \
  --output_model_path "${PSEUDOGS_MODEL}" \
  --output_dir "${BUILD_DIR}" \
  --vggt_root "${VGGT_ROOT}" \
  --images_subdir "${IMAGES_SUBDIR}" \
  --load_iteration "${LOAD_ITERATION}" \
  --output_iteration "${OUTPUT_ITERATION}" \
  --max_views "${MAX_VIEWS}" \
  --chunk_size "${CHUNK_SIZE}" \
  --min_alpha "${MIN_ALPHA}" \
  --min_vggt_confidence "${MIN_VGGT_CONFIDENCE}" \
  --min_views "${MIN_VIEWS}" \
  --min_weight "${MIN_WEIGHT}" \
  --zbuffer_tolerance_abs "${ZBUFFER_TOLERANCE_ABS}" \
  --zbuffer_tolerance_rel "${ZBUFFER_TOLERANCE_REL}" \
  --max_depth_residual_abs "${MAX_DEPTH_RESIDUAL_ABS}" \
  --max_depth_residual_ratio "${MAX_DEPTH_RESIDUAL_RATIO}" \
  --blend "${BLEND}"

if [[ "${RUN_BASE_RENDER}" == "1" ]]; then
  python -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${BASE_MODEL}" \
    --output_dir "${BASE_RENDER_DIR}" \
    --images_subdir "${RENDER_IMAGES_SUBDIR}" \
    --iteration "${LOAD_ITERATION}" \
    --split "${RENDER_SPLIT}"
fi

if [[ "${RUN_RENDER}" == "1" ]]; then
  python -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${PSEUDOGS_MODEL}" \
    --output_dir "${RENDER_DIR}" \
    --images_subdir "${RENDER_IMAGES_SUBDIR}" \
    --iteration "${OUTPUT_ITERATION}" \
    --split "${RENDER_SPLIT}"
fi

if [[ "${RUN_RENDER_COMPARE}" == "1" && "${RUN_BASE_RENDER}" == "1" && "${RUN_RENDER}" == "1" ]]; then
  python -u "${SOF_ROOT}/scripts/compare_render_dirs_no_gt.py" \
    --reference_dir "${BASE_RENDER_DIR}/${RENDER_SPLIT}/ours_${LOAD_ITERATION}/renders" \
    --candidate_dir "${RENDER_DIR}/${RENDER_SPLIT}/ours_${OUTPUT_ITERATION}/renders" \
    --output_dir "${RENDER_COMPARE_DIR}"
fi

echo "[done] pseudoGS payload : ${BUILD_DIR}/pseudogs_payload_v0.npz"
echo "[done] summary          : ${BUILD_DIR}/summary.json"
echo "[done] pseudoGS model   : ${PSEUDOGS_MODEL}"
echo "[done] pseudoGS ply     : ${PSEUDOGS_MODEL}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
if [[ "${RUN_RENDER}" == "1" ]]; then
  echo "[done] renders no gt    : ${RENDER_DIR}/${RENDER_SPLIT}/ours_${OUTPUT_ITERATION}/renders"
fi
if [[ "${RUN_BASE_RENDER}" == "1" ]]; then
  echo "[done] base renders     : ${BASE_RENDER_DIR}/${RENDER_SPLIT}/ours_${LOAD_ITERATION}/renders"
fi
if [[ "${RUN_RENDER_COMPARE}" == "1" && "${RUN_BASE_RENDER}" == "1" && "${RUN_RENDER}" == "1" ]]; then
  echo "[done] render diff      : ${RENDER_COMPARE_DIR}/summary.json"
  echo "[done] render absdiff   : ${RENDER_COMPARE_DIR}/absdiff"
fi
