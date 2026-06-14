#!/usr/bin/env bash
set -euo pipefail

SOF_ROOT="${SOF_ROOT:-/root/autodl-tmp/SOFSR}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
VGGT_ROOT_DEFAULT="${WORK_ROOT}/vggt"
VGGT_ROOT="${VGGT_ROOT:-${VGGT_ROOT_DEFAULT}}"

LR_SOF_MODEL="${LR_SOF_MODEL:-${SCENE_ROOT}/_hrgsrefiner_assets/kitchen_sof_vanilla_images8_v1/soflr30k}"
LR_MESH_PATH="${LR_MESH_PATH:-${LR_SOF_MODEL}/test/ours_30000/lr_sof_mesh_v0_7.ply}"
RUN_NAME="${RUN_NAME:-soflr_vggt_bound_gs_corr_v0p2_probe}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/vggt_bound_gs_correction_v0/${SCENE_NAME}/${RUN_NAME}}"
CORRECTION_DIR="${CORRECTION_DIR:-${RUN_ROOT}/correction}"
CORRECTED_MODEL="${CORRECTED_MODEL:-${RUN_ROOT}/corrected_soflr_model}"
PROBE_ONLY_MODEL="${PROBE_ONLY_MODEL:-${RUN_ROOT}/probe_only_model}"
RENDER_DIR="${RENDER_DIR:-${RUN_ROOT}/renders_no_gt}"
PROBE_RENDER_DIR="${PROBE_RENDER_DIR:-${RUN_ROOT}/probe_only_renders_no_gt}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
RENDER_IMAGES_SUBDIR="${RENDER_IMAGES_SUBDIR:-images_2}"
LOAD_ITERATION="${LOAD_ITERATION:-30000}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-30000}"
MAX_VIEWS="${MAX_VIEWS:-12}"
FACE_K="${FACE_K:-8}"
BINDING_CHUNK_SIZE="${BINDING_CHUNK_SIZE:-50000}"
MAX_CORRECTION_RATIO="${MAX_CORRECTION_RATIO:-0.0015}"
APPLY_MODE="${APPLY_MODE:-probe_duplicate}"
PROBE_POSITION_MODE="${PROBE_POSITION_MODE:-residual_only}"
PROBE_OPACITY_SCALE="${PROBE_OPACITY_SCALE:-0.08}"
PROBE_SCALE_MULTIPLIER="${PROBE_SCALE_MULTIPLIER:-0.75}"
PROBE_MAX_COUNT="${PROBE_MAX_COUNT:-50000}"
PROBE_SEED="${PROBE_SEED:-0}"
NORMAL_OFFSET_SHRINK="${NORMAL_OFFSET_SHRINK:-1.0}"
TANGENT_OFFSET_SHRINK="${TANGENT_OFFSET_SHRINK:-1.0}"
CORRECTION_SCALE="${CORRECTION_SCALE:-0.25}"
MIN_CORRECTION_CONFIDENCE="${MIN_CORRECTION_CONFIDENCE:-0.05}"
MIN_CORRECTION_VIEWS="${MIN_CORRECTION_VIEWS:-2}"
MAX_SURFACE_DISTANCE="${MAX_SURFACE_DISTANCE:-0.0}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_PROBE_RENDER="${RUN_PROBE_RENDER:-1}"
RENDER_SPLIT="${RENDER_SPLIT:-test}"

echo "[soflr-vggt-bound-gs-v0] scene root     : ${SCENE_ROOT}"
echo "[soflr-vggt-bound-gs-v0] SOFLR model    : ${LR_SOF_MODEL}"
echo "[soflr-vggt-bound-gs-v0] LR mesh        : ${LR_MESH_PATH}"
echo "[soflr-vggt-bound-gs-v0] run root       : ${RUN_ROOT}"
echo "[soflr-vggt-bound-gs-v0] max views      : ${MAX_VIEWS}"
echo "[soflr-vggt-bound-gs-v0] correction     : ratio=${MAX_CORRECTION_RATIO} scale=${CORRECTION_SCALE}"
echo "[soflr-vggt-bound-gs-v0] apply mode     : ${APPLY_MODE}"
echo "[soflr-vggt-bound-gs-v0] probe layer    : pos=${PROBE_POSITION_MODE} opacity=${PROBE_OPACITY_SCALE} scale=${PROBE_SCALE_MULTIPLIER} max=${PROBE_MAX_COUNT}"
echo "[soflr-vggt-bound-gs-v0] offset shrink  : normal=${NORMAL_OFFSET_SHRINK} tangent=${TANGENT_OFFSET_SHRINK}"
echo "[soflr-vggt-bound-gs-v0] apply gates    : conf>=${MIN_CORRECTION_CONFIDENCE} views>=${MIN_CORRECTION_VIEWS} surface<=${MAX_SURFACE_DISTANCE}"

mkdir -p "${CORRECTION_DIR}" "${CORRECTED_MODEL}" "${RENDER_DIR}"

python -u "${SOF_ROOT}/build_soflr_vggt_bound_gs_correction_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --soflr_model_path "${LR_SOF_MODEL}" \
  --lr_mesh_path "${LR_MESH_PATH}" \
  --output_dir "${CORRECTION_DIR}" \
  --vggt_root "${VGGT_ROOT}" \
  --images_subdir "${IMAGES_SUBDIR}" \
  --load_iteration "${LOAD_ITERATION}" \
  --max_views "${MAX_VIEWS}" \
  --face_k "${FACE_K}" \
  --binding_chunk_size "${BINDING_CHUNK_SIZE}" \
  --max_correction_ratio "${MAX_CORRECTION_RATIO}"

python -u "${SOF_ROOT}/apply_soflr_bound_gs_surface_correction_v0.py" \
  --base_model_path "${LR_SOF_MODEL}" \
  --base_iteration "${LOAD_ITERATION}" \
  --correction_payload "${CORRECTION_DIR}/correction_payload_v0.npz" \
  --output_model_path "${CORRECTED_MODEL}" \
  --output_iteration "${OUTPUT_ITERATION}" \
  --apply_mode "${APPLY_MODE}" \
  --probe_position_mode "${PROBE_POSITION_MODE}" \
  --probe_only_model_path "${PROBE_ONLY_MODEL}" \
  --probe_opacity_scale "${PROBE_OPACITY_SCALE}" \
  --probe_scale_multiplier "${PROBE_SCALE_MULTIPLIER}" \
  --probe_max_count "${PROBE_MAX_COUNT}" \
  --probe_seed "${PROBE_SEED}" \
  --normal_offset_shrink "${NORMAL_OFFSET_SHRINK}" \
  --tangent_offset_shrink "${TANGENT_OFFSET_SHRINK}" \
  --correction_scale "${CORRECTION_SCALE}" \
  --min_correction_confidence "${MIN_CORRECTION_CONFIDENCE}" \
  --min_correction_views "${MIN_CORRECTION_VIEWS}" \
  --max_surface_distance "${MAX_SURFACE_DISTANCE}"

if [[ "${RUN_RENDER}" == "1" ]]; then
  python -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${CORRECTED_MODEL}" \
    --output_dir "${RENDER_DIR}" \
    --images_subdir "${RENDER_IMAGES_SUBDIR}" \
    --iteration "${OUTPUT_ITERATION}" \
    --split "${RENDER_SPLIT}"
fi

if [[ "${RUN_PROBE_RENDER}" == "1" && "${APPLY_MODE}" == "probe_duplicate" ]]; then
  python -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${PROBE_ONLY_MODEL}" \
    --output_dir "${PROBE_RENDER_DIR}" \
    --images_subdir "${RENDER_IMAGES_SUBDIR}" \
    --iteration "${OUTPUT_ITERATION}" \
    --split "${RENDER_SPLIT}"
fi

echo "[done] correction payload : ${CORRECTION_DIR}/correction_payload_v0.npz"
echo "[done] correction summary : ${CORRECTION_DIR}/summary.json"
echo "[done] corrected model    : ${CORRECTED_MODEL}"
echo "[done] corrected ply      : ${CORRECTED_MODEL}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
if [[ "${APPLY_MODE}" == "probe_duplicate" ]]; then
  echo "[done] probe-only model   : ${PROBE_ONLY_MODEL}"
  echo "[done] probe-only ply     : ${PROBE_ONLY_MODEL}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
fi
if [[ "${RUN_RENDER}" == "1" ]]; then
  echo "[done] renders no gt      : ${RENDER_DIR}/${RENDER_SPLIT}/ours_${OUTPUT_ITERATION}/renders"
fi
if [[ "${RUN_PROBE_RENDER}" == "1" && "${APPLY_MODE}" == "probe_duplicate" ]]; then
  echo "[done] probe renders      : ${PROBE_RENDER_DIR}/${RENDER_SPLIT}/ours_${OUTPUT_ITERATION}/renders"
fi
