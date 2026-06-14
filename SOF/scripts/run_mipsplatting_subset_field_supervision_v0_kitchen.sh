#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="${SOF_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k_rerun_v0}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
MIP_ITERATION="${MIP_ITERATION:-30000}"
SPLATTING_CONFIG_PATH="${SPLATTING_CONFIG_PATH:-${MIP_MODEL_PATH}/config.json}"

STATE_RUN_NAME="${STATE_RUN_NAME:-mip30k_rerun_gs2mesh_surface_state_v0_relaxed_carrier_v1}"
STATE_DIR="${STATE_DIR:-${SOF_ROOT}/output/gaussian_surface_sort/${SCENE_NAME}/${STATE_RUN_NAME}}"
SURFACE_STATE_PAYLOAD="${SURFACE_STATE_PAYLOAD:-${STATE_DIR}/gaussian_surface_state_v0.pt}"
SUBSET_MASK_KEY="${SUBSET_MASK_KEY:-surface_candidate}"

RUN_NAME="${RUN_NAME:-${MIP_EXPERIMENT_NAME}_${SUBSET_MASK_KEY}_subset_field_supervision_v0_images2_4k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/mip_subset_field_supervision_v0/${SCENE_NAME}/${RUN_NAME}}"
SUBSET_EXPORT_ROOT="${SUBSET_EXPORT_ROOT:-${OUTPUT_ROOT}/export}"
SUBSET_MODEL_PATH="${SUBSET_MODEL_PATH:-${SUBSET_EXPORT_ROOT}/subset_model}"

TRAIN_IMAGES_SUBDIR="${TRAIN_IMAGES_SUBDIR:-images_2}"
SUPERVISION_SPLIT="${SUPERVISION_SPLIT:-train}"
SUPERVISION_MAX_VIEWS="${SUPERVISION_MAX_VIEWS:-0}"
BASELINE_RENDER_RESOLUTION="${BASELINE_RENDER_RESOLUTION:-1}"
BASELINE_RENDER_ROOT="${BASELINE_RENDER_ROOT:-${MIP_MODEL_PATH}/${SUPERVISION_SPLIT}/ours_${MIP_ITERATION}/renders}"
NAMED_SUPERVISION_DIR="${NAMED_SUPERVISION_DIR:-${OUTPUT_ROOT}/named_mip_rgb_supervision_${TRAIN_IMAGES_SUBDIR}_${SUPERVISION_SPLIT}}"

EXPORT_PREVIEW_IMAGES_SUBDIR="${EXPORT_PREVIEW_IMAGES_SUBDIR:-images_2}"
EXPORT_PREVIEW_SPLIT="${EXPORT_PREVIEW_SPLIT:-test}"
EXPORT_PREVIEW_MAX_VIEWS="${EXPORT_PREVIEW_MAX_VIEWS:-8}"

ITERATIONS="${ITERATIONS:-4000}"
TRAIN_DATA_DEVICE="${TRAIN_DATA_DEVICE:-cpu}"
KERNEL_SIZE="${KERNEL_SIZE:-0.1}"
START_PLY_ACTIVE_SH_DEGREE="${START_PLY_ACTIVE_SH_DEGREE:--1}"
RAY_JITTER="${RAY_JITTER:-0}"
RESAMPLE_GT_IMAGE="${RESAMPLE_GT_IMAGE:-0}"
SAMPLE_MORE_HIGHRES="${SAMPLE_MORE_HIGHRES:-0}"

EVAL_IMAGES_SUBDIR="${EVAL_IMAGES_SUBDIR:-images_2}"
EVAL_RENDER_RESOLUTION="${EVAL_RENDER_RESOLUTION:-1}"
RUN_RENDER_AFTER="${RUN_RENDER_AFTER:-1}"
RUN_METRICS_AFTER="${RUN_METRICS_AFTER:-1}"
AUTO_PREPARE_BASELINE_RENDERS="${AUTO_PREPARE_BASELINE_RENDERS:-1}"
FORCE_REEXPORT="${FORCE_REEXPORT:-0}"
FORCE_PREPARE_SUPERVISION="${FORCE_PREPARE_SUPERVISION:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[mip-subset-field-v0] missing scene root: ${SCENE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${MIP_MODEL_PATH}" ]]; then
  echo "[mip-subset-field-v0] missing baseline model path: ${MIP_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SURFACE_STATE_PAYLOAD}" ]]; then
  echo "[mip-subset-field-v0] missing surface-state payload: ${SURFACE_STATE_PAYLOAD}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[mip-subset-field-v0] scene                : ${SCENE_ROOT}"
echo "[mip-subset-field-v0] baseline model       : ${MIP_MODEL_PATH} iter=${MIP_ITERATION}"
echo "[mip-subset-field-v0] surface payload      : ${SURFACE_STATE_PAYLOAD}"
echo "[mip-subset-field-v0] subset mask key      : ${SUBSET_MASK_KEY}"
echo "[mip-subset-field-v0] output root          : ${OUTPUT_ROOT}"
echo "[mip-subset-field-v0] train supervision    : ${TRAIN_IMAGES_SUBDIR} split=${SUPERVISION_SPLIT}"
echo "[mip-subset-field-v0] eval render          : ${EVAL_IMAGES_SUBDIR}"
echo "[mip-subset-field-v0] iterations           : ${ITERATIONS}"

if [[ ! -d "${BASELINE_RENDER_ROOT}" ]]; then
  if [[ "${AUTO_PREPARE_BASELINE_RENDERS}" != "1" ]]; then
    echo "[mip-subset-field-v0] missing baseline render root: ${BASELINE_RENDER_ROOT}" >&2
    exit 1
  fi
  echo
  echo "[0/4] prepare baseline train renders"
  OMP_NUM_THREADS=1 "${PYTHON_BIN}" -u "${SOF_ROOT}/render.py" \
    -m "${MIP_MODEL_PATH}" \
    -s "${SCENE_ROOT}" \
    -i "${TRAIN_IMAGES_SUBDIR}" \
    --iteration "${MIP_ITERATION}" \
    --init_type sfm \
    --eval \
    --skip_test
fi

SUBSET_PLY_PATH="${SUBSET_MODEL_PATH}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply"
if [[ "${FORCE_REEXPORT}" == "1" || ! -f "${SUBSET_PLY_PATH}" ]]; then
  echo
  echo "[1/4] export subset field"
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/export_gaussian_mask_subset_v0.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${MIP_MODEL_PATH}" \
    --iteration "${MIP_ITERATION}" \
    --mask_payload_path "${SURFACE_STATE_PAYLOAD}" \
    --mask_key "${SUBSET_MASK_KEY}" \
    --output_root "${SUBSET_EXPORT_ROOT}" \
    --images_subdir "${EXPORT_PREVIEW_IMAGES_SUBDIR}" \
    --split "${EXPORT_PREVIEW_SPLIT}" \
    --max_views "${EXPORT_PREVIEW_MAX_VIEWS}"
else
  echo
  echo "[1/4] reuse existing subset field"
fi

SUPERVISION_SUMMARY_PATH="${NAMED_SUPERVISION_DIR}/prepare_named_mip_render_supervision_v0_summary.json"
if [[ "${FORCE_PREPARE_SUPERVISION}" == "1" || ! -f "${SUPERVISION_SUMMARY_PATH}" ]]; then
  echo
  echo "[2/4] prepare camera-name mip supervision"
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/prepare_named_mip_render_supervision_v0.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${MIP_MODEL_PATH}" \
    --images_subdir "${TRAIN_IMAGES_SUBDIR}" \
    --split "${SUPERVISION_SPLIT}" \
    --max_views "${SUPERVISION_MAX_VIEWS}" \
    --render_root "${BASELINE_RENDER_ROOT}" \
    --output_dir "${NAMED_SUPERVISION_DIR}"
else
  echo
  echo "[2/4] reuse existing named mip supervision"
fi

FINAL_CHECKPOINT_PATH="${SUBSET_MODEL_PATH}/chkpnt${ITERATIONS}.pth"
if [[ "${FORCE_TRAIN}" == "1" || ! -f "${FINAL_CHECKPOINT_PATH}" ]]; then
  echo
  echo "[3/4] train subset-only field against baseline mip renders"
  TRAIN_CMD=(
    "${PYTHON_BIN}" -u "${SOF_ROOT}/train_vanilla_mip_from_init_v0.py"
    -s "${SCENE_ROOT}"
    -i "${TRAIN_IMAGES_SUBDIR}"
    -m "${SUBSET_MODEL_PATH}"
    --eval
    --data_device "${TRAIN_DATA_DEVICE}"
    --kernel_size "${KERNEL_SIZE}"
    --vanilla_mip_mode
    --start_ply "${SUBSET_PLY_PATH}"
    --start_ply_active_sh_degree "${START_PLY_ACTIVE_SH_DEGREE}"
    --global_image_dir "${NAMED_SUPERVISION_DIR}"
    --iterations "${ITERATIONS}"
    --test_iterations "${ITERATIONS}"
    --save_iterations "${ITERATIONS}"
    --checkpoint_iterations "${ITERATIONS}"
  )
  if [[ "${RAY_JITTER}" == "1" ]]; then
    TRAIN_CMD+=(--ray_jitter)
  fi
  if [[ "${RESAMPLE_GT_IMAGE}" == "1" ]]; then
    TRAIN_CMD+=(--resample_gt_image)
  fi
  if [[ "${SAMPLE_MORE_HIGHRES}" == "1" ]]; then
    TRAIN_CMD+=(--sample_more_highres)
  fi
  if [[ -f "${SPLATTING_CONFIG_PATH}" ]]; then
    TRAIN_CMD+=(--splatting_config "${SPLATTING_CONFIG_PATH}")
  fi
  "${TRAIN_CMD[@]}"
else
  echo
  echo "[3/4] checkpoint exists, skipping training: ${FINAL_CHECKPOINT_PATH}"
fi

if [[ "${RUN_RENDER_AFTER}" == "1" ]]; then
  echo
  echo "[4/4] render eval split"
  "${PYTHON_BIN}" -u "${SOF_ROOT}/render.py" \
    -m "${SUBSET_MODEL_PATH}" \
    -s "${SCENE_ROOT}" \
    -i "${EVAL_IMAGES_SUBDIR}" \
    --iteration "${ITERATIONS}" \
    --init_type sfm \
    --eval \
    --skip_train
fi

if [[ "${RUN_METRICS_AFTER}" == "1" ]]; then
  echo
  echo "[metrics] summarize eval PSNR/SSIM"
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/summarize_sof_render_metrics_v0.py" \
    --model_dir "${SUBSET_MODEL_PATH}" \
    --iteration "${ITERATIONS}" \
    --split test
fi

echo
echo "[done] subset model root   : ${SUBSET_MODEL_PATH}"
echo "[done] subset init ply     : ${SUBSET_PLY_PATH}"
echo "[done] supervision dir     : ${NAMED_SUPERVISION_DIR}"
echo "[done] eval renders        : ${SUBSET_MODEL_PATH}/test/ours_${ITERATIONS}/renders"
if [[ "${RUN_METRICS_AFTER}" == "1" ]]; then
  echo "[done] metrics             : ${SUBSET_MODEL_PATH}/results_psnr_ssim.json"
fi
