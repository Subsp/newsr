#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
PYTHON_BIN="${PYTHON_BIN:-python}"

START_PRESET="${START_PRESET:-cleanup_volume_stress}"

if [[ -z "${START_MODEL_PATH:-}" ]]; then
  case "${START_PRESET}" in
    cleanup_volume_stress)
      START_RUN_NAME="${START_RUN_NAME:-mip30k_volume_stress_iterclean_v0}"
      START_MODEL_PATH="${SOF_ROOT}/output/cleanup_mip_view_aligned_volume_artifacts_v0/${SCENE_NAME}/${START_RUN_NAME}/cleaned_mip_model_view_volume_v1"
      START_ITERATION="${START_ITERATION:-30000}"
      ;;
    puremip_recover)
      START_RUN_NAME="${START_RUN_NAME:-mip30k_volume_stress_iterclean_v0_conservative_v0_puremip}"
      START_MODEL_PATH="${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${START_RUN_NAME}/recovered_mip_model_lr_v0"
      START_ITERATION="${START_ITERATION:-31600}"
      ;;
    miphr_recover)
      START_RUN_NAME="${START_RUN_NAME:-mip30k_volume_stress_iterclean_v0_mip_hr_anchor_v0_miphr_v1}"
      START_MODEL_PATH="${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${START_RUN_NAME}/recovered_mip_model_lr_miphr_v1"
      START_ITERATION="${START_ITERATION:-31600}"
      ;;
    accepted_initrepair_miphr)
      START_RUN_NAME="${START_RUN_NAME:-view_aligned_volume_delete_v1_initrepair_v0_prune_more_v1_mip_hr_anchor_v0_miphr_v1}"
      START_MODEL_PATH="${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${START_RUN_NAME}/recovered_mip_model_lr_miphr_v1"
      START_ITERATION="${START_ITERATION:-31600}"
      ;;
    *)
      echo "[cleaned-mip-vanilla-densify-v0] unknown START_PRESET=${START_PRESET}" >&2
      echo "  valid: cleanup_volume_stress, puremip_recover, miphr_recover, accepted_initrepair_miphr" >&2
      exit 1
      ;;
  esac
else
  START_RUN_NAME="${START_RUN_NAME:-custom_start}"
  START_ITERATION="${START_ITERATION:-30000}"
fi

START_PLY="${START_PLY:-${START_MODEL_PATH}/point_cloud/iteration_${START_ITERATION}/point_cloud.ply}"
START_ACTIVE_SH_DEGREE="${START_ACTIVE_SH_DEGREE:-3}"

RUN_NAME="${RUN_NAME:-${START_RUN_NAME}_vanilla_mip_densify_continue_v0}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/vanilla_mip_densify_continue_v0/${SCENE_NAME}/${RUN_NAME}/continued_mip_model_v0}"

TRAIN_IMAGES_SUBDIR="${TRAIN_IMAGES_SUBDIR:-images_8}"
RENDER_IMAGES_SUBDIR="${RENDER_IMAGES_SUBDIR:-images_2}"
RENDER_SPLIT="${RENDER_SPLIT:-test}"
RENDER_MAX_VIEWS="${RENDER_MAX_VIEWS:-0}"
RUN_RENDER="${RUN_RENDER:-1}"

CONTINUE_STEPS="${CONTINUE_STEPS:-2000}"
FINAL_ITERATION="${FINAL_ITERATION:-$((START_ITERATION + CONTINUE_STEPS))}"
TRAIN_DATA_DEVICE="${TRAIN_DATA_DEVICE:-cpu}"
KERNEL_SIZE="${KERNEL_SIZE:-0.1}"
RAY_JITTER="${RAY_JITTER:-0}"
RESAMPLE_GT_IMAGE="${RESAMPLE_GT_IMAGE:-0}"
SAMPLE_MORE_HIGHRES="${SAMPLE_MORE_HIGHRES:-0}"

DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-${START_ITERATION}}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-${FINAL_ITERATION}}"
DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-100}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
OPACITY_RESET_INTERVAL="${OPACITY_RESET_INTERVAL:-3000}"

SPLATTING_CONFIG_PATH="${SPLATTING_CONFIG_PATH:-}"
if [[ -z "${SPLATTING_CONFIG_PATH}" ]]; then
  if [[ -f "${START_MODEL_PATH}/config.json" ]]; then
    SPLATTING_CONFIG_PATH="${START_MODEL_PATH}/config.json"
  elif [[ -f "${SOF_ROOT}/configs/hierarchical.json" ]]; then
    SPLATTING_CONFIG_PATH="${SOF_ROOT}/configs/hierarchical.json"
  fi
fi

RENDER_DIR="${RENDER_DIR:-${OUTPUT_MODEL_PATH}/vanilla_densify_renders_no_gt_v0}"
RENDER_CONTACT_SHEET="${RENDER_CONTACT_SHEET:-${RENDER_DIR}/contact_sheet_${RENDER_IMAGES_SUBDIR}_${RENDER_SPLIT}_${FINAL_ITERATION}.png}"

if [[ ! -f "${START_PLY}" ]]; then
  echo "[cleaned-mip-vanilla-densify-v0] missing start ply: ${START_PLY}" >&2
  exit 1
fi

echo "[cleaned-mip-vanilla-densify-v0] scene       : ${SCENE_ROOT}"
echo "[cleaned-mip-vanilla-densify-v0] preset      : ${START_PRESET}"
echo "[cleaned-mip-vanilla-densify-v0] start model : ${START_MODEL_PATH}"
echo "[cleaned-mip-vanilla-densify-v0] start ply   : ${START_PLY}"
echo "[cleaned-mip-vanilla-densify-v0] output      : ${OUTPUT_MODEL_PATH}"
echo "[cleaned-mip-vanilla-densify-v0] train       : ${START_ITERATION} -> ${FINAL_ITERATION}, images=${TRAIN_IMAGES_SUBDIR}"
echo "[cleaned-mip-vanilla-densify-v0] densify     : from=${DENSIFY_FROM_ITER} until=${DENSIFY_UNTIL_ITER} interval=${DENSIFICATION_INTERVAL} grad=${DENSIFY_GRAD_THRESHOLD}"
echo "[cleaned-mip-vanilla-densify-v0] render      : images=${RENDER_IMAGES_SUBDIR} split=${RENDER_SPLIT} max_views=${RENDER_MAX_VIEWS}"
echo "[cleaned-mip-vanilla-densify-v0] splat config: ${SPLATTING_CONFIG_PATH:-<default>}"

TRAIN_CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/train_vanilla_mip_from_init_v0.py"
  -s "${SCENE_ROOT}"
  -i "${TRAIN_IMAGES_SUBDIR}"
  -m "${OUTPUT_MODEL_PATH}"
  --eval
  --data_device "${TRAIN_DATA_DEVICE}"
  --kernel_size "${KERNEL_SIZE}"
  --vanilla_mip_mode
  --start_ply "${START_PLY}"
  --start_ply_iteration "${START_ITERATION}"
  --start_ply_active_sh_degree "${START_ACTIVE_SH_DEGREE}"
  --iterations "${FINAL_ITERATION}"
  --test_iterations "${FINAL_ITERATION}"
  --save_iterations "${FINAL_ITERATION}"
  --checkpoint_iterations "${FINAL_ITERATION}"
  --densify_from_iter "${DENSIFY_FROM_ITER}"
  --densify_until_iter "${DENSIFY_UNTIL_ITER}"
  --densification_interval "${DENSIFICATION_INTERVAL}"
  --densify_grad_threshold "${DENSIFY_GRAD_THRESHOLD}"
  --opacity_reset_interval "${OPACITY_RESET_INTERVAL}"
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
if [[ -n "${SPLATTING_CONFIG_PATH}" && -f "${SPLATTING_CONFIG_PATH}" ]]; then
  TRAIN_CMD+=(--splatting_config "${SPLATTING_CONFIG_PATH}")
fi

"${TRAIN_CMD[@]}"

if [[ "${RUN_RENDER}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${OUTPUT_MODEL_PATH}" \
    --output_dir "${RENDER_DIR}" \
    --images_subdir "${RENDER_IMAGES_SUBDIR}" \
    --iteration "${FINAL_ITERATION}" \
    --split "${RENDER_SPLIT}" \
    --max_views "${RENDER_MAX_VIEWS}"

  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/make_render_contact_sheet.py" \
    --render_dir "${RENDER_DIR}/${RENDER_SPLIT}/ours_${FINAL_ITERATION}/renders" \
    --output_path "${RENDER_CONTACT_SHEET}" \
    --max_images 16 \
    --columns 4 \
    --thumb_width 360
fi

echo "[done] output model : ${OUTPUT_MODEL_PATH}"
echo "[done] output ply   : ${OUTPUT_MODEL_PATH}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud.ply"
echo "[done] checkpoint   : ${OUTPUT_MODEL_PATH}/chkpnt${FINAL_ITERATION}.pth"
if [[ "${RUN_RENDER}" == "1" ]]; then
  echo "[done] renders      : ${RENDER_DIR}/${RENDER_SPLIT}/ours_${FINAL_ITERATION}/renders"
  echo "[done] contact sheet: ${RENDER_CONTACT_SHEET}"
fi
