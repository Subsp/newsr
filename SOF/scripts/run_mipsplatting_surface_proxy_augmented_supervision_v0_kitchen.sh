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
SURFACE_MASK_KEY="${SURFACE_MASK_KEY:-surface_candidate}"
DONOR_MASK_KEY="${DONOR_MASK_KEY:-surface_complement_active}"

RUN_NAME="${RUN_NAME:-${MIP_EXPERIMENT_NAME}_${SURFACE_MASK_KEY}_plus_${DONOR_MASK_KEY}_surface_proxy_augmented_supervision_v0_images2_4k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/mip_surface_proxy_augmented_supervision_v0/${SCENE_NAME}/${RUN_NAME}}"
EXPORT_ROOT="${EXPORT_ROOT:-${OUTPUT_ROOT}/export}"
AUGMENTED_MODEL_PATH="${AUGMENTED_MODEL_PATH:-${EXPORT_ROOT}/augmented_model}"

TRAIN_IMAGES_SUBDIR="${TRAIN_IMAGES_SUBDIR:-images_2}"
SUPERVISION_SPLIT="${SUPERVISION_SPLIT:-train}"
SUPERVISION_MAX_VIEWS="${SUPERVISION_MAX_VIEWS:-0}"
BASELINE_RENDER_ROOT="${BASELINE_RENDER_ROOT:-${MIP_MODEL_PATH}/${SUPERVISION_SPLIT}/ours_${MIP_ITERATION}/renders}"
NAMED_SUPERVISION_DIR="${NAMED_SUPERVISION_DIR:-${OUTPUT_ROOT}/named_mip_rgb_supervision_${TRAIN_IMAGES_SUBDIR}_${SUPERVISION_SPLIT}}"

EXPORT_PREVIEW_IMAGES_SUBDIR="${EXPORT_PREVIEW_IMAGES_SUBDIR:-images_2}"
EXPORT_PREVIEW_SPLIT="${EXPORT_PREVIEW_SPLIT:-test}"
EXPORT_PREVIEW_MAX_VIEWS="${EXPORT_PREVIEW_MAX_VIEWS:-8}"

PROXY_MAX_DONORS="${PROXY_MAX_DONORS:-200000}"
PROXY_ANCHOR_MODE="${PROXY_ANCHOR_MODE:-ray_surface}"
PROXY_OUTPUT_MODE="${PROXY_OUTPUT_MODE:-replace_cloned_donors}"
PROXY_MESH_PATH="${PROXY_MESH_PATH:-${MESH_PATH:-}}"
PROXY_RAY_CAMERA_STRIDE="${PROXY_RAY_CAMERA_STRIDE:-1}"
PROXY_RAY_MAX_VIEWS="${PROXY_RAY_MAX_VIEWS:-0}"
PROXY_RAY_DEPTH_MIN="${PROXY_RAY_DEPTH_MIN:-0.01}"
PROXY_RAY_CHUNK="${PROXY_RAY_CHUNK:-262144}"
PROXY_RAY_SURFACE_MODE="${PROXY_RAY_SURFACE_MODE:-push_away}"
PROXY_RAY_MAX_HIT_TO_DONOR_GAP="${PROXY_RAY_MAX_HIT_TO_DONOR_GAP:-0.0}"
PROXY_RAY_MAX_HIT_TO_DONOR_GAP_RATIO="${PROXY_RAY_MAX_HIT_TO_DONOR_GAP_RATIO:-0.0}"
PROXY_RAY_MIN_DEPTH_SCALE="${PROXY_RAY_MIN_DEPTH_SCALE:-0.5}"
PROXY_RAY_PROJECTED_CENTER_TOLERANCE_PX="${PROXY_RAY_PROJECTED_CENTER_TOLERANCE_PX:-1.0}"
PROXY_RAY_PRESERVE_SCREEN_SCALE="${PROXY_RAY_PRESERVE_SCREEN_SCALE:-1}"
PROXY_RAY_FALLBACK_TO_ANCHOR="${PROXY_RAY_FALLBACK_TO_ANCHOR:-0}"
PROXY_RAY_RELAX_UNANCHORED_WITH_MESH="${PROXY_RAY_RELAX_UNANCHORED_WITH_MESH:-1}"
PROXY_RAY_RELAXED_MAX_HIT_TO_DONOR_GAP="${PROXY_RAY_RELAXED_MAX_HIT_TO_DONOR_GAP:-0.0}"
PROXY_RAY_RELAXED_MAX_HIT_TO_DONOR_GAP_RATIO="${PROXY_RAY_RELAXED_MAX_HIT_TO_DONOR_GAP_RATIO:-8.0}"
PROXY_RAY_RELAXED_PROJECTED_CENTER_TOLERANCE_PX="${PROXY_RAY_RELAXED_PROJECTED_CENTER_TOLERANCE_PX:-4.0}"
if [[ -z "${PROXY_TANGENT_OFFSET_SCALE+x}" ]]; then
  if [[ "${PROXY_ANCHOR_MODE}" == "ray_surface" ]]; then
    PROXY_TANGENT_OFFSET_SCALE="0.0"
  else
    PROXY_TANGENT_OFFSET_SCALE="1.0"
  fi
fi
PROXY_TANGENT_SCALE_MULTIPLIER="${PROXY_TANGENT_SCALE_MULTIPLIER:-1.0}"
PROXY_NORMAL_SCALE_RATIO="${PROXY_NORMAL_SCALE_RATIO:-0.35}"
if [[ -z "${PROXY_NORMAL_OFFSET_RATIO+x}" ]]; then
  if [[ "${PROXY_ANCHOR_MODE}" == "ray_surface" ]]; then
    PROXY_NORMAL_OFFSET_RATIO="0.02"
  else
    PROXY_NORMAL_OFFSET_RATIO="0.10"
  fi
fi
PROXY_OPACITY_SCALE="${PROXY_OPACITY_SCALE:-1.0}"
MIN_PROXY_SCALE="${MIN_PROXY_SCALE:-1e-4}"

if [[ -n "${CONTINUE_STEPS:-}" ]]; then
  START_PLY_ITERATION="${START_PLY_ITERATION:-${MIP_ITERATION}}"
  FINAL_ITERATION="${FINAL_ITERATION:-$((START_PLY_ITERATION + CONTINUE_STEPS))}"
  ITERATIONS="${ITERATIONS:-${FINAL_ITERATION}}"
else
  START_PLY_ITERATION="${START_PLY_ITERATION:-0}"
  ITERATIONS="${ITERATIONS:-4000}"
  FINAL_ITERATION="${FINAL_ITERATION:-${ITERATIONS}}"
fi
TRAIN_DATA_DEVICE="${TRAIN_DATA_DEVICE:-cpu}"
KERNEL_SIZE="${KERNEL_SIZE:-0.1}"
START_PLY_ACTIVE_SH_DEGREE="${START_PLY_ACTIVE_SH_DEGREE:--1}"
RAY_JITTER="${RAY_JITTER:-0}"
RESAMPLE_GT_IMAGE="${RESAMPLE_GT_IMAGE:-0}"
SAMPLE_MORE_HIGHRES="${SAMPLE_MORE_HIGHRES:-0}"
DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-500}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-15000}"
DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-100}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
OPACITY_RESET_INTERVAL="${OPACITY_RESET_INTERVAL:-3000}"

EVAL_IMAGES_SUBDIR="${EVAL_IMAGES_SUBDIR:-images_2}"
RUN_RENDER_AFTER="${RUN_RENDER_AFTER:-1}"
RUN_METRICS_AFTER="${RUN_METRICS_AFTER:-1}"
RUN_TRAIN_AFTER="${RUN_TRAIN_AFTER:-1}"
AUTO_PREPARE_BASELINE_RENDERS="${AUTO_PREPARE_BASELINE_RENDERS:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
FORCE_PREPARE_SUPERVISION="${FORCE_PREPARE_SUPERVISION:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

if [[ "${RUN_TRAIN_AFTER}" != "1" ]]; then
  RUN_RENDER_AFTER=0
  RUN_METRICS_AFTER=0
fi

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[mip-surface-proxy-v0] missing scene root: ${SCENE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${MIP_MODEL_PATH}" ]]; then
  echo "[mip-surface-proxy-v0] missing baseline model path: ${MIP_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SURFACE_STATE_PAYLOAD}" ]]; then
  echo "[mip-surface-proxy-v0] missing surface-state payload: ${SURFACE_STATE_PAYLOAD}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[mip-surface-proxy-v0] scene                : ${SCENE_ROOT}"
echo "[mip-surface-proxy-v0] baseline model       : ${MIP_MODEL_PATH} iter=${MIP_ITERATION}"
echo "[mip-surface-proxy-v0] surface payload      : ${SURFACE_STATE_PAYLOAD}"
echo "[mip-surface-proxy-v0] surface mask key     : ${SURFACE_MASK_KEY}"
echo "[mip-surface-proxy-v0] donor mask key       : ${DONOR_MASK_KEY}"
echo "[mip-surface-proxy-v0] output root          : ${OUTPUT_ROOT}"
echo "[mip-surface-proxy-v0] proxy max donors     : ${PROXY_MAX_DONORS}"
echo "[mip-surface-proxy-v0] proxy anchor/output  : ${PROXY_ANCHOR_MODE}/${PROXY_OUTPUT_MODE}"
echo "[mip-surface-proxy-v0] proxy ray mode       : ${PROXY_RAY_SURFACE_MODE}"
echo "[mip-surface-proxy-v0] proxy ray relax      : ${PROXY_RAY_RELAX_UNANCHORED_WITH_MESH} gap=${PROXY_RAY_RELAXED_MAX_HIT_TO_DONOR_GAP} ratio=${PROXY_RAY_RELAXED_MAX_HIT_TO_DONOR_GAP_RATIO} center_px=${PROXY_RAY_RELAXED_PROJECTED_CENTER_TOLERANCE_PX}"
if [[ -n "${PROXY_MESH_PATH}" ]]; then
  echo "[mip-surface-proxy-v0] proxy mesh          : ${PROXY_MESH_PATH}"
else
  echo "[mip-surface-proxy-v0] proxy mesh          : infer from surface-state summary"
fi
echo "[mip-surface-proxy-v0] train supervision    : ${TRAIN_IMAGES_SUBDIR} split=${SUPERVISION_SPLIT}"
echo "[mip-surface-proxy-v0] eval render          : ${EVAL_IMAGES_SUBDIR}"
echo "[mip-surface-proxy-v0] train iteration      : ${START_PLY_ITERATION} -> ${FINAL_ITERATION}"
echo "[mip-surface-proxy-v0] densify              : from=${DENSIFY_FROM_ITER} until=${DENSIFY_UNTIL_ITER} interval=${DENSIFICATION_INTERVAL} grad=${DENSIFY_GRAD_THRESHOLD}"

if [[ "${RUN_TRAIN_AFTER}" == "1" && ! -d "${BASELINE_RENDER_ROOT}" ]]; then
  if [[ "${AUTO_PREPARE_BASELINE_RENDERS}" != "1" ]]; then
    echo "[mip-surface-proxy-v0] missing baseline render root: ${BASELINE_RENDER_ROOT}" >&2
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

AUGMENTED_PLY_PATH="${AUGMENTED_MODEL_PATH}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply"
if [[ "${FORCE_REBUILD}" == "1" || ! -f "${AUGMENTED_PLY_PATH}" ]]; then
  echo
  echo "[1/4] build surface-proxy augmented field"
  BUILD_CMD=(
    "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/build_surface_proxy_augmented_field_v0.py"
    --scene_root "${SCENE_ROOT}"
    --model_path "${MIP_MODEL_PATH}"
    --iteration "${MIP_ITERATION}"
    --mask_payload_path "${SURFACE_STATE_PAYLOAD}"
    --surface_mask_key "${SURFACE_MASK_KEY}"
    --donor_mask_key "${DONOR_MASK_KEY}"
    --output_root "${EXPORT_ROOT}"
    --images_subdir "${EXPORT_PREVIEW_IMAGES_SUBDIR}"
    --split "${EXPORT_PREVIEW_SPLIT}"
    --max_views "${EXPORT_PREVIEW_MAX_VIEWS}"
    --max_donors "${PROXY_MAX_DONORS}"
    --proxy_anchor_mode "${PROXY_ANCHOR_MODE}"
    --proxy_output_mode "${PROXY_OUTPUT_MODE}"
    --mesh_path "${PROXY_MESH_PATH}"
    --proxy_ray_camera_stride "${PROXY_RAY_CAMERA_STRIDE}"
    --proxy_ray_max_views "${PROXY_RAY_MAX_VIEWS}"
    --proxy_ray_depth_min "${PROXY_RAY_DEPTH_MIN}"
    --proxy_ray_chunk "${PROXY_RAY_CHUNK}"
    --proxy_ray_surface_mode "${PROXY_RAY_SURFACE_MODE}"
    --proxy_ray_max_hit_to_donor_gap "${PROXY_RAY_MAX_HIT_TO_DONOR_GAP}"
    --proxy_ray_max_hit_to_donor_gap_ratio "${PROXY_RAY_MAX_HIT_TO_DONOR_GAP_RATIO}"
    --proxy_ray_min_depth_scale "${PROXY_RAY_MIN_DEPTH_SCALE}"
    --proxy_ray_projected_center_tolerance_px "${PROXY_RAY_PROJECTED_CENTER_TOLERANCE_PX}"
    --proxy_ray_relaxed_max_hit_to_donor_gap "${PROXY_RAY_RELAXED_MAX_HIT_TO_DONOR_GAP}"
    --proxy_ray_relaxed_max_hit_to_donor_gap_ratio "${PROXY_RAY_RELAXED_MAX_HIT_TO_DONOR_GAP_RATIO}"
    --proxy_ray_relaxed_projected_center_tolerance_px "${PROXY_RAY_RELAXED_PROJECTED_CENTER_TOLERANCE_PX}"
    --proxy_tangent_offset_scale "${PROXY_TANGENT_OFFSET_SCALE}"
    --proxy_tangent_scale_multiplier "${PROXY_TANGENT_SCALE_MULTIPLIER}"
    --proxy_normal_scale_ratio "${PROXY_NORMAL_SCALE_RATIO}"
    --proxy_normal_offset_ratio "${PROXY_NORMAL_OFFSET_RATIO}"
    --proxy_opacity_scale "${PROXY_OPACITY_SCALE}"
    --min_proxy_scale "${MIN_PROXY_SCALE}"
  )
  if [[ "${PROXY_RAY_FALLBACK_TO_ANCHOR}" == "1" ]]; then
    BUILD_CMD+=(--proxy_ray_fallback_to_anchor)
  fi
  if [[ "${PROXY_RAY_RELAX_UNANCHORED_WITH_MESH}" == "1" ]]; then
    BUILD_CMD+=(--proxy_ray_relax_unanchored_with_mesh)
  fi
  if [[ "${PROXY_RAY_PRESERVE_SCREEN_SCALE}" != "1" ]]; then
    BUILD_CMD+=(--proxy_ray_disable_preserve_screen_scale)
  fi
  OMP_NUM_THREADS=1 "${BUILD_CMD[@]}"
else
  echo
  echo "[1/4] reuse existing augmented field"
fi

SUPERVISION_SUMMARY_PATH="${NAMED_SUPERVISION_DIR}/prepare_named_mip_render_supervision_v0_summary.json"
if [[ "${RUN_TRAIN_AFTER}" != "1" ]]; then
  echo
  echo "[2/4] skip camera-name mip supervision by RUN_TRAIN_AFTER=0"
elif [[ "${FORCE_PREPARE_SUPERVISION}" == "1" || ! -f "${SUPERVISION_SUMMARY_PATH}" ]]; then
  echo
  echo "[2/4] prepare camera-name mip supervision"
  OMP_NUM_THREADS=1 "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/prepare_named_mip_render_supervision_v0.py" \
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

FINAL_CHECKPOINT_PATH="${AUGMENTED_MODEL_PATH}/chkpnt${FINAL_ITERATION}.pth"
if [[ "${RUN_TRAIN_AFTER}" != "1" ]]; then
  echo
  echo "[3/4] skip training by RUN_TRAIN_AFTER=0"
elif [[ "${FORCE_TRAIN}" == "1" || ! -f "${FINAL_CHECKPOINT_PATH}" ]]; then
  echo
  echo "[3/4] train surface-proxy augmented field against baseline mip renders"
  TRAIN_CMD=(
    "${PYTHON_BIN}" -u "${SOF_ROOT}/train_vanilla_mip_from_init_v0.py"
    -s "${SCENE_ROOT}"
    -i "${TRAIN_IMAGES_SUBDIR}"
    -m "${AUGMENTED_MODEL_PATH}"
    --eval
    --data_device "${TRAIN_DATA_DEVICE}"
    --kernel_size "${KERNEL_SIZE}"
    --vanilla_mip_mode
    --start_ply "${AUGMENTED_PLY_PATH}"
    --start_ply_iteration "${START_PLY_ITERATION}"
    --start_ply_active_sh_degree "${START_PLY_ACTIVE_SH_DEGREE}"
    --global_image_dir "${NAMED_SUPERVISION_DIR}"
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
  if [[ -f "${SPLATTING_CONFIG_PATH}" ]]; then
    TRAIN_CMD+=(--splatting_config "${SPLATTING_CONFIG_PATH}")
  fi
  OMP_NUM_THREADS=1 "${TRAIN_CMD[@]}"
else
  echo
  echo "[3/4] checkpoint exists, skipping training: ${FINAL_CHECKPOINT_PATH}"
fi

if [[ "${RUN_RENDER_AFTER}" == "1" ]]; then
  echo
  echo "[4/4] render eval split"
  OMP_NUM_THREADS=1 "${PYTHON_BIN}" -u "${SOF_ROOT}/render.py" \
    -m "${AUGMENTED_MODEL_PATH}" \
    -s "${SCENE_ROOT}" \
    -i "${EVAL_IMAGES_SUBDIR}" \
    --iteration "${FINAL_ITERATION}" \
    --init_type sfm \
    --eval \
    --skip_train
fi

if [[ "${RUN_METRICS_AFTER}" == "1" ]]; then
  echo
  echo "[metrics] summarize eval PSNR/SSIM"
  OMP_NUM_THREADS=1 "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/summarize_sof_render_metrics_v0.py" \
    --model_dir "${AUGMENTED_MODEL_PATH}" \
    --iteration "${FINAL_ITERATION}" \
    --split test
fi

echo
echo "[done] augmented model root: ${AUGMENTED_MODEL_PATH}"
echo "[done] augmented init ply  : ${AUGMENTED_PLY_PATH}"
echo "[done] supervision dir     : ${NAMED_SUPERVISION_DIR}"
echo "[done] eval renders        : ${AUGMENTED_MODEL_PATH}/test/ours_${FINAL_ITERATION}/renders"
if [[ "${RUN_METRICS_AFTER}" == "1" ]]; then
  echo "[done] metrics             : ${AUGMENTED_MODEL_PATH}/results_psnr_ssim.json"
fi
