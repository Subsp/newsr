#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PRIORS_DIR="${PRIORS_DIR:-${SCENE_ROOT}/priors}"
VGGT_ROOT="${VGGT_ROOT:-${WORK_ROOT}/vggt}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
MIP_INPUT_MODEL_PATH="${MIP_INPUT_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input}"
MIP_ITERATION="${MIP_ITERATION:-30000}"
PREPARE_INPUT_IF_MISSING="${PREPARE_INPUT_IF_MISSING:-1}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${SOF_ROOT}/output/hrgs_train_formal/${SCENE_NAME}_mipstart_v0}"
REFINER_CHECKPOINT="${REFINER_CHECKPOINT:-}"
if [[ -z "${REFINER_CHECKPOINT}" ]]; then
  REF_LIST="$(find "${TRAIN_OUTPUT_DIR}/checkpoints" -maxdepth 1 -name 'hrgsrefiner_step_*.pt' | sort)"
  if [[ -z "${REF_LIST}" ]]; then
    echo "[eval-hrgs-mipstart] no refiner checkpoints found under ${TRAIN_OUTPUT_DIR}/checkpoints" >&2
    exit 1
  fi
  REFINER_CHECKPOINT="$(printf '%s\n' "${REF_LIST}" | tail -n 1)"
fi

REFINER_TAG="$(basename "${REFINER_CHECKPOINT}" .pt)"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-mipstart_${REFINER_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/hrgs_eval}"
RUNNER_OUT="${RUNNER_OUT:-${OUTPUT_ROOT}/${SCENE_NAME}_${CHECKPOINT_TAG}}"
MESHGS_OUT="${MESHGS_OUT:-${OUTPUT_ROOT}/${SCENE_NAME}_meshgs_${CHECKPOINT_TAG}}"
MERGED_OUT="${MERGED_OUT:-${OUTPUT_ROOT}/${SCENE_NAME}_merged_${CHECKPOINT_TAG}}"
JOINT_OUT="${JOINT_OUT:-${OUTPUT_ROOT}/${SCENE_NAME}_joint_${CHECKPOINT_TAG}}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
DOWNSTREAM_IMAGES_SUBDIR="${DOWNSTREAM_IMAGES_SUBDIR:-images_2}"
MAX_VIEWS="${MAX_VIEWS:-2}"
VISIBILITY_DOWNSAMPLE="${VISIBILITY_DOWNSAMPLE:-8}"
VISIBILITY_TOPK="${VISIBILITY_TOPK:-4}"
VISIBILITY_MAX_VISIBLE="${VISIBILITY_MAX_VISIBLE:-30000}"
VISIBILITY_MAX_PATCH_RADIUS="${VISIBILITY_MAX_PATCH_RADIUS:-1}"
MESHGS_ITERATIONS="${MESHGS_ITERATIONS:-200}"
JOINT_FINAL_ITER="${JOINT_FINAL_ITER:-550}"
SURFACE_MIN_CONFIDENCE="${SURFACE_MIN_CONFIDENCE:-0.0}"
SURFACE_MAX_DISAGREEMENT="${SURFACE_MAX_DISAGREEMENT:-1.0}"
SURFACE_MIN_VIEWS="${SURFACE_MIN_VIEWS:-1}"
DETAIL_ATTACH_SCALE="${DETAIL_ATTACH_SCALE:-0.1}"
DETAIL_UPDATE_SCALE="${DETAIL_UPDATE_SCALE:-1.0}"
DETAIL_DETAIL_SCALE="${DETAIL_DETAIL_SCALE:-1.0}"
DETAIL_PRIOR_COLOR_SCALE="${DETAIL_PRIOR_COLOR_SCALE:-1.0}"
DETAIL_RADIUS_GATE="${DETAIL_RADIUS_GATE:-1}"
DETAIL_RADIUS_IMAGES_SUBDIR="${DETAIL_RADIUS_IMAGES_SUBDIR:-images_2}"
DETAIL_RADIUS_REF_PX="${DETAIL_RADIUS_REF_PX:-96.0}"
DETAIL_RADIUS_MIN_GATE="${DETAIL_RADIUS_MIN_GATE:-0.1}"
DETAIL_RADIUS_MAX_VIEWS="${DETAIL_RADIUS_MAX_VIEWS:-32}"
RUN_MERGED_RENDER_SANITY="${RUN_MERGED_RENDER_SANITY:-0}"
USE_ACTION_PAYLOAD="${USE_ACTION_PAYLOAD:-1}"
USE_ROUTE_PAYLOAD="${USE_ROUTE_PAYLOAD:-1}"
ROUTE_PAYLOAD="${ROUTE_PAYLOAD:-${RUNNER_OUT}/route_payload_v0.pt}"
ROUTE_SOURCE_ACTION_PAYLOAD="${ROUTE_SOURCE_ACTION_PAYLOAD:-${RUNNER_OUT}/gs_action_payload.pt}"
ROUTE_SOURCE_CARRIER_PAYLOAD="${ROUTE_SOURCE_CARRIER_PAYLOAD:-${RUNNER_OUT}/carrier_payload.npz}"
ROUTE_IMAGES_SUBDIR="${ROUTE_IMAGES_SUBDIR:-${DETAIL_RADIUS_IMAGES_SUBDIR}}"
ROUTE_MAX_VIEWS="${ROUTE_MAX_VIEWS:-${DETAIL_RADIUS_MAX_VIEWS}}"
ROUTE_RADIUS_REF_PX="${ROUTE_RADIUS_REF_PX:-${DETAIL_RADIUS_REF_PX}}"
ROUTE_RADIUS_TEMPERATURE_PX="${ROUTE_RADIUS_TEMPERATURE_PX:-24.0}"
ROUTE_RADIUS_GATE_MIN="${ROUTE_RADIUS_GATE_MIN:-${DETAIL_RADIUS_MIN_GATE}}"
ROUTE_SURFACE_CONFIDENCE_FLOOR="${ROUTE_SURFACE_CONFIDENCE_FLOOR:-0.05}"
ROUTE_SURFACE_DISTANCE_SCALE="${ROUTE_SURFACE_DISTANCE_SCALE:-2.0}"
ROUTE_OPACITY_CENTER="${ROUTE_OPACITY_CENTER:-0.35}"
ROUTE_OPACITY_TEMPERATURE="${ROUTE_OPACITY_TEMPERATURE:-0.15}"
ROUTE_DETAIL_BOOST="${ROUTE_DETAIL_BOOST:-1.0}"
ROUTE_EXECUTION_MODE="${ROUTE_EXECUTION_MODE:-detail_preserve_v0p8}"
ROUTE_DETAIL_PROTECT_BETA="${ROUTE_DETAIL_PROTECT_BETA:-0.7}"
ROUTE_DETAIL_ATTACH_STRENGTH="${ROUTE_DETAIL_ATTACH_STRENGTH:-0.0}"
ROUTE_DETAIL_GEOMETRY_STRENGTH="${ROUTE_DETAIL_GEOMETRY_STRENGTH:-0.0}"
ROUTE_PRIOR_COLOR_STRENGTH_SCALE="${ROUTE_PRIOR_COLOR_STRENGTH_SCALE:-0.25}"
ROUTE_SUPPRESS_UPDATE_FLOOR="${ROUTE_SUPPRESS_UPDATE_FLOOR:-0.15}"
ROUTE_SUPPRESS_OPACITY_SCALE="${ROUTE_SUPPRESS_OPACITY_SCALE:-0.25}"
ROUTE_MIN_SCALE_GATE="${ROUTE_MIN_SCALE_GATE:-0.3}"
ROUTE_SCALE_GATE_POWER="${ROUTE_SCALE_GATE_POWER:-1.0}"
ROUTE_SCALE_GATE_SUPPRESS_COUPLING="${ROUTE_SCALE_GATE_SUPPRESS_COUPLING:-1.0}"
ROUTE_RISKY_USEFUL_OPACITY_COUPLING="${ROUTE_RISKY_USEFUL_OPACITY_COUPLING:-0.35}"
ROUTE_RISKY_USEFUL_SCALE_COUPLING="${ROUTE_RISKY_USEFUL_SCALE_COUPLING:-0.2}"
ROUTE_HARMFUL_SCALE_COUPLING="${ROUTE_HARMFUL_SCALE_COUPLING:-1.0}"
GAUSSIAN_ACTION_UPDATE_SCALE="${GAUSSIAN_ACTION_UPDATE_SCALE:-1.0}"
GAUSSIAN_ACTION_ATTACH_SCALE="${GAUSSIAN_ACTION_ATTACH_SCALE:-1.0}"
GAUSSIAN_ACTION_DETAIL_SCALE="${GAUSSIAN_ACTION_DETAIL_SCALE:-1.0}"
GAUSSIAN_ACTION_PRIOR_COLOR_SCALE="${GAUSSIAN_ACTION_PRIOR_COLOR_SCALE:-1.0}"
FREEZE_DETAIL_GEOMETRY="${FREEZE_DETAIL_GEOMETRY:-}"
if [[ -z "${FREEZE_DETAIL_GEOMETRY}" ]]; then
  if [[ "${USE_ROUTE_PAYLOAD}" == "1" ]]; then
    FREEZE_DETAIL_GEOMETRY="1"
  else
    FREEZE_DETAIL_GEOMETRY="0"
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${PREPARE_INPUT_IF_MISSING}" == "1" && ! -e "${MIP_INPUT_MODEL_PATH}/point_cloud" ]]; then
  echo "[eval-hrgs-mipstart] preparing SOF-native mip input field ..."
  SCENE_NAME="${SCENE_NAME}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
  MIP_MODEL_PATH="${MIP_MODEL_PATH}" \
  OUTPUT_MODEL_PATH="${MIP_INPUT_MODEL_PATH}" \
  MIP_ITERATION="${MIP_ITERATION}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SOF_ROOT}/scripts/run_prepare_hrgs_mip_input_scene.sh"
fi

if [[ ! -e "${MIP_INPUT_MODEL_PATH}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply" ]]; then
  echo "[eval-hrgs-mipstart] adapted mip start ply not found under ${MIP_INPUT_MODEL_PATH}" >&2
  exit 1
fi

echo "[eval-hrgs-mipstart] scene root         : ${SCENE_ROOT}"
echo "[eval-hrgs-mipstart] mip model path      : ${MIP_MODEL_PATH}"
echo "[eval-hrgs-mipstart] input field path    : ${MIP_INPUT_MODEL_PATH}"
echo "[eval-hrgs-mipstart] refiner checkpoint  : ${REFINER_CHECKPOINT}"
echo "[eval-hrgs-mipstart] checkpoint tag      : ${CHECKPOINT_TAG}"
echo "[eval-hrgs-mipstart] source grid         : ${SOURCE_IMAGES_SUBDIR}"
echo "[eval-hrgs-mipstart] target grid         : ${TARGET_IMAGES_SUBDIR}"
echo "[eval-hrgs-mipstart] downstream grid     : ${DOWNSTREAM_IMAGES_SUBDIR}"
echo "[eval-hrgs-mipstart] visibility ds       : ${VISIBILITY_DOWNSAMPLE}"
echo "[eval-hrgs-mipstart] detail attach scale : ${DETAIL_ATTACH_SCALE}"
echo "[eval-hrgs-mipstart] detail radius gate  : ${DETAIL_RADIUS_GATE}"
echo "[eval-hrgs-mipstart] use action payload  : ${USE_ACTION_PAYLOAD}"
echo "[eval-hrgs-mipstart] use route payload   : ${USE_ROUTE_PAYLOAD}"
if [[ "${USE_ROUTE_PAYLOAD}" == "1" ]]; then
  echo "[eval-hrgs-mipstart] route payload path  : ${ROUTE_PAYLOAD}"
  echo "[eval-hrgs-mipstart] route exec mode     : ${ROUTE_EXECUTION_MODE}"
  echo "[eval-hrgs-mipstart] route detail boost  : ${ROUTE_DETAIL_BOOST}"
fi
echo "[eval-hrgs-mipstart] action scales       : update=${GAUSSIAN_ACTION_UPDATE_SCALE} attach=${GAUSSIAN_ACTION_ATTACH_SCALE} detail=${GAUSSIAN_ACTION_DETAIL_SCALE} prior=${GAUSSIAN_ACTION_PRIOR_COLOR_SCALE}"
echo "[eval-hrgs-mipstart] freeze detail geom  : ${FREEZE_DETAIL_GEOMETRY}"
echo "[eval-hrgs-mipstart] runner out          : ${RUNNER_OUT}"
echo "[eval-hrgs-mipstart] meshgs out          : ${MESHGS_OUT}"
echo "[eval-hrgs-mipstart] merged out          : ${MERGED_OUT}"
echo "[eval-hrgs-mipstart] joint out           : ${JOINT_OUT}"

echo
echo "[1/4] scene runner with trained refiner"
"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/run_hrgsrefiner_scene.py" \
  --scene_root "${SCENE_ROOT}" \
  --gs_model_path "${MIP_INPUT_MODEL_PATH}" \
  --output_dir "${RUNNER_OUT}" \
  --refiner_checkpoint "${REFINER_CHECKPOINT}" \
  --vggt_root "${VGGT_ROOT}" \
  --source_images_subdir "${SOURCE_IMAGES_SUBDIR}" \
  --target_images_subdir "${TARGET_IMAGES_SUBDIR}" \
  --priors_dir "${PRIORS_DIR}" \
  --require_priors \
  --max_views "${MAX_VIEWS}" \
  --visibility_downsample "${VISIBILITY_DOWNSAMPLE}" \
  --visibility_topk "${VISIBILITY_TOPK}" \
  --visibility_max_visible "${VISIBILITY_MAX_VISIBLE}" \
  --visibility_max_patch_radius "${VISIBILITY_MAX_PATCH_RADIUS}"

echo
if [[ "${USE_ROUTE_PAYLOAD}" == "1" ]]; then
  echo "[2/5] export heuristic route payload"
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/export_gaussian_route_payload_v0.py" \
    --detail_model_path "${MIP_INPUT_MODEL_PATH}" \
    --carrier_payload "${ROUTE_SOURCE_CARRIER_PAYLOAD}" \
    --action_payload "${ROUTE_SOURCE_ACTION_PAYLOAD}" \
    --output_path "${ROUTE_PAYLOAD}" \
    --scene_root "${SCENE_ROOT}" \
    --detail_iteration "${MIP_ITERATION}" \
    --images_subdir "${ROUTE_IMAGES_SUBDIR}" \
    --max_views "${ROUTE_MAX_VIEWS}" \
    --radius_ref_px "${ROUTE_RADIUS_REF_PX}" \
    --radius_temperature_px "${ROUTE_RADIUS_TEMPERATURE_PX}" \
    --radius_gate_min "${ROUTE_RADIUS_GATE_MIN}" \
    --surface_confidence_floor "${ROUTE_SURFACE_CONFIDENCE_FLOOR}" \
    --surface_distance_scale "${ROUTE_SURFACE_DISTANCE_SCALE}" \
    --opacity_center "${ROUTE_OPACITY_CENTER}" \
    --opacity_temperature "${ROUTE_OPACITY_TEMPERATURE}" \
    --suppress_update_floor "${ROUTE_SUPPRESS_UPDATE_FLOOR}" \
    --detail_boost "${ROUTE_DETAIL_BOOST}"
fi

echo
echo "[3/5] carrier payload -> meshGS prior"
CUDA_LAUNCH_BLOCKING=1 "${PYTHON_BIN}" "${SOF_ROOT}/train_meshgs_prior_v0.py" \
  --splatting_config configs/hierarchical.json \
  -s "${SCENE_ROOT}" \
  -i "${DOWNSTREAM_IMAGES_SUBDIR}" \
  -m "${MESHGS_OUT}" \
  --carrier_payload "${RUNNER_OUT}/carrier_payload.npz" \
  --prior_dir "${PRIORS_DIR}" \
  --meshgs_min_confidence 0.0 \
  --meshgs_min_views 1 \
  --meshgs_max_disagreement 1.0 \
  --iterations "${MESHGS_ITERATIONS}" \
  --save_iterations "${MESHGS_ITERATIONS}"

echo
echo "[4/5] merge G_surface + G_detail init"
MERGE_ROUTE_PAYLOAD=""
if [[ "${USE_ROUTE_PAYLOAD}" == "1" ]]; then
  MERGE_ROUTE_PAYLOAD="${ROUTE_PAYLOAD}"
fi
SCENE_NAME="${SCENE_NAME}" \
SCENE_ROOT="${SCENE_ROOT}" \
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
DETAIL_MODEL_PATH="${MIP_INPUT_MODEL_PATH}" \
DETAIL_ITERATION="${MIP_ITERATION}" \
CARRIER_PAYLOAD="${RUNNER_OUT}/carrier_payload.npz" \
ACTION_PAYLOAD="${RUNNER_OUT}/gs_action_payload.pt" \
ROUTE_PAYLOAD="${MERGE_ROUTE_PAYLOAD}" \
OUTPUT_MODEL_PATH="${MERGED_OUT}" \
CHECKPOINT_TAG="${CHECKPOINT_TAG}" \
SURFACE_MIN_CONFIDENCE="${SURFACE_MIN_CONFIDENCE}" \
SURFACE_MAX_DISAGREEMENT="${SURFACE_MAX_DISAGREEMENT}" \
SURFACE_MIN_VIEWS="${SURFACE_MIN_VIEWS}" \
DETAIL_ATTACH_SCALE="${DETAIL_ATTACH_SCALE}" \
DETAIL_UPDATE_SCALE="${DETAIL_UPDATE_SCALE}" \
DETAIL_DETAIL_SCALE="${DETAIL_DETAIL_SCALE}" \
DETAIL_PRIOR_COLOR_SCALE="${DETAIL_PRIOR_COLOR_SCALE}" \
DETAIL_RADIUS_GATE="${DETAIL_RADIUS_GATE}" \
DETAIL_RADIUS_IMAGES_SUBDIR="${DETAIL_RADIUS_IMAGES_SUBDIR}" \
DETAIL_RADIUS_REF_PX="${DETAIL_RADIUS_REF_PX}" \
DETAIL_RADIUS_MIN_GATE="${DETAIL_RADIUS_MIN_GATE}" \
DETAIL_RADIUS_MAX_VIEWS="${DETAIL_RADIUS_MAX_VIEWS}" \
ROUTE_EXECUTION_MODE="${ROUTE_EXECUTION_MODE}" \
ROUTE_DETAIL_PROTECT_BETA="${ROUTE_DETAIL_PROTECT_BETA}" \
ROUTE_DETAIL_ATTACH_STRENGTH="${ROUTE_DETAIL_ATTACH_STRENGTH}" \
ROUTE_DETAIL_GEOMETRY_STRENGTH="${ROUTE_DETAIL_GEOMETRY_STRENGTH}" \
ROUTE_PRIOR_COLOR_STRENGTH_SCALE="${ROUTE_PRIOR_COLOR_STRENGTH_SCALE}" \
ROUTE_SUPPRESS_UPDATE_FLOOR="${ROUTE_SUPPRESS_UPDATE_FLOOR}" \
ROUTE_SUPPRESS_OPACITY_SCALE="${ROUTE_SUPPRESS_OPACITY_SCALE}" \
ROUTE_MIN_SCALE_GATE="${ROUTE_MIN_SCALE_GATE}" \
ROUTE_SCALE_GATE_POWER="${ROUTE_SCALE_GATE_POWER}" \
ROUTE_SCALE_GATE_SUPPRESS_COUPLING="${ROUTE_SCALE_GATE_SUPPRESS_COUPLING}" \
ROUTE_RISKY_USEFUL_OPACITY_COUPLING="${ROUTE_RISKY_USEFUL_OPACITY_COUPLING}" \
ROUTE_RISKY_USEFUL_SCALE_COUPLING="${ROUTE_RISKY_USEFUL_SCALE_COUPLING}" \
ROUTE_HARMFUL_SCALE_COUPLING="${ROUTE_HARMFUL_SCALE_COUPLING}" \
RUN_RENDER_SANITY="${RUN_MERGED_RENDER_SANITY}" \
RENDER_SANITY_IMAGES_SUBDIR="${DOWNSTREAM_IMAGES_SUBDIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${SOF_ROOT}/scripts/run_prepare_hrgs_surface_detail_init.sh"

echo
echo "[5/5] merged init + action payload -> joint finetune"
JOINT_CMD=(
  "${PYTHON_BIN}" "${SOF_ROOT}/train.py"
  --splatting_config configs/hierarchical.json
  -s "${SCENE_ROOT}"
  -i "${DOWNSTREAM_IMAGES_SUBDIR}"
  -m "${JOINT_OUT}"
  --start_ply "${MERGED_OUT}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply"
  --gaussian_action_min_weight 0.0
  --gaussian_action_update_scale "${GAUSSIAN_ACTION_UPDATE_SCALE}"
  --gaussian_action_attach_scale "${GAUSSIAN_ACTION_ATTACH_SCALE}"
  --gaussian_action_detail_scale "${GAUSSIAN_ACTION_DETAIL_SCALE}"
  --gaussian_action_prior_color_scale "${GAUSSIAN_ACTION_PRIOR_COLOR_SCALE}"
  --iterations "${JOINT_FINAL_ITER}"
  --test_iterations "${JOINT_FINAL_ITER}"
  --save_iterations "${JOINT_FINAL_ITER}"
  --checkpoint_iterations "${JOINT_FINAL_ITER}"
  --eval
)

if [[ "${USE_ACTION_PAYLOAD}" == "1" ]]; then
  JOINT_CMD+=(--gaussian_action_payload "${MERGED_OUT}/merged_action_payload.pt")
fi
if [[ "${FREEZE_DETAIL_GEOMETRY}" == "1" ]]; then
  JOINT_CMD+=(
    --freeze_geometry_mask_payload "${MERGED_OUT}/masks/detail_mask.pt"
    --freeze_geometry_mask_key selected_mask
  )
fi

CUDA_LAUNCH_BLOCKING=1 "${JOINT_CMD[@]}"

echo
echo "[done] runner out : ${RUNNER_OUT}"
if [[ "${USE_ROUTE_PAYLOAD}" == "1" ]]; then
  echo "[done] route path : ${ROUTE_PAYLOAD}"
fi
echo "[done] meshgs out : ${MESHGS_OUT}"
echo "[done] merged out : ${MERGED_OUT}"
echo "[done] joint out  : ${JOINT_OUT}"
