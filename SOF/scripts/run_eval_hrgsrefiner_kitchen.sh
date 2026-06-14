#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
PRIORS_DIR="${PRIORS_DIR:-${SCENE_ROOT}/priors}"
VGGT_ROOT="${VGGT_ROOT:-${WORK_ROOT}/vggt}"

BASE_GS_MODEL_PATH="${BASE_GS_MODEL_PATH:-${SOF_ROOT}/output/hrgs_joint_smoke/kitchen_sof_lr_base_500}"
BASE_CKPT="${BASE_CKPT:-${BASE_GS_MODEL_PATH}/chkpnt500.pth}"
BASE_START_PLY="${BASE_START_PLY:-}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${SOF_ROOT}/output/hrgs_train_formal/${SCENE_NAME}_v0}"
REFINER_CHECKPOINT="${REFINER_CHECKPOINT:-}"
if [[ -z "${REFINER_CHECKPOINT}" ]]; then
  REF_LIST="$(find "${TRAIN_OUTPUT_DIR}/checkpoints" -maxdepth 1 -name 'hrgsrefiner_step_*.pt' | sort)"
  if [[ -z "${REF_LIST}" ]]; then
    echo "[eval-hrgs-kitchen] no refiner checkpoints found under ${TRAIN_OUTPUT_DIR}/checkpoints" >&2
    exit 1
  fi
  REFINER_CHECKPOINT="$(printf '%s\n' "${REF_LIST}" | tail -n 1)"
fi

CHECKPOINT_TAG="${CHECKPOINT_TAG:-$(basename "${REFINER_CHECKPOINT}" .pt)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/hrgs_eval}"
RUNNER_OUT="${RUNNER_OUT:-${OUTPUT_ROOT}/${SCENE_NAME}_${CHECKPOINT_TAG}}"
MESHGS_OUT="${MESHGS_OUT:-${OUTPUT_ROOT}/${SCENE_NAME}_meshgs_${CHECKPOINT_TAG}}"
JOINT_OUT="${JOINT_OUT:-${OUTPUT_ROOT}/${SCENE_NAME}_joint_${CHECKPOINT_TAG}}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
DOWNSTREAM_IMAGES_SUBDIR="${DOWNSTREAM_IMAGES_SUBDIR:-${TARGET_IMAGES_SUBDIR}}"
MAX_VIEWS="${MAX_VIEWS:-2}"
MESHGS_ITERATIONS="${MESHGS_ITERATIONS:-200}"
JOINT_FINAL_ITER="${JOINT_FINAL_ITER:-550}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONUNBUFFERED=1

REQUIRED_PATHS=("${SCENE_ROOT}" "${PRIORS_DIR}" "${VGGT_ROOT}" "${BASE_GS_MODEL_PATH}" "${REFINER_CHECKPOINT}")
if [[ -n "${BASE_START_PLY}" ]]; then
  REQUIRED_PATHS+=("${BASE_START_PLY}")
else
  REQUIRED_PATHS+=("${BASE_CKPT}")
fi

for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[eval-hrgs-kitchen] required path not found: ${path}" >&2
    exit 1
  fi
done

echo "[eval-hrgs-kitchen] scene root         : ${SCENE_ROOT}"
echo "[eval-hrgs-kitchen] base gs model      : ${BASE_GS_MODEL_PATH}"
if [[ -n "${BASE_START_PLY}" ]]; then
  echo "[eval-hrgs-kitchen] base start ply     : ${BASE_START_PLY}"
else
  echo "[eval-hrgs-kitchen] base checkpoint    : ${BASE_CKPT}"
fi
echo "[eval-hrgs-kitchen] refiner checkpoint : ${REFINER_CHECKPOINT}"
echo "[eval-hrgs-kitchen] runner out         : ${RUNNER_OUT}"
echo "[eval-hrgs-kitchen] meshgs out         : ${MESHGS_OUT}"
echo "[eval-hrgs-kitchen] joint out          : ${JOINT_OUT}"
echo "[eval-hrgs-kitchen] source grid        : ${SOURCE_IMAGES_SUBDIR}"
echo "[eval-hrgs-kitchen] target grid        : ${TARGET_IMAGES_SUBDIR}"
echo "[eval-hrgs-kitchen] downstream grid    : ${DOWNSTREAM_IMAGES_SUBDIR}"

echo
echo "[1/3] scene runner with trained refiner"
"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/run_hrgsrefiner_scene.py" \
  --scene_root "${SCENE_ROOT}" \
  --gs_model_path "${BASE_GS_MODEL_PATH}" \
  --output_dir "${RUNNER_OUT}" \
  --refiner_checkpoint "${REFINER_CHECKPOINT}" \
  --vggt_root "${VGGT_ROOT}" \
  --source_images_subdir "${SOURCE_IMAGES_SUBDIR}" \
  --target_images_subdir "${TARGET_IMAGES_SUBDIR}" \
  --priors_dir "${PRIORS_DIR}" \
  --require_priors \
  --max_views "${MAX_VIEWS}"

echo
echo "[2/3] carrier payload -> meshGS prior"
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
echo "[3/3] action payload -> joint finetune"
JOINT_CMD=(
  CUDA_LAUNCH_BLOCKING=1 "${PYTHON_BIN}" "${SOF_ROOT}/train.py"
  --splatting_config configs/hierarchical.json
  -s "${SCENE_ROOT}"
  -i "${DOWNSTREAM_IMAGES_SUBDIR}"
  -m "${JOINT_OUT}"
)
if [[ -n "${BASE_START_PLY}" ]]; then
  JOINT_CMD+=(--start_ply "${BASE_START_PLY}")
else
  JOINT_CMD+=(--start_checkpoint "${BASE_CKPT}")
fi
JOINT_CMD+=(
  --gaussian_action_payload "${RUNNER_OUT}/gs_action_payload.pt"
  --gaussian_action_min_weight 0.0
  --gaussian_action_update_scale 1.0
  --iterations "${JOINT_FINAL_ITER}"
  --test_iterations "${JOINT_FINAL_ITER}"
  --save_iterations "${JOINT_FINAL_ITER}"
  --checkpoint_iterations "${JOINT_FINAL_ITER}"
  --eval
)
env "${JOINT_CMD[@]}"

echo
echo "[done] runner out : ${RUNNER_OUT}"
echo "[done] meshgs out : ${MESHGS_OUT}"
echo "[done] joint out  : ${JOINT_OUT}"
