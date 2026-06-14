#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DIAGNOSTIC_MODE="${DIAGNOSTIC_MODE:-both}"
if [[ "${DIAGNOSTIC_MODE}" != "gradient" && "${DIAGNOSTIC_MODE}" != "dropout" && "${DIAGNOSTIC_MODE}" != "both" ]]; then
  echo "[training-diagnostics] unsupported DIAGNOSTIC_MODE: ${DIAGNOSTIC_MODE}" >&2
  exit 1
fi

ARCHIVE_ROOT="${ARCHIVE_ROOT:-/root/autodl-tmp/archive}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${ARCHIVE_ROOT}/${SCENE_NAME}}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
SOURCE_IMAGES_DIR="${SOURCE_IMAGES_DIR:-${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}}"
TARGET_IMAGES_DIR="${TARGET_IMAGES_DIR:-${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}}"

ALIAS_ROOT="${ALIAS_ROOT:-${ARCHIVE_ROOT}/aliases}"
ALIAS_DIR="${ALIAS_DIR:-${ALIAS_ROOT}/${SCENE_NAME}_images8bicubic_to_images2}"

BASE_ITER="${BASE_ITER:-30000}"
DIAG_RUN_ITERS="${DIAG_RUN_ITERS:-600}"
FINAL_ITER="${FINAL_ITER:-$((BASE_ITER + DIAG_RUN_ITERS))}"
DIAGNOSTIC_FROM_ITER="${DIAGNOSTIC_FROM_ITER:-${BASE_ITER}}"
EXPERIMENT_ROOT_NAME="${EXPERIMENT_ROOT_NAME:-training_diagnostics_v0}"
RUN_NAME_DEFAULT="${DIAGNOSTIC_MODE}_from_${BASE_ITER}_to_${FINAL_ITER}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_DEFAULT}}"

TRAIN_BASELINE_IF_MISSING="${TRAIN_BASELINE_IF_MISSING:-0}"
PREPARE_ALIAS_IF_MISSING="${PREPARE_ALIAS_IF_MISSING:-1}"
RUN_RENDER_AFTER="${RUN_RENDER_AFTER:-0}"
RUN_AGGREGATE_AFTER="${RUN_AGGREGATE_AFTER:-1}"
CHECK_INPUTS="${CHECK_INPUTS:-1}"
DRY_RUN="${DRY_RUN:-0}"

BASELINE_SPLATTING_CONFIG="${BASELINE_SPLATTING_CONFIG:-configs/hierarchical.json}"
BASELINE_MODEL_DIR="${BASELINE_MODEL_DIR:-${SOF_ROOT}/output/${SCENE_NAME}_sof_lr_ablation_v1/early4k_soft}"
BASELINE_CKPT="${BASELINE_CKPT:-${BASELINE_MODEL_DIR}/chkpnt${BASE_ITER}.pth}"

RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/${SCENE_NAME}_${EXPERIMENT_ROOT_NAME}/${RUN_NAME}}"
MODEL_DIR="${MODEL_DIR:-${RUN_ROOT}/model}"
DIAGNOSTIC_SUBDIR="${DIAGNOSTIC_SUBDIR:-training_diagnostics_v0}"
DIAGNOSTIC_DIR="${MODEL_DIR}/${DIAGNOSTIC_SUBDIR}"

DIAGNOSTIC_BASIS_MODE="${DIAGNOSTIC_BASIS_MODE:-gaussian_frame}"
DIAGNOSTIC_SURFACE_PAYLOAD="${DIAGNOSTIC_SURFACE_PAYLOAD:-}"

GRAD_SNAPSHOT_INTERVAL="${GRAD_SNAPSHOT_INTERVAL:-100}"
GRAD_TILE_SIZE="${GRAD_TILE_SIZE:-32}"

DROPOUT_INTERVAL="${DROPOUT_INTERVAL:-200}"
DROPOUT_NUM_MASKS="${DROPOUT_NUM_MASKS:-4}"
DROPOUT_TILE_SIZE="${DROPOUT_TILE_SIZE:-48}"
DROPOUT_KEEP_RATIO="${DROPOUT_KEEP_RATIO:-0.75}"
DROPOUT_LOSS_MODE="${DROPOUT_LOSS_MODE:-masked_l1}"
DROPOUT_ALPHA_THRESHOLD="${DROPOUT_ALPHA_THRESHOLD:-0.1}"
DROPOUT_MIN_ACTIVE_PIXELS="${DROPOUT_MIN_ACTIVE_PIXELS:-256}"

AGGREGATE_LATEST_N="${AGGREGATE_LATEST_N:-12}"
PYTHON_BIN="${PYTHON_BIN:-}"

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]]; then
    return 0
  fi

  local candidates=()
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python")
  fi
  candidates+=(
    "/opt/miniconda3/envs/local_gaussian/bin/python"
    "/opt/miniconda3/bin/python"
    "/root/miniconda3/envs/sof/bin/python"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      PYTHON_BIN="${candidate}"
      return 0
    fi
  done

  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return 0
  fi

  echo "[training-diagnostics] failed to locate a usable python interpreter." >&2
  exit 1
}

run_cmd() {
  echo
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

resolve_python_bin

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1 || {
    echo "[training-diagnostics] python does not have torch: ${PYTHON_BIN}" >&2
    exit 1
  }
fi

mkdir -p "${ALIAS_ROOT}" "${RUN_ROOT}" "${MODEL_DIR}"

if [[ "${CHECK_INPUTS}" == "1" && "${DRY_RUN}" != "1" ]]; then
  for path in "${SCENE_ROOT}" "${TARGET_IMAGES_DIR}"; do
    if [[ ! -e "${path}" ]]; then
      echo "[training-diagnostics] required path not found: ${path}" >&2
      exit 1
    fi
  done
  if [[ ! -d "${SCENE_ROOT}/sparse/0" ]]; then
    echo "[training-diagnostics] missing sparse/0: ${SCENE_ROOT}/sparse/0" >&2
    exit 1
  fi
fi

echo "[training-diagnostics] scene              : ${SCENE_NAME}"
echo "[training-diagnostics] mode               : ${DIAGNOSTIC_MODE}"
echo "[training-diagnostics] scene root         : ${SCENE_ROOT}"
echo "[training-diagnostics] alias dir          : ${ALIAS_DIR}"
echo "[training-diagnostics] baseline model     : ${BASELINE_MODEL_DIR}"
echo "[training-diagnostics] baseline ckpt      : ${BASELINE_CKPT}"
echo "[training-diagnostics] run root           : ${RUN_ROOT}"
echo "[training-diagnostics] model dir          : ${MODEL_DIR}"
echo "[training-diagnostics] diagnostic dir     : ${DIAGNOSTIC_DIR}"
echo "[training-diagnostics] python            : ${PYTHON_BIN}"
echo "[training-diagnostics] iterations         : ${BASE_ITER} -> ${FINAL_ITER}"
echo "[training-diagnostics] diagnostic from    : ${DIAGNOSTIC_FROM_ITER}"

if [[ "${PREPARE_ALIAS_IF_MISSING}" == "1" && ( ! -d "${ALIAS_DIR}" || ! -e "${ALIAS_DIR}/sparse/0" ) ]]; then
  echo
  echo "[1/5] prepare pseudo-scene alias"
  (
    cd "${SOF_ROOT}"
    run_cmd "${PYTHON_BIN}" scripts/prepare_colmap_pseudo_sr_scene.py \
      --scene_root "${SCENE_ROOT}" \
      --scene_alias_dir "${ALIAS_DIR}" \
      --source_images_subdir "${SOURCE_IMAGES_SUBDIR}" \
      --target_images_subdir "${TARGET_IMAGES_SUBDIR}" \
      --resize_filter bicubic
  )
else
  echo
  echo "[1/5] reuse existing alias: ${ALIAS_DIR}"
fi

echo
echo "[2/5] ensure early4k_soft baseline"
if [[ ! -f "${BASELINE_CKPT}" ]]; then
  if [[ "${TRAIN_BASELINE_IF_MISSING}" != "1" && "${DRY_RUN}" != "1" ]]; then
    echo "[training-diagnostics] missing baseline checkpoint: ${BASELINE_CKPT}" >&2
    exit 1
  fi
  mkdir -p "${BASELINE_MODEL_DIR}"
  (
    cd "${SOF_ROOT}"
    run_cmd "${PYTHON_BIN}" train.py \
      --splatting_config "${BASELINE_SPLATTING_CONFIG}" \
      -s "${ALIAS_DIR}" \
      --eval \
      -m "${BASELINE_MODEL_DIR}" \
      --iterations "${BASE_ITER}" \
      --test_iterations "${BASE_ITER}" \
      --save_iterations "${BASE_ITER}" \
      --checkpoint_iterations "${BASE_ITER}" \
      --distortion_from_iter 4000 \
      --depth_normal_from_iter 4000 \
      --lambda_distortion 200 \
      --lambda_depth_normal 0.02 \
      --lambda_smoothness 0.005
  )
fi

COMMON_TRAIN_ARGS=(
  -s "${ALIAS_DIR}"
  -m "${MODEL_DIR}"
  --eval
  --data_device cpu
  --splatting_config "${BASELINE_MODEL_DIR}/config.json"
  --start_checkpoint "${BASELINE_CKPT}"
  --iterations "${FINAL_ITER}"
  --test_iterations "${FINAL_ITER}"
  --save_iterations "${FINAL_ITER}"
  --checkpoint_iterations "${FINAL_ITER}"
  --distortion_from_iter 4000
  --depth_normal_from_iter 4000
  --lambda_distortion 200
  --lambda_depth_normal 0.02
  --lambda_smoothness 0.005
  --densify_until_iter 0
  --diagnostic_output_subdir "${DIAGNOSTIC_SUBDIR}"
  --diagnostic_basis_mode "${DIAGNOSTIC_BASIS_MODE}"
)

if [[ -n "${DIAGNOSTIC_SURFACE_PAYLOAD}" ]]; then
  COMMON_TRAIN_ARGS+=(
    --diagnostic_surface_payload "${DIAGNOSTIC_SURFACE_PAYLOAD}"
  )
fi

if [[ "${DIAGNOSTIC_MODE}" == "gradient" || "${DIAGNOSTIC_MODE}" == "both" ]]; then
  COMMON_TRAIN_ARGS+=(
    --enable_gradient_tracking
    --gradient_tracking_from_iter "${DIAGNOSTIC_FROM_ITER}"
    --gradient_tracking_snapshot_interval "${GRAD_SNAPSHOT_INTERVAL}"
    --gradient_tracking_tile_size "${GRAD_TILE_SIZE}"
  )
fi

if [[ "${DIAGNOSTIC_MODE}" == "dropout" || "${DIAGNOSTIC_MODE}" == "both" ]]; then
  COMMON_TRAIN_ARGS+=(
    --enable_2d_dropout_diagnostic
    --dropout_diagnostic_from_iter "${DIAGNOSTIC_FROM_ITER}"
    --dropout_diagnostic_interval "${DROPOUT_INTERVAL}"
    --dropout_diagnostic_num_masks "${DROPOUT_NUM_MASKS}"
    --dropout_diagnostic_tile_size "${DROPOUT_TILE_SIZE}"
    --dropout_diagnostic_keep_ratio "${DROPOUT_KEEP_RATIO}"
    --dropout_diagnostic_loss_mode "${DROPOUT_LOSS_MODE}"
    --dropout_diagnostic_alpha_threshold "${DROPOUT_ALPHA_THRESHOLD}"
    --dropout_diagnostic_min_active_pixels "${DROPOUT_MIN_ACTIVE_PIXELS}"
  )
fi

echo
echo "[3/5] finetune with training diagnostics"
(
  cd "${SOF_ROOT}"
  run_cmd "${PYTHON_BIN}" train.py "${COMMON_TRAIN_ARGS[@]}"
)

if [[ "${RUN_AGGREGATE_AFTER}" == "1" ]]; then
  echo
  echo "[4/5] aggregate diagnostic snapshots"
  (
    cd "${SOF_ROOT}"
    run_cmd "${PYTHON_BIN}" visualize_training_diagnostics_v0.py \
      --diagnostic_dir "${DIAGNOSTIC_DIR}" \
      --latest_n "${AGGREGATE_LATEST_N}"
  )
else
  echo
  echo "[4/5] skip aggregate view"
fi

if [[ "${RUN_RENDER_AFTER}" == "1" ]]; then
  echo
  echo "[5/5] render final checkpoint"
  (
    cd "${SOF_ROOT}"
    run_cmd "${PYTHON_BIN}" render.py \
      -m "${MODEL_DIR}" \
      -s "${SCENE_ROOT}" \
      -i "${TARGET_IMAGES_SUBDIR}" \
      --iteration "${FINAL_ITER}" \
      --eval \
      --skip_train \
      --data_device cpu
  )
else
  echo
  echo "[5/5] skip final render"
fi

echo
echo "[training-diagnostics] done."
echo "[training-diagnostics] model dir      : ${MODEL_DIR}"
echo "[training-diagnostics] diagnostic dir : ${DIAGNOSTIC_DIR}"
