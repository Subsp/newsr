#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"

SCENES_ROOT="${SCENES_ROOT:-${WORKSPACE_ROOT}}"
SCENES="${SCENES:-bicycle bonsai counter flowers garden stump treehill kitchen room}"

ALIAS_ROOT="${ALIAS_ROOT:-${SOF_ROOT}/data_aliases}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREPARE_TOOL="${PREPARE_TOOL:-${SOF_ROOT}/scripts/prepare_colmap_pseudo_sr_scene.py}"
DOWNSAMPLE_TOOL="${DOWNSAMPLE_TOOL:-${SOF_ROOT}/scripts/generate_downsampled_images.py}"
AGGREGATE_TOOL="${AGGREGATE_TOOL:-${SOF_ROOT}/scripts/aggregate_results_full.py}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
GENERATE_IMAGES8_FROM="${GENERATE_IMAGES8_FROM:-images_2}"
GENERATE_IMAGES8_SCALE="${GENERATE_IMAGES8_SCALE:-4}"
RESIZE_FILTER="${RESIZE_FILTER:-bicubic}"

ITERATIONS="${ITERATIONS:-30000}"
SPLATTING_CONFIG="${SPLATTING_CONFIG:-configs/hierarchical.json}"

PREPARE_IMAGES8="${PREPARE_IMAGES8:-1}"
OVERWRITE_IMAGES8="${OVERWRITE_IMAGES8:-0}"
RUN_PREPARE_ALIAS="${RUN_PREPARE_ALIAS:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_RENDER="${RUN_RENDER:-0}"
RUN_METRICS="${RUN_METRICS:-0}"
RUN_AGGREGATE_METRICS="${RUN_AGGREGATE_METRICS:-1}"

if [[ ! -f "${PREPARE_TOOL}" ]]; then
  echo "[sof-mip360-pseudogt] pseudo scene tool not found: ${PREPARE_TOOL}" >&2
  exit 1
fi
if [[ ! -f "${DOWNSAMPLE_TOOL}" ]]; then
  echo "[sof-mip360-pseudogt] downsample tool not found: ${DOWNSAMPLE_TOOL}" >&2
  exit 1
fi
if [[ ! -f "${AGGREGATE_TOOL}" ]]; then
  echo "[sof-mip360-pseudogt] aggregate tool not found: ${AGGREGATE_TOOL}" >&2
  exit 1
fi

mkdir -p "${ALIAS_ROOT}" "${OUTPUT_ROOT}"
MODEL_DIRS=()

echo "[sof-mip360-pseudogt] scenes: ${SCENES}"
echo "[sof-mip360-pseudogt] scene root base: ${SCENES_ROOT}"
echo "[sof-mip360-pseudogt] alias root: ${ALIAS_ROOT}"
echo "[sof-mip360-pseudogt] output root: ${OUTPUT_ROOT}"

for scene in ${SCENES}; do
  SCENE_ROOT="${SCENES_ROOT}/${scene}"
  SOURCE_DIR="${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}"
  TARGET_DIR="${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}"
  GENERATE_SOURCE_DIR="${SCENE_ROOT}/${GENERATE_IMAGES8_FROM}"
  PSEUDO_SCENE_DIR="${ALIAS_ROOT}/${scene}_images8bicubic_to_images2"
  MODEL_DIR="${OUTPUT_ROOT}/${scene}_pseudogt_images8bicubic_to_images2"
  MODEL_DIRS+=("${MODEL_DIR}")

  echo
  echo "[sof-mip360-pseudogt] ===== scene: ${scene} ====="

  if [[ ! -d "${SCENE_ROOT}" ]]; then
    echo "[sof-mip360-pseudogt] scene root not found: ${SCENE_ROOT}" >&2
    exit 1
  fi
  if [[ ! -d "${TARGET_DIR}" ]]; then
    echo "[sof-mip360-pseudogt] target image dir not found: ${TARGET_DIR}" >&2
    exit 1
  fi

  if [[ "${PREPARE_IMAGES8}" == "1" && ! -d "${SOURCE_DIR}" ]]; then
    if [[ ! -d "${GENERATE_SOURCE_DIR}" ]]; then
      echo "[sof-mip360-pseudogt] generate source not found: ${GENERATE_SOURCE_DIR}" >&2
      exit 1
    fi
    echo "[sof-mip360-pseudogt] generate ${SOURCE_IMAGES_SUBDIR} from ${GENERATE_IMAGES8_FROM}"
    DOWNSAMPLE_CMD=(
      "${PYTHON_BIN}" "${DOWNSAMPLE_TOOL}"
      --source_dir "${GENERATE_SOURCE_DIR}"
      --output_dir "${SOURCE_DIR}"
      --scale "${GENERATE_IMAGES8_SCALE}"
      --resize_filter "${RESIZE_FILTER}"
    )
    if [[ "${OVERWRITE_IMAGES8}" == "1" ]]; then
      DOWNSAMPLE_CMD+=(--overwrite)
    fi
    "${DOWNSAMPLE_CMD[@]}"
  fi

  if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "[sof-mip360-pseudogt] source image dir not found: ${SOURCE_DIR}" >&2
    exit 1
  fi

  mkdir -p "${PSEUDO_SCENE_DIR}" "${MODEL_DIR}"

  if [[ "${RUN_PREPARE_ALIAS}" == "1" ]]; then
    echo "[sof-mip360-pseudogt] prepare pseudo scene alias"
    "${PYTHON_BIN}" "${PREPARE_TOOL}" \
      --scene_root "${SCENE_ROOT}" \
      --scene_alias_dir "${PSEUDO_SCENE_DIR}" \
      --source_images_subdir "${SOURCE_IMAGES_SUBDIR}" \
      --target_images_subdir "${TARGET_IMAGES_SUBDIR}" \
      --resize_filter "${RESIZE_FILTER}"
  fi

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo "[sof-mip360-pseudogt] train"
    (
      cd "${SOF_ROOT}"
      "${PYTHON_BIN}" train.py \
        --splatting_config "${SPLATTING_CONFIG}" \
        -s "${PSEUDO_SCENE_DIR}" \
        --eval \
        -m "${MODEL_DIR}" \
        --iterations "${ITERATIONS}"
    )
  fi

  if [[ "${RUN_RENDER}" == "1" ]]; then
    echo "[sof-mip360-pseudogt] render"
    (
      cd "${SOF_ROOT}"
      "${PYTHON_BIN}" render.py \
        -m "${MODEL_DIR}" \
        -s "${SCENE_ROOT}" \
        -i "${TARGET_IMAGES_SUBDIR}" \
        --eval \
        --skip_train \
        --data_device cpu
    )
  fi

  if [[ "${RUN_METRICS}" == "1" ]]; then
    echo "[sof-mip360-pseudogt] metrics"
    (
      cd "${SOF_ROOT}"
      "${PYTHON_BIN}" metrics.py -m "${MODEL_DIR}"
    )
  fi

  echo "[sof-mip360-pseudogt] scene ready:"
  echo "  scene_root   : ${SCENE_ROOT}"
  echo "  pseudo_scene : ${PSEUDO_SCENE_DIR}"
  echo "  model_dir    : ${MODEL_DIR}"
done

if [[ "${RUN_METRICS}" == "1" && "${RUN_AGGREGATE_METRICS}" == "1" ]]; then
  SUMMARY_PATH="${OUTPUT_ROOT}/mip360_pseudogt_metrics_summary.json"
  echo
  echo "[sof-mip360-pseudogt] aggregate metrics"
  "${PYTHON_BIN}" "${AGGREGATE_TOOL}" \
    --model_dirs "${MODEL_DIRS[@]}" \
    --output_path "${SUMMARY_PATH}"
fi
