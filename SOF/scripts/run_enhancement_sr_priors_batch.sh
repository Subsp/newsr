#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"

ARCHIVE_ROOT="${ARCHIVE_ROOT:-${WORKSPACE_ROOT}/archive}"
SCENES="${SCENES:-bicycle bonsai counter flowers garden stump treehill kitchen room}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
PREPARE_IMAGES8="${PREPARE_IMAGES8:-1}"
GENERATE_IMAGES8_SCALE="${GENERATE_IMAGES8_SCALE:-4}"
RESIZE_FILTER="${RESIZE_FILTER:-bicubic}"

BACKEND="${BACKEND:-swinir}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-priors_${BACKEND}}"
OVERWRITE_PRIORS="${OVERWRITE_PRIORS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

echo "[enhancement-sr-priors] scenes        : ${SCENES}"
echo "[enhancement-sr-priors] archive root  : ${ARCHIVE_ROOT}"
echo "[enhancement-sr-priors] backend       : ${BACKEND}"
echo "[enhancement-sr-priors] source subdir : ${SOURCE_IMAGES_SUBDIR}"
echo "[enhancement-sr-priors] output subdir : ${OUTPUT_SUBDIR}"

for scene in ${SCENES}; do
  SCENE_ROOT="${ARCHIVE_ROOT}/${scene}"
  TARGET_DIR="${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}"
  SOURCE_DIR="${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}"
  OUTPUT_DIR="${SCENE_ROOT}/${OUTPUT_SUBDIR}"

  echo
  echo "[enhancement-sr-priors] ===== scene: ${scene} ====="

  if [[ ! -d "${SCENE_ROOT}" ]]; then
    echo "[enhancement-sr-priors] scene root not found: ${SCENE_ROOT}" >&2
    exit 1
  fi
  if [[ ! -d "${TARGET_DIR}" ]]; then
    echo "[enhancement-sr-priors] target image dir not found: ${TARGET_DIR}" >&2
    exit 1
  fi

  if [[ ! -d "${SOURCE_DIR}" ]]; then
    if [[ "${PREPARE_IMAGES8}" != "1" ]]; then
      echo "[enhancement-sr-priors] source image dir missing and PREPARE_IMAGES8=0: ${SOURCE_DIR}" >&2
      exit 1
    fi
    echo "[enhancement-sr-priors] generate ${SOURCE_IMAGES_SUBDIR} from ${TARGET_IMAGES_SUBDIR}"
    (
      cd "${SOF_ROOT}"
      "${PYTHON_BIN}" scripts/generate_downsampled_images.py \
        --source_dir "${TARGET_DIR}" \
        --output_dir "${SOURCE_DIR}" \
        --scale "${GENERATE_IMAGES8_SCALE}" \
        --resize_filter "${RESIZE_FILTER}"
    )
  fi

  mkdir -p "${OUTPUT_DIR}"

  input_count=$(find "${SOURCE_DIR}" -maxdepth 1 -type f | wc -l | tr -d ' ')
  prior_count=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')
  if [[ "${OVERWRITE_PRIORS}" != "1" && "${prior_count}" -ge "${input_count}" && "${input_count}" -gt 0 ]]; then
    echo "[enhancement-sr-priors] priors already exist (${prior_count}/${input_count}), skip"
    continue
  fi

  CMD=(
    "${PYTHON_BIN}" "${SOF_ROOT}/scripts/generate_enhancement_sr_priors.py"
    --input_dir "${SOURCE_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --backend "${BACKEND}"
    --device "${DEVICE}"
  )
  if [[ "${OVERWRITE_PRIORS}" == "1" ]]; then
    CMD+=(--overwrite)
  fi

  "${CMD[@]}"
done

echo
echo "[enhancement-sr-priors] done"
