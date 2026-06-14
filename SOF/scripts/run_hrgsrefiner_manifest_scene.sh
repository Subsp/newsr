#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
REPO_ROOT="${REPO_ROOT:-${WORK_ROOT}/SRtestrepo}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
VGGT_ROOT="${VGGT_ROOT:-${WORK_ROOT}/vggt}"
SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
PRIOR_DIR="${PRIOR_DIR:-${SCENE_ROOT}/priors}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/SOF/output/hrgsrefiner}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${SCENE_NAME}_x8to2}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_DIR}/hrgs_manifest.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIRE_PRIORS="${REQUIRE_PRIORS:-1}"

if [[ ! -d "${REPO_ROOT}/SOF" ]]; then
  echo "[hrgs-manifest] SOF repo not found under: ${REPO_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}/SOF"

ARGS=(
  scripts/prepare_hrgsrefiner_manifest.py
  --scene_root "${SCENE_ROOT}"
  --output "${MANIFEST_PATH}"
  --source_images_subdir "${SOURCE_IMAGES_SUBDIR}"
  --target_images_subdir "${TARGET_IMAGES_SUBDIR}"
  --priors_dir "${PRIOR_DIR}"
  --vggt_root "${VGGT_ROOT}"
  --repo_root "${REPO_ROOT}"
)

if [[ "${REQUIRE_PRIORS}" == "1" ]]; then
  ARGS+=(--require_priors)
fi

echo "[hrgs-manifest] repo root        : ${REPO_ROOT}"
echo "[hrgs-manifest] scene root       : ${SCENE_ROOT}"
echo "[hrgs-manifest] source subdir    : ${SOURCE_IMAGES_SUBDIR}"
echo "[hrgs-manifest] target subdir    : ${TARGET_IMAGES_SUBDIR}"
echo "[hrgs-manifest] priors dir       : ${PRIOR_DIR}"
echo "[hrgs-manifest] vggt root        : ${VGGT_ROOT}"
echo "[hrgs-manifest] output manifest  : ${MANIFEST_PATH}"

"${PYTHON_BIN}" "${ARGS[@]}"
