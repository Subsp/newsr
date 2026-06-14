#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HBSR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
INPUT_DIR="${INPUT_DIR:-${SCENE_ROOT}/images_8}"
GT_DIR="${GT_DIR:-${SCENE_ROOT}/images_2}"
COLMAP_SPARSE_DIR="${COLMAP_SPARSE_DIR:-${SCENE_ROOT}/sparse/0}"

PYTHON_EXE="${PYTHON_EXE:-python}"
FLASHVSR_REPO="${FLASHVSR_REPO:-/root/autodl-tmp/FlashVSR}"
FLASHVSR_PYTHON="${FLASHVSR_PYTHON:-/root/miniconda3/envs/flashvsr/bin/python}"
FLASHVSR_MODEL_DIR="${FLASHVSR_MODEL_DIR:-/root/autodl-tmp/hub/FlashVSR-v1.1}"
FLASHVSR_MODEL_TYPE="${FLASHVSR_MODEL_TYPE:-tiny}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RAW_OUTPUT_ROOT="${RAW_OUTPUT_ROOT:-/root/autodl-tmp/priors/kitchen_video_flashvsr_${FLASHVSR_MODEL_TYPE}_seqmat_x8to2_raw}"
FUSED_OUTPUT_ROOT="${FUSED_OUTPUT_ROOT:-/root/autodl-tmp/priors/kitchen_video_flashvsr_${FLASHVSR_MODEL_TYPE}_seqmat_x8to2_fused}"

VIEW_GROUP_MAX_LEN="${VIEW_GROUP_MAX_LEN:-6}"
VIEW_GROUP_MIN_LEN="${VIEW_GROUP_MIN_LEN:-3}"
VIEW_GROUP_THRESHOLDS="${VIEW_GROUP_THRESHOLDS:-30,50}"
VIEW_DIR_WEIGHT="${VIEW_DIR_WEIGHT:-0.0}"

FLASHVSR_SPATIAL_TILE_W="${FLASHVSR_SPATIAL_TILE_W:-1600}"
FLASHVSR_SPATIAL_TILE_H="${FLASHVSR_SPATIAL_TILE_H:-1088}"
FLASHVSR_SPATIAL_OVERLAP="${FLASHVSR_SPATIAL_OVERLAP:-128}"

LEVELS="${LEVELS:-2}"
HF_WEIGHT="${HF_WEIGHT:-0.75}"
DELTA_CLIP="${DELTA_CLIP:-0.10}"
CONSISTENCY_TAU="${CONSISTENCY_TAU:-0.08}"
ENERGY_FLOOR="${ENERGY_FLOOR:-0.002}"
GAIN_POWER="${GAIN_POWER:-1.0}"
WRITE_DEBUG="${WRITE_DEBUG:-0}"
RUN_RAW_ANALYSIS="${RUN_RAW_ANALYSIS:-1}"
RUN_FUSED_METRICS="${RUN_FUSED_METRICS:-1}"

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[flashvsr-recover-x8to2] scene root not found: ${SCENE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[flashvsr-recover-x8to2] input dir not found: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${COLMAP_SPARSE_DIR}" ]]; then
  echo "[flashvsr-recover-x8to2] COLMAP sparse dir not found: ${COLMAP_SPARSE_DIR}" >&2
  exit 1
fi

if [[ ! -d "${FLASHVSR_REPO}" ]]; then
  echo "[flashvsr-recover-x8to2] FlashVSR repo not found: ${FLASHVSR_REPO}" >&2
  exit 1
fi

if [[ ! -x "${FLASHVSR_PYTHON}" ]]; then
  echo "[flashvsr-recover-x8to2] FlashVSR python missing: ${FLASHVSR_PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${FLASHVSR_MODEL_DIR}" ]]; then
  echo "[flashvsr-recover-x8to2] FlashVSR model dir not found: ${FLASHVSR_MODEL_DIR}" >&2
  exit 1
fi

case "${FLASHVSR_MODEL_TYPE}" in
  tiny|tiny_long)
    ;;
  *)
    echo "[flashvsr-recover-x8to2] unsupported FLASHVSR_MODEL_TYPE=${FLASHVSR_MODEL_TYPE}" >&2
    exit 1
    ;;
esac

echo "[flashvsr-recover-x8to2] HBSR_ROOT=${HBSR_ROOT}"
echo "[flashvsr-recover-x8to2] SCENE_ROOT=${SCENE_ROOT}"
echo "[flashvsr-recover-x8to2] INPUT_DIR=${INPUT_DIR}"
echo "[flashvsr-recover-x8to2] GT_DIR=${GT_DIR}"
echo "[flashvsr-recover-x8to2] COLMAP_SPARSE_DIR=${COLMAP_SPARSE_DIR}"
echo "[flashvsr-recover-x8to2] FLASHVSR_MODEL_TYPE=${FLASHVSR_MODEL_TYPE}"
echo "[flashvsr-recover-x8to2] RAW_OUTPUT_ROOT=${RAW_OUTPUT_ROOT}"
echo "[flashvsr-recover-x8to2] FUSED_OUTPUT_ROOT=${FUSED_OUTPUT_ROOT}"

mkdir -p "${RAW_OUTPUT_ROOT}" "${FUSED_OUTPUT_ROOT}"
cd "${HBSR_ROOT}"

RAW_CMD=(
  "${PYTHON_EXE}" "hybrid_sdfgs/tools/generate_video_sr_priors.py"
  "--model" "flashvsr"
  "--repo_root" "${FLASHVSR_REPO}"
  "--python_exe" "${FLASHVSR_PYTHON}"
  "--input_dir" "${INPUT_DIR}"
  "--output_root" "${RAW_OUTPUT_ROOT}"
  "--flashvsr_model_type" "${FLASHVSR_MODEL_TYPE}"
  "--flashvsr_model_dir" "${FLASHVSR_MODEL_DIR}"
  "--flashvsr_fps" "24"
  "--flashvsr_scale" "4.0"
  "--flashvsr_align_mode" "ceil"
  "--flashvsr_dtype" "bf16"
  "--flashvsr_device" "cuda"
  "--flashvsr_kv_ratio" "3.0"
  "--flashvsr_local_range" "11"
  "--flashvsr_sparse_ratio" "2.0"
  "--flashvsr_spatial_tile_w" "${FLASHVSR_SPATIAL_TILE_W}"
  "--flashvsr_spatial_tile_h" "${FLASHVSR_SPATIAL_TILE_H}"
  "--flashvsr_spatial_overlap" "${FLASHVSR_SPATIAL_OVERLAP}"
  "--flashvsr_view_group_mode" "seqmat_pose_als"
  "--flashvsr_colmap_sparse_dir" "${COLMAP_SPARSE_DIR}"
  "--flashvsr_view_group_max_len" "${VIEW_GROUP_MAX_LEN}"
  "--flashvsr_view_group_min_len" "${VIEW_GROUP_MIN_LEN}"
  "--flashvsr_view_group_thresholds" "${VIEW_GROUP_THRESHOLDS}"
  "--flashvsr_view_dir_weight" "${VIEW_DIR_WEIGHT}"
)

echo "[flashvsr-recover-x8to2] generating raw seqmat priors"
printf '  %q' "${RAW_CMD[@]}"
printf '\n'
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${RAW_CMD[@]}"

if [[ "${RUN_RAW_ANALYSIS}" == "1" && -d "${GT_DIR}" ]]; then
  RAW_ANALYSIS_DIR="${RAW_OUTPUT_ROOT}/analysis"
  echo "[flashvsr-recover-x8to2] analyzing raw priors -> ${RAW_ANALYSIS_DIR}"
  "${PYTHON_EXE}" "hybrid_sdfgs/tools/analyze_flashvsr_prior_quality.py" \
    --prior_dir "${RAW_OUTPUT_ROOT}/priors" \
    --input_dir "${INPUT_DIR}" \
    --gt_dir "${GT_DIR}" \
    --output_dir "${RAW_ANALYSIS_DIR}" \
    --view_group_mode "seqmat_pose_als" \
    --colmap_sparse_dir "${COLMAP_SPARSE_DIR}" \
    --view_group_max_len "${VIEW_GROUP_MAX_LEN}" \
    --view_group_min_len "${VIEW_GROUP_MIN_LEN}" \
    --view_group_thresholds "${VIEW_GROUP_THRESHOLDS}" \
    --view_dir_weight "${VIEW_DIR_WEIGHT}"
else
  echo "[flashvsr-recover-x8to2] skip raw analysis (RUN_RAW_ANALYSIS=${RUN_RAW_ANALYSIS}, gt_dir=${GT_DIR})"
fi

FUSE_CMD=(
  "${PYTHON_EXE}" "hybrid_sdfgs/standalone_x8to2_prior/run_x8to2.py"
  "--input_dir" "${INPUT_DIR}"
  "--prior_dir" "${RAW_OUTPUT_ROOT}/priors"
  "--output_dir" "${FUSED_OUTPUT_ROOT}"
  "--scale" "4"
  "--levels" "${LEVELS}"
  "--hf_weight" "${HF_WEIGHT}"
  "--delta_clip" "${DELTA_CLIP}"
  "--consistency_tau" "${CONSISTENCY_TAU}"
  "--energy_floor" "${ENERGY_FLOOR}"
  "--gain_power" "${GAIN_POWER}"
)

if [[ -d "${GT_DIR}" ]]; then
  FUSE_CMD+=("--gt_dir" "${GT_DIR}")
fi

if [[ "${WRITE_DEBUG}" == "1" ]]; then
  FUSE_CMD+=("--write_debug")
fi

echo "[flashvsr-recover-x8to2] running conservative x8->x2 fusion"
printf '  %q' "${FUSE_CMD[@]}"
printf '\n'
"${FUSE_CMD[@]}"

export FUSED_OUTPUT_ROOT
"${PYTHON_EXE}" - <<'PY'
import os
import shutil
from pathlib import Path

root = Path(os.environ["FUSED_OUTPUT_ROOT"])
renders = root / "renders"
priors = root / "priors"
if not renders.is_dir():
    raise SystemExit(f"renders directory missing: {renders}")
if priors.is_symlink() or priors.is_file():
    priors.unlink()
elif priors.is_dir():
    shutil.rmtree(priors)
priors.symlink_to(renders, target_is_directory=True)
print(f"[flashvsr-recover-x8to2] linked priors -> {renders}")
PY

if [[ "${RUN_FUSED_METRICS}" == "1" && -d "${GT_DIR}" ]]; then
  echo "[flashvsr-recover-x8to2] evaluating fused renders"
  "${PYTHON_EXE}" "hybrid_sdfgs/standalone_x8to2_prior/eval_with_existing_metrics.py" \
    --output_dir "${FUSED_OUTPUT_ROOT}"
else
  echo "[flashvsr-recover-x8to2] skip fused metrics (RUN_FUSED_METRICS=${RUN_FUSED_METRICS}, gt_dir=${GT_DIR})"
fi

echo "[flashvsr-recover-x8to2] done"
echo "[flashvsr-recover-x8to2] raw priors   : ${RAW_OUTPUT_ROOT}/priors"
echo "[flashvsr-recover-x8to2] fused priors : ${FUSED_OUTPUT_ROOT}/priors"
echo "[flashvsr-recover-x8to2] train with   : --external_prior_root ${FUSED_OUTPUT_ROOT}"
