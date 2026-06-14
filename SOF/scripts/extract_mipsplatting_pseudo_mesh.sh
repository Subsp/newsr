#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SOF_ROOT}/.." && pwd)"

MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${WORKSPACE_ROOT}/mip-splatting}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-${1:-}}"
ITERATION="${ITERATION:-}"

DEVICE="${DEVICE:-cuda}"
OPACITY_MIN="${OPACITY_MIN:-0.02}"
SHEETNESS_MIN="${SHEETNESS_MIN:-2.0}"
MAX_SURFELS="${MAX_SURFELS:-60000}"
SAMPLES_PER_SURFEL="${SAMPLES_PER_SURFEL:-3}"
NORMAL_AXIS="${NORMAL_AXIS:-min_scale}"
FOCUS_MODE="${FOCUS_MODE:-main_object}"
FOCUS_CLUSTER_RATIO="${FOCUS_CLUSTER_RATIO:-0.50}"
FOCUS_INLIER_QUANTILE="${FOCUS_INLIER_QUANTILE:-0.90}"
FOCUS_PADDING_RATIO="${FOCUS_PADDING_RATIO:-0.08}"
FOCUS_CENTER="${FOCUS_CENTER:-}"
FOCUS_EXTENT="${FOCUS_EXTENT:-}"
POISSON_DEPTH="${POISSON_DEPTH:-9}"
DENSITY_PRUNE_QUANTILE="${DENSITY_PRUNE_QUANTILE:-0.05}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-pseudo_mesh_2dgs}"

usage() {
  cat <<'EOF'
Usage:
  MODEL_DIR=/path/to/model bash scripts/extract_mipsplatting_pseudo_mesh.sh

Or:
  bash scripts/extract_mipsplatting_pseudo_mesh.sh /path/to/model

Optional env vars:
  ITERATION=30100
  OUTPUT_DIR=/custom/output/dir
  MIPSPLATTING_ROOT=/path/to/mip-splatting
  PYTHON_BIN=python

Extraction knobs:
  OPACITY_MIN=0.02
  SHEETNESS_MIN=2.0
  MAX_SURFELS=60000
  SAMPLES_PER_SURFEL=3
  NORMAL_AXIS=min_scale
  FOCUS_MODE=main_object|global|manual
  FOCUS_CLUSTER_RATIO=0.50
  FOCUS_INLIER_QUANTILE=0.90
  FOCUS_PADDING_RATIO=0.08
  FOCUS_CENTER="x,y,z"      # required if FOCUS_MODE=manual
  FOCUS_EXTENT="sx,sy,sz"   # required if FOCUS_MODE=manual
  POISSON_DEPTH=9
  DENSITY_PRUNE_QUANTILE=0.05
EOF
}

if [[ -z "${MODEL_DIR}" ]]; then
  usage
  echo
  echo "[pseudo-mesh] MODEL_DIR is required." >&2
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[pseudo-mesh] MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -d "${MIPSPLATTING_ROOT}" ]]; then
  echo "[pseudo-mesh] MIPSPLATTING_ROOT does not exist: ${MIPSPLATTING_ROOT}" >&2
  exit 1
fi

POINT_CLOUD_ROOT="${MODEL_DIR}/point_cloud"
if [[ ! -d "${POINT_CLOUD_ROOT}" ]]; then
  echo "[pseudo-mesh] point_cloud directory is missing: ${POINT_CLOUD_ROOT}" >&2
  exit 1
fi

if [[ -z "${ITERATION}" ]]; then
  latest_dir="$(find "${POINT_CLOUD_ROOT}" -maxdepth 1 -type d -name 'iteration_*' | sort -V | tail -n 1 || true)"
  if [[ -z "${latest_dir}" ]]; then
    echo "[pseudo-mesh] Could not auto-detect an iteration_* directory under ${POINT_CLOUD_ROOT}" >&2
    exit 1
  fi
  ITERATION="${latest_dir##*_}"
fi

POINT_CLOUD_PLY="${POINT_CLOUD_PLY:-${POINT_CLOUD_ROOT}/iteration_${ITERATION}/point_cloud.ply}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_DIR}/${OUTPUT_SUBDIR}/iteration_${ITERATION}}"

if [[ ! -f "${POINT_CLOUD_PLY}" ]]; then
  echo "[pseudo-mesh] point cloud is missing: ${POINT_CLOUD_PLY}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${MIPSPLATTING_ROOT}"

CMD=(
  "${PYTHON_BIN}" hybrid_sdfgs/tools/extract_2dgs_surface.py
  --point_cloud_ply "${POINT_CLOUD_PLY}"
  --output_dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --opacity_min "${OPACITY_MIN}"
  --sheetness_min "${SHEETNESS_MIN}"
  --max_surfels "${MAX_SURFELS}"
  --samples_per_surfel "${SAMPLES_PER_SURFEL}"
  --normal_axis "${NORMAL_AXIS}"
  --focus_mode "${FOCUS_MODE}"
  --focus_cluster_ratio "${FOCUS_CLUSTER_RATIO}"
  --focus_inlier_quantile "${FOCUS_INLIER_QUANTILE}"
  --focus_padding_ratio "${FOCUS_PADDING_RATIO}"
  --poisson_depth "${POISSON_DEPTH}"
  --density_prune_quantile "${DENSITY_PRUNE_QUANTILE}"
)

if [[ "${FOCUS_MODE}" == "manual" ]]; then
  if [[ -z "${FOCUS_CENTER}" || -z "${FOCUS_EXTENT}" ]]; then
    echo "[pseudo-mesh] FOCUS_MODE=manual requires FOCUS_CENTER and FOCUS_EXTENT." >&2
    exit 1
  fi
  CMD+=(--focus_center "${FOCUS_CENTER}" --focus_extent "${FOCUS_EXTENT}")
fi

echo "[pseudo-mesh] mip-splatting root : ${MIPSPLATTING_ROOT}"
echo "[pseudo-mesh] model dir          : ${MODEL_DIR}"
echo "[pseudo-mesh] iteration          : ${ITERATION}"
echo "[pseudo-mesh] point cloud        : ${POINT_CLOUD_PLY}"
echo "[pseudo-mesh] output dir         : ${OUTPUT_DIR}"
echo "[pseudo-mesh] running:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "[pseudo-mesh] done"
echo "[pseudo-mesh] mesh    : ${OUTPUT_DIR}/surface_mesh_poisson.ply"
echo "[pseudo-mesh] surfels : ${OUTPUT_DIR}/surfel_points.ply"
echo "[pseudo-mesh] info    : ${OUTPUT_DIR}/surface_extract_info.json"
