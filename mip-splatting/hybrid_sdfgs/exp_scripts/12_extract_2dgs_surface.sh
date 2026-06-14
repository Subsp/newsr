#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="${HBSR_ROOT:-/root/autodl-tmp/HBSR}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/HBSR/outputs/hybrid_gsbootstrap_sdfdensify_quickply_kitchen_x8to2}"
ITERATION="${ITERATION:-7000}"
POINT_CLOUD_PLY="${POINT_CLOUD_PLY:-${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PATH}/surface_2dgs/iteration_${ITERATION}}"

DEVICE="${DEVICE:-cuda}"
OPACITY_MIN="${OPACITY_MIN:-0.02}"
SHEETNESS_MIN="${SHEETNESS_MIN:-2.0}"
MAX_SURFELS="${MAX_SURFELS:-60000}"
SAMPLES_PER_SURFEL="${SAMPLES_PER_SURFEL:-3}"
FOCUS_MODE="${FOCUS_MODE:-main_object}"
FOCUS_CLUSTER_RATIO="${FOCUS_CLUSTER_RATIO:-0.50}"
FOCUS_INLIER_QUANTILE="${FOCUS_INLIER_QUANTILE:-0.90}"
FOCUS_PADDING_RATIO="${FOCUS_PADDING_RATIO:-0.08}"
POISSON_DEPTH="${POISSON_DEPTH:-9}"
DENSITY_PRUNE_QUANTILE="${DENSITY_PRUNE_QUANTILE:-0.05}"

mkdir -p "${OUTPUT_DIR}"
cd "${HBSR_ROOT}"

CMD=(
  python hybrid_sdfgs/tools/extract_2dgs_surface.py
  --point_cloud_ply "${POINT_CLOUD_PLY}"
  --output_dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --opacity_min "${OPACITY_MIN}"
  --sheetness_min "${SHEETNESS_MIN}"
  --max_surfels "${MAX_SURFELS}"
  --samples_per_surfel "${SAMPLES_PER_SURFEL}"
  --focus_mode "${FOCUS_MODE}"
  --focus_cluster_ratio "${FOCUS_CLUSTER_RATIO}"
  --focus_inlier_quantile "${FOCUS_INLIER_QUANTILE}"
  --focus_padding_ratio "${FOCUS_PADDING_RATIO}"
  --poisson_depth "${POISSON_DEPTH}"
  --density_prune_quantile "${DENSITY_PRUNE_QUANTILE}"
)

echo "[2dgs-surface] running:"
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
