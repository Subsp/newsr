#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="${HBSR_ROOT:-/root/autodl-tmp/HBSR}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/HBSR/outputs/hybrid_gsbootstrap_sdfdensify_4x_kitchen_x8to2}"
ITERATION="${ITERATION:-30000}"
POINT_CLOUD_PLY="${POINT_CLOUD_PLY:-${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PATH}/pseudo_sdf_viz/iteration_${ITERATION}}"
DEVICE="${DEVICE:-cuda}"

SLICE_AXIS="${SLICE_AXIS:-z}"
SLICE_VALUE="${SLICE_VALUE:-}"
SLICE_RESOLUTION="${SLICE_RESOLUTION:-512}"
EXPORT_MESH="${EXPORT_MESH:-0}"
MESH_RESOLUTION="${MESH_RESOLUTION:-128}"
MESH_FOCUS_MODE="${MESH_FOCUS_MODE:-main_object}"
MESH_FOCUS_CLUSTER_RATIO="${MESH_FOCUS_CLUSTER_RATIO:-0.35}"
MESH_FOCUS_INLIER_QUANTILE="${MESH_FOCUS_INLIER_QUANTILE:-0.85}"

mkdir -p "${OUTPUT_DIR}"
cd "${HBSR_ROOT}"

CMD=(
  python hybrid_sdfgs/tools/visualize_gs_bootstrap_pseudosdf.py
  --point_cloud_ply "${POINT_CLOUD_PLY}"
  --output_dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --slice_axis "${SLICE_AXIS}"
  --slice_resolution "${SLICE_RESOLUTION}"
  --export_slice_png
  --export_slice_npz
)

if [[ -n "${SLICE_VALUE}" ]]; then
  CMD+=(--slice_value "${SLICE_VALUE}")
fi

if [[ "${EXPORT_MESH}" == "1" ]]; then
  CMD+=(
    --export_mesh
    --mesh_resolution "${MESH_RESOLUTION}"
    --mesh_focus_mode "${MESH_FOCUS_MODE}"
    --mesh_focus_cluster_ratio "${MESH_FOCUS_CLUSTER_RATIO}"
    --mesh_focus_inlier_quantile "${MESH_FOCUS_INLIER_QUANTILE}"
  )
fi

echo "[pseudoSDF-viz] running:"
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
