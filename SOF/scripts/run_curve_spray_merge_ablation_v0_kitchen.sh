#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting}"

RUN_TAG="${RUN_TAG:-mip30k_lr30000_spray_curve_candidate_graph_v0}"
MODEL_DIR="${MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_sr_hf_curve_spray_v0/${SCENE_NAME}/${RUN_TAG}}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/mip30k_rerun_check_directsrc_r1_v0}"
ITERATION="${ITERATION:-30000}"
SPLIT="${SPLIT:-train}"
MAX_VIEWS="${MAX_VIEWS:-8}"
VIEW_SELECT_MODE="${VIEW_SELECT_MODE:-first}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/curve_spray_ablation/${RUN_TAG}}"
CHECK_ROOT="${CHECK_ROOT:-${WORK_ROOT}/check/curve_spray_ablation/${RUN_TAG}}"
OVERWRITE="${OVERWRITE:-0}"

PRIOR_TAU_SCALE="${PRIOR_TAU_SCALE:-20.0}"
PRIOR_SCALE_MULTIPLIER="${PRIOR_SCALE_MULTIPLIER:-2.0}"
PRIOR_FILTER_MULTIPLIER="${PRIOR_FILTER_MULTIPLIER:-1.0}"
DELTA_VIS_SCALE="${DELTA_VIS_SCALE:-80.0}"

BASE_PLY="${BASE_MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
MERGED_PLY="${MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
TAGS_PATH="${MODEL_DIR}/point_cloud/iteration_${ITERATION}/gaussian_tags.pt"

for required in "${BASE_MODEL_DIR}" "${BASE_PLY}" "${MODEL_DIR}" "${MERGED_PLY}" "${TAGS_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[curve-spray-ablation-v0] required path not found: ${required}" >&2
    exit 1
  fi
done

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_ROOT}" "${CHECK_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}" "${CHECK_ROOT}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}:${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[curve-spray-ablation-v0] base   : ${BASE_MODEL_DIR}"
echo "[curve-spray-ablation-v0] merged : ${MODEL_DIR}"
echo "[curve-spray-ablation-v0] output : ${OUTPUT_ROOT}"
echo "[curve-spray-ablation-v0] check  : ${CHECK_ROOT}"
echo "[curve-spray-ablation-v0] views  : ${SPLIT}/${MAX_VIEWS} mode=${VIEW_SELECT_MODE}"

BASE_EXPORT="${OUTPUT_ROOT}/_base_full"
MERGED_EXPORT="${OUTPUT_ROOT}/_merged_full"
MERGED_NO_PRIOR_EXPORT="${OUTPUT_ROOT}/_merged_no_prior"
PRIOR_ONLY_EXPORT="${OUTPUT_ROOT}/_prior_only"
PRIOR_BOOST_EXPORT="${OUTPUT_ROOT}/_prior_boost"

export_variant() {
  local output_root="$1"
  local model_path="$2"
  local source="$3"
  local key="$4"
  local mode="$5"
  local tau="$6"
  local scale="$7"
  local filter="$8"
  shift 8
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${model_path}" \
    --output_root "${output_root}" \
    --images_subdir images_2 \
    --iteration "${ITERATION}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}" \
    --view_select_mode "${VIEW_SELECT_MODE}" \
    --selection_source "${source}" \
    --selection_key "${key}" \
    --selection_mode "${mode}" \
    --tau_scale "${tau}" \
    --scale_multiplier "${scale}" \
    --filter_multiplier "${filter}" \
    "$@"
}

echo "[curve-spray-ablation-v0] render base_full"
export_variant "${BASE_EXPORT}" "${BASE_MODEL_DIR}" lineage full full 1.0 1.0 1.0

echo "[curve-spray-ablation-v0] render merged_full"
export_variant "${MERGED_EXPORT}" "${MODEL_DIR}" tracking full full 1.0 1.0 1.0

echo "[curve-spray-ablation-v0] render merged_no_prior"
export_variant "${MERGED_NO_PRIOR_EXPORT}" "${MODEL_DIR}" tracking prior_injected selected_removed 1.0 1.0 1.0

echo "[curve-spray-ablation-v0] render prior_only"
export_variant "${PRIOR_ONLY_EXPORT}" "${MODEL_DIR}" tracking prior_injected selected_only 1.0 1.0 1.0 --save_alpha

echo "[curve-spray-ablation-v0] render prior_boost"
export_variant \
  "${PRIOR_BOOST_EXPORT}" \
  "${MODEL_DIR}" \
  tracking \
  prior_injected \
  selected_only \
  "${PRIOR_TAU_SCALE}" \
  "${PRIOR_SCALE_MULTIPLIER}" \
  "${PRIOR_FILTER_MULTIPLIER}" \
  --save_alpha

BASE_DIR="${BASE_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
MERGED_DIR="${MERGED_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
MERGED_NO_PRIOR_DIR="${MERGED_NO_PRIOR_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
PRIOR_ONLY_DIR="${PRIOR_ONLY_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
PRIOR_BOOST_DIR="${PRIOR_BOOST_EXPORT}/${SPLIT}/ours_${ITERATION}/renders"
PRIOR_ALPHA_DIR="${PRIOR_BOOST_EXPORT}/${SPLIT}/ours_${ITERATION}/alpha"

mkdir -p \
  "${CHECK_ROOT}/base_full" \
  "${CHECK_ROOT}/merged_full" \
  "${CHECK_ROOT}/merged_no_prior" \
  "${CHECK_ROOT}/prior_only" \
  "${CHECK_ROOT}/prior_boost" \
  "${CHECK_ROOT}/prior_boost_alpha"

shopt -s nullglob
for image_path in "${BASE_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/base_full/"; done
for image_path in "${MERGED_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/merged_full/"; done
for image_path in "${MERGED_NO_PRIOR_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/merged_no_prior/"; done
for image_path in "${PRIOR_ONLY_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/prior_only/"; done
for image_path in "${PRIOR_BOOST_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/prior_boost/"; done
for image_path in "${PRIOR_ALPHA_DIR}"/*.png; do cp "${image_path}" "${CHECK_ROOT}/prior_boost_alpha/"; done
shopt -u nullglob

compare_pair() {
  local name="$1"
  local base_dir="$2"
  local current_dir="$3"
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/compare_render_dirs_v0.py" \
    --base_dir "${base_dir}" \
    --current_dir "${current_dir}" \
    --output_dir "${CHECK_ROOT}/delta_${name}" \
    --match_policy stem \
    --vis_scale "${DELTA_VIS_SCALE}" \
    --overwrite
}

echo "[curve-spray-ablation-v0] compare base_vs_merged_no_prior"
compare_pair "base_vs_merged_no_prior" "${CHECK_ROOT}/base_full" "${CHECK_ROOT}/merged_no_prior"

echo "[curve-spray-ablation-v0] compare base_vs_merged_full"
compare_pair "base_vs_merged_full" "${CHECK_ROOT}/base_full" "${CHECK_ROOT}/merged_full"

echo "[curve-spray-ablation-v0] compare merged_no_prior_vs_merged_full"
compare_pair "merged_no_prior_vs_merged_full" "${CHECK_ROOT}/merged_no_prior" "${CHECK_ROOT}/merged_full"

BASE_PLY="${BASE_PLY}" MERGED_PLY="${MERGED_PLY}" TAGS_PATH="${TAGS_PATH}" OUT="${CHECK_ROOT}/gaussian_prefix_ablation.json" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

base_ply = Path(os.environ["BASE_PLY"])
merged_ply = Path(os.environ["MERGED_PLY"])
tags_path = Path(os.environ["TAGS_PATH"])
out = Path(os.environ["OUT"])

base = PlyData.read(str(base_ply))["vertex"].data
merged = PlyData.read(str(merged_ply))["vertex"].data
base_n = int(base.shape[0])
merged_n = int(merged.shape[0])

fields = list(base.dtype.names or [])
field_rows = {}
max_abs_all = 0.0
for name in fields:
    if name not in merged.dtype.names:
        continue
    a = np.asarray(base[name])
    b = np.asarray(merged[name][:base_n])
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    field_rows[name] = {"max_abs": max_abs, "mean_abs": mean_abs}
    max_abs_all = max(max_abs_all, max_abs)

tags = torch.load(tags_path, map_location="cpu")
source = tags["source_tag"].reshape(-1).to(torch.int64)
counts = {
    "total": int(source.numel()),
    "original": int((source == 0).sum().item()),
    "prior_injected": int((source == 1).sum().item()),
    "non_original": int((source != 0).sum().item()),
}
summary = {
    "version": "curve_spray_merge_ablation_v0",
    "base_ply": str(base_ply),
    "merged_ply": str(merged_ply),
    "tags_path": str(tags_path),
    "base_gaussians": base_n,
    "merged_gaussians": merged_n,
    "extra_gaussians": merged_n - base_n,
    "source_tag_counts": counts,
    "base_prefix_identical": bool(max_abs_all == 0.0),
    "base_prefix_max_abs_all_fields": max_abs_all,
    "field_abs_diff": field_rows,
}
out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

cat > "${CHECK_ROOT}/README.txt" <<EOF
Curve spray merge ablation.

base model:
  ${BASE_MODEL_DIR}
merged model:
  ${MODEL_DIR}
iteration:
  ${ITERATION}

Main proof checks:
  ${CHECK_ROOT}/gaussian_prefix_ablation.json
    PLY-level check: merged first base_count vertices vs original base PLY.
  ${CHECK_ROOT}/delta_base_vs_merged_no_prior/summary.json
    Render-level check: same exporter/filter, newborn muted. Should be near zero if base is unchanged.

Contribution checks:
  ${CHECK_ROOT}/delta_base_vs_merged_full/summary.json
    Real-strength added effect.
  ${CHECK_ROOT}/delta_merged_no_prior_vs_merged_full/summary.json
    Isolated newborn contribution.

Images:
  ${CHECK_ROOT}/base_full
  ${CHECK_ROOT}/merged_no_prior
  ${CHECK_ROOT}/merged_full
  ${CHECK_ROOT}/prior_only
  ${CHECK_ROOT}/prior_boost
  ${CHECK_ROOT}/prior_boost_alpha
EOF

echo "[curve-spray-ablation-v0] shallow outputs:"
echo "  ${CHECK_ROOT}/README.txt"
echo "  ${CHECK_ROOT}/gaussian_prefix_ablation.json"
echo "  ${CHECK_ROOT}/delta_base_vs_merged_no_prior/summary.json"
echo "  ${CHECK_ROOT}/delta_base_vs_merged_full/summary.json"
echo "  ${CHECK_ROOT}/delta_merged_no_prior_vs_merged_full/summary.json"
echo "  ${CHECK_ROOT}/base_full"
echo "  ${CHECK_ROOT}/merged_no_prior"
echo "  ${CHECK_ROOT}/merged_full"
echo "  ${CHECK_ROOT}/prior_only"
echo "  ${CHECK_ROOT}/prior_boost"
