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

BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/${BASE_EXPERIMENT_NAME}}"
BASE_ITERATION="${BASE_ITERATION:-30000}"
BASE_PLY="${BASE_MODEL_DIR}/point_cloud/iteration_${BASE_ITERATION}/point_cloud.ply"

EVIDENCE_NAME="${EVIDENCE_NAME:-qwen_vosr_sr_hf_effective_verywide_8view_v0}"
TRACK_NAME="${TRACK_NAME:-${BASE_EXPERIMENT_NAME}_${EVIDENCE_NAME}_curve_tracks_v0}"
TRACK_ROOT="${TRACK_ROOT:-${SCENE_ASSET_ROOT}/sr_hf_curve_tracks/${TRACK_NAME}}"
TRACK_PAYLOAD="${TRACK_PAYLOAD:-${TRACK_ROOT}/sr_hf_curve_tracks_v0.npz}"

OUTPUT_NAME="${OUTPUT_NAME:-${BASE_EXPERIMENT_NAME}_spray_sr_hf_curve_tracks_v0}"
OUTPUT_MODEL_DIR="${OUTPUT_MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_sr_hf_curve_spray_v0/${SCENE_NAME}/${OUTPUT_NAME}}"
NEWBORN_MODEL_DIR="${NEWBORN_MODEL_DIR:-${OUTPUT_MODEL_DIR}_newborn_only}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-${BASE_ITERATION}}"
MERGE_SCRIPT="${MERGE_SCRIPT:-}"

SELECTION="${SELECTION:-keep}"
FALLBACK_TO_KEEP="${FALLBACK_TO_KEEP:-1}"
OVERWRITE="${OVERWRITE:-0}"
MAX_TRACKS="${MAX_TRACKS:-0}"
MAX_TOTAL_NEWBORN="${MAX_TOTAL_NEWBORN:-0}"
SAMPLE_SPACING_PX="${SAMPLE_SPACING_PX:-4.0}"
SAMPLE_SPACING_MIN="${SAMPLE_SPACING_MIN:-0.003}"
SAMPLE_SPACING_MAX="${SAMPLE_SPACING_MAX:-0.020}"
MAX_SAMPLES_PER_TRACK="${MAX_SAMPLES_PER_TRACK:-12}"
SCALE_LONG_FACTOR="${SCALE_LONG_FACTOR:-0.75}"
SCALE_SHORT_PX="${SCALE_SHORT_PX:-0.55}"
SCALE_NORMAL_PX="${SCALE_NORMAL_PX:-0.35}"
SCALE_MIN="${SCALE_MIN:-0.0004}"
SCALE_MAX="${SCALE_MAX:-0.015}"
OPACITY_FLOOR="${OPACITY_FLOOR:-0.015}"
OPACITY_SCALE="${OPACITY_SCALE:-0.10}"
OPACITY_POWER="${OPACITY_POWER:-0.75}"
OPACITY_MIN="${OPACITY_MIN:-0.008}"
OPACITY_MAX="${OPACITY_MAX:-0.12}"
COLOR_GAIN="${COLOR_GAIN:-1.0}"
JITTER_PERP="${JITTER_PERP:-0.0}"
SEED="${SEED:-12345}"
WRITE_CPU_MERGED_PREVIEW="${WRITE_CPU_MERGED_PREVIEW:-0}"

for required in "${BASE_MODEL_DIR}" "${BASE_PLY}" "${TRACK_PAYLOAD}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[spray-curve-tracks-v0] required path not found: ${required}" >&2
    exit 1
  fi
done

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_MODEL_DIR}" "${NEWBORN_MODEL_DIR}"
fi

echo "[spray-curve-tracks-v0] base model : ${BASE_MODEL_DIR}"
echo "[spray-curve-tracks-v0] base ply   : ${BASE_PLY}"
echo "[spray-curve-tracks-v0] tracks     : ${TRACK_PAYLOAD}"
echo "[spray-curve-tracks-v0] newborn   : ${NEWBORN_MODEL_DIR}"
echo "[spray-curve-tracks-v0] output    : ${OUTPUT_MODEL_DIR}"
echo "[spray-curve-tracks-v0] selection : ${SELECTION}"
echo "[spray-curve-tracks-v0] sampling  : spacing=${SAMPLE_SPACING_PX}px max_samples=${MAX_SAMPLES_PER_TRACK}"
echo "[spray-curve-tracks-v0] opacity   : floor=${OPACITY_FLOOR} scale=${OPACITY_SCALE} max=${OPACITY_MAX}"

SPRAY_ARGS=(
  --base_model_dir "${BASE_MODEL_DIR}"
  --base_iteration "${BASE_ITERATION}"
  --track_payload "${TRACK_PAYLOAD}"
  --output_model_dir "${OUTPUT_MODEL_DIR}"
  --newborn_model_dir "${NEWBORN_MODEL_DIR}"
  --selection "${SELECTION}"
  --max_tracks "${MAX_TRACKS}"
  --max_total_newborn "${MAX_TOTAL_NEWBORN}"
  --sample_spacing_px "${SAMPLE_SPACING_PX}"
  --sample_spacing_min "${SAMPLE_SPACING_MIN}"
  --sample_spacing_max "${SAMPLE_SPACING_MAX}"
  --max_samples_per_track "${MAX_SAMPLES_PER_TRACK}"
  --scale_long_factor "${SCALE_LONG_FACTOR}"
  --scale_short_px "${SCALE_SHORT_PX}"
  --scale_normal_px "${SCALE_NORMAL_PX}"
  --scale_min "${SCALE_MIN}"
  --scale_max "${SCALE_MAX}"
  --opacity_floor "${OPACITY_FLOOR}"
  --opacity_scale "${OPACITY_SCALE}"
  --opacity_power "${OPACITY_POWER}"
  --opacity_min "${OPACITY_MIN}"
  --opacity_max "${OPACITY_MAX}"
  --color_gain "${COLOR_GAIN}"
  --jitter_perp "${JITTER_PERP}"
  --seed "${SEED}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  SPRAY_ARGS+=(--overwrite)
fi
if [[ "${FALLBACK_TO_KEEP}" == "1" ]]; then
  SPRAY_ARGS+=(--fallback_to_keep)
fi
if [[ "${WRITE_CPU_MERGED_PREVIEW}" == "1" ]]; then
  SPRAY_ARGS+=(--write_cpu_merged_preview)
fi

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/spray_sr_hf_curve_tracks_to_gaussian_layer_v0.py" "${SPRAY_ARGS[@]}"

NEWBORN_PLY="${NEWBORN_MODEL_DIR}/point_cloud/iteration_${BASE_ITERATION}/point_cloud.ply"
if [[ ! -f "${NEWBORN_PLY}" ]]; then
  echo "[spray-curve-tracks-v0] newborn PLY not found after spray: ${NEWBORN_PLY}" >&2
  exit 1
fi

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}:${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${MERGE_SCRIPT}" ]]; then
  if [[ -f "${SOF_ROOT}/merge_gaussian_plys_v0.py" ]]; then
    MERGE_SCRIPT="${SOF_ROOT}/merge_gaussian_plys_v0.py"
  else
    MERGE_SCRIPT="${SOF_ROOT}/scripts/merge_gaussian_plys_v0.py"
  fi
fi
if [[ ! -f "${MERGE_SCRIPT}" ]]; then
  echo "[spray-curve-tracks-v0] merge script not found: ${MERGE_SCRIPT}" >&2
  exit 1
fi
echo "[spray-curve-tracks-v0] merge     : ${MERGE_SCRIPT}"
"${PYTHON_BIN}" "${MERGE_SCRIPT}" \
  --source_path "${SCENE_ROOT}" \
  --model_path "${OUTPUT_MODEL_DIR}" \
  --images images_2 \
  --resolution 1 \
  --eval \
  --base_ply "${BASE_PLY}" \
  --extra_ply "${NEWBORN_PLY}" \
  --copy_config_from "${BASE_MODEL_DIR}" \
  --output_model_path "${OUTPUT_MODEL_DIR}" \
  --output_iteration "${OUTPUT_ITERATION}"

NEWBORN_METADATA="${NEWBORN_MODEL_DIR}/point_cloud/iteration_${BASE_ITERATION}/sprayed_sr_hf_curve_tracks_metadata_v0.npz"
MERGED_METADATA="${OUTPUT_MODEL_DIR}/point_cloud/iteration_${OUTPUT_ITERATION}/sprayed_sr_hf_curve_tracks_metadata_v0.npz"
if [[ -f "${NEWBORN_METADATA}" ]]; then
  mkdir -p "$(dirname -- "${MERGED_METADATA}")"
  cp "${NEWBORN_METADATA}" "${MERGED_METADATA}"
  echo "[spray-curve-tracks-v0] metadata:"
  echo "  ${MERGED_METADATA}"
fi

echo "[spray-curve-tracks-v0] done model:"
echo "  ${OUTPUT_MODEL_DIR}"
echo "[spray-curve-tracks-v0] merged ply:"
echo "  ${OUTPUT_MODEL_DIR}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
echo "[spray-curve-tracks-v0] tags:"
echo "  ${OUTPUT_MODEL_DIR}/point_cloud/iteration_${OUTPUT_ITERATION}/gaussian_tags.pt"
echo "[spray-curve-tracks-v0] summary:"
echo "  ${OUTPUT_MODEL_DIR}/spray_sr_hf_curve_tracks_to_gaussian_layer_v0_summary.json"
