#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_mip_vanilla_images8_v1/mip30k_sof_native_input_init_early4ksoft_v1}"
ITERATION="${ITERATION:--1}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-8}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"

RUN_LAYER_VARIANTS="${RUN_LAYER_VARIANTS:-1}"
RUN_CHILD_ABLATIONS="${RUN_CHILD_ABLATIONS:-1}"
RUN_FILTER_ABLATIONS="${RUN_FILTER_ABLATIONS:-0}"

SAVE_ALPHA="${SAVE_ALPHA:-1}"
SAVE_DEPTH="${SAVE_DEPTH:-1}"
SAVE_PREMUL="${SAVE_PREMUL:-1}"

MAKE_CONTACT_SHEETS="${MAKE_CONTACT_SHEETS:-1}"
CONTACT_MAX_IMAGES="${CONTACT_MAX_IMAGES:-8}"
CONTACT_COLUMNS="${CONTACT_COLUMNS:-4}"

MODEL_BASENAME="${MODEL_BASENAME:-$(basename "${MODEL_PATH}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/prepare_lineage_diagnostics_v0/${SCENE_NAME}/${MODEL_BASENAME}}"

if [[ ! -e "${SCENE_ROOT}" ]]; then
  echo "[prepare-lineage-diag-v0] scene root not found: ${SCENE_ROOT}" >&2
  exit 1
fi
if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "[prepare-lineage-diag-v0] model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

render_variant() {
  local label="$1"
  shift
  local out_root="${OUTPUT_ROOT}/${label}"
  local args=(
    "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py"
    --scene_root "${SCENE_ROOT}"
    --model_path "${MODEL_PATH}"
    --output_root "${out_root}"
    --images_subdir "${IMAGES_SUBDIR}"
    --iteration "${ITERATION}"
    --split "${SPLIT}"
    --max_views "${MAX_VIEWS}"
    "$@"
  )
  if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
    args+=(--white_background)
  fi
  if [[ "${SAVE_ALPHA}" == "1" ]]; then
    args+=(--save_alpha)
  fi
  if [[ "${SAVE_DEPTH}" == "1" ]]; then
    args+=(--save_depth)
  fi
  if [[ "${SAVE_PREMUL}" == "1" ]]; then
    args+=(--save_premul)
  fi
  echo "[prepare-lineage-diag-v0] render ${label}"
  "${args[@]}"
  if [[ "${MAKE_CONTACT_SHEETS}" == "1" ]]; then
    local render_dir
    render_dir="$("${PYTHON_BIN}" - <<'PY' "${out_root}/summary.json" "${SPLIT}"
import json, sys
summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
split = sys.argv[2]
print(summary["renders"][split]["render_root"])
PY
)"
    local sheet_path="${out_root}/contact_sheet_${IMAGES_SUBDIR}_${SPLIT}.png"
    "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/make_render_contact_sheet.py" \
      --render_dir "${render_dir}" \
      --output_path "${sheet_path}" \
      --max_images "${CONTACT_MAX_IMAGES}" \
      --columns "${CONTACT_COLUMNS}"
  fi
}

if [[ "${RUN_LAYER_VARIANTS}" == "1" ]]; then
  render_variant "full" \
    --selection_source lineage \
    --selection_key full \
    --selection_mode full

  render_variant "children_only" \
    --selection_source lineage \
    --selection_key children \
    --selection_mode selected_only

  render_variant "non_children_only" \
    --selection_source lineage \
    --selection_key non_children \
    --selection_mode selected_only

  render_variant "softened_children_only" \
    --selection_source lineage \
    --selection_key softened_children \
    --selection_mode selected_only

  render_variant "unsoftened_children_only" \
    --selection_source lineage \
    --selection_key unsoftened_children \
    --selection_mode selected_only

  render_variant "children_removed_full" \
    --selection_source lineage \
    --selection_key children \
    --selection_mode selected_removed

  render_variant "softened_removed_full" \
    --selection_source lineage \
    --selection_key softened_children \
    --selection_mode selected_removed
fi

if [[ "${RUN_CHILD_ABLATIONS}" == "1" ]]; then
  render_variant "children_restsh_zero" \
    --selection_source lineage \
    --selection_key children \
    --selection_mode full \
    --rest_scale 0.0

  render_variant "children_tau_0p5" \
    --selection_source lineage \
    --selection_key children \
    --selection_mode full \
    --tau_scale 0.5

  render_variant "children_scale_0p7" \
    --selection_source lineage \
    --selection_key children \
    --selection_mode full \
    --scale_multiplier 0.7
fi

if [[ "${RUN_FILTER_ABLATIONS}" == "1" ]]; then
  render_variant "children_filter_0p5" \
    --selection_source lineage \
    --selection_key children \
    --selection_mode full \
    --filter_multiplier 0.5
fi

"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/summarize_prepare_lineage_diagnostics_v0.py" \
  --output_root "${OUTPUT_ROOT}" \
  --split "${SPLIT}" \
  --summary_name lineage_diagnostics_index.json

echo
echo "[done] diagnostics root : ${OUTPUT_ROOT}"
echo "[done] diagnostics index: ${OUTPUT_ROOT}/lineage_diagnostics_index.json"
