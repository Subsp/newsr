#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_mip_vanilla_images8_v1/mip30k_sof_native_input_init_early4ksoft_v1_debug}"
ITERATION="${ITERATION:--1}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-8}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"

MAKE_CONTACT_SHEETS="${MAKE_CONTACT_SHEETS:-1}"
CONTACT_MAX_IMAGES="${CONTACT_MAX_IMAGES:-8}"
CONTACT_COLUMNS="${CONTACT_COLUMNS:-4}"

MODEL_BASENAME="${MODEL_BASENAME:-$(basename "${MODEL_PATH}")}"
STAGE_ROOT="${STAGE_ROOT:-${MODEL_PATH}/debug_prepare_stages}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/prepare_stage_diagnostics_v0/${SCENE_NAME}/${MODEL_BASENAME}}"

if [[ ! -e "${SCENE_ROOT}" ]]; then
  echo "[prepare-stage-diag-v0] scene root not found: ${SCENE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${STAGE_ROOT}" ]]; then
  echo "[prepare-stage-diag-v0] debug stage root not found: ${STAGE_ROOT}" >&2
  echo "[prepare-stage-diag-v0] rerun prepare with DEBUG_DUMP_PREPARE_STAGES=1 first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

shopt -s nullglob
stages=("${STAGE_ROOT}"/debug_stage_*)
shopt -u nullglob
if (( ${#stages[@]} == 0 )); then
  echo "[prepare-stage-diag-v0] no debug_stage_* directories under ${STAGE_ROOT}" >&2
  exit 1
fi

for stage_path in "${stages[@]}"; do
  if [[ ! -d "${stage_path}" ]]; then
    continue
  fi
  stage_name="$(basename "${stage_path}")"
  out_root="${OUTPUT_ROOT}/${stage_name}"
  render_args=(
    "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py"
    --scene_root "${SCENE_ROOT}"
    --model_path "${stage_path}"
    --output_dir "${out_root}"
    --images_subdir "${IMAGES_SUBDIR}"
    --iteration "${ITERATION}"
    --split "${SPLIT}"
    --max_views "${MAX_VIEWS}"
  )
  if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
    render_args+=(--white_background)
  fi
  echo "[prepare-stage-diag-v0] render ${stage_name}"
  "${render_args[@]}"

  if [[ "${MAKE_CONTACT_SHEETS}" == "1" ]]; then
    render_dir="$("${PYTHON_BIN}" - <<'PY' "${out_root}/render_model_no_gt_summary.json" "${SPLIT}"
import json, sys
summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
split = sys.argv[2]
print(summary["renders"][split]["render_root"])
PY
)"
    sheet_path="${out_root}/contact_sheet_${IMAGES_SUBDIR}_${SPLIT}.png"
    "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/make_render_contact_sheet.py" \
      --render_dir "${render_dir}" \
      --output_path "${sheet_path}" \
      --max_images "${CONTACT_MAX_IMAGES}" \
      --columns "${CONTACT_COLUMNS}"
  fi
done

"${PYTHON_BIN}" - <<'PY' "${OUTPUT_ROOT}" "${SPLIT}" "${IMAGES_SUBDIR}" > "${OUTPUT_ROOT}/prepare_stage_diagnostics_v0_summary.json"
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1]).resolve()
split = sys.argv[2]
images_subdir = sys.argv[3]
rows = []
for child in sorted(output_root.iterdir()):
    summary_path = child / "render_model_no_gt_summary.json"
    if not child.is_dir() or not summary_path.is_file():
        continue
    summary = json.load(open(summary_path, "r", encoding="utf-8"))
    stage_summary_path = summary.get("model_path")
    debug_summary = None
    if stage_summary_path:
        candidate = Path(stage_summary_path) / "summary.json"
        if candidate.is_file():
            debug_summary = json.load(open(candidate, "r", encoding="utf-8"))
    render_info = summary.get("renders", {}).get(split, {})
    contact_sheet = child / f"contact_sheet_{images_subdir}_{split}.png"
    rows.append({
        "stage": child.name,
        "stage_index": None if debug_summary is None else debug_summary.get("stage_index"),
        "stage_name": None if debug_summary is None else debug_summary.get("stage_name"),
        "num_gaussians": None if debug_summary is None else debug_summary.get("num_gaussians"),
        "render_root": render_info.get("render_root"),
        "selected_indices": render_info.get("selected_indices"),
        "contact_sheet": str(contact_sheet) if contact_sheet.is_file() else None,
        "render_summary": str(summary_path),
        "debug_summary": None if debug_summary is None else str(Path(stage_summary_path) / "summary.json"),
    })
payload = {
    "mode": "prepare_stage_diagnostics_v0",
    "output_root": str(output_root),
    "split": split,
    "stage_count": len(rows),
    "stages": rows,
}
print(json.dumps(payload, indent=2))
PY

echo
echo "[done] stage diagnostics root : ${OUTPUT_ROOT}"
echo "[done] stage summary          : ${OUTPUT_ROOT}/prepare_stage_diagnostics_v0_summary.json"
