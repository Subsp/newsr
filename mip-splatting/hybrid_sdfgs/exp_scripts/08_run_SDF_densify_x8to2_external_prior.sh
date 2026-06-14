#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HBSR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
INPUT_SUBDIR="${INPUT_SUBDIR:-images_8}"
GT_SUBDIR="${GT_SUBDIR:-images_2}"
PRIOR_DIR="${PRIOR_DIR:-/root/autodl-tmp/priors/kitchen_video_flashvsr_seqmat_x8to2/priors}"
OUTPUT_PATH="${OUTPUT_PATH:-${HBSR_ROOT}/outputs/SDF_densify_x8to2_external_prior}"
PYTHON_EXE="${PYTHON_EXE:-python}"

LEVELS="${LEVELS:-2}"
HF_WEIGHT="${HF_WEIGHT:-0.75}"
DELTA_CLIP="${DELTA_CLIP:-0.10}"
CONSISTENCY_TAU="${CONSISTENCY_TAU:-0.08}"
ENERGY_FLOOR="${ENERGY_FLOOR:-0.002}"
GAIN_POWER="${GAIN_POWER:-1.0}"
WRITE_DEBUG="${WRITE_DEBUG:-0}"
RUN_EXISTING_METRICS="${RUN_EXISTING_METRICS:-1}"

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[SDF_densify-x8to2] scene root not found: ${SCENE_ROOT}"
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/${INPUT_SUBDIR}" ]]; then
  echo "[SDF_densify-x8to2] input dir not found: ${SCENE_ROOT}/${INPUT_SUBDIR}"
  exit 1
fi

if [[ ! -d "${PRIOR_DIR}" ]]; then
  echo "[SDF_densify-x8to2] prior dir not found: ${PRIOR_DIR}"
  exit 1
fi

mkdir -p "${OUTPUT_PATH}"
cd "${HBSR_ROOT}"

CMD=(
  "${PYTHON_EXE}" "hybrid_sdfgs/standalone_x8to2_prior/run_x8to2.py"
  "--scene_root" "${SCENE_ROOT}"
  "--prior_dir" "${PRIOR_DIR}"
  "--output_dir" "${OUTPUT_PATH}"
  "--input_subdir" "${INPUT_SUBDIR}"
  "--gt_subdir" "${GT_SUBDIR}"
  "--levels" "${LEVELS}"
  "--hf_weight" "${HF_WEIGHT}"
  "--delta_clip" "${DELTA_CLIP}"
  "--consistency_tau" "${CONSISTENCY_TAU}"
  "--energy_floor" "${ENERGY_FLOOR}"
  "--gain_power" "${GAIN_POWER}"
)

if [[ "${WRITE_DEBUG}" == "1" ]]; then
  CMD+=("--write_debug")
fi

echo "[SDF_densify-x8to2] running standalone prior fusion"
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

if [[ "${RUN_EXISTING_METRICS}" == "1" && -d "${SCENE_ROOT}/${GT_SUBDIR}" ]]; then
  echo "[SDF_densify-x8to2] running existing metrics wrapper"
  "${PYTHON_EXE}" "hybrid_sdfgs/standalone_x8to2_prior/eval_with_existing_metrics.py" \
    --output_dir "${OUTPUT_PATH}"
else
  echo "[SDF_densify-x8to2] skip existing metrics (RUN_EXISTING_METRICS=${RUN_EXISTING_METRICS}, gt=${SCENE_ROOT}/${GT_SUBDIR})"
fi

echo "[SDF_densify-x8to2] done: ${OUTPUT_PATH}"
echo "[SDF_densify-x8to2] prior dir consumed directly: ${PRIOR_DIR}"
