#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

pick_first_existing_dir() {
  for p in "$@"; do
    if [[ -d "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

HBSR_ROOT="${HBSR_ROOT:-${WORKSPACE_ROOT}}"
PYTHON_EXE="${PYTHON_EXE:-python}"
INPUT_DIR="${INPUT_DIR:-/root/autodl-tmp/kitchen/images_8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/priors/kitchen_video_uav_x8to2}"

UAV_REPO="${UAV_REPO:-$(pick_first_existing_dir \
  /root/autodl-tmp/HBSR/video_sr_models/Upscale-A-Video \
  /Users/ltl/Desktop/codex_playground/video_sr_models/Upscale-A-Video)}"

UAV_CKPT_SRC="${UAV_CKPT_SRC:-$(pick_first_existing_dir \
  /root/autodl-tmp/upscale_a_video \
  /root/autodl-tmp/HBSR/upscale_a_video \
  /Users/ltl/Desktop/codex_playground/upscale_a_video)}"

if [[ -z "${UAV_REPO}" || ! -d "${UAV_REPO}" ]]; then
  echo "[uav-x8to2] UAV repo not found. Set UAV_REPO=/path/to/Upscale-A-Video" >&2
  exit 1
fi

if [[ -z "${UAV_CKPT_SRC}" || ! -d "${UAV_CKPT_SRC}" ]]; then
  echo "[uav-x8to2] UAV checkpoint folder not found. Set UAV_CKPT_SRC=/path/to/upscale_a_video" >&2
  exit 1
fi

TARGET_CKPT_ROOT="${UAV_REPO}/pretrained_models"
TARGET_CKPT_LINK="${TARGET_CKPT_ROOT}/upscale_a_video"
mkdir -p "${TARGET_CKPT_ROOT}"
ln -sfn "${UAV_CKPT_SRC}" "${TARGET_CKPT_LINK}"

echo "[uav-x8to2] HBSR_ROOT=${HBSR_ROOT}"
echo "[uav-x8to2] UAV_REPO=${UAV_REPO}"
echo "[uav-x8to2] UAV_CKPT_SRC=${UAV_CKPT_SRC}"
echo "[uav-x8to2] INPUT_DIR=${INPUT_DIR}"
echo "[uav-x8to2] OUTPUT_ROOT=${OUTPUT_ROOT}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=1
fi

echo "[uav-x8to2] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[uav-x8to2] MKL_NUM_THREADS=${MKL_NUM_THREADS}"

${PYTHON_EXE} "${HBSR_ROOT}/hybrid_sdfgs/tools/check_upscale_a_video_checkpoints.py" \
  --root "${TARGET_CKPT_LINK}"

CMD=(
  "${PYTHON_EXE}"
  "${HBSR_ROOT}/hybrid_sdfgs/tools/generate_video_sr_priors.py"
  --model uav
  --repo_root "${UAV_REPO}"
  --python_exe "${PYTHON_EXE}"
  --input_dir "${INPUT_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --uav_noise_level 120
  --uav_guidance_scale 6
  --uav_inference_steps 30
  --uav_a_prompt "best quality, extremely detailed"
  --uav_n_prompt "blur, worst quality"
  --uav_color_fix AdaIn
  --uav_save_suffix x8to2
)

echo "[run] ${CMD[*]}"
"${CMD[@]}"

echo "[uav-x8to2] done: ${OUTPUT_ROOT}"
