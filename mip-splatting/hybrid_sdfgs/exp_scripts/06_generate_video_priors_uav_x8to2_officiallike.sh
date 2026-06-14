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
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/priors/kitchen_video_uav_x8to2_officiallike}"

UAV_REPO="${UAV_REPO:-$(pick_first_existing_dir \
  /root/autodl-tmp/Upscale-A-Video \
  /root/autodl-tmp/HBSR/video_sr_models/Upscale-A-Video \
  /Users/ltl/Desktop/codex_playground/video_sr_models/Upscale-A-Video)}"

UAV_CKPT_SRC="${UAV_CKPT_SRC:-$(pick_first_existing_dir \
  /root/autodl-tmp/Upscale-A-Video/pretrained_models/upscale_a_video \
  /root/autodl-tmp/upscale_a_video \
  /root/autodl-tmp/HBSR/upscale_a_video \
  /Users/ltl/Desktop/codex_playground/upscale_a_video)}"

UAV_NOISE_LEVEL="${UAV_NOISE_LEVEL:-120}"
UAV_GUIDANCE_SCALE="${UAV_GUIDANCE_SCALE:-6}"
UAV_INFERENCE_STEPS="${UAV_INFERENCE_STEPS:-30}"
UAV_A_PROMPT="${UAV_A_PROMPT:-best quality, extremely detailed}"
UAV_N_PROMPT="${UAV_N_PROMPT:-blur, worst quality}"
UAV_COLOR_FIX="${UAV_COLOR_FIX:-AdaIn}"
UAV_SAVE_SUFFIX="${UAV_SAVE_SUFFIX:-x8to2_officiallike}"
UAV_USE_LLAVA="${UAV_USE_LLAVA:-0}"
UAV_USE_VIDEO_VAE="${UAV_USE_VIDEO_VAE:-0}"
UAV_PERFORM_TILE="${UAV_PERFORM_TILE:-0}"
UAV_TILE_SIZE="${UAV_TILE_SIZE:-256}"
UAV_PROPAGATION_STEPS="${UAV_PROPAGATION_STEPS:-}"

if [[ -z "${UAV_REPO}" || ! -d "${UAV_REPO}" ]]; then
  echo "[uav-x8to2-officiallike] UAV repo not found. Set UAV_REPO=/path/to/Upscale-A-Video" >&2
  exit 1
fi

if [[ -z "${UAV_CKPT_SRC}" || ! -d "${UAV_CKPT_SRC}" ]]; then
  echo "[uav-x8to2-officiallike] UAV checkpoint folder not found. Set UAV_CKPT_SRC=/path/to/upscale_a_video" >&2
  exit 1
fi

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=1
fi

TARGET_CKPT_ROOT="${UAV_REPO}/pretrained_models"
TARGET_CKPT_LINK="${TARGET_CKPT_ROOT}/upscale_a_video"
mkdir -p "${TARGET_CKPT_ROOT}"
ln -sfn "${UAV_CKPT_SRC}" "${TARGET_CKPT_LINK}"

echo "[uav-x8to2-officiallike] HBSR_ROOT=${HBSR_ROOT}"
echo "[uav-x8to2-officiallike] UAV_REPO=${UAV_REPO}"
echo "[uav-x8to2-officiallike] UAV_CKPT_SRC=${UAV_CKPT_SRC}"
echo "[uav-x8to2-officiallike] INPUT_DIR=${INPUT_DIR}"
echo "[uav-x8to2-officiallike] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[uav-x8to2-officiallike] UAV_NOISE_LEVEL=${UAV_NOISE_LEVEL}"
echo "[uav-x8to2-officiallike] UAV_GUIDANCE_SCALE=${UAV_GUIDANCE_SCALE}"
echo "[uav-x8to2-officiallike] UAV_INFERENCE_STEPS=${UAV_INFERENCE_STEPS}"
echo "[uav-x8to2-officiallike] UAV_COLOR_FIX=${UAV_COLOR_FIX}"
echo "[uav-x8to2-officiallike] UAV_SAVE_SUFFIX=${UAV_SAVE_SUFFIX}"
echo "[uav-x8to2-officiallike] UAV_USE_LLAVA=${UAV_USE_LLAVA}"
echo "[uav-x8to2-officiallike] UAV_USE_VIDEO_VAE=${UAV_USE_VIDEO_VAE}"
echo "[uav-x8to2-officiallike] UAV_PERFORM_TILE=${UAV_PERFORM_TILE}"
echo "[uav-x8to2-officiallike] UAV_TILE_SIZE=${UAV_TILE_SIZE}"
echo "[uav-x8to2-officiallike] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[uav-x8to2-officiallike] MKL_NUM_THREADS=${MKL_NUM_THREADS}"

"${PYTHON_EXE}" "${HBSR_ROOT}/hybrid_sdfgs/tools/check_upscale_a_video_checkpoints.py" \
  --root "${TARGET_CKPT_LINK}"

CMD=(
  "${PYTHON_EXE}"
  "${HBSR_ROOT}/hybrid_sdfgs/tools/generate_video_sr_priors.py"
  --model uav
  --repo_root "${UAV_REPO}"
  --python_exe "${PYTHON_EXE}"
  --input_dir "${INPUT_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --uav_noise_level "${UAV_NOISE_LEVEL}"
  --uav_guidance_scale "${UAV_GUIDANCE_SCALE}"
  --uav_inference_steps "${UAV_INFERENCE_STEPS}"
  --uav_a_prompt "${UAV_A_PROMPT}"
  --uav_n_prompt "${UAV_N_PROMPT}"
  --uav_color_fix "${UAV_COLOR_FIX}"
  --uav_save_suffix "${UAV_SAVE_SUFFIX}"
)

if [[ "${UAV_USE_LLAVA}" == "1" ]]; then
  CMD+=(--uav_use_llava)
fi

if [[ "${UAV_USE_VIDEO_VAE}" == "1" ]]; then
  CMD+=(--uav_use_video_vae)
fi

if [[ "${UAV_PERFORM_TILE}" == "1" ]]; then
  CMD+=(--uav_perform_tile --uav_tile_size "${UAV_TILE_SIZE}")
fi

if [[ -n "${UAV_PROPAGATION_STEPS}" ]]; then
  CMD+=(--uav_propagation_steps "${UAV_PROPAGATION_STEPS}")
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"

echo "[uav-x8to2-officiallike] done: ${OUTPUT_ROOT}/priors"
