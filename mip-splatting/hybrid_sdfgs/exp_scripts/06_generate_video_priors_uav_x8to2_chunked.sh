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
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/priors/kitchen_video_uav_x8to2_chunked}"
CHUNK_SIZE="${CHUNK_SIZE:-93}"
UAV_PERFORM_TILE="${UAV_PERFORM_TILE:-0}"
UAV_TILE_SIZE="${UAV_TILE_SIZE:-256}"

UAV_REPO="${UAV_REPO:-$(pick_first_existing_dir \
  /root/autodl-tmp/Upscale-A-Video \
  /root/autodl-tmp/HBSR/video_sr_models/Upscale-A-Video \
  /Users/ltl/Desktop/codex_playground/video_sr_models/Upscale-A-Video)}"

UAV_CKPT_SRC="${UAV_CKPT_SRC:-$(pick_first_existing_dir \
  /root/autodl-tmp/Upscale-A-Video/pretrained_models/upscale_a_video \
  /root/autodl-tmp/upscale_a_video \
  /Users/ltl/Desktop/codex_playground/upscale_a_video)}"

if [[ -z "${UAV_REPO}" || ! -d "${UAV_REPO}" ]]; then
  echo "[uav-x8to2-chunked] UAV repo not found. Set UAV_REPO=/path/to/Upscale-A-Video" >&2
  exit 1
fi

if [[ -z "${UAV_CKPT_SRC}" || ! -d "${UAV_CKPT_SRC}" ]]; then
  echo "[uav-x8to2-chunked] UAV checkpoint folder not found. Set UAV_CKPT_SRC=/path/to/upscale_a_video" >&2
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

echo "[uav-x8to2-chunked] HBSR_ROOT=${HBSR_ROOT}"
echo "[uav-x8to2-chunked] UAV_REPO=${UAV_REPO}"
echo "[uav-x8to2-chunked] UAV_CKPT_SRC=${UAV_CKPT_SRC}"
echo "[uav-x8to2-chunked] INPUT_DIR=${INPUT_DIR}"
echo "[uav-x8to2-chunked] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[uav-x8to2-chunked] CHUNK_SIZE=${CHUNK_SIZE}"
echo "[uav-x8to2-chunked] UAV_PERFORM_TILE=${UAV_PERFORM_TILE}"
echo "[uav-x8to2-chunked] UAV_TILE_SIZE=${UAV_TILE_SIZE}"
echo "[uav-x8to2-chunked] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[uav-x8to2-chunked] MKL_NUM_THREADS=${MKL_NUM_THREADS}"

${PYTHON_EXE} "${HBSR_ROOT}/hybrid_sdfgs/tools/check_upscale_a_video_checkpoints.py" \
  --root "${TARGET_CKPT_LINK}"

STAGE_ROOT="${OUTPUT_ROOT}/_chunk_stage"
CHUNK_INPUT_ROOT="${STAGE_ROOT}/inputs"
CHUNK_OUTPUT_ROOT="${STAGE_ROOT}/outputs"
FINAL_PRIOR_ROOT="${OUTPUT_ROOT}/priors"

rm -rf "${STAGE_ROOT}"
mkdir -p "${CHUNK_INPUT_ROOT}" "${CHUNK_OUTPUT_ROOT}" "${FINAL_PRIOR_ROOT}"

export INPUT_DIR CHUNK_INPUT_ROOT CHUNK_SIZE

mapfile -t CHUNK_DIRS < <("${PYTHON_EXE}" - <<'PY'
import os
from pathlib import Path

input_dir = Path(os.environ["INPUT_DIR"])
chunk_input_root = Path(os.environ["CHUNK_INPUT_ROOT"])
chunk_size = int(os.environ["CHUNK_SIZE"])
exts = {".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"}
paths = sorted([p for p in input_dir.iterdir() if p.suffix in exts], key=lambda p: p.name)
if not paths:
    raise SystemExit("No input images found.")
for chunk_idx in range(0, len(paths), chunk_size):
    chunk_dir = chunk_input_root / f"chunk_{chunk_idx // chunk_size:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for src in paths[chunk_idx:chunk_idx + chunk_size]:
        dst = chunk_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    print(str(chunk_dir))
PY
)

NUM_CHUNKS="${#CHUNK_DIRS[@]}"
if [[ "${NUM_CHUNKS}" -eq 0 ]]; then
  echo "[uav-x8to2-chunked] no chunks created" >&2
  exit 1
fi

for idx in "${!CHUNK_DIRS[@]}"; do
  CHUNK_DIR="${CHUNK_DIRS[$idx]}"
  CHUNK_NAME="$(basename "${CHUNK_DIR}")"
  CHUNK_OUT="${CHUNK_OUTPUT_ROOT}/${CHUNK_NAME}"
  echo "[uav-x8to2-chunked] chunk $((idx + 1))/${NUM_CHUNKS}: ${CHUNK_NAME}"

  "${PYTHON_EXE}" "${HBSR_ROOT}/hybrid_sdfgs/tools/generate_video_sr_priors.py" \
    --model uav \
    --repo_root "${UAV_REPO}" \
    --python_exe "${PYTHON_EXE}" \
    --input_dir "${CHUNK_DIR}" \
    --output_root "${CHUNK_OUT}" \
    --uav_noise_level 120 \
    --uav_guidance_scale 6 \
    --uav_inference_steps 30 \
    --uav_a_prompt "best quality, extremely detailed" \
    --uav_n_prompt "blur, worst quality" \
    --uav_color_fix AdaIn \
    --uav_save_suffix "${CHUNK_NAME}" \
    $([[ "${UAV_PERFORM_TILE}" == "1" ]] && printf '%s ' --uav_perform_tile --uav_tile_size "${UAV_TILE_SIZE}")

  if [[ ! -d "${CHUNK_OUT}/priors" ]]; then
    echo "[uav-x8to2-chunked] missing priors for ${CHUNK_NAME}" >&2
    exit 1
  fi

  find "${CHUNK_OUT}/priors" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.webp' \) \
    -exec cp {} "${FINAL_PRIOR_ROOT}/" \;
done

COUNT="$(find "${FINAL_PRIOR_ROOT}" -maxdepth 1 -type f | wc -l | tr -d ' ')"
echo "[uav-x8to2-chunked] done: ${OUTPUT_ROOT} (frames=${COUNT})"
