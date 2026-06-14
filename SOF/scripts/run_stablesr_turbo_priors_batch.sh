#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"

STABLESR_ROOT="${STABLESR_ROOT:-${WORKSPACE_ROOT}/StableSR}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${WORKSPACE_ROOT}/archive}"
SCENES="${SCENES:-bicycle bonsai counter flowers garden stump treehill kitchen room}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"

PREPARE_IMAGES8="${PREPARE_IMAGES8:-1}"
OVERWRITE_PRIORS="${OVERWRITE_PRIORS:-0}"
RESIZE_FILTER="${RESIZE_FILTER:-bicubic}"
GENERATE_IMAGES8_SCALE="${GENERATE_IMAGES8_SCALE:-4}"

STABLESR_ENV_NAME="${STABLESR_ENV_NAME:-stablesr}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STABLESR_CKPT="${STABLESR_CKPT:-${WORKSPACE_ROOT}/stablesr_turbo.ckpt}"
VQGAN_CKPT="${VQGAN_CKPT:-${WORKSPACE_ROOT}/vqgan_cfw_00011.ckpt}"
OPENCLIP_BIN="${OPENCLIP_BIN:-${WORKSPACE_ROOT}/open_clip_pytorch_model.bin}"
LOCALCLIP_CONFIG="${LOCALCLIP_CONFIG:-${STABLESR_ROOT}/configs/stableSRNew/v2-finetune_text_T_512_localclip.yaml}"

mkdir -p "${ARCHIVE_ROOT}"

for path in "${SOF_ROOT}" "${STABLESR_ROOT}" "${STABLESR_CKPT}" "${VQGAN_CKPT}" "${OPENCLIP_BIN}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[stablesr-priors] required path not found: ${path}" >&2
    exit 1
  fi
done

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${STABLESR_ENV_NAME}"

unset OMP_NUM_THREADS
unset MKL_NUM_THREADS

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="${STABLESR_ROOT}:${WORKSPACE_ROOT}/taming-transformers:${WORKSPACE_ROOT}/CLIP:${PYTHONPATH:-}"

if [[ ! -f "${LOCALCLIP_CONFIG}" ]]; then
  cp "${STABLESR_ROOT}/configs/stableSRNew/v2-finetune_text_T_512.yaml" "${LOCALCLIP_CONFIG}"
fi

LOCALCLIP_CONFIG="${LOCALCLIP_CONFIG}" \
OPENCLIP_BIN="${OPENCLIP_BIN}" \
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

p = Path(os.environ["LOCALCLIP_CONFIG"])
s = p.read_text()
old = '''    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder
      params:
        freeze: True
        layer: "penultimate"
'''
new = f'''    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder
      params:
        freeze: True
        layer: "penultimate"
        arch: "ViT-H-14"
        version: "{os.environ["OPENCLIP_BIN"]}"
'''

if 'version: "' in s and os.environ["OPENCLIP_BIN"] in s:
    print(f"[stablesr-priors] localclip config already patched: {p}")
else:
    if old not in s:
        raise SystemExit(f"target cond_stage_config block not found in {p}")
    p.write_text(s.replace(old, new))
    print(f"[stablesr-priors] patched localclip config: {p}")
PY

STABLESR_ROOT="${STABLESR_ROOT}" \
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

p = Path(os.environ["STABLESR_ROOT"]) / "scripts" / "util_image.py"
s = p.read_text()
old = """        self.im_res[:, :, h_start:h_end, w_start:w_end] += pch_res * self.weight
        self.pixel_count[:, :, h_start:h_end, w_start:w_end] += self.weight
"""
new = """        # Edge patches can be smaller than the nominal tile size when only one
        # image dimension exceeds the VQGAN tile limit. In that case, crop the
        # precomputed Gaussian weights to the actual patch size before blending.
        weight = self.weight[:, :, :pch_res.shape[-2], :pch_res.shape[-1]]
        self.im_res[:, :, h_start:h_end, w_start:w_end] += pch_res * weight
        self.pixel_count[:, :, h_start:h_end, w_start:w_end] += weight
"""

if "weight = self.weight[:, :, :pch_res.shape[-2], :pch_res.shape[-1]]" in s:
    print(f"[stablesr-priors] StableSR util_image already patched: {p}")
elif old in s:
    p.write_text(s.replace(old, new))
    print(f"[stablesr-priors] patched StableSR util_image: {p}")
else:
    raise SystemExit(f"target update_gaussian block not found in {p}")
PY

echo "[stablesr-priors] scenes        : ${SCENES}"
echo "[stablesr-priors] archive root  : ${ARCHIVE_ROOT}"
echo "[stablesr-priors] stablesr root : ${STABLESR_ROOT}"
echo "[stablesr-priors] output subdir : priors"

for scene in ${SCENES}; do
  SCENE_ROOT="${ARCHIVE_ROOT}/${scene}"
  TARGET_DIR="${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}"
  SOURCE_DIR="${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}"
  PRIOR_DIR="${SCENE_ROOT}/priors"

  echo
  echo "[stablesr-priors] ===== scene: ${scene} ====="

  if [[ ! -d "${SCENE_ROOT}" ]]; then
    echo "[stablesr-priors] scene root not found: ${SCENE_ROOT}" >&2
    exit 1
  fi
  if [[ ! -d "${TARGET_DIR}" ]]; then
    echo "[stablesr-priors] target image dir not found: ${TARGET_DIR}" >&2
    exit 1
  fi

  if [[ ! -d "${SOURCE_DIR}" ]]; then
    if [[ "${PREPARE_IMAGES8}" != "1" ]]; then
      echo "[stablesr-priors] source image dir missing and PREPARE_IMAGES8=0: ${SOURCE_DIR}" >&2
      exit 1
    fi
    echo "[stablesr-priors] generate ${SOURCE_IMAGES_SUBDIR} from ${TARGET_IMAGES_SUBDIR}"
    (
      cd "${SOF_ROOT}"
      "${PYTHON_BIN}" scripts/generate_downsampled_images.py \
        --source_dir "${TARGET_DIR}" \
        --output_dir "${SOURCE_DIR}" \
        --scale "${GENERATE_IMAGES8_SCALE}" \
        --resize_filter "${RESIZE_FILTER}"
    )
  fi

  mkdir -p "${PRIOR_DIR}"

  input_count=$(find "${SOURCE_DIR}" -maxdepth 1 -type f | wc -l | tr -d ' ')
  prior_count=$(find "${PRIOR_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')

  if [[ "${OVERWRITE_PRIORS}" != "1" && "${prior_count}" -ge "${input_count}" && "${input_count}" -gt 0 ]]; then
    echo "[stablesr-priors] priors already exist (${prior_count}/${input_count}), skip"
    continue
  fi

  echo "[stablesr-priors] input dir  : ${SOURCE_DIR}"
  echo "[stablesr-priors] prior dir  : ${PRIOR_DIR}"

  (
    cd "${STABLESR_ROOT}"
    "${PYTHON_BIN}" scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
      --config "${LOCALCLIP_CONFIG}" \
      --ckpt "${STABLESR_CKPT}" \
      --vqgan_ckpt "${VQGAN_CKPT}" \
      --init-img "${SOURCE_DIR}" \
      --outdir "${PRIOR_DIR}" \
      --ddpm_steps 4 \
      --dec_w 0.5 \
      --seed 42 \
      --n_samples 1 \
      --colorfix_type wavelet \
      --upscale 4 \
      --input_size 512 \
      --tile_overlap 32 \
      --vqgantile_size 2048 \
      --vqgantile_stride 2048
  )
done

echo
echo "[stablesr-priors] done"
