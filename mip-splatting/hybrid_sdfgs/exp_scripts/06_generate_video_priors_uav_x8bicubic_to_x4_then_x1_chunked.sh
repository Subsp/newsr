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
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/kitchen}"
INPUT_DIR="${INPUT_DIR:-${DATASET_ROOT}/images_8}"
REF_DIR="${REF_DIR:-${DATASET_ROOT}/images_4}"
GT_DIR="${GT_DIR:-${DATASET_ROOT}/images}"
GT2_DIR="${GT2_DIR:-${DATASET_ROOT}/images_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/priors/kitchen_video_uav_x8bicubic_to_x4_then_x1_chunked}"
CHUNK_SIZE="${CHUNK_SIZE:-48}"
UAV_PERFORM_TILE="${UAV_PERFORM_TILE:-0}"
UAV_TILE_SIZE="${UAV_TILE_SIZE:-512}"

UAV_REPO="${UAV_REPO:-$(pick_first_existing_dir \
  /root/autodl-tmp/Upscale-A-Video \
  /root/autodl-tmp/HBSR/video_sr_models/Upscale-A-Video \
  /Users/ltl/Desktop/codex_playground/video_sr_models/Upscale-A-Video)}"

UAV_CKPT_SRC="${UAV_CKPT_SRC:-$(pick_first_existing_dir \
  /root/autodl-tmp/Upscale-A-Video/pretrained_models/upscale_a_video \
  /root/autodl-tmp/upscale_a_video \
  /Users/ltl/Desktop/codex_playground/upscale_a_video)}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[uav-x8b4to1] input dir not found: ${INPUT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${REF_DIR}" ]]; then
  echo "[uav-x8b4to1] reference dir not found: ${REF_DIR}" >&2
  exit 1
fi
if [[ ! -d "${GT_DIR}" ]]; then
  echo "[uav-x8b4to1] GT dir not found: ${GT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${GT2_DIR}" ]]; then
  echo "[uav-x8b4to1] GT images_2 dir not found: ${GT2_DIR}" >&2
  exit 1
fi
if [[ -z "${UAV_REPO}" || ! -d "${UAV_REPO}" ]]; then
  echo "[uav-x8b4to1] UAV repo not found. Set UAV_REPO=/path/to/Upscale-A-Video" >&2
  exit 1
fi
if [[ -z "${UAV_CKPT_SRC}" || ! -d "${UAV_CKPT_SRC}" ]]; then
  echo "[uav-x8b4to1] UAV checkpoint folder not found. Set UAV_CKPT_SRC=/path/to/upscale_a_video" >&2
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

echo "[uav-x8b4to1] HBSR_ROOT=${HBSR_ROOT}"
echo "[uav-x8b4to1] UAV_REPO=${UAV_REPO}"
echo "[uav-x8b4to1] UAV_CKPT_SRC=${UAV_CKPT_SRC}"
echo "[uav-x8b4to1] INPUT_DIR=${INPUT_DIR}"
echo "[uav-x8b4to1] REF_DIR=${REF_DIR}"
echo "[uav-x8b4to1] GT_DIR=${GT_DIR}"
echo "[uav-x8b4to1] GT2_DIR=${GT2_DIR}"
echo "[uav-x8b4to1] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[uav-x8b4to1] CHUNK_SIZE=${CHUNK_SIZE}"
echo "[uav-x8b4to1] UAV_PERFORM_TILE=${UAV_PERFORM_TILE}"
echo "[uav-x8b4to1] UAV_TILE_SIZE=${UAV_TILE_SIZE}"

${PYTHON_EXE} "${HBSR_ROOT}/hybrid_sdfgs/tools/check_upscale_a_video_checkpoints.py" \
  --root "${TARGET_CKPT_LINK}"

STAGE_ROOT="${OUTPUT_ROOT}/_bicubic_stage"
BICUBIC_INPUT_DIR="${STAGE_ROOT}/images_8_to_images_4_bicubic"
mkdir -p "${BICUBIC_INPUT_DIR}"
rm -rf "${BICUBIC_INPUT_DIR}"
mkdir -p "${BICUBIC_INPUT_DIR}"

export INPUT_DIR REF_DIR BICUBIC_INPUT_DIR
"${PYTHON_EXE}" - <<'PY'
import os
from pathlib import Path
from PIL import Image

input_dir = Path(os.environ["INPUT_DIR"])
ref_dir = Path(os.environ["REF_DIR"])
out_dir = Path(os.environ["BICUBIC_INPUT_DIR"])
exts = {".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"}

inputs = sorted([p for p in input_dir.iterdir() if p.suffix in exts], key=lambda p: p.name)
if not inputs:
    raise SystemExit(f"No input images found in {input_dir}")

for src in inputs:
    ref = ref_dir / src.name
    if not ref.exists():
        raise SystemExit(f"Missing reference frame in images_4: {ref}")
    with Image.open(src) as im_src, Image.open(ref) as im_ref:
        target_size = im_ref.size
        resized = im_src.convert("RGB").resize(target_size, Image.BICUBIC)
        resized.save(out_dir / src.name)

print(f"[uav-x8b4to1] bicubic prepared: {len(inputs)} frames -> {out_dir}")
PY

mkdir -p "${OUTPUT_ROOT}"
ln -sfn "${GT_DIR}" "${OUTPUT_ROOT}/ref_images"
ln -sfn "${GT2_DIR}" "${OUTPUT_ROOT}/ref_images_2"
ln -sfn "${BICUBIC_INPUT_DIR}" "${OUTPUT_ROOT}/input_bicubic_images_4"

QUAD_STAGE_ROOT="${OUTPUT_ROOT}/_quad_stage"
QUAD_INPUT_ROOT="${QUAD_STAGE_ROOT}/inputs"
QUAD_OUTPUT_ROOT="${QUAD_STAGE_ROOT}/outputs"
FINAL_PRIOR_ROOT="${OUTPUT_ROOT}/priors"
rm -rf "${QUAD_STAGE_ROOT}" "${FINAL_PRIOR_ROOT}"
mkdir -p "${QUAD_INPUT_ROOT}" "${QUAD_OUTPUT_ROOT}" "${FINAL_PRIOR_ROOT}"

export BICUBIC_INPUT_DIR QUAD_INPUT_ROOT
"${PYTHON_EXE}" - <<'PY'
import os
from pathlib import Path
from PIL import Image

src_root = Path(os.environ["BICUBIC_INPUT_DIR"])
quad_root = Path(os.environ["QUAD_INPUT_ROOT"])
exts = {".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"}
frames = sorted([p for p in src_root.iterdir() if p.suffix in exts], key=lambda p: p.name)
if not frames:
    raise SystemExit(f"No bicubic frames found in {src_root}")

quad_names = ["quad_tl", "quad_tr", "quad_bl", "quad_br"]
for name in quad_names:
    (quad_root / name).mkdir(parents=True, exist_ok=True)

for src in frames:
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        mid_x = w // 2
        mid_y = h // 2
        boxes = {
            "quad_tl": (0, 0, mid_x, mid_y),
            "quad_tr": (mid_x, 0, w, mid_y),
            "quad_bl": (0, mid_y, mid_x, h),
            "quad_br": (mid_x, mid_y, w, h),
        }
        for name, box in boxes.items():
            crop = rgb.crop(box)
            crop.save(quad_root / name / src.name)

print(f"[uav-x8b4to1] prepared 4 symmetric quadrants under {quad_root}")
PY

ln -sfn "${QUAD_INPUT_ROOT}" "${OUTPUT_ROOT}/input_quadrants_images_4"

CHUNKED_SCRIPT="${HBSR_ROOT}/hybrid_sdfgs/exp_scripts/06_generate_video_priors_uav_x8to2_chunked.sh"
if [[ ! -f "${CHUNKED_SCRIPT}" ]]; then
  echo "[uav-x8b4to1] chunked UAV script not found: ${CHUNKED_SCRIPT}" >&2
  exit 1
fi

QUAD_NAMES=(quad_tl quad_tr quad_bl quad_br)
for QUAD_NAME in "${QUAD_NAMES[@]}"; do
  echo "[uav-x8b4to1] running UAV on ${QUAD_NAME}"
  HBSR_ROOT="${HBSR_ROOT}" \
  UAV_REPO="${UAV_REPO}" \
  UAV_CKPT_SRC="${UAV_CKPT_SRC}" \
  INPUT_DIR="${QUAD_INPUT_ROOT}/${QUAD_NAME}" \
  OUTPUT_ROOT="${QUAD_OUTPUT_ROOT}/${QUAD_NAME}" \
  PYTHON_EXE="${PYTHON_EXE}" \
  CHUNK_SIZE="${CHUNK_SIZE}" \
  UAV_PERFORM_TILE="${UAV_PERFORM_TILE}" \
  UAV_TILE_SIZE="${UAV_TILE_SIZE}" \
  bash "${CHUNKED_SCRIPT}"
done

export GT_DIR QUAD_OUTPUT_ROOT FINAL_PRIOR_ROOT
"${PYTHON_EXE}" - <<'PY'
import os
from pathlib import Path
from PIL import Image

gt_root = Path(os.environ["GT_DIR"])
quad_output_root = Path(os.environ["QUAD_OUTPUT_ROOT"])
final_prior_root = Path(os.environ["FINAL_PRIOR_ROOT"])
exts = {".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"}
gt_frames = sorted([p for p in gt_root.iterdir() if p.suffix in exts], key=lambda p: p.name)
if not gt_frames:
    raise SystemExit(f"No GT frames found in {gt_root}")

quad_dirs = {
    "quad_tl": quad_output_root / "quad_tl" / "priors",
    "quad_tr": quad_output_root / "quad_tr" / "priors",
    "quad_bl": quad_output_root / "quad_bl" / "priors",
    "quad_br": quad_output_root / "quad_br" / "priors",
}
for name, path in quad_dirs.items():
    if not path.is_dir():
        raise SystemExit(f"Missing UAV prior dir for {name}: {path}")

for gt_path in gt_frames:
    stem = gt_path.name
    patches = {}
    for name, path in quad_dirs.items():
        patch_path = path / stem
        if not patch_path.exists():
            raise SystemExit(f"Missing patch {patch_path}")
        patches[name] = Image.open(patch_path).convert("RGB")
    with Image.open(gt_path) as gt_im:
        full_w, full_h = gt_im.size
    canvas = Image.new("RGB", (full_w, full_h))
    mid_x = patches["quad_tl"].size[0]
    mid_y = patches["quad_tl"].size[1]
    canvas.paste(patches["quad_tl"], (0, 0))
    canvas.paste(patches["quad_tr"], (mid_x, 0))
    canvas.paste(patches["quad_bl"], (0, mid_y))
    canvas.paste(patches["quad_br"], (mid_x, mid_y))
    canvas.save(final_prior_root / stem)
    for patch in patches.values():
        patch.close()

print(f"[uav-x8b4to1] stitched {len(gt_frames)} frames -> {final_prior_root}")
PY

echo "[uav-x8b4to1] done: ${OUTPUT_ROOT}"
echo "[uav-x8b4to1] refs:"
echo "  bicubic x4 input : ${OUTPUT_ROOT}/input_bicubic_images_4"
echo "  quadrant inputs   : ${OUTPUT_ROOT}/input_quadrants_images_4"
echo "  GT images        : ${OUTPUT_ROOT}/ref_images"
echo "  GT images_2      : ${OUTPUT_ROOT}/ref_images_2"
