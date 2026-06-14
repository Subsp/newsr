#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="${HBSR_ROOT:-/root/autodl-tmp/HBSR}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/kitchen}"
TRAIN_IMAGES="${TRAIN_IMAGES:-images_8}"
RENDER_IMAGES="${RENDER_IMAGES:-images_4}"
EVAL_IMAGES="${EVAL_IMAGES:-images_2}"
OUTPUT_PATH="${OUTPUT_PATH:-/root/autodl-tmp/HBSR/outputs/mipsplatting_baseline_${TRAIN_IMAGES}_kitchen}"
EXPORT_ROOT="${EXPORT_ROOT:-/root/autodl-tmp/priors/kitchen_mipsplatting_train_render_${RENDER_IMAGES}_from_${TRAIN_IMAGES}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ITERATIONS="${ITERATIONS:-30000}"
TEST_ITERATIONS="${TEST_ITERATIONS:-7000 30000}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-7000 30000}"

if [[ ! -d "${DATASET_ROOT}/${TRAIN_IMAGES}" ]]; then
  echo "[mips-render-export] train image dir not found: ${DATASET_ROOT}/${TRAIN_IMAGES}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}/${RENDER_IMAGES}" ]]; then
  echo "[mips-render-export] render image dir not found: ${DATASET_ROOT}/${RENDER_IMAGES}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}/${EVAL_IMAGES}" ]]; then
  echo "[mips-render-export] eval image dir not found: ${DATASET_ROOT}/${EVAL_IMAGES}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_PATH}" "${EXPORT_ROOT}"

echo "[mips-render-export] HBSR_ROOT=${HBSR_ROOT}"
echo "[mips-render-export] DATASET_ROOT=${DATASET_ROOT}"
echo "[mips-render-export] TRAIN_IMAGES=${TRAIN_IMAGES}"
echo "[mips-render-export] RENDER_IMAGES=${RENDER_IMAGES}"
echo "[mips-render-export] EVAL_IMAGES=${EVAL_IMAGES}"
echo "[mips-render-export] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[mips-render-export] EXPORT_ROOT=${EXPORT_ROOT}"

cd "${HBSR_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py \
  -s "${DATASET_ROOT}" \
  -i "${TRAIN_IMAGES}" \
  -m "${OUTPUT_PATH}" \
  --eval \
  --white_background \
  --disable_gui \
  --iterations "${ITERATIONS}" \
  --test_iterations ${TEST_ITERATIONS} \
  --save_iterations ${SAVE_ITERATIONS}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python render.py \
  -s "${DATASET_ROOT}" \
  -i "${RENDER_IMAGES}" \
  -m "${OUTPUT_PATH}" \
  --iteration -1 \
  --resolution -1 \
  --skip_test \
  --white_background

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python render.py \
  -s "${DATASET_ROOT}" \
  -i "${EVAL_IMAGES}" \
  -m "${OUTPUT_PATH}" \
  --iteration -1 \
  --resolution -1 \
  --skip_train \
  --white_background

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python metrics.py \
  -m "${OUTPUT_PATH}" \
  -r -1

export OUTPUT_PATH EXPORT_ROOT DATASET_ROOT TRAIN_IMAGES RENDER_IMAGES EVAL_IMAGES
python - <<'PY'
import os
import shutil
from pathlib import Path

output_path = Path(os.environ["OUTPUT_PATH"])
export_root = Path(os.environ["EXPORT_ROOT"])
dataset_root = Path(os.environ["DATASET_ROOT"])
train_images = os.environ["TRAIN_IMAGES"]
render_images = os.environ["RENDER_IMAGES"]
eval_images = os.environ["EVAL_IMAGES"]

train_root = output_path / "train"
ours_dirs = sorted(
    [p for p in train_root.iterdir() if p.is_dir() and p.name.startswith("ours_")],
    key=lambda p: int(p.name.split("_", 1)[1]),
)
if not ours_dirs:
    raise SystemExit(f"No rendered train directory found in {train_root}")
render_dir = ours_dirs[-1] / "test_preds_-1"
gt_dir = ours_dirs[-1] / "gt_-1"
if not render_dir.is_dir():
    raise SystemExit(f"Rendered train preds not found: {render_dir}")
if not gt_dir.is_dir():
    raise SystemExit(f"Rendered train gt not found: {gt_dir}")

priors_dir = export_root / "priors"
render_gt_dir = export_root / f"gt_{render_images}"
if priors_dir.exists():
    shutil.rmtree(priors_dir)
if render_gt_dir.exists():
    shutil.rmtree(render_gt_dir)
shutil.copytree(render_dir, priors_dir)
shutil.copytree(gt_dir, render_gt_dir)

for link_name, target in {
    f"input_{train_images}": dataset_root / train_images,
    f"render_ref_{render_images}": dataset_root / render_images,
    f"eval_ref_{eval_images}": dataset_root / eval_images,
    "model_output": output_path,
}.items():
    link_path = export_root / link_name
    if link_path.is_symlink() or link_path.is_file():
        link_path.unlink()
    elif link_path.is_dir():
        shutil.rmtree(link_path)
    link_path.symlink_to(target)

print(f"[mips-render-export] exported priors -> {priors_dir}")
print(f"[mips-render-export] exported train GT -> {render_gt_dir}")
print(f"[mips-render-export] latest render dir -> {ours_dirs[-1]}")
PY

echo "[mips-render-export] done"
