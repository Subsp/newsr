#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ARCHIVE_ROOT="${ARCHIVE_ROOT:-/root/autodl-tmp/archive}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${ARCHIVE_ROOT}/${SCENE_NAME}}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"

SOURCE_IMAGES_DIR="${SOURCE_IMAGES_DIR:-${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}}"
TARGET_IMAGES_DIR="${TARGET_IMAGES_DIR:-${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}}"
PRIOR_DIR="${PRIOR_DIR:-${SCENE_ROOT}/priors}"

ALIAS_ROOT="${ALIAS_ROOT:-${ARCHIVE_ROOT}/aliases}"
ALIAS_DIR="${ALIAS_DIR:-${ALIAS_ROOT}/${SCENE_NAME}_images8bicubic_to_images2}"

BASE_ITER="${BASE_ITER:-30000}"
FINAL_ITER="${FINAL_ITER:-32000}"
SPLATTING_CONFIG="${SPLATTING_CONFIG:-configs/hierarchical.json}"

BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-vanilla3dgs_no_sof_reg}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${SOF_ROOT}/output/${SCENE_NAME}_vanilla3dgs_lr_ablation_v1/${BASE_EXPERIMENT_NAME}}"
BASELINE_CKPT="${BASELINE_CKPT:-${BASE_MODEL_DIR}/chkpnt${BASE_ITER}.pth}"
BASELINE_RENDER_DIR="${BASE_MODEL_DIR}/test/ours_${BASE_ITER}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-stablesr_turbo_vanilla3dgs_no_sof_reg_prior_v1}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/${SCENE_NAME}_vanilla3dgs_prior_repro/${EXPERIMENT_NAME}}"
MASK_OUT_DIR="${RUN_ROOT}/direct_prior_masks_v0"
MASK_DIR="${MASK_OUT_DIR}/direct_prior_masks"
GS_OUT_DIR="${RUN_ROOT}/direct_prior_gs_v0"
GS_PAYLOAD="${GS_OUT_DIR}/edge_region_gaussians_v0.pt"
MODEL_DIR="${RUN_ROOT}/direct_prior_no_sof_reg_v1"
CURRENT_RENDER_DIR="${MODEL_DIR}/test/ours_${FINAL_ITER}"
COMPARE_OUT="${RUN_ROOT}/baseline_compare.json"

PYTHON_BIN="${PYTHON_BIN:-python}"

EDGE_MIN_TOUCH_VIEWS="${EDGE_MIN_TOUCH_VIEWS:-2}"
EDGE_MIN_VISIBLE_VIEWS="${EDGE_MIN_VISIBLE_VIEWS:-2}"
EDGE_MIN_TOUCH_RATIO="${EDGE_MIN_TOUCH_RATIO:-0.0}"
EDGE_RADIUS_SCALE="${EDGE_RADIUS_SCALE:-1.0}"
EDGE_MIN_TOUCH_RADIUS_PX="${EDGE_MIN_TOUCH_RADIUS_PX:-1}"
EDGE_MAX_TOUCH_RADIUS_PX="${EDGE_MAX_TOUCH_RADIUS_PX:-16}"

LAMBDA_PRIOR_EDGE="${LAMBDA_PRIOR_EDGE:-0.3}"
PRIOR_EDGE_UPDATE_SCALE="${PRIOR_EDGE_UPDATE_SCALE:-0.5}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sof

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${ALIAS_ROOT}" "${RUN_ROOT}" "${MODEL_DIR}" "${BASE_MODEL_DIR}"

for path in "${SCENE_ROOT}" "${TARGET_IMAGES_DIR}" "${PRIOR_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[vanilla3dgs-no-sof-reg-prior] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${SCENE_ROOT}/sparse/0" ]]; then
  echo "[vanilla3dgs-no-sof-reg-prior] missing sparse/0: ${SCENE_ROOT}/sparse/0" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_IMAGES_DIR}" ]]; then
  echo "[vanilla3dgs-no-sof-reg-prior] ${SOURCE_IMAGES_SUBDIR} missing, generating from ${TARGET_IMAGES_SUBDIR}"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" scripts/generate_downsampled_images.py \
      --source_dir "${TARGET_IMAGES_DIR}" \
      --output_dir "${SOURCE_IMAGES_DIR}" \
      --scale 4 \
      --resize_filter bicubic
  )
fi

echo "[vanilla3dgs-no-sof-reg-prior] scene              : ${SCENE_NAME}"
echo "[vanilla3dgs-no-sof-reg-prior] scene root         : ${SCENE_ROOT}"
echo "[vanilla3dgs-no-sof-reg-prior] source images      : ${SOURCE_IMAGES_DIR}"
echo "[vanilla3dgs-no-sof-reg-prior] target images      : ${TARGET_IMAGES_DIR}"
echo "[vanilla3dgs-no-sof-reg-prior] prior dir          : ${PRIOR_DIR}"
echo "[vanilla3dgs-no-sof-reg-prior] alias dir          : ${ALIAS_DIR}"
echo "[vanilla3dgs-no-sof-reg-prior] baseline model     : ${BASE_MODEL_DIR}"
echo "[vanilla3dgs-no-sof-reg-prior] run root           : ${RUN_ROOT}"
echo "[vanilla3dgs-no-sof-reg-prior] experiment         : ${EXPERIMENT_NAME}"
echo "[vanilla3dgs-no-sof-reg-prior] edge min touch     : ${EDGE_MIN_TOUCH_VIEWS}"
echo "[vanilla3dgs-no-sof-reg-prior] edge min visible   : ${EDGE_MIN_VISIBLE_VIEWS}"
echo "[vanilla3dgs-no-sof-reg-prior] edge min ratio     : ${EDGE_MIN_TOUCH_RATIO}"
echo "[vanilla3dgs-no-sof-reg-prior] lambda prior edge  : ${LAMBDA_PRIOR_EDGE}"
echo "[vanilla3dgs-no-sof-reg-prior] prior update scale : ${PRIOR_EDGE_UPDATE_SCALE}"

echo
echo "[1/8] prepare pseudo-scene alias"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" scripts/prepare_colmap_pseudo_sr_scene.py \
    --scene_root "${SCENE_ROOT}" \
    --scene_alias_dir "${ALIAS_DIR}" \
    --source_images_subdir "${SOURCE_IMAGES_SUBDIR}" \
    --target_images_subdir "${TARGET_IMAGES_SUBDIR}" \
    --resize_filter bicubic
)

echo
echo "[2/8] train 30000-step no-SOF-reg baseline"
if [[ ! -f "${BASELINE_CKPT}" ]]; then
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" train.py \
      --splatting_config "${SPLATTING_CONFIG}" \
      -s "${ALIAS_DIR}" \
      -m "${BASE_MODEL_DIR}" \
      --eval \
      --iterations "${BASE_ITER}" \
      --test_iterations "${BASE_ITER}" \
      --save_iterations "${BASE_ITER}" \
      --checkpoint_iterations "${BASE_ITER}" \
      --lambda_distortion 0.0 \
      --lambda_depth_normal 0.0 \
      --lambda_smoothness 0.0 \
      --lambda_opacity_field 0.0 \
      --lambda_extent 0.0 \
      --distortion_from_iter 99999999 \
      --depth_normal_from_iter 99999999 \
      --scale_reg 0.0 \
      --opacity_reg 0.0 \
      --min_scale_reg 0.0
  )
fi

if [[ ! -d "${BASELINE_RENDER_DIR}" ]]; then
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" render.py \
      -m "${BASE_MODEL_DIR}" \
      -s "${SCENE_ROOT}" \
      -i "${TARGET_IMAGES_SUBDIR}" \
      --iteration "${BASE_ITER}" \
      --eval \
      --skip_train \
      --data_device cpu
  )
fi

echo
echo "[3/8] metrics for 30000-step baseline"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" metrics.py -m "${BASE_MODEL_DIR}"
)

echo
echo "[4/8] prepare direct prior masks"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" prepare_direct_prior_masks_v0.py \
    --prior_dir "${PRIOR_DIR}" \
    --anchor_dir "${TARGET_IMAGES_DIR}" \
    --output_dir "${MASK_OUT_DIR}" \
    --blur_kernel 9 \
    --lowfreq_threshold 0.08 \
    --highfreq_gain_threshold 0.015 \
    --prior_highfreq_threshold 0.02 \
    --confidence_threshold 0.15 \
    --dilate_kernel 3
)

echo
echo "[5/8] select GS touched by direct prior masks"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" select_edge_region_gaussians_v0.py \
    -s "${ALIAS_DIR}" \
    -m "${BASE_MODEL_DIR}" \
    --eval \
    --data_device cpu \
    --iteration "${BASE_ITER}" \
    --start_checkpoint "${BASELINE_CKPT}" \
    --edge_mask_dir "${MASK_DIR}" \
    --output_dir "${GS_OUT_DIR}" \
    --min_touch_views "${EDGE_MIN_TOUCH_VIEWS}" \
    --min_visible_views "${EDGE_MIN_VISIBLE_VIEWS}" \
    --min_touch_ratio "${EDGE_MIN_TOUCH_RATIO}" \
    --radius_scale "${EDGE_RADIUS_SCALE}" \
    --min_touch_radius_px "${EDGE_MIN_TOUCH_RADIUS_PX}" \
    --max_touch_radius_px "${EDGE_MAX_TOUCH_RADIUS_PX}"
)

echo
echo "[6/8] 2000-step no-SOF-reg prior finetune"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" train.py \
    -s "${ALIAS_DIR}" \
    -m "${MODEL_DIR}" \
    --eval \
    --data_device cpu \
    --splatting_config "${BASE_MODEL_DIR}/config.json" \
    --start_checkpoint "${BASELINE_CKPT}" \
    --iterations "${FINAL_ITER}" \
    --test_iterations "${FINAL_ITER}" \
    --save_iterations "${FINAL_ITER}" \
    --checkpoint_iterations "${FINAL_ITER}" \
    --lambda_distortion 0.0 \
    --lambda_depth_normal 0.0 \
    --lambda_smoothness 0.0 \
    --lambda_opacity_field 0.0 \
    --lambda_extent 0.0 \
    --distortion_from_iter 99999999 \
    --depth_normal_from_iter 99999999 \
    --scale_reg 0.0 \
    --opacity_reg 0.0 \
    --min_scale_reg 0.0 \
    --prior_edge_dir "${PRIOR_DIR}" \
    --prior_edge_mask_dir "${MASK_DIR}" \
    --lambda_prior_edge "${LAMBDA_PRIOR_EDGE}" \
    --prior_edge_loss_mode detail_v1 \
    --prior_edge_detail_alpha 0.4 \
    --prior_edge_detail_alpha_final 0.7 \
    --prior_edge_detail_warmup_iters 2000 \
    --prior_edge_detail_weight 1.0 \
    --prior_edge_lowfreq_weight 0.05 \
    --prior_edge_grad_weight 0.05 \
    --prior_edge_lowfreq_threshold 0.08 \
    --prior_edge_lowfreq_anchor gt \
    --prior_edge_detail_min_gain 0.005 \
    --prior_edge_confidence_power 1.5 \
    --prior_edge_update_scale "${PRIOR_EDGE_UPDATE_SCALE}" \
    --optimize_gaussian_mask_payload "${GS_PAYLOAD}" \
    --optimize_gaussian_mask_key selected_mask \
    --prior_edge_min_pixels 64 \
    --prior_edge_touch_min_radius_px "${EDGE_MIN_TOUCH_RADIUS_PX}" \
    --prior_edge_touch_radius_scale "${EDGE_RADIUS_SCALE}" \
    --prior_edge_touch_max_radius_px "${EDGE_MAX_TOUCH_RADIUS_PX}" \
    --densify_until_iter 0
)

echo
echo "[7/8] render 32000-step prior finetune"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" render.py \
    -m "${MODEL_DIR}" \
    -s "${SCENE_ROOT}" \
    -i "${TARGET_IMAGES_SUBDIR}" \
    --iteration "${FINAL_ITER}" \
    --eval \
    --skip_train \
    --data_device cpu
)

echo
echo "[8/8] metrics and baseline delta"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" metrics.py -m "${MODEL_DIR}"
)

BASELINE_RENDER_DIR="${BASELINE_RENDER_DIR}" \
CURRENT_RENDER_DIR="${CURRENT_RENDER_DIR}" \
COMPARE_OUT="${COMPARE_OUT}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def scores(root_str):
    root = Path(root_str)
    renders_dir = root / "renders"
    gt_dir = root / "gt"
    render_files = sorted([p for p in renders_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    gt_files = sorted([p for p in gt_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if not render_files or len(render_files) != len(gt_files):
        raise RuntimeError(f"render/gt mismatch under {root}")

    psnrs, ssims = [], []
    for rp, gp in zip(render_files, gt_files):
        r = np.array(Image.open(rp).convert("RGB"))
        g = np.array(Image.open(gp).convert("RGB"))
        psnrs.append(peak_signal_noise_ratio(g, r, data_range=255))
        ssims.append(structural_similarity(g, r, channel_axis=2, data_range=255))

    return {
        "mean_psnr": float(np.mean(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
        "n_views": len(psnrs),
    }


baseline = scores(os.environ["BASELINE_RENDER_DIR"])
current = scores(os.environ["CURRENT_RENDER_DIR"])
summary = {
    "baseline": baseline,
    "current": current,
    "delta": {
        "psnr": current["mean_psnr"] - baseline["mean_psnr"],
        "ssim": current["mean_ssim"] - baseline["mean_ssim"],
    },
}

out = Path(os.environ["COMPARE_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"saved to: {out}")
PY

echo
echo "[done] baseline model : ${BASE_MODEL_DIR}"
echo "[done] prior model    : ${MODEL_DIR}"
echo "[done] compare json   : ${COMPARE_OUT}"
