#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-${SOF_ROOT}/output/kitchen_pseudogt_images8bicubic_to_images2}"
SCENE_DIR="${SCENE_DIR:-${SOF_ROOT}/data_aliases/kitchen_images8bicubic_to_images2}"
PRIOR_DIR="${PRIOR_DIR:-${WORKSPACE_ROOT}/priors/StableSRpriors}"
LR_IMAGE_DIR="${LR_IMAGE_DIR:-${WORKSPACE_ROOT}/kitchen/images_8}"
EVAL_SCENE_DIR="${EVAL_SCENE_DIR:-${WORKSPACE_ROOT}/kitchen}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SOF_ROOT}/output/kitchen_pseudogt_sticky_prior}"
BASE_ITER="${BASE_ITER:-30000}"
VIEW_LIMIT="${VIEW_LIMIT:-8}"

WARMUP_STEPS="${WARMUP_STEPS:-300}"
COMPETITION_STEPS="${COMPETITION_STEPS:-500}"
POSITION_LR_MAX_STEPS="${POSITION_LR_MAX_STEPS:-50000}"
DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-100}"

GRID_STRIDE="${GRID_STRIDE:-96}"
GRID_BORDER="${GRID_BORDER:-32}"
PATCH_RADIUS="${PATCH_RADIUS:-1}"
CANDIDATE_MODE="${CANDIDATE_MODE:-auto_mask}"
INJECTION_PRESET="${INJECTION_PRESET:-sticky_single_anchor}"

RUN_INJECT="${RUN_INJECT:-0}"
RUN_WARMUP="${RUN_WARMUP:-0}"
RUN_REOPEN="${RUN_REOPEN:-0}"
RUN_RENDER="${RUN_RENDER:-0}"

WARMUP_END=$((BASE_ITER + WARMUP_STEPS))
FINAL_ITER=$((WARMUP_END + COMPETITION_STEPS))

INJECT_DIR="${EXPERIMENT_ROOT}/inject"
WARMUP_DIR="${EXPERIMENT_ROOT}/warmup_model"
REOPEN_DIR="${EXPERIMENT_ROOT}/competition_model"
INJECT_CKPT="${INJECT_DIR}/chkpnt${BASE_ITER}_priorinject.pth"
WARMUP_CKPT="${WARMUP_DIR}/chkpnt${WARMUP_END}.pth"

for path in "${BASE_MODEL_DIR}" "${SCENE_DIR}" "${PRIOR_DIR}" "${LR_IMAGE_DIR}" "${EVAL_SCENE_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[sticky-prior] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ ! -f "${BASE_MODEL_DIR}/config.json" ]]; then
  echo "[sticky-prior] missing base config: ${BASE_MODEL_DIR}/config.json" >&2
  exit 1
fi

mkdir -p "${INJECT_DIR}" "${WARMUP_DIR}" "${REOPEN_DIR}"

INJECT_CMD=(
  "${PYTHON_BIN}" inject_prior_gaussians.py
  -s "${SCENE_DIR}"
  -m "${BASE_MODEL_DIR}"
  --iteration "${BASE_ITER}"
  --prior_dir "${PRIOR_DIR}"
  --lr_image_dir "${LR_IMAGE_DIR}"
  --candidate_mode "${CANDIDATE_MODE}"
  --grid_stride "${GRID_STRIDE}"
  --grid_border "${GRID_BORDER}"
  --patch_radius "${PATCH_RADIUS}"
  --view_limit "${VIEW_LIMIT}"
  --injection_preset "${INJECTION_PRESET}"
  --output_checkpoint "${INJECT_CKPT}"
  --output_summary "${INJECT_DIR}/prior_injection_summary_iter${BASE_ITER}.json"
  --output_preview_dir "${INJECT_DIR}/preview"
)

WARMUP_CMD=(
  "${PYTHON_BIN}" train.py
  -s "${SCENE_DIR}"
  -m "${WARMUP_DIR}"
  --eval
  --data_device cpu
  --splatting_config "${BASE_MODEL_DIR}/config.json"
  --start_checkpoint "${INJECT_CKPT}"
  --iterations "${WARMUP_END}"
  --test_iterations "${WARMUP_END}"
  --save_iterations "${WARMUP_END}"
  --checkpoint_iterations "${WARMUP_END}"
  --densify_from_iter 999999
  --densify_until_iter 999999
  --position_lr_max_steps "${POSITION_LR_MAX_STEPS}"
)

REOPEN_CMD=(
  "${PYTHON_BIN}" train.py
  -s "${SCENE_DIR}"
  -m "${REOPEN_DIR}"
  --eval
  --data_device cpu
  --splatting_config "${BASE_MODEL_DIR}/config.json"
  --start_checkpoint "${WARMUP_CKPT}"
  --iterations "${FINAL_ITER}"
  --test_iterations "${FINAL_ITER}"
  --save_iterations "${FINAL_ITER}"
  --checkpoint_iterations "${FINAL_ITER}"
  --densify_from_iter "${WARMUP_END}"
  --densify_until_iter "${FINAL_ITER}"
  --densification_interval "${DENSIFICATION_INTERVAL}"
  --position_lr_max_steps "${POSITION_LR_MAX_STEPS}"
)

RENDER_CMD=(
  "${PYTHON_BIN}" render.py
  -m "${REOPEN_DIR}"
  -s "${EVAL_SCENE_DIR}"
  -i images_2
  --eval
  --skip_train
  --data_device cpu
)

echo "[sticky-prior] experiment root : ${EXPERIMENT_ROOT}"
echo "[sticky-prior] base model      : ${BASE_MODEL_DIR}"
echo "[sticky-prior] pseudo scene    : ${SCENE_DIR}"
echo "[sticky-prior] prior dir       : ${PRIOR_DIR}"
echo "[sticky-prior] lr image dir    : ${LR_IMAGE_DIR}"
echo "[sticky-prior] eval scene      : ${EVAL_SCENE_DIR}"
echo "[sticky-prior] base iter       : ${BASE_ITER}"
echo "[sticky-prior] warmup end      : ${WARMUP_END}"
echo "[sticky-prior] final iter      : ${FINAL_ITER}"
echo "[sticky-prior] injection preset: ${INJECTION_PRESET}"
echo
echo "[sticky-prior] next commands:"
printf '  (cd %q &&' "${SOF_ROOT}"
printf ' %q' "${INJECT_CMD[@]}"
printf ')\n'
printf '  (cd %q &&' "${SOF_ROOT}"
printf ' %q' "${WARMUP_CMD[@]}"
printf ')\n'
printf '  (cd %q &&' "${SOF_ROOT}"
printf ' %q' "${REOPEN_CMD[@]}"
printf ')\n'
printf '  (cd %q &&' "${SOF_ROOT}"
printf ' %q' "${RENDER_CMD[@]}"
printf ')\n'

if [[ "${RUN_INJECT}" == "1" ]]; then
  echo "[sticky-prior] step 1/4: inject sticky prior anchors"
  (
    cd "${SOF_ROOT}"
    "${INJECT_CMD[@]}"
  )
fi

if [[ "${RUN_WARMUP}" == "1" ]]; then
  echo "[sticky-prior] step 2/4: warmup without prune/densify"
  (
    cd "${SOF_ROOT}"
    "${WARMUP_CMD[@]}"
  )
fi

if [[ "${RUN_REOPEN}" == "1" ]]; then
  echo "[sticky-prior] step 3/4: reopen densify/prune competition"
  (
    cd "${SOF_ROOT}"
    "${REOPEN_CMD[@]}"
  )
fi

if [[ "${RUN_RENDER}" == "1" ]]; then
  echo "[sticky-prior] step 4/4: render final checkpoint"
  (
    cd "${SOF_ROOT}"
    "${RENDER_CMD[@]}"
  )
fi
