#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SOF_ROOT}/scripts/run_train_hrgsrefiner_kitchen.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SOF_ROOT}/scripts/run_eval_hrgsrefiner_kitchen.sh}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SCENE_NAME}_v0}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${SOF_ROOT}/output/hrgs_train_formal}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${TRAIN_OUTPUT_ROOT}/${EXPERIMENT_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${TRAIN_OUTPUT_DIR}/checkpoints}"

NUM_VIEWS="${NUM_VIEWS:-4}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-32}"

PHASE1_END="${PHASE1_END:-500}"
PHASE2_END="${PHASE2_END:-2000}"
PHASE3_END="${PHASE3_END:-5000}"

PHASE1_MV_W="${PHASE1_MV_W:-0.0}"
PHASE2_MV_W="${PHASE2_MV_W:-0.0}"
PHASE3_MV_W="${PHASE3_MV_W:-0.2}"

RUN_EVAL_STEP500="${RUN_EVAL_STEP500:-1}"
RUN_EVAL_STEP2000="${RUN_EVAL_STEP2000:-1}"
RUN_EVAL_STEP5000="${RUN_EVAL_STEP5000:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SOF_GEOMETRY_MODEL="${SOF_GEOMETRY_MODEL:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_sof_vanilla_images2_v1/sof30k}"
SOF_ORACLE_ROOT="${SOF_ORACLE_ROOT:-${SCENE_ASSET_ROOT}/oracle/formal_oracle_sof30k}"

CKPT_500="${CHECKPOINT_DIR}/hrgsrefiner_step_$(printf '%06d' "${PHASE1_END}").pt"
CKPT_2000="${CHECKPOINT_DIR}/hrgsrefiner_step_$(printf '%06d' "${PHASE2_END}").pt"
CKPT_5000="${CHECKPOINT_DIR}/hrgsrefiner_step_$(printf '%06d' "${PHASE3_END}").pt"

mkdir -p "${CHECKPOINT_DIR}"

for path in "${SCENE_ROOT}" "${SOF_GEOMETRY_MODEL}" "${SOF_ORACLE_ROOT}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[hrgs-formal-5k] required path not found: ${path}" >&2
    exit 1
  fi
done

echo "[hrgs-formal-5k] scene root         : ${SCENE_ROOT}"
echo "[hrgs-formal-5k] asset root         : ${SCENE_ASSET_ROOT}"
echo "[hrgs-formal-5k] train output dir   : ${TRAIN_OUTPUT_DIR}"
echo "[hrgs-formal-5k] num views          : ${NUM_VIEWS}"
echo "[hrgs-formal-5k] samples per scene  : ${SAMPLES_PER_SCENE}"
echo "[hrgs-formal-5k] phase targets      : ${PHASE1_END} -> ${PHASE2_END} -> ${PHASE3_END}"
echo "[hrgs-formal-5k] mv weights         : ${PHASE1_MV_W} / ${PHASE2_MV_W} / ${PHASE3_MV_W}"

run_phase() {
  local target_step="$1"
  local resume_ckpt="$2"
  local mv_w="$3"
  local phase_label="$4"

  echo
  echo "[hrgs-formal-5k] ${phase_label}: train to ${target_step}"
  (
    cd "${SOF_ROOT}"
    WORK_ROOT="${WORK_ROOT}" \
    SCENE_NAME="${SCENE_NAME}" \
    SCENE_ROOT="${SCENE_ROOT}" \
    SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
    GS_MODEL_PATH="${SOF_GEOMETRY_MODEL}" \
    ORACLE_ROOT="${SOF_ORACLE_ROOT}" \
    OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT}" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
    NUM_VIEWS="${NUM_VIEWS}" \
    SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE}" \
    MAX_STEPS="${target_step}" \
    LOSS_MV_W="${mv_w}" \
    REFINER_CHECKPOINT="${resume_ckpt}" \
    bash "${TRAIN_SCRIPT}"
  )
}

run_eval() {
  local ckpt_path="$1"
  local tag="$2"

  echo
  echo "[hrgs-formal-5k] eval ${tag}"
  (
    cd "${SOF_ROOT}"
    WORK_ROOT="${WORK_ROOT}" \
    SCENE_NAME="${SCENE_NAME}" \
    SCENE_ROOT="${SCENE_ROOT}" \
    TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
    REFINER_CHECKPOINT="${ckpt_path}" \
    CHECKPOINT_TAG="${tag}" \
    bash "${EVAL_SCRIPT}"
  )
}

if [[ "${SKIP_EXISTING}" != "1" || ! -f "${CKPT_500}" ]]; then
  run_phase "${PHASE1_END}" "" "${PHASE1_MV_W}" "phase1"
else
  echo
  echo "[hrgs-formal-5k] phase1 checkpoint exists, skipping: ${CKPT_500}"
fi

if [[ "${RUN_EVAL_STEP500}" == "1" ]]; then
  run_eval "${CKPT_500}" "step_$(printf '%06d' "${PHASE1_END}")"
fi

if [[ "${SKIP_EXISTING}" != "1" || ! -f "${CKPT_2000}" ]]; then
  run_phase "${PHASE2_END}" "${CKPT_500}" "${PHASE2_MV_W}" "phase2"
else
  echo
  echo "[hrgs-formal-5k] phase2 checkpoint exists, skipping: ${CKPT_2000}"
fi

if [[ "${RUN_EVAL_STEP2000}" == "1" ]]; then
  run_eval "${CKPT_2000}" "step_$(printf '%06d' "${PHASE2_END}")"
fi

if [[ "${SKIP_EXISTING}" != "1" || ! -f "${CKPT_5000}" ]]; then
  run_phase "${PHASE3_END}" "${CKPT_2000}" "${PHASE3_MV_W}" "phase3"
else
  echo
  echo "[hrgs-formal-5k] phase3 checkpoint exists, skipping: ${CKPT_5000}"
fi

if [[ "${RUN_EVAL_STEP5000}" == "1" ]]; then
  run_eval "${CKPT_5000}" "step_$(printf '%06d' "${PHASE3_END}")"
fi

echo
echo "[done] final checkpoint : ${CKPT_5000}"
echo "[done] train output     : ${TRAIN_OUTPUT_DIR}"
