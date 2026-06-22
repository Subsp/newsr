#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting}"

RUN_TAG="${RUN_TAG:-mip30k_rerun_check_directsrc_r1_v0_spray_2dgs_effective_hf_gaulayer_one_v0}"
MODEL_DIR="${MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_2dgs_hf_spray_v0/${SCENE_NAME}/${RUN_TAG}}"
ITERATION="${ITERATION:-30000}"
SPLIT="${SPLIT:-train}"
MAX_VIEWS="${MAX_VIEWS:-1}"
MODE="${MODE:-alpha}"
BOOST_TAU_SCALE="${BOOST_TAU_SCALE:-${ALPHA_TAU_SCALE:-10.0}}"
BOOST_SCALE_MULTIPLIER="${BOOST_SCALE_MULTIPLIER:-2.0}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
SEED="${SEED:-12345}"
VARIANT_NAME="${VARIANT_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/spray_one_render/${RUN_TAG}}"
OVERWRITE="${OVERWRITE:-1}"
KEEP_EXPORT_ROOT="${KEEP_EXPORT_ROOT:-0}"

if [[ -z "${VARIANT_NAME}" ]]; then
  if [[ "${MODE}" == "alpha" || "${MODE}" == "boost_alpha" ]]; then
    VARIANT_NAME="tau$(printf '%04d' "$(python - <<PY
print(round(float("${BOOST_TAU_SCALE}") * 100))
PY
)")_scale$(printf '%03d' "$(python - <<PY
print(round(float("${BOOST_SCALE_MULTIPLIER}") * 100))
PY
)")"
  elif [[ "${MODE}" == "random" || "${MODE}" == "boost_random" ]]; then
    VARIANT_NAME="keep$(printf '%03d' "$(python - <<PY
print(round(float("${KEEP_RATIO}") * 100))
PY
)")_tau$(printf '%04d' "$(python - <<PY
print(round(float("${BOOST_TAU_SCALE}") * 100))
PY
)")_scale$(printf '%03d' "$(python - <<PY
print(round(float("${BOOST_SCALE_MULTIPLIER}") * 100))
PY
)")"
  else
    echo "[spray-ablation-one-v0] invalid MODE=${MODE}; use alpha/boost_alpha or random/boost_random" >&2
    exit 1
  fi
fi

POINT_DIR="${MODEL_DIR}/point_cloud/iteration_${ITERATION}"
TAGS_PATH="${POINT_DIR}/gaussian_tags.pt"
if [[ ! -f "${POINT_DIR}/point_cloud.ply" ]]; then
  echo "[spray-ablation-one-v0] point cloud not found: ${POINT_DIR}/point_cloud.ply" >&2
  exit 1
fi
if [[ ( "${MODE}" == "random" || "${MODE}" == "boost_random" ) && ! -f "${TAGS_PATH}" ]]; then
  echo "[spray-ablation-one-v0] tags not found for random mode: ${TAGS_PATH}" >&2
  exit 1
fi

FINAL_DIR="${OUTPUT_ROOT}/${VARIANT_NAME}"
EXPORT_ROOT="${FINAL_DIR}/_export"
PAYLOAD_PATH="${FINAL_DIR}/random_drop_payload.pt"
if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${FINAL_DIR}"
fi
mkdir -p "${FINAL_DIR}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}:${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[spray-ablation-one-v0] model   : ${MODEL_DIR}"
echo "[spray-ablation-one-v0] mode    : ${MODE}"
echo "[spray-ablation-one-v0] variant : ${VARIANT_NAME}"
echo "[spray-ablation-one-v0] output  : ${FINAL_DIR}"
echo "[spray-ablation-one-v0] boost   : tau=${BOOST_TAU_SCALE} scale=${BOOST_SCALE_MULTIPLIER}"

EXPORT_ARGS=(
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_DIR}"
  --output_root "${EXPORT_ROOT}"
  --images_subdir images_2
  --iteration "${ITERATION}"
  --split "${SPLIT}"
  --max_views "${MAX_VIEWS}"
)

if [[ "${MODE}" == "alpha" || "${MODE}" == "boost_alpha" ]]; then
  EXPORT_ARGS+=(
    --selection_source tracking
    --selection_key prior_injected
    --selection_mode full
    --tau_scale "${BOOST_TAU_SCALE}"
    --scale_multiplier "${BOOST_SCALE_MULTIPLIER}"
  )
elif [[ "${MODE}" == "random" || "${MODE}" == "boost_random" ]]; then
  TAGS_PATH="${TAGS_PATH}" PAYLOAD_PATH="${PAYLOAD_PATH}" KEEP_RATIO="${KEEP_RATIO}" SEED="${SEED}" "${PYTHON_BIN}" - <<'PY'
import os
import torch

tags = torch.load(os.environ["TAGS_PATH"], map_location="cpu")
source = tags["source_tag"].reshape(-1)
prior = source == 1
keep_ratio = max(0.0, min(1.0, float(os.environ["KEEP_RATIO"])))
seed = int(os.environ["SEED"])
generator = torch.Generator(device="cpu")
generator.manual_seed(seed)
rand = torch.rand((int(prior.sum().item()),), generator=generator)
keep_prior = torch.zeros_like(prior)
keep_prior[prior] = rand < keep_ratio
drop_prior = prior & ~keep_prior
torch.save(
    {
        "drop_prior": drop_prior,
        "keep_prior": keep_prior,
        "prior": prior,
        "keep_ratio": torch.tensor(keep_ratio),
        "seed": torch.tensor(seed),
    },
    os.environ["PAYLOAD_PATH"],
)
print(
    f"[spray-ablation-one-v0] random prior keep={int(keep_prior.sum())}/"
    f"{int(prior.sum())} drop={int(drop_prior.sum())}"
)
PY
  EXPORT_ARGS+=(
    --selection_source payload
    --mask_payload_path "${PAYLOAD_PATH}"
    --selection_key keep_prior
    --selection_mode full
    --tau_scale "${BOOST_TAU_SCALE}"
    --scale_multiplier "${BOOST_SCALE_MULTIPLIER}"
    --post_mute_selection_source payload
    --post_mute_mask_payload_path "${PAYLOAD_PATH}"
    --post_mute_selection_key drop_prior
  )
else
  echo "[spray-ablation-one-v0] invalid MODE=${MODE}; use alpha/boost_alpha or random/boost_random" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py" "${EXPORT_ARGS[@]}"

RENDER_DIR="${EXPORT_ROOT}/${SPLIT}/ours_${ITERATION}/renders"
first_render="$(find "${RENDER_DIR}" -maxdepth 1 -type f -name '*.png' | sort | head -1)"
if [[ -z "${first_render}" ]]; then
  echo "[spray-ablation-one-v0] no rendered image found under ${RENDER_DIR}" >&2
  exit 1
fi
cp "${first_render}" "${FINAL_DIR}/$(basename "${first_render}")"

cat > "${FINAL_DIR}/README.txt" <<EOF
Single sprayed 2DGS merged-render ablation.

model: ${MODEL_DIR}
iteration: ${ITERATION}
split: ${SPLIT}
mode: ${MODE}
boost_tau_scale: ${BOOST_TAU_SCALE}
boost_scale_multiplier: ${BOOST_SCALE_MULTIPLIER}
keep_ratio: ${KEEP_RATIO}
seed: ${SEED}

Primary image:
  ${FINAL_DIR}/$(basename "${first_render}")

Nested temporary export:
  ${EXPORT_ROOT}
EOF

if [[ "${KEEP_EXPORT_ROOT}" != "1" ]]; then
  rm -rf "${EXPORT_ROOT}"
fi

echo "[spray-ablation-one-v0] image:"
echo "  ${FINAL_DIR}/$(basename "${first_render}")"
echo "[spray-ablation-one-v0] readme:"
echo "  ${FINAL_DIR}/README.txt"
