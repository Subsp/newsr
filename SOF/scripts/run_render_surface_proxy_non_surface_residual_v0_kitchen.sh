#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="${SOF_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_ROOT="${MODEL_ROOT:-${SOF_ROOT}/output/proxy_ray_surface/push_full_softalpha_relaxed_nearmesh_continue3k_v0/export/augmented_model}"
RENDER_ITERATION="${RENDER_ITERATION:-33000}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-8}"

STATE_RUN_NAME="${STATE_RUN_NAME:-mip30k_rerun_gs2mesh_surface_state_v0_relaxed_carrier_v1}"
SURFACE_STATE_PAYLOAD="${SURFACE_STATE_PAYLOAD:-${SOF_ROOT}/output/gaussian_surface_sort/${SCENE_NAME}/${STATE_RUN_NAME}/gaussian_surface_state_v0.pt}"
SURFACE_MASK_KEY="${SURFACE_MASK_KEY:-surface_candidate}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/proxy_ray_surface/non_surface_residual_render_v0}"
MASK_PAYLOAD_PATH="${MASK_PAYLOAD_PATH:-${OUTPUT_ROOT}/non_surface_residual_mask_payload.pt}"
SAVE_ALPHA="${SAVE_ALPHA:-1}"
SAVE_DEPTH="${SAVE_DEPTH:-0}"
SAVE_PREMUL="${SAVE_PREMUL:-0}"

if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "[render-non-surface-residual] missing model root: ${MODEL_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SURFACE_STATE_PAYLOAD}" ]]; then
  echo "[render-non-surface-residual] missing surface-state payload: ${SURFACE_STATE_PAYLOAD}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[render-non-surface-residual] scene     : ${SCENE_ROOT}"
echo "[render-non-surface-residual] model     : ${MODEL_ROOT} iter=${RENDER_ITERATION}"
echo "[render-non-surface-residual] surface   : ${SURFACE_STATE_PAYLOAD} key=${SURFACE_MASK_KEY}"
echo "[render-non-surface-residual] output    : ${OUTPUT_ROOT}"

MODEL_ROOT="${MODEL_ROOT}" \
RENDER_ITERATION="${RENDER_ITERATION}" \
SURFACE_STATE_PAYLOAD="${SURFACE_STATE_PAYLOAD}" \
SURFACE_MASK_KEY="${SURFACE_MASK_KEY}" \
MASK_PAYLOAD_PATH="${MASK_PAYLOAD_PATH}" \
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import torch

model_root = Path(os.environ["MODEL_ROOT"]).expanduser().resolve()
iteration = int(os.environ["RENDER_ITERATION"])
surface_payload_path = Path(os.environ["SURFACE_STATE_PAYLOAD"]).expanduser().resolve()
surface_key = os.environ["SURFACE_MASK_KEY"]
out_path = Path(os.environ["MASK_PAYLOAD_PATH"]).expanduser().resolve()

surface_payload = torch.load(surface_payload_path, map_location="cpu")
if surface_key not in surface_payload:
    raise KeyError(f"surface key {surface_key!r} not found in {surface_payload_path}")
orig_surface = surface_payload[surface_key].reshape(-1).bool().cpu()
orig_total = int(orig_surface.numel())

tags_path = model_root / "point_cloud" / f"iteration_{iteration}" / "gaussian_tags.pt"
if not tags_path.is_file():
    fallback = model_root / "point_cloud" / "iteration_30000" / "gaussian_tags.pt"
    if fallback.is_file():
        tags_path = fallback
    else:
        raise FileNotFoundError(f"missing gaussian tags: {tags_path}")

tags = torch.load(tags_path, map_location="cpu")
source_tag = tags["source_tag"].reshape(-1).long().cpu()
seed_id = tags.get("seed_id")
if seed_id is None:
    seed_id = torch.full_like(source_tag, -1)
else:
    seed_id = seed_id.reshape(-1).long().cpu()

aug_total = int(source_tag.numel())
original_slots = source_tag == 0
proxy_slots = source_tag == 1

proxy_seed = seed_id[proxy_slots]
proxy_seed = proxy_seed[(proxy_seed >= 0) & (proxy_seed < orig_total)]
removed = torch.zeros((orig_total,), dtype=torch.bool)
if proxy_seed.numel() > 0:
    removed[proxy_seed] = True

migration_masks = model_root / "surface_proxy_migration_masks.pt"
if int(original_slots.sum()) != orig_total - int(removed.sum()) and migration_masks.is_file():
    migration = torch.load(migration_masks, map_location="cpu")
    selected_ids = migration.get("selected_donor_ids")
    if selected_ids is not None:
        selected_ids = selected_ids.reshape(-1).long()
        selected_ids = selected_ids[(selected_ids >= 0) & (selected_ids < orig_total)]
        selected_removed = torch.zeros((orig_total,), dtype=torch.bool)
        if selected_ids.numel() > 0:
            selected_removed[selected_ids] = True
        if int(original_slots.sum()) == orig_total - int(selected_removed.sum()):
            removed = selected_removed

kept_orig_ids = torch.arange(orig_total, dtype=torch.long)[~removed]
if int(original_slots.sum()) != int(kept_orig_ids.numel()):
    raise RuntimeError(
        "Cannot align augmented original slots to source ids: "
        f"original_slots={int(original_slots.sum())} kept_orig_ids={int(kept_orig_ids.numel())} "
        f"orig_total={orig_total} proxy={int(proxy_slots.sum())} tags={tags_path}"
    )

surface_carrier = torch.zeros((aug_total,), dtype=torch.bool)
original_positions = torch.nonzero(original_slots, as_tuple=False).reshape(-1)
surface_carrier[original_positions] = orig_surface[kept_orig_ids]
surface_carrier[proxy_slots] = True
non_surface_residual = ~surface_carrier

out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "surface_carrier": surface_carrier,
        "non_surface_residual": non_surface_residual,
        "original_surface_kept": surface_carrier & original_slots,
        "proxy_surface": proxy_slots,
        "original_non_surface_residual": non_surface_residual & original_slots,
        "source_tag": source_tag,
        "seed_id": seed_id,
        "metadata": {
            "model_root": str(model_root),
            "iteration": iteration,
            "tags_path": str(tags_path),
            "surface_payload_path": str(surface_payload_path),
            "surface_key": surface_key,
            "source_gaussians": orig_total,
            "augmented_gaussians": aug_total,
            "surface_carrier_count": int(surface_carrier.sum()),
            "non_surface_residual_count": int(non_surface_residual.sum()),
            "proxy_surface_count": int(proxy_slots.sum()),
            "original_surface_kept_count": int((surface_carrier & original_slots).sum()),
            "original_non_surface_residual_count": int((non_surface_residual & original_slots).sum()),
        },
    },
    out_path,
)
print("[render-non-surface-residual] saved mask:", out_path)
print("[render-non-surface-residual] surface_carrier:", int(surface_carrier.sum()), "/", aug_total)
print("[render-non-surface-residual] non_surface_residual:", int(non_surface_residual.sum()), "/", aug_total)
print("[render-non-surface-residual] proxy_surface:", int(proxy_slots.sum()))
print("[render-non-surface-residual] original_surface_kept:", int((surface_carrier & original_slots).sum()))
PY

ARGS=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_ROOT}"
  --iteration "${RENDER_ITERATION}"
  --output_root "${OUTPUT_ROOT}"
  --images_subdir "${IMAGES_SUBDIR}"
  --split "${SPLIT}"
  --max_views "${MAX_VIEWS}"
  --selection_source payload
  --selection_key non_surface_residual
  --mask_payload_path "${MASK_PAYLOAD_PATH}"
  --selection_mode selected_only
)
if [[ "${SAVE_ALPHA}" == "1" ]]; then
  ARGS+=(--save_alpha)
fi
if [[ "${SAVE_DEPTH}" == "1" ]]; then
  ARGS+=(--save_depth)
fi
if [[ "${SAVE_PREMUL}" == "1" ]]; then
  ARGS+=(--save_premul)
fi

"${ARGS[@]}"

echo
echo "[done] residual renders: ${OUTPUT_ROOT}/${SPLIT}/ours_${RENDER_ITERATION}/renders"
if [[ "${SAVE_ALPHA}" == "1" ]]; then
  echo "[done] residual alpha  : ${OUTPUT_ROOT}/${SPLIT}/ours_${RENDER_ITERATION}/alpha"
fi
echo "[done] mask payload    : ${MASK_PAYLOAD_PATH}"
