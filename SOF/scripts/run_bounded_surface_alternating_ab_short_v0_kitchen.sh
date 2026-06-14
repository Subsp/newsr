#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PHASE_MODE="${PHASE_MODE:-appearance_only}"
RUN_TAG="${RUN_TAG:-short_v0}"

case "${PHASE_MODE}" in
  appearance_only)
    APPEARANCE_STEPS="${APPEARANCE_STEPS:-300}"
    GEOMETRY_STEPS=0
    PRUNE_AFTER_CYCLE="${PRUNE_AFTER_CYCLE:-0}"
    ;;
  geometry_only)
    APPEARANCE_STEPS=0
    GEOMETRY_STEPS="${GEOMETRY_STEPS:-300}"
    PRUNE_AFTER_CYCLE="${PRUNE_AFTER_CYCLE:-0}"
    ;;
  alternating)
    APPEARANCE_STEPS="${APPEARANCE_STEPS:-150}"
    GEOMETRY_STEPS="${GEOMETRY_STEPS:-150}"
    PRUNE_AFTER_CYCLE="${PRUNE_AFTER_CYCLE:-0}"
    ;;
  *)
    echo "[bounded-surface-alternating-v0] unsupported PHASE_MODE=${PHASE_MODE}" >&2
    exit 1
    ;;
esac

CYCLES="${CYCLES:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-0}"
SAVE_EVERY_CYCLES="${SAVE_EVERY_CYCLES:-1}"
MAX_VIEWS="${MAX_VIEWS:-0}"
RUN_NAME="${RUN_NAME:-${START_RUN_NAME:-debug_stage_00b3_after_scale_canonicalize_geometry_only_v0}_bounded_${PHASE_MODE}_${RUN_TAG}}"

export PHASE_MODE
export APPEARANCE_STEPS
export GEOMETRY_STEPS
export PRUNE_AFTER_CYCLE
export CYCLES
export TOTAL_STEPS
export SAVE_EVERY_CYCLES
export MAX_VIEWS
export RUN_NAME

bash "${SOF_ROOT}/scripts/run_bounded_surface_alternating_v0_kitchen.sh"
