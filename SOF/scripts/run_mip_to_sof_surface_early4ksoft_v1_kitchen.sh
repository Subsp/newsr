#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}" \
MIP_TO_SOF_PROFILE="${MIP_TO_SOF_PROFILE:-early4ksoft_v1}" \
PREPARE_INPUT_PROFILE="${PREPARE_INPUT_PROFILE:-early4ksoft_v1}" \
bash "${SCRIPT_DIR}/run_mip_to_sof_surface_v0_kitchen.sh"
