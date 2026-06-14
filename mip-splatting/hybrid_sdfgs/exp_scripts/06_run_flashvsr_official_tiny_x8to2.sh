#!/usr/bin/env bash
set -euo pipefail

FLASHVSR_WAN_DIR="${FLASHVSR_WAN_DIR:-/root/autodl-tmp/FlashVSR/examples/WanVSR}"
PYTHON_EXE="${PYTHON_EXE:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-/root/autodl-tmp/HBSR/hybrid_sdfgs/tools/infer_flashvsr_v1.1_tiny_x8to2_official.py}"

cd "${FLASHVSR_WAN_DIR}"
"${PYTHON_EXE}" "${SCRIPT_PATH}"
