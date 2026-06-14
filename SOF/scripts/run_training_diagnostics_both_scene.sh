#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec env DIAGNOSTIC_MODE=both "${SCRIPT_DIR}/run_training_diagnostics_scene.sh" "$@"
