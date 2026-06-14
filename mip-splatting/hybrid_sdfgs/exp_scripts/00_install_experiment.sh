#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HBSR_ROOT="${HBSR_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

resolve_executable() {
  local candidate="$1"
  if [[ "${candidate}" == */* ]]; then
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
    fi
    return 0
  fi

  command -v "${candidate}" 2>/dev/null || true
}

PYTHON_EXE="$(resolve_executable "${PYTHON_BIN}")"
if [[ -z "${PYTHON_EXE}" && "${PYTHON_BIN}" == "python" ]]; then
  PYTHON_EXE="$(resolve_executable python3)"
fi

if [[ -z "${PYTHON_EXE}" ]]; then
  echo "[install] python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${HBSR_ROOT}"

echo "[install] repo root : ${HBSR_ROOT}"
echo "[install] python    : ${PYTHON_EXE}"

"${PYTHON_EXE}" -m pip install -U pip "setuptools<81" wheel packaging cmake ninja

"${PYTHON_EXE}" -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

"${PYTHON_EXE}" -m pip install -r hybrid_sdfgs/requirements.unified.txt

# Defensively normalize OpenCV/NumPy ABI in reused environments.
"${PYTHON_EXE}" -m pip uninstall -y opencv-python opencv-python-headless || true
"${PYTHON_EXE}" -m pip install --no-cache-dir numpy==1.26.4 opencv-python-headless==4.10.0.84

"${PYTHON_EXE}" -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
"${PYTHON_EXE}" -m pip install --no-build-isolation ./submodules/simple-knn
"${PYTHON_EXE}" -m pip install --no-build-isolation \
  git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

"${PYTHON_EXE}" -m pip check

"${PYTHON_EXE}" - <<'PY'
import torch
import diff_gaussian_rasterization
import simple_knn
import tinycudann
import numpy
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_ok:", torch.cuda.is_available())
print("numpy:", numpy.__version__)
print("extensions: ok (diff-gaussian-rasterization, simple-knn, tinycudann)")
PY

echo "[install] done"
