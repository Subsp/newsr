#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENABLE_GITHUB_SSH_REWRITE="${ENABLE_GITHUB_SSH_REWRITE:-1}"
INSTALL_SIMPLE_KNN="${INSTALL_SIMPLE_KNN:-1}"
INSTALL_FUSED_SSIM="${INSTALL_FUSED_SSIM:-1}"
TETRA_CUDA_ARCHITECTURES="${TETRA_CUDA_ARCHITECTURES:-native}"

if [[ "${ENABLE_GITHUB_SSH_REWRITE}" == "1" ]]; then
  git config --global url."git@github.com:".insteadOf https://github.com/
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  elif [[ -d "/usr/local/cuda" ]]; then
    CUDA_HOME="/usr/local/cuda"
  else
    for candidate in /usr/local/cuda-*; do
      if [[ -x "${candidate}/bin/nvcc" ]]; then
        CUDA_HOME="${candidate}"
        break
      fi
    done
  fi
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  else
    echo "[install-server-extensions] Could not infer CUDA_HOME. Set it manually first." >&2
    exit 1
  fi
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Some environments export this as an empty string, which breaks CMake CUDA detection.
unset CMAKE_CUDA_ARCHITECTURES || true

TORCH_CMAKE_PREFIX="$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"

echo "[install-server-extensions] CUDA_HOME=${CUDA_HOME}"
echo "[install-server-extensions] TORCH_CMAKE_PREFIX=${TORCH_CMAKE_PREFIX}"
echo "[install-server-extensions] INSTALL_SIMPLE_KNN=${INSTALL_SIMPLE_KNN}"
echo "[install-server-extensions] TETRA_CUDA_ARCHITECTURES=${TETRA_CUDA_ARCHITECTURES}"

cd "${REPO_ROOT}"

if [[ "${INSTALL_SIMPLE_KNN}" == "1" ]]; then
  if ! "${PYTHON_BIN}" -m pip install --no-build-isolation submodules/simple-knn/; then
    echo "[install-server-extensions] simple-knn install failed; Gaussian init will fall back to scipy cKDTree."
  fi
fi
"${PYTHON_BIN}" -m pip install --no-build-isolation submodules/diff-gaussian-rasterization/

if [[ "${INSTALL_FUSED_SSIM}" == "1" ]]; then
  if ! "${PYTHON_BIN}" -c "import fused_ssim" >/dev/null 2>&1; then
    if ! "${PYTHON_BIN}" -m pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim/; then
      echo "[install-server-extensions] fused-ssim install failed; SOF training will fall back to utils.loss_utils.ssim."
    fi
  fi
fi

cd "${REPO_ROOT}/submodules/tetra-triangulation"
cmake . \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX}" \
  -DCMAKE_CUDA_ARCHITECTURES="${TETRA_CUDA_ARCHITECTURES}"
make -j"$(nproc)"
"${PYTHON_BIN}" -m pip install -e . --no-build-isolation

echo "[install-server-extensions] done"
