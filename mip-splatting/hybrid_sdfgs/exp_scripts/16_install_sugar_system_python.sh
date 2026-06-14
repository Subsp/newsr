#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HBSR_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SUGAR_ROOT="${HBSR_ROOT}/../SuGaR"
if [[ ! -d "${DEFAULT_SUGAR_ROOT}" ]]; then
  DEFAULT_SUGAR_ROOT="/root/autodl-tmp/SuGaR"
fi

PYTHON_BIN="${PYTHON_BIN:-}"
SUGAR_ROOT="${SUGAR_ROOT:-${DEFAULT_SUGAR_ROOT}}"
SUGAR_ENV_DIR="${SUGAR_ENV_DIR:-${HBSR_ROOT}/.venvs/sugar-system-py}"
NVDIFFRAST_ROOT="${NVDIFFRAST_ROOT:-${SUGAR_ROOT}/nvdiffrast}"

TORCH_VERSION="${TORCH_VERSION:-2.0.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.15.2}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.0.2}"
PYTORCH3D_VERSION="${PYTORCH3D_VERSION:-0.7.4}"
OPEN3D_VERSION="${OPEN3D_VERSION:-0.17.0}"
TIMM_VERSION="${TIMM_VERSION:-0.4.12}"
OPENCV_PYTHON_HEADLESS_VERSION="${OPENCV_PYTHON_HEADLESS_VERSION:-4.10.0.84}"

INSTALL_TORCH="${INSTALL_TORCH:-1}"
INSTALL_PYTORCH3D="${INSTALL_PYTORCH3D:-1}"
INSTALL_NVDIFFRAST="${INSTALL_NVDIFFRAST:-1}"
INSTALL_SWINIR_RUNTIME="${INSTALL_SWINIR_RUNTIME:-1}"
BOOTSTRAP_SYSTEM_PYTHON="${BOOTSTRAP_SYSTEM_PYTHON:-0}"

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

is_conda_python() {
  local candidate="$1"
  case "${candidate}" in
    *"/conda/"*|*"/miniconda"*|*"/anaconda"*|*"/micromamba"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

discover_system_python() {
  local candidates=()
  local candidate
  local resolved

  if [[ -n "${PYTHON_BIN}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi

  candidates+=(
    "/usr/bin/python3.10"
    "/usr/local/bin/python3.10"
    "/usr/bin/python3.9"
    "/usr/local/bin/python3.9"
    "/usr/bin/python3"
    "/usr/local/bin/python3"
    "python3"
    "python"
  )

  for candidate in "${candidates[@]}"; do
    resolved="$(resolve_executable "${candidate}")"
    if [[ -z "${resolved}" ]]; then
      continue
    fi
    if is_conda_python "${resolved}"; then
      continue
    fi
    printf '%s\n' "${resolved}"
    return 0
  done

  return 1
}

bootstrap_system_python_with_apt() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[sugar-install] BOOTSTRAP_SYSTEM_PYTHON=1 was set, but apt-get is unavailable." >&2
    return 1
  fi

  if [[ "$(id -u)" != "0" ]]; then
    echo "[sugar-install] BOOTSTRAP_SYSTEM_PYTHON=1 requires root privileges." >&2
    return 1
  fi

  echo "[sugar-install] bootstrapping system python via apt-get"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    build-essential
}

PYTHON_EXE="$(discover_system_python || true)"
if [[ -z "${PYTHON_EXE}" && "${BOOTSTRAP_SYSTEM_PYTHON}" == "1" ]]; then
  bootstrap_system_python_with_apt
  PYTHON_EXE="$(discover_system_python || true)"
fi

if [[ -z "${PYTHON_EXE}" ]]; then
  echo "[sugar-install] no usable system python was found." >&2
  echo "[sugar-install] this script intentionally avoids conda/micromamba interpreters." >&2
  echo "[sugar-install] rerun with BOOTSTRAP_SYSTEM_PYTHON=1, or install python3/python3-venv/python3-dev first." >&2
  exit 1
fi

if [[ ! -d "${SUGAR_ROOT}" ]]; then
  echo "[sugar-install] repo not found: ${SUGAR_ROOT}" >&2
  exit 1
fi

PYTHON_MM="$("${PYTHON_EXE}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "${PYTHON_MM}" in
  3.8)
    PYTORCH3D_PY_TAG="py38"
    NUMPY_VERSION="${NUMPY_VERSION:-1.24.4}"
    ;;
  3.9)
    PYTORCH3D_PY_TAG="py39"
    NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
    ;;
  3.10)
    PYTORCH3D_PY_TAG="py310"
    NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
    ;;
  *)
    echo "[sugar-install] unsupported python version: ${PYTHON_MM}" >&2
    echo "[sugar-install] please use a system python 3.8, 3.9, or 3.10." >&2
    exit 1
    ;;
esac

PYTORCH3D_WHEEL_URL="${PYTORCH3D_WHEEL_URL:-https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${PYTORCH3D_PY_TAG}_cu118_pyt201/download.html}"

echo "[sugar-install] repo: ${SUGAR_ROOT}"
echo "[sugar-install] python: ${PYTHON_EXE} (${PYTHON_MM})"
echo "[sugar-install] venv: ${SUGAR_ENV_DIR}"

if ! "${PYTHON_EXE}" -m venv --help >/dev/null 2>&1; then
  if [[ "${BOOTSTRAP_SYSTEM_PYTHON}" == "1" ]]; then
    bootstrap_system_python_with_apt
  fi
fi

if ! "${PYTHON_EXE}" -m venv --help >/dev/null 2>&1; then
  echo "[sugar-install] python venv module is unavailable for ${PYTHON_EXE}" >&2
  echo "[sugar-install] install the OS package that provides venv support, or rerun with BOOTSTRAP_SYSTEM_PYTHON=1." >&2
  exit 1
fi

if [[ ! -d "${SUGAR_ENV_DIR}" ]]; then
  "${PYTHON_EXE}" -m venv "${SUGAR_ENV_DIR}"
fi

VENV_PYTHON="${SUGAR_ENV_DIR}/bin/python"
VENV_PIP="${SUGAR_ENV_DIR}/bin/pip"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "[sugar-install] venv python missing: ${VENV_PYTHON}" >&2
  exit 1
fi

export PATH="${SUGAR_ENV_DIR}/bin:${PATH}"

"${VENV_PYTHON}" -m pip install -U pip "setuptools<81" wheel packaging cmake ninja

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  "${VENV_PYTHON}" -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"
fi

"${VENV_PYTHON}" -m pip install \
  "numpy==${NUMPY_VERSION}" \
  "pillow>=10,<11" \
  "tqdm>=4.66,<5" \
  "rich>=13,<14" \
  "plotly>=5.18,<6" \
  "requests>=2.31,<3" \
  "plyfile==0.8.1" \
  "fvcore>=0.1.5" \
  "iopath>=0.1.10" \
  "open3d==${OPEN3D_VERSION}" \
  "PyMCubes==0.1.4"

if [[ "${INSTALL_SWINIR_RUNTIME}" == "1" ]]; then
  "${VENV_PYTHON}" -m pip install \
    "opencv-python-headless==${OPENCV_PYTHON_HEADLESS_VERSION}" \
    "timm==${TIMM_VERSION}"
fi

if [[ "${INSTALL_PYTORCH3D}" == "1" ]]; then
  "${VENV_PYTHON}" -m pip install "pytorch3d==${PYTORCH3D_VERSION}" -f "${PYTORCH3D_WHEEL_URL}"
fi

"${VENV_PYTHON}" -m pip install --no-build-isolation -e "${SUGAR_ROOT}/gaussian_splatting/submodules/diff-gaussian-rasterization"
"${VENV_PYTHON}" -m pip install --no-build-isolation -e "${SUGAR_ROOT}/gaussian_splatting/submodules/simple-knn"

if [[ "${INSTALL_NVDIFFRAST}" == "1" ]]; then
  if [[ ! -d "${NVDIFFRAST_ROOT}" ]]; then
    git clone https://github.com/NVlabs/nvdiffrast "${NVDIFFRAST_ROOT}"
  fi
  "${VENV_PYTHON}" -m pip install --no-build-isolation "${NVDIFFRAST_ROOT}"
fi

"${VENV_PYTHON}" -m pip check

VERIFY_SCRIPT='
import importlib
modules = [
    "torch",
    "torchvision",
    "torchaudio",
    "pytorch3d",
    "open3d",
    "requests",
    "mcubes",
    "plotly",
    "rich",
    "plyfile",
    "diff_gaussian_rasterization",
    "simple_knn",
]
for name in modules:
    importlib.import_module(name)
print("verified:", ", ".join(modules))
'

if [[ "${INSTALL_SWINIR_RUNTIME}" == "1" ]]; then
  VERIFY_SCRIPT+='
import importlib
for name in ["cv2", "timm"]:
    importlib.import_module(name)
print("verified: cv2, timm")
'
fi

if [[ "${INSTALL_NVDIFFRAST}" == "1" ]]; then
  VERIFY_SCRIPT+='
import importlib
importlib.import_module("nvdiffrast")
print("verified: nvdiffrast")
'
fi

"${VENV_PYTHON}" - <<PY
${VERIFY_SCRIPT}
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_ok:", torch.cuda.is_available())
PY

echo "[sugar-install] done"
echo "[sugar-install] activate with: source ${SUGAR_ENV_DIR}/bin/activate"
if [[ "${INSTALL_SWINIR_RUNTIME}" == "1" ]]; then
  echo "[sugar-install] SwinIR runtime packages are available in the same venv."
fi
