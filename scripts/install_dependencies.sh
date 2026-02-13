#!/usr/bin/env bash
set -euo pipefail

# Safe bootstrap for this project.
# - Installs only missing OS packages via apt.
# - Installs/updates Python deps from requirements files.
# - Never installs/removes NVIDIA driver packages.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
INSTALL_GPU=0
ALLOW_EOL_UBUNTU=0

usage() {
  cat <<'EOF'
Usage:
  scripts/install_dependencies.sh [options]

Options:
  --gpu                 Install GPU Python extras from requirements-gpu.txt
  --venv PATH           Virtualenv path (default: .venv in project root)
  --allow-eol-ubuntu    Continue even on EOL Ubuntu releases (<20.04)
  -h, --help            Show this help

Safety:
  - This script DOES NOT install or modify NVIDIA drivers.
  - This script DOES NOT install CUDA driver/toolkit apt packages.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      INSTALL_GPU=1
      shift
      ;;
    --venv)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --venv requires a path argument."
        exit 2
      fi
      VENV_PATH="$2"
      shift 2
      ;;
    --allow-eol-ubuntu)
      ALLOW_EOL_UBUNTU=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

cd "${PROJECT_ROOT}"

if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
  echo "[INFO] NVIDIA GPU detected: ${GPU_NAME:-unknown}, driver=${DRIVER_VERSION:-unknown}"
  echo "[INFO] NVIDIA driver packages will not be touched."
fi

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
else
  echo "[ERROR] /etc/os-release not found. Unsupported Linux distribution."
  exit 1
fi

if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "[WARN] Detected distro: ${ID:-unknown}. This script is optimized for Ubuntu."
fi

if [[ "${ID:-}" == "ubuntu" ]]; then
  UBUNTU_VERSION="${VERSION_ID:-0}"
  UBUNTU_MAJOR="${UBUNTU_VERSION%%.*}"
  if [[ "${UBUNTU_MAJOR}" =~ ^[0-9]+$ ]] && (( UBUNTU_MAJOR < 20 )); then
    echo "[WARN] Detected Ubuntu ${UBUNTU_VERSION} (EOL)."
    echo "[WARN] apt repositories may be archived/broken; latest packages are not guaranteed."
    if (( ALLOW_EOL_UBUNTU == 0 )); then
      echo "[ERROR] Refusing to continue on EOL Ubuntu without --allow-eol-ubuntu."
      echo "[ERROR] Recommended: use Ubuntu 22.04+ or Docker image flow from README."
      exit 1
    fi
  fi
fi

ensure_sudo() {
  if [[ "$(id -u)" -eq 0 ]]; then
    echo ""
  elif command -v sudo >/dev/null 2>&1; then
    echo "sudo"
  else
    echo "[ERROR] sudo is required to install system packages." >&2
    exit 1
  fi
}

SUDO_CMD="$(ensure_sudo)"

APT_PACKAGES=(
  ca-certificates
  curl
  git
  wget
  build-essential
  pkg-config
  python3
  python3-venv
  python3-pip
  python3-dev
)

MISSING_APT=()
for pkg in "${APT_PACKAGES[@]}"; do
  if dpkg-query -W -f='${Status}' "${pkg}" 2>/dev/null | grep -q "install ok installed"; then
    echo "[OK] apt package already installed: ${pkg}"
  else
    MISSING_APT+=("${pkg}")
  fi
done

if (( ${#MISSING_APT[@]} > 0 )); then
  echo "[INFO] Installing missing apt packages: ${MISSING_APT[*]}"
  ${SUDO_CMD} apt-get update
  ${SUDO_CMD} apt-get install -y --no-install-recommends "${MISSING_APT[@]}"
else
  echo "[OK] All required apt packages are already installed."
fi

resolve_python_bin() {
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo "[ERROR] Python 3 is required but was not found." >&2
  exit 1
}

PYTHON_BIN="$(resolve_python_bin)"
PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[INFO] Using Python interpreter: ${PYTHON_BIN} (version ${PYTHON_VERSION})"

if ! ${PYTHON_BIN} -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  echo "[ERROR] Python >= 3.10 is required for this project."
  exit 1
fi

if ! ${PYTHON_BIN} -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  echo "[WARN] Python 3.11 is recommended (detected ${PYTHON_VERSION})."
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[INFO] Creating virtual environment: ${VENV_PATH}"
  ${PYTHON_BIN} -m venv "${VENV_PATH}"
else
  echo "[OK] Virtual environment already exists: ${VENV_PATH}"
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

echo "[INFO] Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

REQ_FILES=("${PROJECT_ROOT}/requirements.txt")
if (( INSTALL_GPU == 1 )); then
  REQ_FILES+=("${PROJECT_ROOT}/requirements-gpu.txt")
fi

for req_file in "${REQ_FILES[@]}"; do
  if [[ ! -f "${req_file}" ]]; then
    echo "[ERROR] Missing requirements file: ${req_file}"
    exit 1
  fi
  echo "[INFO] Installing/updating Python dependencies from ${req_file}"
  python -m pip install --upgrade --upgrade-strategy only-if-needed -r "${req_file}"
done

echo "[INFO] Installation completed."
echo "[INFO] Virtual environment: ${VENV_PATH}"
echo "[INFO] Activate with: source ${VENV_PATH}/bin/activate"
