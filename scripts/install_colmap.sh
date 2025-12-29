#!/usr/bin/env bash
set -euo pipefail

###############################################
# COLMAP 3.8 install helper
# Notes:
#   - Colmap 3.8 bundles legacy SiftGPU code that is NOT compatible with CUDA >= 12
#   - If your nvcc toolkit version is 12 or newer this script will (by default) fall back to a CPU-only build
#   - To force a CUDA build (at your own risk) export COLMAP_FORCE_CUDA=1
#   - To explicitly request CPU-only build export COLMAP_CPU_ONLY=1
#   - Recommended toolkit for GPU build: CUDA 11.4 – 11.8 (tested)
###############################################

root_folder=$(realpath $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..)
source "${root_folder}/scripts/load_env.sh" 2>/dev/null || true

mkdir -p "${root_folder}/external"
cd "${root_folder}/external"

if [ -d "${root_folder}/external/colmap_v3.8" ]; then
    echo "[INFO] Removing previous colmap_v3.8 directory" >&2
    sudo rm -rf "${root_folder}/external/colmap_v3.8"
fi

echo "[INFO] Installing system dependencies for COLMAP 3.8 ..." >&2
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends --no-install-suggests \
        build-essential cmake git \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev

echo "[INFO] Cloning COLMAP v3.8 ..." >&2
git clone --recursive -b 3.8 https://github.com/colmap/colmap colmap_v3.8 --depth=1
cd colmap_v3.8

# Detect nvcc version if available
CUDA_TOOLKIT_VER_MAJOR=""; CUDA_TOOLKIT_VER_MINOR=""
if command -v nvcc >/dev/null 2>&1; then
    nvcc_raw=$(nvcc --version | grep -i "release" || true)
    # Example: Cuda compilation tools, release 11.8, V11.8.89
    ver=$(echo "$nvcc_raw" | sed -E 's/.*release ([0-9]+)\.([0-9]+).*/\1 \2/' || true)
    CUDA_TOOLKIT_VER_MAJOR=$(echo "$ver" | awk '{print $1}')
    CUDA_TOOLKIT_VER_MINOR=$(echo "$ver" | awk '{print $2}')
fi

echo "[INFO] Detected nvcc version: ${CUDA_TOOLKIT_VER_MAJOR:-none}.${CUDA_TOOLKIT_VER_MINOR:-x}" >&2 || true

GPU_BUILD=1
if [ -n "${COLMAP_CPU_ONLY:-}" ]; then
    GPU_BUILD=0
    echo "[INFO] COLMAP_CPU_ONLY set -> forcing CPU-only build" >&2
elif [ -n "${COLMAP_FORCE_CUDA:-}" ]; then
    echo "[WARN] COLMAP_FORCE_CUDA set -> attempting CUDA build even if toolkit is >=12" >&2
else
    # Automatic safeguard: if CUDA >=12 fallback
    if [ -n "$CUDA_TOOLKIT_VER_MAJOR" ] && [ "$CUDA_TOOLKIT_VER_MAJOR" -ge 12 ]; then
        echo "[WARN] Detected CUDA toolkit >= 12 (found ${CUDA_TOOLKIT_VER_MAJOR}.${CUDA_TOOLKIT_VER_MINOR})." >&2
        echo "[WARN] Legacy SiftGPU in COLMAP 3.8 fails to compile with CUDA 12 (removed texture reference API)." >&2
        echo "[WARN] Falling back to CPU-only build. For GPU support install CUDA 11.8 and re-run, or export COLMAP_FORCE_CUDA=1." >&2
        GPU_BUILD=0
    fi
fi

CMAKE_EXTRA_ARGS=()
if [ "$GPU_BUILD" -eq 0 ]; then
    CMAKE_EXTRA_ARGS+=("-DCUDA_ENABLED=OFF")
    echo "[INFO] Configuring CPU-only COLMAP build..." >&2
else
    # Provide broad architectures (may be pruned by CMake). Using explicit list safer than 'all' for older CMake.
    : "${CMAKE_CUDA_ARCH_LIST:=50;52;60;61;70;75;80;86}"
    CMAKE_EXTRA_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCH_LIST}")
    echo "[INFO] Configuring GPU COLMAP build (CUDA architectures: ${CMAKE_CUDA_ARCH_LIST})" >&2
fi

echo "[INFO] Running CMake configure ..." >&2
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "${CMAKE_EXTRA_ARGS[@]}"

echo "[INFO] Building & Installing ..." >&2
cmake --build build --target install -- -j"$(nproc)"

echo "[INFO] COLMAP 3.8 installation finished." >&2
if [ "$GPU_BUILD" -eq 0 ]; then
    echo "[INFO] Built without CUDA. To enable GPU SIFT: install CUDA 11.x (e.g., 11.8) and re-run after removing colmap_v3.8." >&2
fi
