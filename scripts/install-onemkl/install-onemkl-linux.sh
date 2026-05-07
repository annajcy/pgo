#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y gnupg wget

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor \
    | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt-get update
sudo apt-get install -y intel-oneapi-mkl-devel

MKL_ROOT="/opt/intel/oneapi/mkl/latest"
MKL_CMAKE_DIR="${MKL_ROOT}/lib/cmake/mkl"

test -f "${MKL_CMAKE_DIR}/MKLConfig.cmake"

if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
        echo "MKLROOT=${MKL_ROOT}"
        echo "MKL_DIR=${MKL_CMAKE_DIR}"
        echo "CMAKE_PREFIX_PATH=${MKL_ROOT}:${CMAKE_PREFIX_PATH:-}"
        echo "LD_LIBRARY_PATH=${MKL_ROOT}/lib/intel64:${LD_LIBRARY_PATH:-}"
        echo "LIBRARY_PATH=${MKL_ROOT}/lib/intel64:${LIBRARY_PATH:-}"
    } >> "${GITHUB_ENV}"
fi

echo "oneMKL is installed at ${MKL_ROOT}"
