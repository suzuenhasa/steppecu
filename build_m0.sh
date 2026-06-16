#!/usr/bin/env bash
# build_m0.sh — fallback direct-nvcc build of the M0 CPU-vs-GPU f2 equivalence
# test, so the structure-lift slice builds even if CMake needs polish.
#
# This compiles the SAME real source files the CMake build wires together
# (architecture.md §4 paths; ROADMAP M0) into one binary, using only nvcc +
# -lcublas. The GoogleTest path is OFF in this build (STEPPE_TEST_WITH_GTEST
# undefined), so the equivalence test runs its self-checking fallback and exits
# non-zero on failure — exactly what a gate needs.
#
# TARGET: sm_120 (RTX 5090) / CUDA 13 — the box where the f2 kernel was VALIDATED.
# Do NOT run this on the local RTX 2070 / CUDA 11.8 dev box (wrong arch).
#
# Build flag -DSTEPPE_HAVE_EMU_TUNING=1 engages the fixed-slice Ozaki tuning
# calls (cublasSetEmulationStrategy / ...MantissaControl / ...MaxMantissaBitCount)
# that the MEASURED precision policy requires (ROADMAP §0; architecture.md §12).
#
# Override the compiler/arch via env: NVCC=/path/to/nvcc ARCH=sm_120 ./build_m0.sh
set -euo pipefail

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_120}"
STD="${STD:-c++20}"   # C++20 is required: device/cuda/check.cuh uses
                      # std::source_location (architecture.md §7) and the project
                      # standard is C++20 (architecture.md §6). The pure-data
                      # contract headers (config/views/f2_estimator/backend) are
                      # C++17-clean, but the CUDA RAII/check headers the backend
                      # TUs pull in are not — so build at C++20, matching CMake.

# Resolve paths relative to this script so it runs from anywhere.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

# Public API headers (include/steppe/...) and the src/ root (core/..., device/...).
INCLUDES=(
    -I "${HERE}/include"
    -I "${HERE}/src"
)

# The real source files (must match the paths the implementation agents author):
#   GPU backend + kernel + scalar CPU reference oracle + host orchestration + test.
SOURCES=(
    "${HERE}/tests/reference/test_f2_equivalence.cu"   # GPU-vs-CPU equivalence harness
    "${HERE}/src/device/cuda/cuda_backend.cu"          # CudaBackend : ComputeBackend
    "${HERE}/src/device/cuda/f2_block_kernel.cu"       # 3-GEMM + fixed-slice Ozaki + assemble
    "${HERE}/src/device/cpu/cpu_backend.cpp"           # CpuBackend  : ComputeBackend (oracle)
    "${HERE}/src/core/fstats/f2_from_blocks.cpp"       # host f2 assembly orchestration
)

OUT="${HERE}/test_f2_equivalence"

echo "==> Building M0 f2 equivalence test for ${ARCH} (CUDA 13, -lcublas, -std=${STD})"
echo "    EMU_TUNING=1 (fixed-slice Ozaki engagement)"
"${NVCC}" -O3 -std="${STD}" -arch="${ARCH}" \
    -DSTEPPE_HAVE_EMU_TUNING=1 \
    "${INCLUDES[@]}" \
    "${SOURCES[@]}" \
    -lcublas \
    -o "${OUT}"

echo "==> Build OK: ${OUT}"
echo "==> Run it on the sm_120 box: ${OUT}"
