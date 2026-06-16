# src/device/CMakeLists.txt
#
# steppe_device — the CUDA-ONLY kernel layer (architecture.md §4, §8; ROADMAP M0).
#
# This is where the validated f2 logic is lifted: the 3-GEMM reformulation +
# fixed-slice-Ozaki precision (cuda/f2_block_kernel.cu), the GPU backend
# (cuda/cuda_backend.cu implementing ComputeBackend), and the scalar CPU
# reference oracle (cpu/cpu_backend.cpp). The DRY device helpers — STEPPE_CUDA_CHECK
# / CUBLAS_CHECK (cuda/check.cuh), RAII DeviceBuffer (cuda/device_buffer.cuh),
# RAII cuBLAS handle (cuda/handles.hpp) — are the allowlisted homes for all CUDA
# resource ownership (architecture.md §2, §7).
#
# THE LAYERING MECHANISM (architecture.md §4 "How CMake enforces it"):
#   * CUDA::cublas / CUDA::cudart are PRIVATE → core/api/tests cannot see a CUDA
#     header through this target. backend.hpp is verified CUDA-free.
#   * the src/ include dir is PUBLIC so consumers (core, tests) can include the
#     CUDA-free seam "device/backend.hpp" without pulling CUDA in.
#   * steppe::core_internal is PUBLIC because the device kernels and the CPU
#     reference BOTH consume the shared f2 estimator primitive + views (the
#     single-source-of-the-formula invariant, architecture.md §8).

add_library(steppe_device STATIC
    cuda/f2_block_kernel.cu       # 3-GEMM reformulation + fixed-slice Ozaki + assemble (single block)
    cuda/f2_blocks_kernel.cu      # M4: size-grouped strided-batched per-block f2 (gather/GEMMs/assemble)
    cuda/decode_af_kernel.cu      # M1: 2-bit unpack + segmented allele-freq reduction → Q/V/N
    cuda/cuda_backend.cu          # implements ComputeBackend on the GPU
    cpu/cpu_backend.cpp)          # REFERENCE backend: scalar host oracle
add_library(steppe::device ALIAS steppe_device)

# backend.hpp (the CUDA-free DI seam) is reached as "device/backend.hpp"; the f2
# estimator + views come via steppe::core_internal as "core/internal/...". Both
# resolve from the src/ root.
target_include_directories(steppe_device PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/..")

target_link_libraries(steppe_device
    PUBLIC  steppe::api steppe::core_internal     # public, CUDA-FREE seam + shared helpers
    PRIVATE steppe::warnings CUDA::cudart CUDA::cublas)  # CUDA stays PRIVATE → layering

# Fixed-slice Ozaki engagement symbols (cublasSetEmulationStrategy /
# cublasSetFixedPointEmulationMantissaControl /
# cublasSetFixedPointEmulationMaxMantissaBitCount) are gated behind this macro in
# the device code. The MEASURED precision policy (FIXED 40-bit default; ROADMAP
# §0) REQUIRES them, so the production build defines it (default ON, §SteppeOptions).
# PRIVATE: it is a device-codegen detail, never leaks to consumers.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(steppe_device PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()

# RDC + separable compilation: the backend TU launches the kernel from another TU
# via a narrow launch_xxx wrapper (architecture.md §7), and device-LTO (Release)
# needs it. Cheap and correct at M0; the multi-arch device-LTO wiring (§6) lands
# with the release matrix.
set_target_properties(steppe_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON)

# Optional fatbin compression, Release-only opt-in (architecture.md §6).
if(STEPPE_COMPRESS_FATBIN)
  target_compile_options(steppe_device PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--compress-mode=size>)
endif()
