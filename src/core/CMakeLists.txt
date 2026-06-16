# src/core/CMakeLists.txt
#
# Two targets (architecture.md §4, §8):
#
#   steppe_core_internal  INTERFACE  — the DRY, host-pure, CUDA-FREE shared
#       helpers that are the "kernel" of the codebase: the Q/V/N views
#       (internal/views.hpp), the shared __host__ __device__ f2 estimator
#       primitive (internal/f2_estimator.hpp — the single source of the f2
#       formula so CPU oracle and GPU feeder cannot diverge), and the shared
#       SNP→block domain rule (domain/block_partition_rule.hpp). Consumed by BOTH
#       core and the device layer; that shared consumption is exactly why it is
#       INTERFACE-only and CUDA-free (f2_estimator.hpp compiles under nvcc via
#       STEPPE_HD == __host__ __device__, and under g++ as plain inline).
#
#   steppe_core           static     — the host orchestration. M0: the f2
#       assembly orchestration (fstats/f2_from_blocks.cpp), which drives the
#       backend through the ComputeBackend seam and never issues a GEMM itself.
#
# Layering: steppe_core links steppe::device PRIVATE, so it reaches the GPU ONLY
# through ComputeBackend (a CUDA-free header). CUDA stays PRIVATE to
# steppe_device, so core physically cannot include a CUDA header.

# --- steppe_core_internal: shared host-pure helpers (INTERFACE) --------------
add_library(steppe_core_internal INTERFACE)
add_library(steppe::core_internal ALIAS steppe_core_internal)

# Headers under src/core/ are included as "core/internal/..." and
# "core/domain/...", and backend.hpp includes "core/internal/views.hpp", so the
# include root is src/ (one level above this dir).
target_include_directories(steppe_core_internal INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/..")

target_link_libraries(steppe_core_internal INTERFACE steppe::api)  # config.hpp, error.hpp
target_compile_features(steppe_core_internal INTERFACE cxx_std_20)

# --- steppe_core: host orchestration (static lib) ----------------------------
# domain/block_partition_rule.cpp — the whole-ordering SNP→block assignment
# (M3). Host-pure, CUDA-free; the single home of the jackknife block id consumed
# by both io and the device kernels (architecture.md §8 DRY invariant). The
# per-SNP `block_of` primitive + the cM→Morgan conversion stay header-inline; only
# the stateful pass compiles here.
add_library(steppe_core STATIC
    fstats/f2_from_blocks.cpp
    domain/block_partition_rule.cpp)
add_library(steppe::core ALIAS steppe_core)

target_link_libraries(steppe_core
    PUBLIC  steppe::api steppe::core_internal
    PRIVATE steppe::device steppe::warnings)   # GPU only via ComputeBackend

# steppe_core has NO CUDA sources of its own (pure host orchestration); it links
# steppe_device PRIVATE only to reach the GPU through ComputeBackend. Do NOT let it
# archive its own nvcc -dlink object for steppe_device's RDC symbols
# (CUDA_RESOLVE_DEVICE_SYMBOLS OFF): if it did, a CUDA executable that links BOTH
# steppe_core and steppe_device (e.g. the M4 equivalence test) would double-resolve
# the __cudaRegisterLinkedBinary symbols and the final host link would miss them.
# With this OFF, the consuming executable device-links steppe_device directly — the
# same path the M0/M1 CUDA tests already use successfully.
set_target_properties(steppe_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
