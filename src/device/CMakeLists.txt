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
    cuda/qpadm_fit_kernels.cu     # M(fit-4): qpAdm fit kernels — f4 gather + loo/total + xtau + on-device small-LA (SVD/ALS/weight/chisq, batched LOO)
    cuda/qpgraph_fit_kernels.cu   # qpGraph: the on-device IDEA-1 fleet (one thread/restart; the path-table fill_pwts + GLS objective + NNLS edge solve)
    cuda/decode_af_kernel.cu      # M1: 2-bit unpack + segmented allele-freq reduction → Q/V/N
    cuda/decode_compact_kernel.cu # device-resident decode seam: autosome keep-mask + [P×M] column gather (C2/M3/M4 cure)
    cuda/dstat_kernel.cu          # qpDstat Part B: per-block per-quadruple normalized-D reduction (the S2 divergence)
    cuda/dates_kernel.cu          # DATES: the cuFFT autocorrelation LD engine kernels (weight/residual/scatter + |F|²/cross-power + lag re-bin)
    cuda/qpfstats_kernel.cu       # qpfstats: zero-NaN-in-ymat + add-ridge-diag + NaN-row-downdate (the smoothing-solve prep kernels)
    cuda/qpfstats_jackknife_kernel.cu  # qpfstats PERF: on-device per-comb block-jackknife (y/ymat) + per-pair recenter (native FP64; §12 carve-out)
    cuda/ratio_block_jackknife_kernel.cu  # SHARED on-device RATIO-block-jackknife (f4ratio M1 + qpDstat M2; native FP64; §12 carve-out)
    cuda/cuda_backend.cu          # implements ComputeBackend on the GPU
    cuda/device_partial.cu        # M4.5: DevicePartial out-of-line special members (opaque resident-handle Impl)
    cuda/device_f2_blocks.cu      # M4.5: DeviceF2Blocks — device-resident FULL result handle (special members + to_host/upload)
    cuda/device_decode_result.cu  # device-resident decode seam: DeviceDecodeResult — autosome-compacted Q/V resident handle (special members + accessors)
    cuda/p2p_combine.cu           # M4.5: combine_f2_partials_resident — device-resident cudaMemcpyPeer disjoint placement combine
    cuda/block_sink.cu            # M5: HostRamSink + DiskSink — triple-buffered persistent-pinned block spill (TIER 1/2)
    cuda/f2_blocks_out.cu         # M5: F2BlocksOut tier-agnostic read-back (to_host/read_block_to_host) + DiskF2Blocks members
    cpu/cpu_backend.cpp           # REFERENCE backend: scalar host oracle
    host_ram.cpp                  # M5: free_host_ram_bytes (sysinfo) + resolve_output_tier (CUDA-free tier policy)
    resources.cpp                 # M4.5: build_resources — one backend+probe PER device (CUDA-free body)
    shard_plan.cpp)               # M4.5: plan_block_shards — block-aligned SNP shard plan (CUDA-free body)
add_library(steppe::device ALIAS steppe_device)

# resources.cpp + shard_plan.cpp are PLAIN .cpp (not CUDA): build_resources reaches
# the GPU only through the CUDA-free factory (make_cuda_backend) + the CUDA-free
# capabilities() probe, and plan_block_shards is host-pure block→device arithmetic
# over core::BlockRange — neither includes a CUDA header (design §1/§2/§7). They link
# into steppe_device (so make_cuda_backend resolves, and the multi-GPU orchestrator
# in steppe_core reaches plan_block_shards through the device target), but are
# themselves device-toolkit-free.

# backend.hpp (the CUDA-free DI seam) is reached as "device/backend.hpp"; the f2
# estimator + views come via steppe::core_internal as "core/internal/...". Both
# resolve from the src/ root.
target_include_directories(steppe_device PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/..")

target_link_libraries(steppe_device
    PUBLIC  steppe::api steppe::core_internal     # public, CUDA-FREE seam + shared helpers
    PRIVATE steppe::warnings CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cufft)  # CUDA stays PRIVATE → layering (cufft: the DATES autocorrelation LD engine)

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
