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
#
# M4.5 SPMG precompute (architecture.md §11.4, §12): the multi-GPU entry point
# (fstats/f2_blocks_multigpu.cpp) shards whole blocks across the G devices of a
# Resources bundle, drives each per-device backend through the CUDA-free
# ComputeBackend seam, and combines the partials in the FIXED g=0..G-1 order — either
# the host-staged fixed-order combine (fstats/f2_combine.cpp, the portable parity
# baseline) or, behind the §4 capability gate (defined ONCE at the `use_p2p`
# computation in fstats/f2_blocks_multigpu.cpp — "THE §4 COMBINE GATE", §8
# single-source), the device-resident cudaMemcpyPeer combine
# (combine_f2_partials_p2p). Both combine tiers are CUDA-free-DECLARED: f2_combine
# .cpp is itself CUDA-free host-pure in steppe::core, and the P2P combine is reached
# through the CUDA-free decl device/p2p_combine.hpp (its CUDA definition,
# cuda/p2p_combine.cu, lives in steppe_device — the same CUDA-free-decl pattern as
# make_cuda_backend). So this entry-point TU names only F2BlockTensor / Resources /
# DeviceShard + the CUDA-free combine decls and stays CUDA-free. Both
# plan_block_shards and combine_f2_partials_p2p resolve through the PRIVATE
# steppe::device link below.
#
# The HOST-PURE, P2P-FREE core of that orchestrator — the block-aligned shard PLAN
# (plan_multigpu_shards) + the per-device concurrent FAN-OUT (compute_multigpu_partials)
# — is factored into fstats/f2_blocks_multigpu_core.cpp (cleanup D1/T1, B9): it
# references ONLY the CUDA-free ComputeBackend seam + plan_block_shards + block_ranges,
# NEVER the device-RDC combine_f2_partials_p2p, so a GPU-FREE host test
# (tests/unit/test_f2_blocks_multigpu.cpp) drives it against a fake backend with no
# device-link. The entry-point TU composes that core with the §4 combine fork.
add_library(steppe_core STATIC
    fstats/f2_from_blocks.cpp
    fstats/f2_combine.cpp
    fstats/f2_blocks_multigpu.cpp
    fstats/f2_blocks_multigpu_core.cpp
    domain/block_partition_rule.cpp
    # Phase 2 qpAdm fit engine (S3-S8). Host-pure, CUDA-FREE: every device op
    # routes through the ComputeBackend seam (CpuBackend reference in M(fit-1);
    # CudaBackend in M(fit-4)). f4_matrix/jackknife/gls_solve are header-only thin
    # drivers; the orchestrator + S7 SE + pchisq compile here.
    qpadm/qpadm_fit.cpp
    qpadm/ranktest.cpp
    qpadm/nested_models.cpp
    # Standalone f4 statistic (run_f4): the SIBLING of run_qpwave — REUSES
    # assemble_f4_quartets + jackknife_cov (fit-engine §6), no ALS/rank, ZERO new math.
    qpadm/f4.cpp
    # Standalone f3 statistic (run_f3): the THREE-slab SIBLING of run_f4 — REUSES
    # jackknife_cov (fit-engine §6), no ALS/rank; the ONE new math seam is the 3-slab combine.
    qpadm/f3.cpp
    # Standalone f4-ratio (run_f4ratio): the SIBLING of run_f4/run_f3 — REUSES
    # assemble_f4_quartets (ONE shared assemble for num+den); the ONE new math seam is the
    # per-block-RATIO weighted block-jackknife (AT2 qpf4ratio jack_mat_stats $est), host-pure.
    qpadm/f4ratio.cpp
    # GPU-ONLY f-stat SWEEP (run_f4_sweep / run_f3_sweep): the production-scale sibling of
    # run_f4/run_f3 — enumerate EVERY C(P,k) item ON THE DEVICE (combinatorial-unrank kernel),
    # reuse assemble_f4_quartets + the diagonal jackknife, filter (|z|) + CUB-compact survivors
    # on-device, return ONLY survivors. NO host enumeration/filter/per-item-loop; maxcomb cap.
    qpadm/fstat_sweep.cpp
    # M(fit-6) S8 ROTATION: the model-space search orchestrator (run_qpadm_search +
    # the base fit_models_batched default) + the GPU-free model->device shard planner.
    qpadm/model_search.cpp
    qpadm/model_search_core.cpp
    # qpDstat Part B (run_dstat): the GENOTYPE-PATH normalized-D — REUSES the extract-f2
    # decode front-end (io reader + decode_af + assign_blocks) + DIVERGES at S2 into the
    # per-SNP D kernel (ComputeBackend::dstat_block_reduce, the .cu) + the num/den block-
    # jackknife (host-pure, the f4ratio ratio-jackknife FAMILY). NEVER reads the f2 cache.
    stats/dstat.cpp
    # M(cli-0) ACCESS LAYER config (architecture.md §9; cli-bindings.md §4.5): the
    # CUDA-FREE, host-pure ConfigBuilder/RunConfig/CliArgs — the §9 precedence merge
    # (compiled defaults < TOML < env STEPPE_* < CLI) + the validating build() that maps
    # the raw flags onto the real structs (DeviceConfig/QpAdmOptions/FilterConfig/
    # PopSelection) and fail-fasts (Status::InvalidConfig) on bad config. It lives in
    # core (not app) so the bindings reuse it and it is host-testable GPU-free like
    # test_config (the io PopSelection header is header-only; no steppe::io link needed).
    config/config_builder.cpp)
add_library(steppe::core ALIAS steppe_core)

target_link_libraries(steppe_core
    PUBLIC  steppe::api steppe::core_internal
    PRIVATE steppe::device steppe::warnings     # GPU only via ComputeBackend
            steppe::io                          # qpDstat Part B (run_dstat): the genotype decode front-end (GenoReader/read_ind/read_snp)
            Threads::Threads)                   # SPMG per-device fan-out (f2_blocks_multigpu)

# steppe_core has NO CUDA sources of its own (pure host orchestration); it links
# steppe_device PRIVATE only to reach the GPU through ComputeBackend. Do NOT let it
# archive its own nvcc -dlink object for steppe_device's RDC symbols
# (CUDA_RESOLVE_DEVICE_SYMBOLS OFF): if it did, a CUDA executable that links BOTH
# steppe_core and steppe_device (e.g. the M4 equivalence test) would double-resolve
# the __cudaRegisterLinkedBinary symbols and the final host link would miss them.
# With this OFF, the consuming executable device-links steppe_device directly — the
# same path the M0/M1 CUDA tests already use successfully.
set_target_properties(steppe_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
