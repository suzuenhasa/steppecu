# tests/CMakeLists.txt
#
# M0 gate (ROADMAP §6 DoD): the CPU-reference-equivalence test — identical Q/V/N
# through CpuBackend and CudaBackend, asserted equal at the f2/Vpair seam within
# the tight tolerance tier (architecture.md §13). Plus the host-only unit test of
# the shared __host__ __device__ f2 estimator numerics (no GPU).
#
# GoogleTest is the project harness (architecture.md §3, §13). To keep the M0
# slice buildable even before third_party/ CPM-fetches GTest, we use it when it
# is discoverable and otherwise fall back to a self-checking executable that
# returns non-zero on failure — which is all CTest needs to gate. The equivalence
# .cu also has the standalone nvcc fallback in /home/suzunik/steppe/build_m0.sh.

# --- Optional GoogleTest (architecture.md §3) --------------------------------
find_package(GTest CONFIG QUIET)
if(GTest_FOUND)
  set(STEPPE_HAVE_GTEST ON)
  message(STATUS "steppe tests: using GoogleTest ${GTest_VERSION}")
else()
  set(STEPPE_HAVE_GTEST OFF)
  message(STATUS "steppe tests: GoogleTest not found — using self-checking \
fallback (returns non-zero on failure; CTest gates on exit code)")
endif()

# --- GPU-vs-CPU equivalence test (the central M0 trust seam) -----------------
# CUDA TU: drives the PRODUCTION backends — steppe::device::make_cpu_backend() (the
# long-double oracle) and steppe::device::make_cuda_backend() (the 3-GEMM GPU
# path) — through the CUDA-free ComputeBackend::compute_f2 seam, NOT an inline
# re-implementation (cleanup X-2/B4: the old inline oracle walked j=i+1 and hid the
# M0 diagonal divergence). Both factories are declared in the CUDA-free
# device/backend_factory.hpp (X-9/B8) and now share namespace steppe::device. It
# diffs the FULL [P × P] matrix INCLUDING the diagonal and exits non-zero on any
# mismatch. Links steppe::core (transitively) + steppe::device (both factories +
# the device-private kernel header for the X-6/B2 emulation_honorable predicate);
# CUDA::cublas is transitively-private through steppe_device.
#
# This test has its OWN main() and is NOT a GoogleTest-style TU (it self-checks
# and exits non-zero on failure), so it does NOT link gtest_main or define
# STEPPE_TEST_WITH_GTEST — that would collide with its main(). CTest gates on the
# exit code either way (architecture.md §13).
add_executable(test_f2_equivalence reference/test_f2_equivalence.cu)
target_link_libraries(test_f2_equivalence
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
# Separable compilation so the executable device-links steppe_device's RDC kernels
# directly (the same path the other CUDA reference tests use).
set_target_properties(test_f2_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# The EmulatedFp64 arm asserts on BOTH lanes via the PRODUCTION emulation_honorable
# predicate (X-6/B2): the predicate reflects steppe_device's own
# STEPPE_HAVE_EMU_TUNING, so the verdict is correct regardless of this target's
# define. Match the library's define anyway so this TU's #if-guarded includes agree
# with steppe_device (both flow from the one cmake option; box build turns it on).
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_equivalence PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()

add_test(NAME f2_equivalence COMMAND test_f2_equivalence)

# --- M1 genotype-decode equivalence test (the M1 trust seam) -----------------
# CUDA TU: reads the raw TGENO triple via the `io` leaf, reproduces derived_acc's
# auto-top-50 × 100k-SNP selection, decodes via CpuBackend + CudaBackend, diffs
# the two against each other and against the numpy oracle, checks the property
# invariants, and runs the decoded Q/V/N through compute_f2 end-to-end. Self-
# checking main() (not a GoogleTest TU); CTest gates on the exit code. Links
# steppe::io (the readers) + steppe::device (both decode backends + compute_f2).
add_executable(test_decode_equivalence reference/test_decode_equivalence.cu)
target_link_libraries(test_decode_equivalence
    PRIVATE steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_decode_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# The decode path itself uses no emulation, but compute_f2 (end-to-end step) runs
# native FP64 here; the EmulatedFp64 tuning is not required for this test. Pass
# the data ROOT (default /workspace/data/aadr) as argv[1] so raw/ + derived_acc/
# resolve from one argument, matching the box layout.
set(STEPPE_AADR_ROOT "/workspace/data/aadr"
    CACHE PATH "AADR data root (contains raw/ TGENO triple + derived_acc/ oracle)")
add_test(NAME decode_equivalence COMMAND test_decode_equivalence "${STEPPE_AADR_ROOT}")

# --- M4 per-block f2 equivalence test (the M4 trust seam) --------------------
# CUDA TU: loads real derived_acc Q/V/N + the .snp prefix via the io leaf, runs
# the SHARED assign_blocks, computes the per-block f2 tensor via the production
# seam (core::compute_f2_blocks) through CpuBackend (per-block long-double oracle)
# + CudaBackend (the size-grouped strided-batched design) at Fp64 and EmuFp64{40},
# and asserts: per-block combined-tolerance tight tier, f2 symmetry, Vpair ==
# independent recount, and single-block == M0 compute_f2. Self-checking main()
# (not a GoogleTest TU); CTest gates on the exit code. Links steppe::io (read_snp)
# + steppe::core (assign_blocks + compute_f2_blocks) + steppe::device (backends).
add_executable(test_f2_blocks_equivalence reference/test_f2_blocks_equivalence.cu)
target_link_libraries(test_f2_blocks_equivalence
    PRIVATE steppe::io steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
# Separable compilation so the executable device-links steppe_device's RDC kernels
# directly (the same path the M0/M1 CUDA tests use). steppe_core carries no device
# link object of its own (CUDA_RESOLVE_DEVICE_SYMBOLS OFF, src/core/CMakeLists.txt),
# so there is exactly one device-link of steppe_device — here.
set_target_properties(test_f2_blocks_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Like test_f2_equivalence: the EmulatedFp64 arm only asserts when the fixed-slice
# Ozaki tuning is compiled in; otherwise it is SKIPPED (a non-assert), so define
# the macro on this target when the option is on (the box build) to assert it.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_blocks_equivalence PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
add_test(NAME f2_blocks_equivalence COMMAND test_f2_blocks_equivalence "${STEPPE_AADR_ROOT}")

# --- B1 emulated-FP64 run-to-run determinism gate ----------------------------
# CUDA TU (mirrors test_f2_blocks_equivalence.cu's data loading): loads the real
# derived_acc Q/V/N + the .snp prefix via the io leaf, runs the SHARED
# assign_blocks, and computes f2 (M0 compute_f2) and the per-block tensor (M4
# compute_f2_blocks) TWICE each through CudaBackend at EmulatedFp64{40}, asserting
# the two runs are BIT-IDENTICAL (exact byte-for-byte equality). This is the §12
# Results-Reproducibility gate (architecture.md §12, §13): it holds ONLY because
# CublasHandle owns the (stream, workspace) invariant and re-applies the
# determinism workspace after the (single, ctor-bound) stream binding — the X-1/B1
# fix. Without it the per-call cublasSetStream reset the workspace (cuBLAS §2.4.7),
# per-chunk on the M4 path, and the emulated-FP64 output was not guaranteed
# bit-stable. Self-checking main() (not a GoogleTest TU); CTest gates on the exit
# code. Links steppe::io (read_snp) + steppe::core (assign_blocks +
# compute_f2_blocks) + steppe::device (the GPU backend).
add_executable(test_f2_determinism reference/test_f2_determinism.cu)
target_link_libraries(test_f2_determinism
    PRIVATE steppe::io steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_determinism PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Engage the fixed-slice Ozaki tuning when the option is on (the box build), so
# the emulated arm actually emulates; the bit-identity assertion holds either way
# (native single-stream is also deterministic), but the box build exercises the
# emulated workspace the B1 fix protects.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_determinism PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
add_test(NAME f2_determinism COMMAND test_f2_determinism "${STEPPE_AADR_ROOT}")

# --- B24 M4 grouped-batch wrapper empty-guard + gather/scatter gate -----------
# CUDA TU, DATA-FREE (sibling of test_f2_empty_guard.cu): drives the three
# device-private M4 wrappers (launch_gather_group / run_f2_gemms_group /
# launch_assemble_blocks_group, f2_blocks_kernel.cuh) DIRECTLY on a tiny synthetic
# feeder-output layout. The B24 verdict gate (cleanup f2_blocks_kernel F6, B24):
#   1. POSITIVE CONTROL — a valid, REORDERED 2-block layout (block_ids_in_group =
#      {1,0}, sizes {2,3} padded to s_pad=4) gathered -> 3 GEMMs -> scattered, with
#      the resident [P x P x n_block] f2 + Vpair tensors diffed BLOCK-FOR-BLOCK
#      against a host oracle built from the SAME shared assemble_f2_numerator /
#      finalize_f2 the GPU calls. This exercises the gather's pad-column zero-fill
#      and the scatter-to-arbitrary-id slab offset (the M4-specific index math; the
#      B25 gather coverage the B24 gate references, folded in so this gate stands
#      alone).
#   2. DEATH CASES (only when STEPPE_ASSERT is active — !NDEBUG; under NDEBUG the
#      guard is compiled out by contract and they are SKIPPED): each wrapper called
#      with a degenerate extent (n_in_group <= 0 — a zero gridDim.z invalid launch;
#      s_pad <= 0 — a zero gridDim.y / k==0 GEMM) must ABORT (SIGABRT from the fired
#      STEPPE_ASSERT) in a forked child, mirroring tests/unit/test_f2_from_blocks.cpp's
#      death harness.
# Self-checking main() (not a GoogleTest TU); CTest gates on the exit code. Links
# steppe::device (the wrappers + the CublasHandle/DeviceBuffer RAII) +
# steppe::core_internal (the shared f2 primitive oracle). No io / no data.
# architecture.md §2 fail-fast, §13; cleanup f2_blocks_kernel F6, B24.
add_executable(test_f2_blocks_group reference/test_f2_blocks_group.cu)
target_link_libraries(test_f2_blocks_group
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_blocks_group PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# The wrappers run native Fp64 here (the gather/scatter index math + the guards are
# precision-neutral), so the fixed-slice Ozaki tuning is not required; match the
# library's define anyway when the option is on (the box build) so any #if-guarded
# include agrees with steppe_device.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_blocks_group PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
add_test(NAME f2_blocks_group COMMAND test_f2_blocks_group)

# --- B25 M4 gather/scatter FOCUSED index-math gate (f2_blocks_kernel F24) ------
# CUDA TU, DATA-FREE (sibling of test_f2_blocks_group.cu): drives the M4 gather +
# scatter wrappers (launch_gather_group / run_f2_gemms_group /
# launch_assemble_blocks_group, f2_blocks_kernel.cuh) on a tiny synthetic REORDERED
# 2-block layout and reads back the INTERMEDIATE buffers to pin the M4-specific
# index math DIRECTLY — coverage that today is only TRANSITIVE (the single-block==M0
# arm of test_f2_blocks_equivalence.cu exercises only the IDENTITY scatter and never
# a pad column; the B24 f2_blocks_group gate diffs only the FINAL assembled tensors,
# inferring the pad/copy contracts rather than asserting them on the slab bytes). The
# B25 verdict gate (cleanup f2_blocks_kernel F24):
#   (i)   PAD-COLUMN ZERO — after launch_gather_group, every pad column c >= sz of
#         dQg/dVg/dSg (both the Qsq and Hc halves of the [2P×s_pad] S slab) is EXACTLY
#         0.0 (bit pattern; the slabs are pre-filled with a nonzero sentinel so the
#         zero-fill is a real kernel write, not zero-init residue).
#   (ii)  GATHERED == FEEDER — every real column c < sz of slab k equals the feeder
#         columns at block_offsets[block_ids_in_group[k]]+c, BIT-FOR-BIT (a pure copy).
#   (iii) SCATTER TO BLOCK-id SLAB — with a reordered block_ids_in_group = {1,0} and
#         two blocks of DIFFERENT size (2 vs 3, hence different f2/Vpair), each resident
#         [P×P×n_block] slab matches ITS OWN block's oracle AND (negative control) NOT
#         the other block's — so a scatter to slab-k instead of block-id (the identity
#         single-block==M0 cannot distinguish) fails.
# Native Fp64 (the gather/scatter index math is precision-neutral), so no emulation
# tuning is required. Self-checking main() (not a GoogleTest TU); CTest gates on the
# exit code. Links steppe::device (the wrappers + the CublasHandle/DeviceBuffer RAII)
# + steppe::core_internal (the shared f2 primitive oracle). No io / no data.
# architecture.md §13; cleanup f2_blocks_kernel F24, B25.
add_executable(test_f2_blocks_gather reference/test_f2_blocks_gather.cu)
target_link_libraries(test_f2_blocks_gather
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_blocks_gather PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Match the library's STEPPE_HAVE_EMU_TUNING define when the option is on (the box
# build) so any #if-guarded include agrees with steppe_device, even though this test
# runs native Fp64.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_blocks_gather PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
add_test(NAME f2_blocks_gather COMMAND test_f2_blocks_gather)

# --- X-6/B2 EmulatedFp64 honorability gate (the dynamic-mantissa-trap close-out)
# CUDA TU, DATA-FREE: drives the production seam (make_cuda_backend → compute_f2)
# on a tiny SYNTHETIC Q/V/N to assert the engaged MODE, not a precision value. It
# pins the ONE honorability predicate (emulation_honorable), that f2_compute_type
# routes through it (so the compute type and math mode can never disagree — the
# C-1/C-2 split close), and end-to-end that an EmulatedFp64 request is EITHER the
# engaged FIXED-slice Ozaki path (tuning ON) OR an OBSERVABLY downgraded native
# Fp64 path with a logged capability tag (tuning OFF) — never silently DYNAMIC
# (architecture.md §9, §12; ROADMAP §0; cleanup X-6/B2). Passes on BOTH lanes.
# Self-checking main() (not a GoogleTest TU); CTest gates on the exit code. Links
# steppe::device only (no io / no data). Includes the device-private kernel header
# for emulation_honorable / f2_compute_type (the src/ root is on steppe_device's
# PUBLIC include dir).
add_executable(test_emu_honorability reference/test_emu_honorability.cu)
target_link_libraries(test_emu_honorability
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_emu_honorability PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Match steppe_device's STEPPE_HAVE_EMU_TUNING so this TU's banner/#if agrees with
# the library (both flow from the one cmake option). The end-to-end arm trusts the
# library's compiled emulation_honorable() regardless, so the test is correct on
# both lanes even if the defines ever diverged.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_emu_honorability PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
add_test(NAME emu_honorability COMMAND test_emu_honorability)

# --- B12 compute_f2 degenerate-extent guard gate (X-7/E-3) -------------------
# CUDA TU, DATA-FREE (mirrors test_emu_honorability.cu): drives the production seam
# (make_cuda_backend -> compute_f2) on degenerate extents to assert the X-7/E-3
# fail-fast guard. The B12 verdict gate: compute_f2 with P<=0 || M<=0 returns an
# EMPTY F2Result CLEANLY (no cuBLAS / CUDA throw from deep in the runtime), matching
# its siblings compute_f2_blocks / decode_af which already early-return; plus a
# positive control that a tiny VALID block still produces a populated [P x P] result
# (the guard did not break the happy path). Self-checking main() (not a GoogleTest
# TU); CTest gates on the exit code. Links steppe::device only (no io / no data).
# architecture.md §2 fail-fast, §13; cleanup X-7/E-3, B12.
add_executable(test_f2_empty_guard reference/test_f2_empty_guard.cu)
target_link_libraries(test_f2_empty_guard
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_empty_guard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME f2_empty_guard COMMAND test_f2_empty_guard)

# --- B22 compute_f2 M>INT_MAX narrowing guard gate (f2_block_kernel E-1) ------
# CUDA TU, DATA-FREE (sibling of test_f2_empty_guard.cu): drives the production seam
# (make_cuda_backend -> compute_f2) with an SNP count M > INT_MAX to assert the B22
# fail-fast guard. The M0 WHOLE-matrix path issues the three f2 GEMMs over ALL M
# SNPs via cublasGemmEx, whose contraction extent `k` is a signed 32-bit int (only
# the cublas*Ex_64 variants take int64_t); MatView::M is `long`, so an unguarded
# static_cast<int>(M) wraps/negates `k` for M > INT_MAX → cuBLAS rejects, or worse
# SILENTLY contracts over fewer SNPs (a wrong-but-plausible f2). The B22 verdict
# gate: compute_f2 with M == INT_MAX+1 THROWS std::runtime_error naming INT_MAX
# (a typed fail-fast, never a silent overflow), the guard is a TIGHT `M > INT_MAX`
# (M == INT_MAX is NOT B22-rejected), and a valid tiny block still computes (the
# guard did not break the happy path). data == nullptr for the over-limit cases:
# the guard fires before any allocation/dereference. Self-checking main() (not a
# GoogleTest TU); CTest gates on the exit code. Links steppe::device only (no io /
# no data). architecture.md §2 fail-fast, §13; cleanup f2_block_kernel E-1, B22.
add_executable(test_f2_int_max_guard reference/test_f2_int_max_guard.cu)
target_link_libraries(test_f2_int_max_guard
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_int_max_guard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME f2_int_max_guard COMMAND test_f2_int_max_guard)

# --- B23 DeviceBuffer<T> n*sizeof(T) overflow guard gate (device_buffer 1.1/7.5)
# CUDA TU, DATA-FREE (sibling of test_f2_int_max_guard.cu): exercises the
# device-private DeviceBuffer<T> RAII owner directly. device_buffer.cuh is a `.cuh`
# (includes <cuda_runtime.h>) — PRIVATE to steppe_device (architecture.md §4) — so
# unlike the host-only unit tests this gate is a CUDA TU. The B23 verdict gate:
# DeviceBuffer<T>(n) with `n` near SIZE_MAX/sizeof(T) must THROW a typed CudaError
# (status cudaErrorMemoryAllocation -> STEPPE_ERR_DEVICE_OOM, §10) rather than
# SILENTLY WRAP the `n * sizeof(T)` size_t multiply and under-allocate (the guard
# fires before cudaMalloc). It also pins the TIGHT inequality (n==SIZE_MAX/sizeof(T)
# is NOT rejected — it fails later on the genuine cudaMalloc, a different DeviceOom),
# the untouched happy path (small alloc is non-null with exact size()/bytes(),
# zero-size is {nullptr,0,0}), and the move-only RAII shape (move-construct nulls
# source, move-assign frees-then-steals, self-move no-op) the §13 compute-sanitizer
# protects. Self-checking main() (not a GoogleTest TU); CTest gates on the exit code.
# Links steppe::device only (no io / no data). architecture.md §2 fail-fast, §11.2,
# §13; cleanup device_buffer 1.1/7.5, B23.
add_executable(test_device_buffer reference/test_device_buffer.cu)
target_link_libraries(test_device_buffer
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_device_buffer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME device_buffer_unit COMMAND test_device_buffer)

# --- Host-only unit test of the shared f2 estimator primitive ----------------
# Pure C++ TU (no CUDA): proves the __host__ __device__ primitive compiles and is
# correct on the host path (STEPPE_HD == empty under g++). architecture.md §13.
add_executable(test_f2_estimator unit/test_f2.cpp)
target_link_libraries(test_f2_estimator
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_f2_estimator PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_f2_estimator PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME f2_estimator_unit COMMAND test_f2_estimator)

# --- Host-only unit test of the shared decode/allele-frequency primitive ------
# Pure C++ TU (no CUDA, no data): proves the __host__ __device__ decode primitives
# in core/internal/decode_af.hpp (genotype_code, genotype_valid, finalize_af)
# compile and are correct on the host path (STEPPE_HD == empty under g++), the
# §13-mandated direct test of the single-source decode primitive (previously only
# covered transitively by the GPU/4GB-data .cu). The B10 verdict gate: finalize_af
# with ploidy=0 -> masked-out {0,0,0} (not NaN/Inf-with-validity), ploidy=1 ->
# haploid (N == 1*AN), ploidy=2 -> diploid (N == 2*AN). architecture.md §13;
# cleanup B10/X-11/T-1. Links only steppe::core_internal (the header-only home).
add_executable(test_decode unit/test_decode.cpp)
target_link_libraries(test_decode
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_decode PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_decode PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME decode_unit COMMAND test_decode)

# --- Host-only unit test of the B7 single-source-home headers ----------------
# Pure C++ TU (no CUDA): pins the launch-math home (core/internal/launch_config.hpp
# — cdiv int+long, grid_for, the warp-justified 32×8 decode block dims) and the
# host/device + debug facility home (core/internal/host_device.hpp — STEPPE_HD,
# STEPPE_DEBUG_ONLY, STEPPE_ASSERT), reached DIRECTLY through their new homes so the
# B7 move out of f2_estimator.hpp is a real single source. architecture.md §8, §13.
add_executable(test_launch_config unit/test_launch_config.cpp)
target_link_libraries(test_launch_config
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_launch_config PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_launch_config PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME launch_config_unit COMMAND test_launch_config)

# --- Host-only unit test of the public config contract (cleanup B9) -----------
# Pure C++ TU (no CUDA, no GPU): includes the CUDA-free include/steppe/config.hpp
# and pins the spec-mandated DeviceConfig defaults — above all that the B9
# `deterministic` field exists, is a bool, and defaults to TRUE (architecture.md
# §9 build()-validation listing, §12), since the §12 stream_count==1 /
# EmulatedFp64-workspace / fixed-order multi-GPU-combine rules the M4.5
# parity-recompute path relies on are inexpressible without it. The checks are
# compile-time static_asserts plus run-time guards; the TU links only steppe::api
# (config.hpp) — its CUDA-free host compile is itself the §4-layering proof.
# architecture.md §4, §9, §12, §13; cleanup include-config 9.1 / B9.
add_executable(test_config unit/test_config.cpp)
target_link_libraries(test_config
    PRIVATE steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_config PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_config PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME config_unit COMMAND test_config)

# --- Host-only unit test of the backend-factory single source (X-9/B8) --------
# Pure C++ TU (no CUDA, no GPU): includes the CUDA-free device/backend_factory.hpp
# and static_asserts that BOTH factories — make_cpu_backend AND make_cuda_backend —
# are declared in the ONE namespace steppe::device with the documented
# std::unique_ptr<ComputeBackend>() signature, reached THROUGH the header (no ad-hoc
# forward declaration). The checks are decltype-only (they never ODR-use the
# factory), so this TU compiles + links WITHOUT steppe::device / its PRIVATE CUDA
# toolkit — which is itself the proof the header is CUDA-free (a CUDA leak would
# fail this host compile). The end-to-end link-through-the-header path is exercised
# by the six reference .cu tests that now reach both factories through this header
# with their forward declarations deleted. architecture.md §4, §8, §13; cleanup X-9/B8.
add_executable(test_backend_factory unit/test_backend_factory.cpp)
target_link_libraries(test_backend_factory
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_backend_factory PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_backend_factory PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME backend_factory_unit COMMAND test_backend_factory)

# --- Host-only unit test of the M4 VRAM budget helper (X-5/B5 + X-13/B26) -----
# Pure C++ TU (no CUDA, no io): exercises device/vram_budget.hpp — the host-pure
# chunk-budget arithmetic extracted out of cuda_backend.cu so the budget is
# unit-testable GPU-free (architecture.md §13). Pins that the budget counts BOTH
# resident f2+vpair tensors (the B26 2× term — the prior path counted one and
# under-budgeted by ~2×), subtracts the cuBLAS workspace before applying
# kMaxVramUtilizationFraction (the X-5 nuance), clamps in size_t before the int
# narrowing (the X-5/F9 wrap), and that one chunk's slabs never exceed the
# fraction-of-net budget. The header is CUDA-free and reached as
# "device/vram_budget.hpp" via steppe::core_internal's src/ include root (the same
# root that resolves "core/internal/..." / "device/backend.hpp"); steppe::api
# carries config.hpp. No CUDA here. architecture.md §8, §11.2, §13.
add_executable(test_vram_budget unit/test_vram_budget.cpp)
target_link_libraries(test_vram_budget
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_vram_budget PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_vram_budget PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME vram_budget_unit COMMAND test_vram_budget)

# --- Host-only unit test of the shared SNP→block partition rule (M3) ----------
# Pure C++ TU (no CUDA, no io): exercises core/domain/block_partition_rule on
# synthetic chromosome/position LAYOUTS (control-flow, not a precision claim —
# ROADMAP §0) and the cM→Morgan conversion site. Links steppe::core for the
# out-of-line assign_blocks (block_partition_rule.cpp). architecture.md §8, §13.
add_executable(test_block_partition unit/test_block_partition.cpp)
target_link_libraries(test_block_partition
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_block_partition PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_block_partition PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME block_partition_unit COMMAND test_block_partition)

# --- Host-only unit test of the block-range INVERSE rule (M4; X-3/B3) ---------
# Pure C++ TU (no CUDA, no io): exercises core::block_ranges — the single-source
# inverse of assign_blocks that BOTH backends now call instead of a hand-rolled
# scan (the X-3 DRY triplication close). Pins the VALID layouts (empty, single,
# multi-block, round-trip vs assign_blocks) and that MALFORMED partitions are
# FAIL-FAST (throw): short/null block_id, out-of-range id, non-monotonic — the OOB
# write/read B3 closes. block_ranges is header-inline (it is reached from BOTH the
# core and device layers only through the header-only steppe::core_internal target,
# so it cannot live out-of-line in block_partition_rule.cpp / steppe_core); linking
# steppe::core also pulls assign_blocks for the round-trip case.
# architecture.md §2 fail-fast, §8, §13.
add_executable(test_block_ranges unit/test_block_ranges.cpp)
target_link_libraries(test_block_ranges
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_block_ranges PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_block_ranges PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME block_ranges_unit COMMAND test_block_ranges)

# Real-AADR INTERNAL-CONSISTENCY check (no synthetic, no statistic): feed the real
# v66 .snp to assign_blocks and assert the measured block count + dense/ordering
# invariants. Path overridable via -DSTEPPE_AADR_SNP=<path>; defaults to the box
# location. The self-checking main() takes the .snp as argv[1]
# (tests/reference/test_f2_equivalence.cu uses the same argv-driven-data pattern).
# Registered only on the non-gtest fallback build, where main() consumes argv[1];
# under gtest the consistency body is compiled out (it lives in the #else arm).
set(STEPPE_AADR_SNP "/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB.snp"
    CACHE FILEPATH "Real AADR .snp file for the block-partition consistency check")
if(NOT STEPPE_HAVE_GTEST)
  add_test(NAME block_partition_aadr_consistency
           COMMAND test_block_partition "${STEPPE_AADR_SNP}")
endif()

# --- Host-only unit test of the f2 composition root (cleanup B11) -------------
# Pure C++ TU (no CUDA, no GPU, no io, no data): drives the f2 orchestration
# (core::compute_f2_block / core::compute_f2_blocks) through a CUDA-FREE recording
# MockBackend — the GPU-free seam test §13 designed the ComputeBackend injection
# point to admit but that did not exist (f2_from_blocks F-12). The B11 verdict
# gate: (1) the orchestration forwards Q/V/N + precision + (M4) block_id.data()/
# n_block to the backend BYTE-FOR-BYTE and returns its result unchanged, and
# (2) a MALFORMED contract — Q/V/N disagreeing on P/M, a short/null block_id, an
# n_block out of range, or a non-decreasing block_id — FAILS FAST (STEPPE_ASSERT
# abort) BEFORE reaching the backend, instead of a silent OOB read/write deep in a
# backend (the death cases are skipped under NDEBUG by contract). Links
# steppe::core for the out-of-line orchestration (f2_from_blocks.cpp); steppe_core
# links steppe::device PRIVATE with CUDA_RESOLVE_DEVICE_SYMBOLS OFF, and this TU
# names only the ABSTRACT ComputeBackend (the mock is its concrete override), so no
# GPU/CUDA symbol is pulled — the CUDA-free host link is itself the §4 proof.
# architecture.md §2 fail-fast, §8 DRY, §13.
add_executable(test_f2_from_blocks unit/test_f2_from_blocks.cpp)
target_link_libraries(test_f2_from_blocks
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_f2_from_blocks PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_f2_from_blocks PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME f2_from_blocks_unit COMMAND test_f2_from_blocks)

# --- Host-only unit test of the SHARED M2 filter predicates (no GPU, no data) -
# Pure C++ TU: exercises the host-pure filter_decision.hpp predicates + the
# include/exclude membership + the --mind per-sample resolution on hand-built
# inputs (control flow / boundaries, NOT a precision claim — ROADMAP §0). Links
# steppe::io (the filter sources live in the io leaf). architecture.md §13.
add_executable(test_filters unit/test_filters.cpp)
target_link_libraries(test_filters
    PRIVATE steppe::io steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_filters PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_filters PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME filters_unit COMMAND test_filters)

# --- Host-only unit test of io::GenoReader (cleanup geno_reader B17) ----------
# Pure C++ TU (no CUDA, no real AADR data): writes tiny SYNTHETIC TGENO .geno
# fixtures to a temp path and exercises GenoReader's happy-path gather plus its
# FAIL-FAST boundary checks. The B17 verdict gate: a HOSTILE / OVERSIZED partition
# (row >= records_present_, or one that would wrap the n_ind*bytes_per_rec size
# multiply) must THROW std::runtime_error rather than silently corrupting the heap
# (pre-fix: a too-small resize + a gather write PAST the allocation). Also pins a
# normal partition decoding correctly, empty-partition reject, SNP-range/partial-
# file edges, and constructor fail-fast. Links steppe::io only (the reader leaf),
# mirroring tests/unit/test_filters.cpp. architecture.md §2 fail-fast, §13.
add_executable(test_geno_reader unit/test_geno_reader.cpp)
target_link_libraries(test_geno_reader
    PRIVATE steppe::io steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_geno_reader PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_geno_reader PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME geno_reader_unit COMMAND test_geno_reader)

# --- Host-only unit test of io::read_snp (cleanup snp_reader B14 + B15) --------
# Pure C++ TU (no CUDA, no real AADR data): writes tiny SYNTHETIC .snp fixtures to
# a temp path and exercises read_snp's TOKEN-COUNT column decision plus its FAIL-
# FAST behavior on malformed records. The B14 verdict gate: a MALFORMED record (too
# few fields, or a non-finite/garbage genpos that std::from_chars rejects) THROWS
# std::runtime_error with the offending 1-based LINE NUMBER rather than silently
# dropping the row (the old extraction-failure fall-through desynced the SNP axis
# from the .geno); a WELL-FORMED file yields the correct chrom codes (numeric pass-
# through + X->23 mapping) and the Morgan genpos as-read. The B15 verdict gate: an
# all-digit chromosome code too large for int (e.g. "99999999999") routes to the
# negative-sentinel path with NO uncaught throw — chrom_code now parses with
# std::from_chars, not std::stoi (whose out_of_range is a logic_error, not the
# runtime_error the header contract promises). Also pins the >=3-but-<6 field
# allele defaulting, max_snps truncation, and the interior-blank-throws /
# trailing-blank-tolerated policy. Links steppe::io only (the reader leaf),
# mirroring tests/unit/test_geno_reader.cpp. architecture.md §2 fail-fast, §12, §13.
add_executable(test_snp_reader unit/test_snp_reader.cpp)
target_link_libraries(test_snp_reader
    PRIVATE steppe::io steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_snp_reader PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_snp_reader PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME snp_reader_unit COMMAND test_snp_reader)

# --- M2 filter ORACLE test (real AADR; the M2 trust seam) --------------------
# CUDA TU (mirrors test_decode_equivalence.cu): reads the raw TGENO triple via the
# io leaf, decodes via the CPU reference backend, builds the filter keep-mask via
# the io filters, and validates it against an INDEPENDENT scalar host oracle on
# real AADR — no-op-when-default, drop-equals-mask bit-identity, exact-mask match,
# --mind exact, monotonicity, autosomes_only block count, strand/tv class. Links
# steppe::io (readers + filters) + steppe::device (CPU decode + compute_f2) +
# steppe::core (assign_blocks for the autosomes tie-in). Self-checking main(); CTest
# gates on the exit code. Takes the data ROOT as argv[1] (default the box layout).
add_executable(test_filter_oracle reference/test_filter_oracle.cu)
target_link_libraries(test_filter_oracle
    PRIVATE steppe::io steppe::device steppe::core steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_filter_oracle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME filter_oracle COMMAND test_filter_oracle "${STEPPE_AADR_ROOT}")
