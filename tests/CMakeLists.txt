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
# CUDA TU: drives the PRODUCTION backends — steppe::core::make_cpu_backend() (the
# long-double oracle) and steppe::device::make_cuda_backend() (the 3-GEMM GPU
# path) — through the CUDA-free ComputeBackend::compute_f2 seam, NOT an inline
# re-implementation (cleanup X-2/B4: the old inline oracle walked j=i+1 and hid the
# M0 diagonal divergence). It diffs the FULL [P × P] matrix INCLUDING the diagonal
# and exits non-zero on any mismatch. Links steppe::core (make_cpu_backend) +
# steppe::device (make_cuda_backend + the device-private kernel header for the X-6/
# B2 emulation_honorable predicate); CUDA::cublas is transitively-private through
# steppe_device.
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
