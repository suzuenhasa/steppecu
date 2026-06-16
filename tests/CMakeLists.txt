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

# --- GPU-vs-CPU equivalence test (the central trust seam) --------------------
# CUDA TU: links steppe::device (CudaBackend + CpuBackend) directly so it can
# diff the two implementations. CUDA::cublas comes transitively-private through
# steppe_device; the test only needs the CUDA-free backend.hpp seam.
#
# This test has its OWN main() and is NOT a GoogleTest-style TU (it self-checks
# and exits non-zero on failure), so it does NOT link gtest_main or define
# STEPPE_TEST_WITH_GTEST — that would collide with its main(). CTest gates on the
# exit code either way (architecture.md §13).
add_executable(test_f2_equivalence reference/test_f2_equivalence.cu)
target_link_libraries(test_f2_equivalence
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# The equivalence test engages the fixed-slice Ozaki tuning (EAGER strategy +
# FIXED mantissa control) under its own STEPPE_HAVE_EMU_TUNING guard, exactly like
# the device layer. Without it the EAGER strategy is absent, cuBLAS declines to
# emulate these tiny GEMMs, the emu output is bit-identical to native, and the
# engagement guard reports the EmulatedFp64 arm SKIPPED (a non-assert). The
# MEASURED precision policy REQUIRES the tuning (ROADMAP §0), so define it on this
# target when the option is on — this is what turns the emu arm from SKIPPED into
# an asserted PASS on the sm_120 / CUDA-13 box.
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
