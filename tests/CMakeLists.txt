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

# --- M-FR-1 transpose+gather+encoding primitive gate (DUAL) -------------------
# CUDA TU: the SNP-major -> canonical individual-major transpose primitive
# (format-readers.md M-FR-1). (A) a SYNTHETIC small SNP-major tile + a KNOWN
# expected individual-major output asserts GPU==CPU==expected (memcmp==0), covering
# the rlen-floor padding + selection/reorder + partial-last-byte edge cases real
# AADR cannot exercise; (B) the convertf-PA (GENO, SNP-major) first-K-SNP x all-inds
# slice transposes byte-identical (memcmp==0) to the raw TGENO (individual-major)
# slice of the SAME data via the existing read_tile — same data two ways, the
# strongest check. Integer/bit-only (no float, no tolerance, no oracle file). The
# REAL part SKIPs exit-0 if the AADR root is absent. Self-checking main() (not a
# GoogleTest TU); CTest gates on the exit code. Links steppe::io (the readers) +
# steppe::device (both transpose backends).
add_executable(test_transpose_canonical reference/test_transpose_canonical.cu)
target_link_libraries(test_transpose_canonical
    PRIVATE steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_transpose_canonical PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME transpose_canonical COMMAND test_transpose_canonical "${STEPPE_AADR_ROOT}")

# --- M-FR-2 Tier-1 PA-reader == TGENO-reader BIT-EXACT gate -------------------
# CUDA TU (format-readers.md §5.1): drives the PRODUCTION dispatch
# core::read_canonical_tile on a GENO (SNP-major PACKEDANCESTRYMAP) prefix
# (converted_pa/v66_HO_pa) AND on the raw TGENO (individual-major) prefix of the
# SAME data, then asserts BYTE-FOR-BYTE: the canonical tile (tail-masked), the
# auto-detected sample_ploidy, and the decoded Q/V/N (memcmp==0) match on BOTH the
# GPU backend and the CPU oracle. convertf is a lossless transcode, so PA and TGENO
# are the SAME genotypes two ways — a correct full PA reader MUST reproduce the
# TGENO decode bit-for-bit. Integer/bit-only (no float, no tolerance, no oracle
# file). SKIPs exit-0 if the AADR root is absent. Self-checking main() (not a
# GoogleTest TU); CTest gates on the exit code. Links steppe::core (the dispatch) +
# steppe::io (the readers) + steppe::device (the transpose + decode backends).
add_executable(test_pa_decode_equivalence reference/test_pa_decode_equivalence.cu)
target_link_libraries(test_pa_decode_equivalence
    PRIVATE steppe::core steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_pa_decode_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME pa_decode_equivalence COMMAND test_pa_decode_equivalence "${STEPPE_AADR_ROOT}")

# --- M-FR-EIG Tier-1 EIGENSTRAT-reader == PA-reader BIT-EXACT gate -------------
# CUDA TU (format-readers.md §5.3, the Tier-3-by-reduction-to-PA gate): drives the
# PRODUCTION dispatch core::read_canonical_tile on an EIGENSTRAT (ASCII SNP-major)
# .geno AND on the convertf-PA (GENO, SNP-major) .geno of the SAME 9-pop subset, then
# asserts BYTE-FOR-BYTE: the canonical tile (NO tail mask — both are SNP-major), the
# auto-detected sample_ploidy, and the decoded Q/V/N (memcmp==0) match on BOTH the GPU
# backend and the CPU oracle. Both fixtures are convertf outputs from the SAME PA source
# with the SAME poplist, so they share identical .ind/.snp and carry the SAME genotypes
# two ways (ASCII vs packed) — a correct EIGENSTRAT reader MUST reproduce the PA decode
# bit-for-bit. Since PA==TGENO==goldens (test_pa_decode_equivalence), this transitively
# pins EIGENSTRAT to the goldens with NO new statistical golden + NO AT2 rerun. Integer/
# bit-only (no float, no tolerance, no oracle file). SKIPs exit-0 if the fixture dir
# (<root>/fixtures_eig) is absent. Self-checking main(); CTest gates on the exit code.
add_executable(test_eigenstrat_decode_equivalence reference/test_eigenstrat_decode_equivalence.cu)
target_link_libraries(test_eigenstrat_decode_equivalence
    PRIVATE steppe::core steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_eigenstrat_decode_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME eigenstrat_decode_equivalence COMMAND test_eigenstrat_decode_equivalence "${STEPPE_AADR_ROOT}")

# --- M-FR PLINK TIER-1: the PLINK (.bed/.bim/.fam) reader == the PA reader, BIT-EXACT --
# CUDA TU: drives the PRODUCTION dispatch (core::read_canonical_tile) on a PLINK .bed AND
# on the convertf-PA .geno of the SAME 9-pop subset (fixtures_eig/v66_fit9_ped vs _pa) and
# asserts BYTE-FOR-BYTE: the canonical tile (NO tail mask — both are SNP-major), the auto-
# detected sample_ploidy, and the decoded Q/V/N (memcmp==0) match on BOTH the GPU backend
# and the CPU oracle. The PLINK reader's LUT (00->2/01->3 missing/10->1/11->0 in A1-copies)
# + the LSB->MSB bit-order flip + ref:=A1 polarity are pinned by this bit-exact compare —
# a wrong LUT/bit-order/polarity diverges immediately (format-readers.md §8.4 highest-risk).
# Since PA==TGENO==goldens (test_pa_decode_equivalence), this transitively pins PLINK to the
# goldens with NO new statistical golden + NO AT2 rerun. SKIPs exit-0 if the fixture dir
# (<root>/fixtures_eig) is absent. Self-checking main(); CTest gates on the exit code.
add_executable(test_plink_decode_equivalence reference/test_plink_decode_equivalence.cu)
target_link_libraries(test_plink_decode_equivalence
    PRIVATE steppe::core steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_plink_decode_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME plink_decode_equivalence COMMAND test_plink_decode_equivalence "${STEPPE_AADR_ROOT}")

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

# --- M4.5 CAP-1/CAP-2 non-throwing tagged-degrade check gate (device-cuda-check)
# CUDA TU, DATA-FREE and NO GPU work (sibling of test_device_buffer.cu): drives the
# device-private check.cuh (a .cuh — PRIVATE to steppe_device, architecture.md §4 —
# so this gate is a CUDA TU) on chosen cudaError_t values, never touching a device.
# The verify gate (cleanup device-cuda-check CAP-1/CAP-2; architecture.md §10, §11.4,
# §12; TODO M4.5): STEPPE_CUDA_WARN on the EXPECTED capability-degrade statuses
# (cudaErrorPeerAccessAlreadyEnabled / cudaErrorPeerAccessNotEnabled) does NOT throw,
# logs one line, and YIELDS the status the multi-GPU probe branches on — while
# STEPPE_CUDA_CHECK on the SAME status DOES throw a CudaError carrying it (the two are
# distinct paths; routing a graceful degrade through the throwing checker would fault
# the budget tier). Also pins the success quiet-pass-through and the source_location
# capture (the A-4 nvcc-fragility regression guard). The runtime's
# cudaGetErrorName/String are host-side string lookups needing no device context, so
# this runs on EVERY lane with no GPU and no data. Self-checking main() (not a
# GoogleTest TU); CTest gates on the exit code. Links steppe::device (check.cuh) +
# steppe::core_internal (the STEPPE_LOG_WARN sink). No io / no data / no GPU.
add_executable(test_cuda_check reference/test_cuda_check.cu)
target_link_libraries(test_cuda_check
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_cuda_check PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME cuda_check_unit COMMAND test_cuda_check)

# --- M4.5 U4/L10 CublasHandle MathModeScope + device-ordinal scaffold gate -----
# CUDA TU, DATA-FREE (sibling of test_device_buffer.cu): exercises the device-
# private handles.hpp RAII cuBLAS-handle wrapper directly. handles.hpp includes
# <cublas_v2.h> — PRIVATE to steppe_device (architecture.md §4) — so this gate is a
# CUDA TU. Unlike test_cuda_check (no GPU), this NEEDS a live device: cublasCreate
# / cublasGetMathMode require a cuBLAS context. The U4/L10 verdict gate (cleanup
# device-cuda-handles 2.3/11.x, L10; overview §(2).1; architecture.md §7, §9,
# §11.4, §12; TODO M4.5):
#   (a) MathModeScope (L10 / f2_block_kernel N-5) captures the live cuBLAS math
#       mode on construction and RESTORES it at scope exit — proven across nesting
#       (inner restores to outer, so the ctor captures the live mode, not a default)
#       and a moved-from scope (which restores NOTHING, so the mode is restored
#       exactly once). This is the M4.5 Fp64-parity-recompute-coexisting-with-
#       EmulatedFp64 scaffold: the §12 oracle pass can engage PEDANTIC for its scope
#       and leave the shared handle exactly as it found it. Parity-NEUTRAL (it only
#       restores state already set imperatively, §12).
#   (b) CublasHandle records the device ordinal current at construction
#       (cudaGetDevice — query only, NEVER cudaSetDevice inside the wrapper; cuBLAS
#       §2.1.2 couples the context to the creation device) and, in DEBUG, the
#       context-mutating methods record-and-ASSERT the current device still matches
#       (the §11.4 multi-GPU precondition; a no-op-true on the single-GPU box).
# Native Fp64 / no numeric claim (it asserts handle STATE, not a precision value),
# so no emulation tuning and no real AADR are required. Self-checking main() (not a
# GoogleTest TU); CTest gates on the exit code. Links steppe::device (handles.hpp +
# check.cuh) + steppe::core_internal (the STEPPE_LOG_WARN sink). No io / no data.
add_executable(test_handles reference/test_handles.cu)
target_link_libraries(test_handles
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_handles PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME handles_unit COMMAND test_handles)

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

# --- M(cli-0) access-layer config-builder unit test (cli-bindings.md §8 M(cli-0)) ----
# Pure C++ TU (no CUDA, no GPU, no data): drives the CUDA-FREE ConfigBuilder/RunConfig/
# CliArgs (src/core/config/) — the §9 precedence merge (compiled defaults < TOML < env
# STEPPE_* < CLI) + the validating build() that maps the raw flags onto the real structs
# (DeviceConfig/QpAdmOptions/FilterConfig/PopSelection) and fail-fasts (InvalidConfig) on
# bad config — plus the Status->exit-code record-and-continue map. The M(cli-0) GATE
# (cli-bindings.md §8): "unit: precedence order + build() rejects ... reuse the
# test_config pattern". It links steppe::core (the config_builder symbol + the io
# PopSelection header path through core's PUBLIC src/ include root) + steppe::api; its
# CUDA-free host compile/link WITHOUT the device layer is itself the §4-layering proof.
# Mirrors test_config's dual GoogleTest / self-checking-main() harness (CTest gates on
# the exit code). architecture.md §4, §9, §10, §13; cli-bindings.md §4.5, §8.
add_executable(test_config_builder unit/test_config_builder.cpp)
target_link_libraries(test_config_builder
    PRIVATE steppe::core steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_config_builder PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_config_builder PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME config_builder_unit COMMAND test_config_builder)

# --- qpAdm parity test (the Phase-2 fit acceptance gate, BOTH backends) ---------
# Reads the committed 9-pop f2 fixture (tests/reference/goldens/at2/fixtures/) — the
# SAME f2 ADMIXTOOLS 2 read internally — and runs the single-model qpAdm fit
# (steppe::run_qpadm, S3->S4->S6->S7) on BOTH backends: the CpuBackend reference
# (M(fit-1): the bit-exact diff oracle + the no-GPU fallback) AND the CUDA backend
# (M(fit-4): the PRODUCTION GPU path — uploads the fixture to VRAM as a
# DeviceF2Blocks, asserts f2 RESIDENT + a real GPU bound, runs the fit on the GPU,
# asserts weights/chisq/dof/p vs the af6a8c2 golden AND vs the CpuBackend oracle to
# 1e-9). Plus the S3/S4 X/Q slice cross-check on both. Self-checking main() (exits
# non-zero on failure; CTest gates on the exit code), so it does NOT link gtest_main.
#
# The GPU block SKIPs cleanly (prints SKIP, still exits via the CpuBackend gate) when
# no CUDA device is visible — so this remains the §4 layering proof that the fit
# reference runs with no GPU, while exercising the real GPU path on box5090. Named .cu
# so it device-links steppe_device's RDC kernels (CUDA_SEPARABLE_COMPILATION) and can
# construct a real CudaBackend. Links steppe::core (run_qpadm) + steppe::device
# (make_cpu_backend / make_cuda_backend + the fit virtuals on both backends); cuSOLVER
# links transitively through steppe_device.
add_executable(test_qpadm_parity reference/test_qpadm_parity.cu)
target_link_libraries(test_qpadm_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpadm_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME qpadm_parity
         COMMAND test_qpadm_parity "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

# --- qpGraph single-graph-fit parity test (the qpGraph acceptance gate, BOTH backends) --
# Reads the committed 9-pop f2 fixture + the WELL-IDENTIFIED golden topology
# (goldens/at2/golden_qpgraph_topology.csv) and runs the single-graph qpGraph fit
# (steppe::run_qpgraph: the topology data model -> the resident f3 basis via
# assemble_f3_triples + jackknife_cov -> the productized IDEA-1 fleet) on BOTH backends:
# the CpuBackend ORACLE (native FP64, the bit-exact diff reference + no-GPU fallback) AND
# the CUDA backend (the PRODUCTION GPU path — f3 basis RESIDENT in VRAM, the fleet on-
# device). Gates score (tight) + the admix weight keyed by parent NAME (pSteppe->aCW) +
# the restart spread (the identifiability witness) vs the admixtools 2.0.10 golden, AND
# CpuBackend == CudaBackend. The GPU block SKIPs cleanly when no device is visible. Named
# .cu so it device-links steppe_device's RDC kernels and can construct a real CudaBackend.
add_executable(test_qpgraph_parity reference/test_qpgraph_parity.cu)
target_link_libraries(test_qpgraph_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpgraph_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME qpgraph_parity
         COMMAND test_qpgraph_parity "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

# --- qpGraph topology ENUMERATOR unit test (the exhaustive-coverage proof, host-only) ---
# Plain C++ TU (NO CUDA): the host enumerator reproduces admixtools generate_all_trees /
# generate_all_graphs EXACTLY — count (105 trees + 1485 non-iso nadmix=1) AND the canonical
# graph_hash set == the committed AT2 fixture (fixtures/at2_enum_5pop.txt). The (2) oracle
# leg: exhaustiveness proven 1:1 against AT2's own enumerator. Also checks every enumerated
# topology parses through parse_qpgraph + the neighbor move set is connected.
add_executable(test_qpgraph_enumerate reference/test_qpgraph_enumerate.cpp)
target_link_libraries(test_qpgraph_enumerate
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
add_test(NAME qpgraph_enumerate
         COMMAND test_qpgraph_enumerate "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

# --- qpGraph topology SEARCH v1 acceptance (oracle C, BOTH backends) ---------------------
# CUDA TU: the exhaustive bounded-space search (run_qpgraph_search) on the committed real-AADR
# 5-pop subset of the f2_fit0_FINAL dir. Gates (1) per-candidate fit == admixtools::qpgraph
# (the AT2 scores of steppe's OWN 1590 enumerated graphs, fixtures/at2_scores_5pop.csv), (2)
# the exhaustive count == AT2 (105/1485), (3) the deterministic global-best argmin == AT2's
# argmin (same canonical hash, CpuBackend == CudaBackend), the heuristic recovers it, and the
# heterogeneous fleet does the fits (GPU-bound). Named .cu so it device-links the RDC kernels.
add_executable(test_qpgraph_search reference/test_qpgraph_search.cu)
target_link_libraries(test_qpgraph_search
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpgraph_search PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME qpgraph_search
         COMMAND test_qpgraph_search "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

# --- qpGraph topology-search PER-CANDIDATE parity (closes the argmin-anchored gap) --------
# CUDA TU: runs the SAME exhaustive search, EXPOSES the full per-candidate scored vector
# (QpGraphSearchResult::candidates), then for EVERY candidate looks steppe's per-candidate score
# up BY THE CANONICAL graph_hash in the trusted oracle goldens/at2/fixtures/at2_scores_5pop.csv
# (1590 rows = AT2 qpgraph() over steppe's OWN enumerated graphs, hash-keyed — steppe's
# enumeration hash IS the file key, so it is a DIRECT lookup, no edge-string re-hash). The tier:
# (a) the global-best + the well-identified (sharp low-score) candidates match AT2 at rtol 1e-6 /
# atol 1e-9; (b) the global-best argmin hash == the AT2 argmin hash; (c) the full 1590 lie within
# the documented ill-conditioning envelope (the flat poorly-identified graphs differ ~0.3-0.5%).
# Additive: the fit/score/enumeration math is unchanged; the oracle is REUSED verbatim (no regen).
# This SUPERSEDES the defective 4-row golden_qpgraph_toposearch_spotcheck.csv (mis-paired). BOTH
# backends. Named .cu so it device-links the RDC kernels.
add_executable(test_qpgraph_search_parity reference/test_qpgraph_search_parity.cu)
target_link_libraries(test_qpgraph_search_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpgraph_search_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME qpgraph_search_parity
         COMMAND test_qpgraph_search_parity "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

# --- qpfstats SMOOTHING-SOLVE CpuBackend==CudaBackend parity (the solve primitive) -------
# CUDA TU: a numerical-equivalence unit test of the qpfstats_smooth solve primitive (A_shared
# = x'x + ridge·I, Cholesky, the multi-column Dtrsm solve, the NaN-row downdate, all-NaN→b=0)
# over a deterministic small design/RHS exercising EVERY branch (no-NaN, partial-NaN, all-NaN
# blocks). CpuBackend (small_linalg LU oracle) == CudaBackend (cublasDsyrk + potrf + gemm +
# Dtrsm pair + downdate) to ~1e-9. The GPU block SKIPs cleanly when no device is visible.
add_executable(test_qpfstats_smooth_parity reference/test_qpfstats_smooth_parity.cu)
target_link_libraries(test_qpfstats_smooth_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpfstats_smooth_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME qpfstats_smooth_parity COMMAND test_qpfstats_smooth_parity)

# --- qpfstats FUSED reduce→jackknife→smooth→recenter parity (the PERF-path gate) ----
# Directly diffs the ON-DEVICE block-jackknife kernels (qpfstats_jackknife_kernel.cu; the
# per-comb y/ymat + the per-pair recenter, native FP64 §12 carve-out) against the long-double
# CpuBackend oracle on a synthetic genotype field, at the golden tier (rtol 1e-6). SKIPs/fails
# fast with no GPU. The real-AADR end-to-end statistic is gated by cli_qpfstats.
add_executable(test_qpfstats_fused_parity reference/test_qpfstats_fused_parity.cu)
target_link_libraries(test_qpfstats_fused_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpfstats_fused_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME qpfstats_fused_parity COMMAND test_qpfstats_fused_parity)

# --- DATES cuFFT LD-engine parity test (the within-codebase FFT-engine gate) -------
# CUDA TU: drives BOTH production backends and asserts the CudaBackend cuFFT autocorrelation
# path (ComputeBackend::dates_curve) matches the CpuBackend FFT-FREE direct-autocorrelation
# oracle on the moment sufficient statistics that drive the corr curve — Σ_g W[g]·W[g+lag] ==
# IFFT(|FFT(W)|²)/n, the within-codebase pin for the GPU FFT engine (the external DATES date
# golden is cli_dates / py_dates). CONSTRUCTED small inputs (a within-codebase numerical-
# equivalence test, NOT a reported result). SKIPs/fails fast with no GPU.
add_executable(test_dates_parity reference/test_dates_parity.cu)
target_link_libraries(test_dates_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_dates_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME dates_parity COMMAND test_dates_parity)

# --- GPU-ONLY f-stat SWEEP parity test (the sweep acceptance gate) ----------------
# CUDA TU: uploads the REAL 9-pop golden f2 fixture RESIDENT to VRAM, runs the GPU-only
# run_f4_sweep / run_f3_sweep (on-device combinatorial unrank -> the SAME assemble gather +
# diagonal jackknife the explicit run_f4/run_f3 use -> on-device |z| filter -> CUB compaction),
# and asserts: every sweep survivor's est/se/z is BIT-EXACT to the explicit run_f4/run_f3 over
# that item (proves NO compute fork); the |z|>=3 filter keeps exactly the explicit |z|>=3 items;
# and an over-cap sweep REFUSES (capped, no compute) without --sure. SKIPs cleanly if no GPU.
add_executable(test_fstat_sweep_parity reference/test_fstat_sweep_parity.cu)
target_link_libraries(test_fstat_sweep_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_fstat_sweep_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME fstat_sweep_parity
         COMMAND test_fstat_sweep_parity "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

# --- M(fit-6) S8 qpAdm ROTATION test (the model-space search acceptance gate) ----
# CUDA TU: reads the REAL AT2 rotation golden (goldens/at2/golden_rot.json) + the
# co-matching f2 fixture (fixtures/f2_rot.bin), uploads f2 RESIDENT to VRAM, and runs
# the BATCHED, MULTI-GPU-SHARDED rotation steppe::run_qpadm_search on the CudaBackend
# (NOT the CpuBackend). It gates: per-model parity vs the golden (weights rtol 1e-6,
# p/feasible loose/decision, f4rank exact) for the validated 84-model set; G=1 vs G=2
# bit-identical + identically-ordered determinism; no regression on the 9-pop + NRBIG
# single-model goldens via a 1-element search; and REPORTS the 1-GPU + 2-GPU
# throughput (models/sec) at the validated N and a synthetic scale-N (timing only, NO
# per-model golden). The GPU blocks SKIP cleanly if <1 / <2 devices visible (box5090
# has 2). Self-checking main() (exits non-zero on failure; CTest gates on the exit
# code). Named .cu so it device-links steppe_device's RDC kernels and constructs a
# real CudaBackend. Links steppe::core (run_qpadm_search) + steppe::device.
add_executable(test_qpadm_rotation reference/test_qpadm_rotation.cu)
target_link_libraries(test_qpadm_rotation
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpadm_rotation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME qpadm_rotation
         COMMAND test_qpadm_rotation "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

# --- M(fit-5) qpAdm DOMAIN-OUTCOME test (the contract acceptance gate, BOTH backends)
# CUDA TU: the M(fit-5) acceptance gate (fit-engine.md §2 M(fit-5); architecture.md
# §13:803; §18 DoD:939) — a deliberately DEGENERATE model returns a per-model
# DOMAIN-OUTCOME STATUS VALUE (RankDeficient / NonSpdCovariance / ChisqUndefined),
# never a crash / exception / silently-Ok NaN, on BOTH the CpuBackend ORACLE and the
# production CudaBackend (f2 RESIDENT in VRAM). Every other qpadm status assertion
# checks `== Status::Ok`; this covers the NON-Ok taxonomy the contract requires.
#
# NO SYNTHETIC DATA: every degenerate model is built from the SAME committed REAL-AADR
# 9-pop f2 fixture (fixtures/f2_fit0_9pop.bin) the parity gate uses, arranged
# degenerate by selecting REAL pops in a degenerate configuration (a left set with a
# duplicated real source ⇒ RankDeficient / fudge=0 non-SPD; an over-parameterized
# dof<=0 model ⇒ ChisqUndefined) — never a made-up matrix. The CpuBackend oracle block
# runs under STEPPE_THOROUGH=1 OR when no GPU is visible (the CI-without-GPU gate); the
# CudaBackend deliverable block runs whenever a CUDA device is visible (default FAST).
# Self-checking main() (exits non-zero on failure; CTest gates on the exit code), so it
# does NOT link gtest_main. Named .cu so it device-links steppe_device's RDC kernels and
# can construct a real CudaBackend. Links steppe::core (run_qpadm) + steppe::device
# (make_cpu_backend / make_cuda_backend + the fit virtuals on both backends).
# architecture.md §10, §13, §18; fit-engine.md §2 M(fit-5); fit-engine-finish-punchlist F2.
add_executable(test_qpadm_domain reference/test_qpadm_domain.cu)
target_link_libraries(test_qpadm_domain
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpadm_domain PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME qpadm_domain
         COMMAND test_qpadm_domain "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

# CUDA TU: the F1 / OQ-12 MISSING-BLOCK acceptance gate (fit-engine.md OQ-12). EVERY other
# qpAdm golden is maxmiss=0 (global SNP intersection ⇒ NO missing blocks); this is the ONE
# that exercises a MISSING block — a REAL-AADR f2 set extracted with maxmiss>0 in which one
# jackknife block has a population pair with Vpair==0. AT2 read_f2(remove_na=TRUE) DROPS that
# block ENTIRELY before the jackknife (NOT impute-0, which biases f4 toward 0 + inflates
# variance); steppe reproduces the drop on BOTH backends. Gates weights/chisq/dof/p/rankdrop +
# the f4 X over survivors vs a pinned REAL AT2 golden (golden_fitNA.json; admixtools 2.0.10 /
# R 4.3.3; maxmiss=0.99) AND CpuBackend==CudaBackend. Without the drop the X/Q differ by ~1e-3
# (above the tight tier) ⇒ the test fails the buggy impute-0 path.
#
# NO SYNTHETIC DATA: the missing block arises from a REAL sparse ancient pop
# (Afghanistan_DarraiKurCave_MBA, a single low-coverage individual) as a right pop; the fixture
# fixtures/f2_fitNA.bin is the RAW (pre-drop) AT2 f2 tensor with the NA pair stored as NaN.
# CpuBackend oracle runs under STEPPE_THOROUGH=1 or no GPU; CudaBackend (f2 RESIDENT) on default.
add_executable(test_qpadm_missing_block reference/test_qpadm_missing_block.cu)
target_link_libraries(test_qpadm_missing_block
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpadm_missing_block PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME qpadm_missing_block
         COMMAND test_qpadm_missing_block "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

# CUDA TU: the F4 standalone qpWave acceptance gate (fit-engine-finish-punchlist F4). The
# first-class PUBLIC run_qpwave entry (include/steppe/qpadm.hpp; impl qpadm_fit.cpp
# run_qpwave_impl) had ZERO direct test coverage — its distinguishing no-target-prepend /
# left[0]-is-reference semantic was unvalidated. This calls run_qpwave DIRECTLY on BOTH the
# CpuBackend ORACLE and the production CudaBackend (f2 RESIDENT in VRAM) and gates est_rank
# / f4rank / the rankdrop table / chisq vs a pinned REAL AT2 qpwave() golden
# (golden_qpwave.json, admixtools 2.0.10 / R 4.3.3, boot=FALSE).
#
# NO SYNTHETIC DATA: both qpWave models REUSE the committed REAL-AADR 9-pop f2 fixture
# (fixtures/f2_fit0_9pop.bin) arranged as qpWave left/right splits over REAL pops (M1
# 3-left est_rank=1; M2 2-left clade est_rank=0) — NO new fixture, NO made-up matrix. The
# CpuBackend oracle block runs under STEPPE_THOROUGH=1 OR when no GPU is visible (the
# CI-without-GPU gate); the CudaBackend deliverable block runs whenever a CUDA device is
# visible (default FAST). Self-checking main() (exits non-zero on failure), so it does NOT
# link gtest_main. Named .cu so it device-links steppe_device's RDC kernels and can
# construct a real CudaBackend. fit-engine.md §4.1 (qpWave tier); fit-engine-finish-punchlist F4.
add_executable(test_qpwave_parity reference/test_qpwave_parity.cu)
target_link_libraries(test_qpwave_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpwave_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME qpwave_parity
         COMMAND test_qpwave_parity "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

# --- M(fit-6) S8 model-search shard planner unit test (GPU-free) -----------------
# Pure C++ TU (no CUDA, no GPU, no io, no data): drives
# steppe::core::qpadm::plan_model_shards (the model->device shard plan) + the
# pre-sized-slot re-sort discipline in isolation — the GPU-free coverage gate for the
# determinism mechanism run_qpadm_search relies on (the proven test_shard_plan /
# test_f2_blocks_multigpu pattern). Links steppe::core (the plan_model_shards
# out-of-line symbol) + steppe::core_internal (the src/ include root resolving
# "core/qpadm/model_search_core.hpp") + steppe::api. Self-checking main().
add_executable(test_model_search_core unit/test_model_search_core.cpp)
target_link_libraries(test_model_search_core
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
add_test(NAME model_search_core_unit COMMAND test_model_search_core)

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

# --- Host-only unit test of the M4.5 BackendCapabilities POD (U2) -------------
# Pure C++ TU (no CUDA, no GPU): includes the CUDA-free device/backend.hpp and
# pins the SHAPE + value-initialized "unknown" defaults of the BackendCapabilities
# POD the M4.5 capability-tier scaffold adds (architecture.md §9 Resources, §11.4
# SPMG capability-tiered combine, §12 parity; cleanup 00-overview §(2).1). The U2
# verdict gate: BackendCapabilities is an aggregate / trivially-copyable /
# standard-layout POD whose every field defaults to the documented zero/false
# ("unknown" tier), AND ComputeBackend::capabilities() has a usable base default
# returning that unknown (so CpuBackend need not override it and backend.hpp
# compiles standalone). The REAL probe values are asserted in U5, not here. The
# checks are compile-time static_asserts plus run-time guards. Like
# test_backend_factory, this links ONLY steppe::core_internal (which exposes the
# src/ include root resolving "device/backend.hpp") + steppe::api — NOT
# steppe::device / its PRIVATE CUDA toolkit; a CUDA leak in the capability struct
# would fail this host compile, which is itself the §4-layering proof.
# architecture.md §4, §9, §11.4, §12, §13.
add_executable(test_backend_capabilities unit/test_backend_capabilities.cpp)
target_link_libraries(test_backend_capabilities
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_backend_capabilities PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_backend_capabilities PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME backend_capabilities_unit COMMAND test_backend_capabilities)

# --- M4.5 U5 CUDA capabilities() probe + device_id threading gate -------------
# CUDA TU, DATA-FREE (sibling of test_f2_empty_guard.cu): drives the PRODUCTION
# CUDA path — make_cuda_backend(device_id) + CudaBackend::capabilities() — on a live
# device. The U2 host-only gate (tests/unit/test_backend_capabilities.cpp) pins the
# BackendCapabilities POD SHAPE + the value-initialized "unknown" base default; THIS
# gate pins the REAL probe VALUES the CUDA override reports, plus the device_id
# binding (F19/F20). The U5 verdict gate (cleanup device-cuda-cuda_backend F19/F20 +
# X-6/B2; architecture.md §9, §11.4, §12; 00-overview §(2).1):
#   TIER-INVARIANT (every lane): compute_major >= 12 (sm_120), total_vram_bytes > 0
#     (the captured `total` the M0/M4 path discarded), free <= total, device_count
#     matches cudaGetDeviceCount, emulated_fp64_honorable == the LIBRARY's X-6/B2
#     emulation_honorable predicate (the probe + GEMM path can never disagree), and a
#     backend constructs on device 0 reporting itself.
#   rtxbox VERDICT (2× RTX PRO 6000): device_count == 2, a backend constructs on
#     device 1 too, and can_access_peer == true (the REAL stock-driver P2P this box
#     has — MEASURED 55.6 GB/s; architecture.md §11.4 capability-tier law). The
#     strict peer assertion is KEYED to the tier (device name contains "PRO"): on the
#     budget GeForce box can_access_peer == false is the EXPECTED tagged degrade to
#     the host-staged combine baseline (a PASS, not a fault), so the ONE sm_120 build
#     serves BOTH boxes (the §0 / ⚡ box-role split). Every field is parity-NEUTRAL
#     (§12) — the gate asserts the PROBE, never a reported number.
# Self-checking main() (not a GoogleTest TU); CTest gates on the exit code. Links
# steppe::device (the factory + capabilities()) + steppe::core_internal (the
# STEPPE_LOG_WARN / STEPPE_CUDA_WARN sink) + the device-private f2_block_kernel.cuh
# for emulation_honorable. No io / no data. architecture.md §9, §11.4, §12, §13.
add_executable(test_backend_capabilities_probe reference/test_backend_capabilities_probe.cu)
target_link_libraries(test_backend_capabilities_probe
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_backend_capabilities_probe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# emulated_fp64_honorable reflects steppe_device's own STEPPE_HAVE_EMU_TUNING; the
# end-to-end check trusts the LIBRARY's compiled emulation_honorable() regardless of
# this target's define. Match the library's define anyway (both flow from the one
# cmake option; the box build turns it on) so this TU's #if-guarded include agrees.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_backend_capabilities_probe PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
add_test(NAME backend_capabilities_probe COMMAND test_backend_capabilities_probe)

# --- M4.5 UNIT I1 build_resources / Resources / PerGpuResources gate ----------
# CUDA TU, DATA-FREE (sibling of test_backend_capabilities_probe.cu): drives the
# M4.5 Resources HOME + builder — steppe::device::build_resources — on live
# devices. The U5 gate above pins one CudaBackend's capabilities() probe; THIS
# gate pins that build_resources BINDS one backend PER device in
# DeviceConfig::devices order (the per-device-instance contract, backend.hpp;
# the fixed g=0..G-1 combine order, architecture.md §11.4/§12) and stores each
# device's ONE probe in PerGpuResources::caps (out-of-band, cleanup §(2).1/§(2).2).
# The I1 verdict gate (architecture.md §9 Resources, §11.4 SPMG, §12 parity):
#   TIER-INVARIANT (every lane): devices={0} -> exactly 1 entry bound to device 0
#     with compute_major >= 12 and total_vram > 0; devices={} (auto) -> all visible
#     ordinals 0..count-1 in order (the §9 DeviceConfig::devices auto-enumerate
#     contract).
#   rtxbox VERDICT (2x RTX PRO 6000, count >= 2): devices={0,1} -> exactly 2 entries
#     bound to devices 0 and 1 IN ORDER, each compute_major >= 12, and — on the PRO
#     tier (device name contains "PRO") — each can_access_peer == true (the REAL
#     stock-driver P2P; architecture.md §11.4 capability-tier law). On a budget
#     GeForce box can_access_peer == false is the EXPECTED tagged degrade to the
#     host-staged combine baseline (a PASS, not a fault), so the strict peer
#     assertion is KEYED to the tier and the 2-device build still succeeds — the
#     ONE sm_120 build serves BOTH boxes (the §0 / ⚡ box-role split). NO real f2
#     compute; every checked field is parity-NEUTRAL (§12) — the gate asserts the
#     device BINDING + the PROBE, never a reported number.
# Self-checking main() (not a GoogleTest TU); CTest gates on the exit code. Links
# steppe::device (build_resources + the factory + capabilities()) + steppe::core_internal
# (the src/ include root resolving "device/resources.hpp") + steppe::api (config.hpp).
# No io / no data. architecture.md §9, §11.4, §12, §13.
add_executable(test_resources_build reference/test_resources_build.cu)
target_link_libraries(test_resources_build
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_resources_build PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# resources.cpp's build_resources reaches the GPU only through the CUDA-free factory
# + capabilities(); no STEPPE_HAVE_EMU_TUNING-gated symbol is touched here, so no
# emulation-tuning define is needed for this TU.
add_test(NAME resources_build COMMAND test_resources_build)

# --- M4.5 MULTI-GPU PARITY GATE (the §12 PARITY LAW; THE verify gate) ----------
# CUDA TU (mirrors test_f2_blocks_equivalence.cu's data loading): loads real
# derived_acc (P=50, fast) AND derived_full (P=768, scale) Q/V/N + the .snp prefix
# via the io leaf, runs the SHARED assign_blocks, and asserts the MULTI-GPU combined
# f2_blocks AND Vpair are BIT-IDENTICAL (exact std::memcmp / ==, NOT a tolerance) to
# the single-GPU reference (core::compute_f2_blocks over the full inputs), across G:
#   (1) G==1 (devices={0}) multi-GPU == single-GPU reference  (the bit-identity floor)
#   (2) G==2 (devices={0,1}, prefer_p2p_combine=false) host-staged == single-GPU
#       reference  (THE core parity claim, architecture.md §11.4 SPMG host-side
#       fixed-order combine, §12 — bit-identical across G AND to single-GPU; NEVER an
#       NCCL AllReduce).
#   (3) G==2 (devices={0,1}, prefer_p2p_combine=true) P2P device-resident
#       (cudaMemcpyPeer) combine == single-GPU reference  (the OPT-IN fast-path moves
#       the same bytes / sums the same fixed order ⇒ identical; runs on the PRO tier).
#   (4) G==2 P2P == G==2 host-staged  (the two-tier NEUTRALITY: both combine tiers
#       bit-identical to each other — the explicit workflow ask).
# Bit-identity holds because the sharding is BLOCK-ALIGNED (each block computed
# entirely on one device from its own contiguous SNP columns ⇒ same slab bits as
# single-GPU) and BOTH combine tiers sum in fixed g=0..G-1 order onto a zero tensor
# (design §0/§3/§4). Strict bit-identity is asserted for EmulatedFp64{40} (the f2
# production default; (4) is strict for both precisions); native Fp64 at G>=2 is the
# documented batched-GEMM shape-sensitivity cell (oracle-tolerance, §12). On a
# single-GPU box the G==2 lane is skipped (logged); an absent dataset directory is
# skipped so the fast gate runs without derived_full. The out-of-band which-path tag
# (Resources::last_combine_path) is asserted: prefer_p2p_combine=false ⇒ HostStaged on
# every tier; on the PRO tier (rtxbox) can_access_peer==true AND the P2P candidate's
# tag == P2pDeviceResident (confirms (3) actually exercised cudaMemcpyPeer, not a
# silent fallback). Self-checking main() (not a GoogleTest TU); CTest gates on the
# exit code. Links steppe::io (read_snp) + steppe::core (assign_blocks +
# compute_f2_blocks + the multi-GPU entry point + the host-staged combine) +
# steppe::device (build_resources + the per-device backends + plan_block_shards + the
# device-resident P2P combine). architecture.md §11.4, §12, §13; design §6.
add_executable(test_f2_multigpu_parity reference/test_f2_multigpu_parity.cu)
target_link_libraries(test_f2_multigpu_parity
    PRIVATE steppe::io steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
# Separable compilation so the executable device-links steppe_device's RDC kernels
# directly (the same path the M4 / U5 reference tests use). steppe_core carries no
# device link object of its own (CUDA_RESOLVE_DEVICE_SYMBOLS OFF, src/core/CMakeLists.txt),
# so there is exactly one device-link of steppe_device — here.
set_target_properties(test_f2_multigpu_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# The parity assertion is BIT-IDENTITY (single-GPU vs sharded, SAME precision), which
# holds on BOTH lanes; match steppe_device's STEPPE_HAVE_EMU_TUNING when the option is
# on (the box build) so the EmulatedFp64 arm exercises the FIXED-slice Ozaki path that
# the per-device compute_f2_blocks engages (it must produce identical bits sharded vs
# not), and so this TU's #if-guarded includes agree with the library.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_multigpu_parity PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
add_test(NAME f2_multigpu_parity COMMAND test_f2_multigpu_parity "${STEPPE_AADR_ROOT}")

# --- M4.5 SPMG single-GPU vs 2-GPU wall-clock BENCH (cleanup B1 fan-out) -------------
# Manual bench (NOT a ctest gate): times core::compute_f2_blocks_multigpu G==1 vs G==2
# end-to-end on real AADR; reports the speedup and catches the single-GPU OOM (large-P
# only fits when sharded on a 32 GB card). Mirrors the parity target's links + emu def.
add_executable(bench_f2_multigpu reference/bench_f2_multigpu.cu)
target_link_libraries(bench_f2_multigpu
    PRIVATE steppe::io steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(bench_f2_multigpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(bench_f2_multigpu PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()

# Manual bench (NOT a ctest gate): times the REAL batched run_qpadm_search rotation
# (CudaBackend::fit_models_batched, f2 RESIDENT in VRAM) SINGLE-GPU over a REAL 1240K
# f2_blocks dir — the honest rotation throughput the CLI-loop ~1/sec sweep mis-measured.
# Mirrors the rotation target's links. Named .cu to device-link steppe_device's kernels.
add_executable(bench_rotation_1240k reference/bench_rotation_1240k.cu)
target_link_libraries(bench_rotation_1240k
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(bench_rotation_1240k PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- f-stats (f4/f3/f4-ratio) SINGLE-GPU batched throughput bench (manual, not a gate) --
add_executable(bench_fstats_1240k reference/bench_fstats_1240k.cu)
target_link_libraries(bench_fstats_1240k
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(bench_fstats_1240k PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- GPU OPTIMIZER SPIKE bench (manual, NOT a ctest gate) ---------------------------
# Compares the two candidate qpGraph GPU optimizer shapes — IDEA1 batched-sequential
# fleet (per-instance bounded projected-Newton/L-BFGS) vs IDEA2 batched-population
# (per-instance CMA-ES/DE, curand-sampled) — on a REAL-f2-grounded nonlinear box-
# constrained graph-GLS objective built from the committed 9-pop AADR fixture. BOTH
# optimizers run entirely on-device (one batched launch each); the bench pins the
# checkable optimum on the host oracle (small_linalg) and reports wall-clock /
# obj-evals / convergence-quality / dispatch count at increasing N (the GPU-bound
# proof). Self-contained: the spike-local fill_pwts + both optimizer kernels live IN
# this TU (NOT the production qpGraph build). Links steppe::core_internal for the
# host-pure small_linalg oracle + its src/ include root, and CUDA::curand for the
# IDEA2 device-side Gaussian sampling (NEW link — no prior cuRAND usage). docs/research
# qpgraph-gpu-design.md / new-tools-roadmap.md §6.
add_executable(bench_optimizers reference/bench_optimizers.cu)
target_link_libraries(bench_optimizers
    PRIVATE steppe::core_internal steppe::api steppe::warnings CUDA::curand)
set_target_properties(bench_optimizers PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- M4.5 SPMG combine GATE-LOGIC test (cleanup B3 / CT2 / config C-1/K-2; B4 / X6) --
# HOST-PURE-LOGIC TU: drives the §4 capability-tier fork in
# core::compute_f2_blocks_multigpu through a CUDA-FREE recording FakeBackend over a
# hand-built Resources (no real GPU compute, no io, no real peer access). It asserts
# the FULL combine-gate truth table for the FOUR-term §4 gate — the workflow's REQUIRED
# objective gate is that enable_peer_access=false forces last_combine_path==HostStaged
# EVEN IF can_access_peer were true (the row the OLD two-term gate got wrong, which
# would have selected P2P and called the forbidden cudaDeviceEnablePeerAccess). It also
# pins: prefer && enable && can_access && G>=2 is the ONLY combo that selects P2P;
# prefer=false or no-peer degrade to HostStaged; the G==1 fast path never runs the gate
# (tag stays None); and (B4/X6) the `&& G >= 2` term is spelled in the predicate and
# bites at G==1 — the production code now MATCHES the four-term predicate documented
# across the seam (it was documented 4-term while shipping 3-term). Parity-NEUTRAL: it
# asserts WHICH transport the gate picks, never a reported number, so the
# f2_multigpu_parity bit-identity is untouched (the G>=2 term changes no reached path).
#
# It is a .cu (not a host .cpp) because compute_f2_blocks_multigpu references the
# device-layer symbols plan_block_shards (host) + combine_f2_partials_p2p (CUDA, RDC),
# so the executable must LINK + device-link steppe::device to RESOLVE them
# (steppe::core links steppe::device PRIVATE; the P2P symbol is resolved, never called
# — every driven row resolves to the host-staged baseline). Same .cu +
# CUDA_SEPARABLE_COMPILATION pattern as test_resources_build.cu. Links steppe::core
# (the entry point + host-staged combine) + steppe::device (resolve the referenced
# device symbols) + steppe::core_internal (the src/ include root) + steppe::api
# (config.hpp). No io / no data / no STEPPE_HAVE_EMU_TUNING (no emulation symbol
# touched). architecture.md §11.4, §12, §13; cleanup B3 / CT2.
add_executable(test_f2_multigpu_gate reference/test_f2_multigpu_gate.cu)
target_link_libraries(test_f2_multigpu_gate
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_multigpu_gate PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME f2_multigpu_gate COMMAND test_f2_multigpu_gate)

# --- Host-only unit test of the SHARED f2-combine validator (cleanup B5 / X7) -
# Pure C++ TU (no CUDA, no GPU, no io, no data): exercises the new single-home guard
# core::validate_f2_partials (src/core/fstats/f2_partials_validate.hpp) — the ONE
# validator BOTH f2 combine tiers (host-staged core::combine_f2_partials_host AND
# device-resident device::combine_f2_partials_p2p) call, so the two tiers reject
# malformed inputs IDENTICALLY (their parity-neutrality, architecture.md §11.4/§12).
# The guard was DUPLICATED byte-for-byte across the two TUs (kept "in lock-step" by a
# comment); B5 single-homes it AND closes the shared short-partial OOB gap (no
# validator checked part.f2/vpair/block_sizes.size() == P*P*n_block, so a partial with
# the right scalar n_block/P but an under-sized vector read out of bounds). The B5
# verdict gate: every VALID case passes (single/two-shard tiling, empty/trailing-empty
# degenerates, the n_block_full==0 corner) and EVERY malformed case FAILS FAST (throws
# std::runtime_error) — count mismatch, negative P/n_block_full, n_block != shard span,
# P disagreement, the SHORT-PARTIAL storage gap (f2/vpair/block_sizes under-sized), and
# shards that do not tile [0, n_block_full). validate_f2_partials is header-only INLINE
# and CUDA-free, reached as "core/fstats/f2_partials_validate.hpp" via
# steppe::core_internal's src/ include root (the same root that resolves
# "device/shard_plan.hpp"); steppe::api carries the public steppe/fstats.hpp. NO CUDA /
# no out-of-line symbol, so this links ONLY steppe::core_internal + steppe::api — a
# CUDA leak in the shared guard would fail this host compile (the §4-layering proof).
# architecture.md §2 fail-fast, §4, §8, §13; cleanup B5 / X7 / C1.
add_executable(test_f2_partials_validate unit/test_f2_partials_validate.cpp)
target_link_libraries(test_f2_partials_validate
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_f2_partials_validate PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_f2_partials_validate PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME f2_partials_validate_unit COMMAND test_f2_partials_validate)

# --- Host-only unit test of the HOST-STAGED f2 combine (cleanup B7 / f2_combine T1)
# Pure C++ TU (no CUDA, no GPU, no io, no data): exercises the PORTABLE PARITY
# BASELINE core::combine_f2_partials_host (src/core/fstats/f2_combine.cpp) — the
# host-staged fixed-order combine that is the ONLY combine path on the budget box
# (no peer access) and the bit-identity reference the device-resident P2P sibling
# must match (architecture.md §11.4, §12). The unit was covered ONLY transitively
# through the GPU-required parity test (test_f2_multigpu_parity.cu); this realizes
# the header's own "unit-testable host-only" promise on the cheap host path.
#
# THE FIX UNDER TEST (B7 / f2_combine N2+P2): the combine PLACES each device's
# compact partial at its block offset via a contiguous std::copy_n (memcpy-grade),
# replacing the prior scalar `+=`-onto-+0.0 triple loop. The placement is STRICTLY
# more faithful to single-GPU: a copy reproduces a −0.0 partial element byte-for-
# byte, where the `+=` flipped it to +0.0 ((+0.0)+(−0.0)==+0.0). The verdict gate:
# (1) two disjoint shards place over the full cover BIT-EXACT (f2/vpair/block_sizes);
# (2) the −0.0 CASE — a −0.0 partial element lands as −0.0 (this case FAILS the old
# `+=` body and PASSES the std::copy_n body, the N2 hardening proof); (3) the G==1
# single-full-shard floor; (4) the degenerates (all-empty -> empty tensor, trailing
# empty shard skipped); (5) fail-fast through the shared validate_f2_partials guard.
# Bit-level placement + control flow + a bit-pattern fidelity claim, NOT a precision
# statistic (ROADMAP §0).
#
# combine_f2_partials_host is out-of-line in steppe_core (f2_combine.cpp); steppe_core
# links steppe::device PRIVATE with CUDA_RESOLVE_DEVICE_SYMBOLS OFF, and this TU names
# only the CUDA-free F2BlockTensor + DeviceShard, so no GPU/CUDA symbol is pulled —
# the CUDA-free host link is itself the §4-layering proof. Reached as
# "core/fstats/f2_combine.hpp" / "device/shard_plan.hpp" via steppe::core_internal's
# src/ include root; steppe::api carries the public steppe/fstats.hpp. Dual harness
# (gtest when present, else self-checking main()); CTest gates on the exit code.
# architecture.md §11.4, §12, §13; cleanup B7 / f2_combine N2/P2/T1.
add_executable(test_f2_combine unit/test_f2_combine.cpp)
target_link_libraries(test_f2_combine
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_f2_combine PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_f2_combine PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME f2_combine_unit COMMAND test_f2_combine)

# --- Host-only unit test of the SPMG block-aligned shard PLANNER (cleanup TST-1) -
# Pure C++ TU (no CUDA, no GPU, no io, no data): exercises the host-pure deterministic
# planner steppe::device::plan_block_shards (src/device/shard_plan.{hpp,cpp}) — the
# block→device mapping that is the STRUCTURAL parity floor (whole-block-on-one-device)
# under BOTH combine tiers (architecture.md §11.4, §12). The unit was covered ONLY
# transitively through the GPU parity test (test_f2_multigpu_parity.cu), which on
# near-uniform AADR blocks can never hit the skew/edge cases; this test realizes the
# header's own "exercisable host-only" promise.
#
# THE FIX UNDER TEST (B6 / X1 / shard_plan D-1): plan_block_shards dropped its redundant
# `block_sizes` parameter and now derives each block's SNP count from `ranges[b].size()`
# (a `long`; the single source of both balance + each shard's [s0,s1)). The plan MUST
# stay BYTE-IDENTICAL to the old (block_sizes, ranges, G) signature — each case asserts
# plan == an independent greedy reference + the documented exact layout, so a divergence
# (which would change the SPMG combine layout and break the locked f2_multigpu_parity
# bit-identity) fails fast here on the cheap host path, before the slow GPU run.
#
# plan_block_shards is host-pure CUDA-FREE (it names only core::BlockRange + std types,
# its TU shard_plan.cpp is a plain .cpp in steppe_device that references no CUDA symbol),
# so this links the out-of-line symbol from steppe::device + the CUDA-free seam via
# steppe::core_internal (the src/ include root that resolves "device/shard_plan.hpp" +
# "core/domain/block_partition_rule.hpp"); steppe::api carries config.hpp. Only
# shard_plan.cpp.o is pulled (no kernel object), so no GPU/CUDA code links — a CUDA leak
# in the planner would fail this host link (the §4 layering proof). Self-checking main()
# (no gtest); CTest gates on the exit code. architecture.md §8, §11.4, §12, §13.
add_executable(test_shard_plan unit/test_shard_plan.cpp)
target_link_libraries(test_shard_plan
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
add_test(NAME shard_plan_unit COMMAND test_shard_plan)

# --- Host-only unit test of the SPMG ORCHESTRATOR CORE (cleanup D1/T1, B9) -----
# Pure C++ TU (no CUDA, no GPU, no io, no data): exercises the HOST-PURE, P2P-FREE
# core of the multi-GPU precompute orchestrator — core::plan_multigpu_shards (the
# block-aligned shard plan) + core::compute_multigpu_partials (the per-device
# concurrent std::jthread fan-out) — factored out of the public entry point
# compute_f2_blocks_multigpu into src/core/fstats/f2_blocks_multigpu_core.{hpp,cpp}
# (the audit's D1 extraction that unlocks the T1 host test). The orchestrator's host
# logic — the zero-copy column sub-view, the dense zero-based local block_id
# transform, the empty-/n_block<G-shard handling, the exception_ptr rethrow — was
# exercised ONLY end-to-end on a GPU (test_f2_multigpu_parity.cu) or through the §4
# gate .cu (test_f2_multigpu_gate.cu, which must device-link steppe::device for the
# RDC P2P symbol). This realizes the §13 "GPU-free-testable seam" promise on the cheap
# host path. The B9/T1 verdict gate:
#   (1) SUB-VIEW + LOCAL-ID — each device g is handed EXACTLY its shard's columns
#       (Q.data + P*s0, M == s1-s0) and a DENSE ZERO-BASED local block_id
#       (block_id[s0+k] - b0), recorded + asserted by a fake backend;
#   (2) FAN-OUT + COMBINE — the G partials, keyed on the GLOBAL block id, recombine via
#       the host-staged fixed-order combine into the full tensor BIT-FOR-BIT vs an
#       independent reference (driven over G=1/2/4 incl. an uneven 8-block/4-device
#       case — the concurrent fan-out writing 4 distinct partials[g] slots, also
#       TSan-checkable for race-freedom);
#   (3) G==1 IDENTITY — the single full shard with local==global ids;
#   (4) EMPTY / n_block<G — trailing empty shards place nothing, prefix still exact;
#   (5) EXCEPTION PROPAGATION — a faulting worker surfaces as a normal throw (the
#       exception_ptr rethrow), never std::terminate from an escaped thread exception.
# Parity-NEUTRAL: it asserts the host-pure transform + placement, never a number — the
# locked f2_multigpu_parity bit-identity is untouched (the extraction is a pure refactor).
#
# WHY THIS LINKS NO CUDA / NEEDS NO device-link. The extracted core references ONLY the
# CUDA-free ComputeBackend seam, plan_block_shards (a plain .cpp symbol in steppe_device,
# the same one test_shard_plan resolves), block_ranges (header-inline), and the
# host-staged combine_f2_partials_host (steppe_core) — NEVER combine_f2_partials_p2p
# (the device-RDC symbol the public entry point references). So this TU links
# steppe::core (the core TU + the host-staged combine) + steppe::device (resolve the
# plain plan_block_shards.o — NO kernel object pulled, NO CUDA_SEPARABLE_COMPILATION) +
# steppe::core_internal (the src/ include root) + steppe::api (config.hpp). A CUDA leak
# in the extracted core would fail this host link (the §4 layering proof). Self-checking
# main() (no gtest); CTest gates on the exit code. architecture.md §2, §4, §7, §11.4,
# §12, §13; cleanup D1/T1, B9.
add_executable(test_f2_blocks_multigpu unit/test_f2_blocks_multigpu.cpp)
target_link_libraries(test_f2_blocks_multigpu
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
add_test(NAME f2_blocks_multigpu_unit COMMAND test_f2_blocks_multigpu)

# --- Host-only unit test of the §9 device-ordinal validator (cleanup B8/C1/T1) -
# Pure C++ TU (no CUDA, no GPU, no io, no data): exercises the §9 build()-validation
# predicate steppe::device::validate_device_order(span<const int>, int) the B8
# cleanup factored out of build_resources — reject DUPLICATE or OUT-OF-RANGE device
# ordinals fail-fast (architecture.md §9 "Validation at build() rejects ... any
# device id in DeviceConfig::devices that is absent or duplicated", §2). Before B8
# the duplicate case ({0,0}) was NOT validated at all (it silently ran two combine
# lanes on ONE GPU), and the validation was entangled with a real device probe so
# its only coverage was a GPU `.cu` — the §13 "exercisable GPU-free" duty unmet. The
# B8/C1/T1 verdict gate: every g=0..G-1 order build_resources produces PASSES; every
# duplicate ({0,0}, {0,1,0}) and every out-of-range (>=visible, negative, device on a
# zero-visible box) THROWS std::runtime_error; out-of-range is checked BEFORE the
# `seen` duplicate index (no OOB). Parity-NEUTRAL: it asserts INPUT rejection, never a
# reached path or a number, so the locked f2_multigpu_parity bit-identity is untouched.
#
# validate_device_order is out-of-line in steppe_device (resources.cpp) and is itself
# CUDA-FREE host arithmetic over (order, visible) — it makes NO CUDA call and touches
# no device. This TU links steppe::device to RESOLVE the out-of-line symbol (like
# test_shard_plan does for plan_block_shards) but calls only validate_device_order, so
# it needs NO GPU at run time. resources.hpp is reached as "device/resources.hpp" via
# the src/ root on steppe::device's PUBLIC include dir; steppe::api carries config.hpp.
# Dual harness (gtest when present, else self-checking main()); CTest gates on the exit
# code. architecture.md §2 fail-fast, §9, §11.4, §12, §13; cleanup device-resources
# C1/T1, B8.
add_executable(test_validate_device_order unit/test_validate_device_order.cpp)
target_link_libraries(test_validate_device_order
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_validate_device_order PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_validate_device_order PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
add_test(NAME validate_device_order_unit COMMAND test_validate_device_order)

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

# --- REGIME-B on-device keep-set PARITY gate (extract_f2 FULL filter on the GPU) --
# CUDA TU (mirrors test_decode_equivalence.cu / test_filter_oracle.cu): the central
# gate for the extract_f2 regime-B filter moved on-device. Decodes the SAME real v66
# AADR tile, runs CudaBackend::decode_af_compact_filter (the regime-B keep-mask kernel
# + the N lockstep CUB compaction) and the verbatim HOST filter oracle, and asserts
# the on-device keep-SET + the lockstep-compacted Q/V/N are BIT-IDENTICAL to the host
# (M_kept exact, chrom/genpos bit-identical, Q/V/N memcmp). Spans the FMA-immune
# integer filters (transversions / maxmiss / monomorphic / autosome) AND the FP-fragile
# pooled-MAF boundary (the only piece the __dmul_rn/__dadd_rn FFMA-immune pin defends).
# SKIPs cleanly (exit 0) with no CUDA device / no raw AADR. Links steppe::io (readers +
# filters) + steppe::device (the CUDA + CPU backends). Self-checking main(); CTest gates
# on the exit code. Takes the data ROOT as argv[1] (default the box layout).
add_executable(test_extract_f2_regimeB_parity reference/test_extract_f2_regimeB_parity.cu)
target_link_libraries(test_extract_f2_regimeB_parity
    PRIVATE steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_extract_f2_regimeB_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME extract_f2_regimeB_parity COMMAND test_extract_f2_regimeB_parity "${STEPPE_AADR_ROOT}")

# --- M(cli-1) `steppe qpadm` CLI acceptance gate (cli-bindings.md §7, §8) ---------
# PLAIN C++ host TU (NO CUDA): writes the COMMITTED real-AADR golden f2 fixtures into a
# temp f2-dir (f2.bin STPF2BK1 + pops.txt + meta.json) — exactly what a real <dir>
# contains, NO SYNTHETIC DATA — then RUNS the BUILT `steppe qpadm` binary for the golden
# model (target=England_BellBeaker, left=Czechia_EBA_CordedWare,Turkey_N, right=the 6
# outgroups), parses its CSV + JSON output, and asserts weights/se/z/p/chisq/dof/
# rankdrop/popdrop == golden_fit0.json + golden_fit1_NRBIG (the large nr=39 case) within
# the SAME two tiers the fit parity test uses — RUNNING ON THE GPU (the steppe qpadm GPU
# path). It SKIPs cleanly (exit 0) when no CUDA device is visible (CI-without-GPU), like
# every reference test. The name->index resolution the parity test HARDCODES is here
# exercised through the CLI's pops.txt — the gap the access layer closes (cli-bindings.md
# §1). Only built behind STEPPE_BUILD_CLI (the CLI is additive; the default ctest is
# unchanged). It links ONLY steppe::core_internal (the src/ include root resolving the
# CUDA-FREE "device/f2_disk_format.hpp" header — the STPF2BK1 writer it shares with the
# app reader) + steppe::api — NOT steppe::device, so this is a pure host TU that device-
# links nothing (its CUDA-free host compile is itself a §4-layering proof); the GPU work
# is in the spawned child. Depends on steppe_app so the binary exists; the binary path +
# the golden dir are passed as argv. cli-bindings.md §4.3, §7, §8.
if(STEPPE_BUILD_CLI)
  add_executable(test_cli_qpadm cli/test_cli_qpadm.cpp)
  target_link_libraries(test_cli_qpadm
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_qpadm steppe_app)
  add_test(NAME cli_qpadm
           COMMAND test_cli_qpadm
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

  # --- M(cli-3) `steppe qpadm-rotate` golden gate (cli-bindings.md §7, §538) ----------
  # PLAIN C++ host TU (NO CUDA): runs the BUILT `steppe qpadm-rotate` over the COMMITTED
  # real-AADR rotation fixture (fixtures/f2_rot.bin) with golden_rot.json's EXACT inputs
  # (target/pool/right, --min-sources 2 --max-sources 3 = 84 models) and diffs every
  # per-model row (weights/p/f4rank/feasible/left/status) against golden_rot.json at the
  # SAME tiers the engine gate test_qpadm_rotation.cu uses. The CLI counterpart of the
  # engine-level qpadm_rotation test. SKIPs cleanly (exit 0) when no CUDA device is
  # visible. Links the CUDA-FREE f2_disk_format header (via core_internal) + the standard
  # host warnings; the GPU work is in the spawned child. Depends on steppe_app so the
  # binary exists; the binary path + the golden dir are argv.
  add_executable(test_cli_rotate cli/test_cli_rotate.cpp)
  target_link_libraries(test_cli_rotate
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_rotate steppe_app)
  add_test(NAME cli_rotate
           COMMAND test_cli_rotate
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

  # --- M(cli-2) `steppe qpwave` golden gate (cli-bindings.md §7, §537) -----------------
  # PLAIN C++ host TU (NO CUDA): runs the BUILT `steppe qpwave` over the COMMITTED real-AADR
  # 9-pop f2 fixture (fixtures/f2_fit0_9pop.bin) with golden_qpwave's EXACT inputs (NO
  # --target — the distinguishing qpWave invocation; left[0]=reference) and diffs the
  # rank-sweep output (est_rank/f4rank/rankdrop chisq/dof/p + the per_rank cross-check)
  # against the pinned f2-object-path constants test_qpwave_parity.cu gates (NO AT2 re-run),
  # at the SAME tiers the engine gate uses. The CLI counterpart of the engine-level
  # qpwave_parity test. SKIPs cleanly (exit 0) when no CUDA device is visible. Links the
  # CUDA-FREE f2_disk_format header (via core_internal) + the standard host warnings; the
  # GPU work is in the spawned child. Depends on steppe_app so the binary exists; the binary
  # path + the golden dir are argv.
  add_executable(test_cli_qpwave cli/test_cli_qpwave.cpp)
  target_link_libraries(test_cli_qpwave
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_qpwave steppe_app)
  add_test(NAME cli_qpwave
           COMMAND test_cli_qpwave
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

  # --- `steppe f4` standalone-f4 golden gate (fit-engine §6) ---------------------------
  # PLAIN C++ host TU (NO CUDA): runs the BUILT `steppe f4` over the COMMITTED real-AADR
  # 9-pop f2 fixture (fixtures/f2_fit0_9pop.bin) for ALL 60 quartets of the REGENERATED,
  # fixture-matched golden (csv/golden_fit0_f4_readf2.csv) and diffs the est/se/z/p table
  # row-for-row at rtol 1e-6 — RUNNING ON THE GPU. Also exercises the single-quartet --pops
  # convenience. The CLI counterpart of the standalone-f4 entry. SKIPs cleanly (exit 0)
  # when no CUDA device is visible. Links the CUDA-FREE f2_disk_format header (via
  # core_internal) + the standard host warnings; the GPU work is in the spawned child.
  # Depends on steppe_app so the binary exists; the binary path + the golden dir are argv.
  add_executable(test_cli_f4 cli/test_cli_f4.cpp)
  target_link_libraries(test_cli_f4
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_f4 steppe_app)
  add_test(NAME cli_f4
           COMMAND test_cli_f4
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

  # --- `steppe qpdstat` qpDstat Part-A golden gate (D-stat/f4 over an f2-dir) -----------
  # PLAIN C++ host TU (NO CUDA): the qpDstat-Part-A clone of the cli_f4 gate. Runs the BUILT
  # `steppe qpdstat` over the COMMITTED real-AADR 9-pop f2 fixture (fixtures/f2_fit0_9pop.bin)
  # for ALL 60 quadruples of the fixture-matched golden (csv/golden_fit0_qpdstat_readf2.csv)
  # and diffs the est/se/z/p table row-for-row at rtol 1e-6 — RUNNING ON THE GPU. The qpdstat
  # f2-path == f4 (f4mode is a no-op without per-SNP genotypes), so est is byte-identical to
  # the f4 golden. Also exercises the single-quadruple --pops convenience AND the --prefix
  # (Part B) fail-fast guard. SKIPs cleanly (exit 0) when no CUDA device is visible.
  add_executable(test_cli_qpdstat cli/test_cli_qpdstat.cpp)
  target_link_libraries(test_cli_qpdstat
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_qpdstat steppe_app)
  add_test(NAME cli_qpdstat
           COMMAND test_cli_qpdstat
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

  # --- `steppe qpdstat --prefix` qpDstat Part-B golden gate (genotype-path NORMALIZED-D) --
  # PLAIN C++ host TU (NO CUDA): runs the BUILT `steppe qpdstat --prefix
  # <STEPPE_AADR_ROOT>/raw/v66.p1_HO.aadr.patch.PUB` (the RAW TGENO prefix steppe reads —
  # --prefix is TGENO-only — the lossless transcode of the convertf-PA v66_HO_pa AT2 made the
  # golden from) over ALL 60 quadruples of the genotype-path golden
  # (csv/golden_fit0_dstat_geno.csv) and diffs the est/se/z table row-for-row at rtol
  # 1e-6 — RUNNING ON THE GPU through the per-SNP D kernel + num/den block-jackknife (the S2
  # divergence; NEVER the f2 cache). The est is the NORMALIZED D (~±0.06), distinct from the
  # f2-path f4 (~10x smaller). AT2 reads only the needed individuals via indvec (steppe reads
  # only the 9-pop union), so no physical subset is needed. The test takes only <steppe-binary>
  # <aadr-root> <golden-dir> and derives the raw TGENO prefix internally. SKIPs cleanly (exit 0)
  # when no CUDA device is visible OR the TGENO prefix / golden is absent. Also exercises the
  # single-quadruple --pops convenience. NO SYNTHETIC DATA (memory real-data-only).
  add_executable(test_cli_dstat_geno cli/test_cli_dstat_geno.cpp)
  target_link_libraries(test_cli_dstat_geno PRIVATE steppe::warnings)
  add_dependencies(test_cli_dstat_geno steppe_app)
  add_test(NAME cli_dstat_geno
           COMMAND test_cli_dstat_geno
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

  # --- `steppe dates` DATES admixture-dating golden gate --------------------------------
  # PLAIN C++ host TU (NO CUDA): runs the BUILT `steppe dates --prefix
  # <STEPPE_AADR_ROOT>/raw/v66.p1_HO.aadr.patch.PUB --target PUR --left CEU,YRI` and asserts
  # the inferred admixture date (generations) reproduces the DATES reference golden (9.742 gen,
  # SE 0.317; tests/reference/goldens/dates/aadr_PUR_CEU_YRI) — RUNNING ON THE GPU through the
  # cuFFT autocorrelation LD engine (NEVER the f2 cache, NEVER a host O(M²) SNP-pair loop). The
  # date is gated at rtol 2% (the raw TGENO's 584131 SNPs vs the reference's convertf-filtered
  # 579720 perturb it marginally), the SE at atol 0.10. SKIPs cleanly (exit 0) when no CUDA
  # device is visible OR the raw AADR is absent. NO SYNTHETIC DATA (memory real-data-only).
  add_executable(test_cli_dates cli/test_cli_dates.cpp)
  target_link_libraries(test_cli_dates PRIVATE steppe::warnings)
  add_dependencies(test_cli_dates steppe_app)
  add_test(NAME cli_dates
           COMMAND test_cli_dates
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}")

  # --- `steppe qpfstats` genotype-path smoothed-f2-TENSOR golden gate --------------------
  # PLAIN C++ host TU (NO CUDA): runs the BUILT `steppe qpfstats --prefix
  # <STEPPE_AADR_ROOT>/raw/v66.p1_HO.aadr.patch.PUB --pops <9 sorted fit0> --out-dir <tmp>`,
  # reads the smoothed f2.bin (STPF2BK1), and diffs every (i,j,block) entry vs the AT2
  # smoothed-f2 tensor golden (csv/golden_qpfstats_geno.csv; 9×9×711 upper-tri incl diag) at
  # rtol 1e-6 — RUNNING ON THE GPU through the qpDstat-B numerator engine over the full
  # f2/f3/f4 popcomb set + the on-device shared-factor smoothing solve (syrk + potrf + gemm +
  # Dtrsm pair). The all-NaN block (536) is matched as 0. SKIPs cleanly (exit 0) when no CUDA
  # device is visible OR the prefix / golden is absent. NO SYNTHETIC DATA.
  add_executable(test_cli_qpfstats cli/test_cli_qpfstats.cpp)
  target_link_libraries(test_cli_qpfstats PRIVATE steppe::warnings)
  add_dependencies(test_cli_qpfstats steppe_app)
  add_test(NAME cli_qpfstats
           COMMAND test_cli_qpfstats
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

  # --- `steppe f3` standalone-f3 golden gate (fit-engine §6) ---------------------------
  # PLAIN C++ host TU (NO CUDA): the THREE-slab clone of the cli_f4 gate. Runs the BUILT
  # `steppe f3` over the COMMITTED real-AADR 9-pop f2 fixture (fixtures/f2_fit0_9pop.bin) for
  # ALL 36 triples of the fixture-matched golden (csv/golden_fit0_f3_readf2.csv) and diffs
  # the est/se/z/p table row-for-row at rtol 1e-6 — RUNNING ON THE GPU. Also exercises the
  # single-triple --pops convenience. SKIPs cleanly (exit 0) when no CUDA device is visible.
  add_executable(test_cli_f3 cli/test_cli_f3.cpp)
  target_link_libraries(test_cli_f3
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_f3 steppe_app)
  add_test(NAME cli_f3
           COMMAND test_cli_f3
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

  # --- `steppe qpgraph` single-graph-fit golden gate -----------------------------------
  # PLAIN C++ host TU (NO CUDA): runs the BUILT `steppe qpgraph` over the COMMITTED real-AADR
  # afprod=FALSE 9-pop f2 fixture (fixtures/f2_qpgraph_9pop.bin) for the WELL-IDENTIFIED golden
  # topology (golden_qpgraph_topology.csv) and asserts the CSV `summary` score + the `edges`
  # admix weight (pSteppe->aCW) match the admixtools 2.0.10 golden — RUNNING ON THE GPU. SKIPs
  # cleanly (exit 0) when no CUDA device is visible.
  add_executable(test_cli_qpgraph cli/test_cli_qpgraph.cpp)
  target_link_libraries(test_cli_qpgraph
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_qpgraph steppe_app)
  add_test(NAME cli_qpgraph
           COMMAND test_cli_qpgraph
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

  # --- `steppe f4-ratio` standalone-f4-ratio golden gate (fit-engine §6) ---------------
  # PLAIN C++ host TU (NO CUDA): the FIVE-column clone of the cli_f4/cli_f3 gate. Runs the
  # BUILT `steppe f4-ratio` over the COMMITTED real-AADR 9-pop f2 fixture
  # (fixtures/f2_fit0_9pop.bin) for ALL 10 5-tuples of the fixture-matched golden
  # (csv/golden_fit0_f4ratio_readf2.csv) and diffs the alpha/se/z table row-for-row at rtol
  # 1e-6 — RUNNING ON THE GPU. Also exercises the single-tuple --pops convenience. SKIPs
  # cleanly (exit 0) when no CUDA device is visible.
  add_executable(test_cli_f4ratio cli/test_cli_f4ratio.cpp)
  target_link_libraries(test_cli_f4ratio
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_f4ratio steppe_app)
  add_test(NAME cli_f4ratio
           COMMAND test_cli_f4ratio
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

  # --- M(cli-4) `steppe extract-f2 -> qpadm` end-to-end gate (cli-bindings.md §7, §8) -
  # PLAIN C++ host TU (NO CUDA): runs the BUILT `steppe extract-f2` on the RAW v66 AADR
  # triple (STEPPE_AADR_ROOT/raw/<prefix>.{geno,snp,ind}) for the golden pop sets, then
  # `steppe qpadm --f2-dir <extracted>` and asserts the result reproduces the committed
  # AT2 goldens (golden_fit0 maxmiss=0 + golden_fitNA maxmiss=0.99) within the established
  # tiers — RUNNING ON THE GPU. It proves the NEW M(cli-4) pieces end-to-end: the
  # io->decode->filter->assign_blocks->precompute chain AND the STPF2BK1 dir WRITER (with
  # REAL vpair — golden_fitNA's dropped block is reproduced ONLY because vpair is real,
  # not zeros). SKIPs cleanly (exit 0) when no CUDA device is visible OR the raw AADR is
  # absent, like f2_blocks_equivalence (the STEPPE_AADR_ROOT raw/ convention). Links ONLY
  # the standard host warnings (it spawns the binary; no library symbols needed). Depends
  # on steppe_app so the binary exists; the binary path + the AADR root + the golden dir
  # are argv. NO SYNTHETIC DATA (memory real-data-only-all-results). cli-bindings.md §4, §7.
  add_executable(test_cli_extract_qpadm cli/test_cli_extract_qpadm.cpp)
  target_link_libraries(test_cli_extract_qpadm PRIVATE steppe::warnings)
  add_dependencies(test_cli_extract_qpadm steppe_app)
  add_test(NAME cli_extract_qpadm
           COMMAND test_cli_extract_qpadm
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2")

  # --- M-FR-2 PA-prefix CLI AUTO-COVER gate (format-readers.md §5.2) -------------------
  # PLAIN C++ host TU (NO CUDA): runs the BUILT steppe TWICE per tool — once with the GENO
  # (SNP-major PACKEDANCESTRYMAP) --prefix <converted_pa/v66_HO_pa> (the prefix that USED TO
  # THROW at geno_reader.cpp:81) and once with the raw TGENO --prefix — and asserts the PA
  # run WORKS (exit 0) AND matches the TGENO run: extract-f2's f2.bin BYTE-IDENTICAL, qpdstat
  # est/se/z row-for-row (rtol 1e-6), qpfstats smoothed f2.bin byte-identical. convertf is a
  # lossless transcode so the two prefixes are the SAME data two ways; a correct PA reader
  # MUST reproduce the TGENO result. NO new golden (the comparison IS PA-vs-TGENO). RUNNING
  # ON THE GPU. SKIPs cleanly (exit 0) when no CUDA device is visible OR the converted_pa PA
  # prefix / raw TGENO prefix is absent. NO SYNTHETIC DATA (memory real-data-only).
  add_executable(test_cli_pa_prefix cli/test_cli_pa_prefix.cpp)
  target_link_libraries(test_cli_pa_prefix PRIVATE steppe::warnings)
  add_dependencies(test_cli_pa_prefix steppe_app)
  add_test(NAME cli_pa_prefix
           COMMAND test_cli_pa_prefix
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}")

  # --- M-FR-EIG EIGENSTRAT-prefix CLI AUTO-COVER gate (format-readers.md §5.3) ---------
  # PLAIN C++ host TU (NO CUDA): runs the BUILT steppe TWICE per tool — once with the
  # EIGENSTRAT (ASCII SNP-major) --prefix <fixtures_eig/v66_fit9_es> (the prefix that USED
  # TO THROW at the GenoReader ctor) and once with the convertf-PA (GENO) --prefix of the
  # SAME 9-pop subset — and asserts the EIGENSTRAT run WORKS (exit 0) AND matches the PA
  # run: extract-f2's f2.bin BYTE-IDENTICAL, qpdstat est/se/z row-for-row (rtol 1e-6). Both
  # fixtures are convertf outputs from the SAME PA source with the SAME poplist (identical
  # .ind/.snp), so they are the SAME genotypes two ways; a correct EIGENSTRAT reader MUST
  # reproduce the PA result. NO new golden (the comparison IS ES-vs-PA, and PA==TGENO==
  # goldens via cli_pa_prefix). RUNNING ON THE GPU. SKIPs cleanly (exit 0) when no CUDA
  # device is visible OR the fixtures_eig prefixes are absent. NO SYNTHETIC DATA.
  add_executable(test_cli_eigenstrat_prefix cli/test_cli_eigenstrat_prefix.cpp)
  target_link_libraries(test_cli_eigenstrat_prefix PRIVATE steppe::warnings)
  add_dependencies(test_cli_eigenstrat_prefix steppe_app)
  add_test(NAME cli_eigenstrat_prefix
           COMMAND test_cli_eigenstrat_prefix
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}")

  # --- M-FR PLINK PLINK-prefix CLI AUTO-COVER gate (format-readers.md §5.3) ------------
  # PLAIN C++ host TU (NO CUDA): runs the BUILT steppe TWICE per tool — once with the PLINK
  # (.bed/.bim/.fam, SNP-major 2-bit LSB-first) --prefix <fixtures_eig/v66_fit9_ped> (the
  # prefix that USED TO THROW at the GenoReader ctor) and once with the convertf-PA (GENO)
  # --prefix of the SAME 9-pop subset — and asserts the PLINK run WORKS (exit 0) AND matches
  # the PA run: extract-f2's f2.bin BYTE-IDENTICAL, qpdstat est/se/z row-for-row (rtol 1e-6).
  # Both fixtures are convertf outputs from the SAME PA source with the SAME poplist, so they
  # are the SAME genotypes two ways; a correct PLINK reader (LUT + bit-flip + ref:=A1) MUST
  # reproduce the PA result. This ALSO exercises the format-aware --prefix expansion (the
  # PLINK prefix has no .geno -> resolve_genotype_triple picks .bed/.bim/.fam) + the
  # read_bim/read_fam dispatch. NO new golden (the comparison IS PLINK-vs-PA, and PA==TGENO==
  # goldens via cli_pa_prefix). RUNNING ON THE GPU. SKIPs cleanly (exit 0) when no CUDA device
  # is visible OR the fixtures_eig prefixes are absent. NO SYNTHETIC DATA.
  add_executable(test_cli_plink_prefix cli/test_cli_plink_prefix.cpp)
  target_link_libraries(test_cli_plink_prefix PRIVATE steppe::warnings)
  add_dependencies(test_cli_plink_prefix steppe_app)
  add_test(NAME cli_plink_prefix
           COMMAND test_cli_plink_prefix
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}")
endif()

# --- M(py-1) nanobind Python gate (cli-bindings.md §7, §8) ----------------------------
# Reproduce the real-AADR golden_fit0 THROUGH the Python API (read_f2 -> qpadm -> the
# pandas-shaped result) + the f2 numpy round-trip + a qpwave/qpadm_search sanity, gated by
# STEPPE_BUILD_PYTHON. SKIPs cleanly (the pytest device-present guard) when no CUDA device
# is visible — identical to the CLI test wiring. STEPPE_BINDINGS_DIR points the conftest
# sys.path shim at the BUILD-tree-staged steppe package (bindings/steppe/_core*.so +
# __init__.py). Depends on _core so the extension is built before the test runs. NO new
# data: it stages the COMMITTED fixtures into a temp f2-dir (memory real-data-only).
if(STEPPE_BUILD_PYTHON)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter)
  add_test(NAME py_qpadm
           COMMAND "${Python_EXECUTABLE}" -m pytest -v
                   "${CMAKE_CURRENT_SOURCE_DIR}/python")
  # STEPPE_AADR_ROOT is passed so test_py_dstat_geno.py finds the raw TGENO prefix steppe
  # reads (<root>/raw/v66.p1_HO.aadr.patch.PUB — the lossless transcode of the convertf-PA
  # v66_HO_pa that AT2 generated the genotype-path D golden from); it SKIPs cleanly when that
  # prefix is absent (the genotype-path data-absent guard).
  set_tests_properties(py_qpadm PROPERTIES
      ENVIRONMENT "STEPPE_BINDINGS_DIR=${CMAKE_BINARY_DIR}/bindings;STEPPE_AADR_ROOT=${STEPPE_AADR_ROOT}")
  # The CTest target depends on the extension via the build graph; CTest itself does not
  # build, so the `cmake --build` step (which builds _core) must precede `ctest`.
endif()
