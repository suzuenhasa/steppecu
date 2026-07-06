# tests/CMakeLists.txt
#
# Builds and registers the steppe test suite: GPU/CPU reference-parity tests
# (CUDA), host-only unit tests, and CLI/Python acceptance gates. Each test is a
# self-checking executable; CTest gates on the exit code.
#
# GoogleTest is used as the harness when discoverable; otherwise tests fall back
# to a self-checking main() that returns non-zero on failure.

# --- Optional GoogleTest -----------------------------------------------------
find_package(GTest CONFIG QUIET)
if(GTest_FOUND)
  set(STEPPE_HAVE_GTEST ON)
  message(STATUS "steppe tests: using GoogleTest ${GTest_VERSION}")
else()
  set(STEPPE_HAVE_GTEST OFF)
  message(STATUS "steppe tests: GoogleTest not found — using self-checking \
fallback (returns non-zero on failure; CTest gates on exit code)")
endif()

# --- steppe_add_test: CTest label taxonomy -----------------------------------
# Every registered test carries one PRIMARY lane label {unit|reference|cli|python}
# plus the orthogonal {gpu} and {slow} labels:
#   * gpu  -> built by nvcc and/or needs a live CUDA device (the .cu reference
#            suite, the binary-spawning cli suite, the python gate); the pure-host
#            unit/ suite carries NO gpu label.
#   * slow -> a long runner.
# Selectors: `ctest -L unit` (host-only lane), `ctest -LE gpu` (CUDA-free lane),
# `ctest -L slow` / `-LE slow` (fast/slow partition). A gpu test also takes
# RESOURCE_LOCK "gpu" so `ctest -j` never oversubscribes the single GPU.
function(steppe_add_test)
  # Classic ${ARGN} form (NOT PARSE_ARGV): a quoted multi-value `LABELS "a;b"`
  # must split into the separate labels a, b — PARSE_ARGV would instead protect
  # the embedded `;` and store one literal label "a;b". Safe here because no
  # COMMAND argument contains a `;` (paths / `$<TARGET_FILE:..>` gen-exprs only).
  cmake_parse_arguments(SAT "" "NAME" "COMMAND;LABELS" ${ARGN})
  if(NOT SAT_NAME)
    message(FATAL_ERROR "steppe_add_test: NAME is required")
  endif()
  if(NOT SAT_LABELS)
    message(FATAL_ERROR "steppe_add_test(${SAT_NAME}): LABELS is required")
  endif()
  add_test(NAME ${SAT_NAME} COMMAND ${SAT_COMMAND})
  set_tests_properties(${SAT_NAME} PROPERTIES LABELS "${SAT_LABELS}")
  if("gpu" IN_LIST SAT_LABELS)
    # Serialize device-touching tests under a future `ctest -j` (single GPU box).
    set_tests_properties(${SAT_NAME} PROPERTIES RESOURCE_LOCK gpu)
  endif()
endfunction()

# --- GPU/CPU f2 parity test --------------------------------------------------
# Exercises production CpuBackend and CudaBackend through
# ComputeBackend::compute_f2 and checks the full PxP matrix, including the
# diagonal. Self-checking executable; CTest gates on the exit code.
add_executable(test_f2_equivalence reference/test_f2_equivalence.cu)
target_link_libraries(test_f2_equivalence
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
# Separable compilation so the executable device-links steppe_device's RDC kernels.
set_target_properties(test_f2_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Match steppe_device's STEPPE_HAVE_EMU_TUNING (both flow from the one cmake
# option) so this TU's #if-guarded includes agree with the library.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_equivalence PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()

steppe_add_test(NAME f2_equivalence COMMAND test_f2_equivalence LABELS "reference;gpu")

# --- Genotype-decode parity test ---------------------------------------------
# Reads the raw TGENO triple, decodes via CpuBackend and CudaBackend, and diffs
# the two against each other and the numpy oracle, then runs the decoded Q/V/N
# through compute_f2 end-to-end. Self-checking executable; CTest gates on exit.
add_executable(test_decode_equivalence reference/test_decode_equivalence.cu)
target_link_libraries(test_decode_equivalence
    PRIVATE steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_decode_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Pass the data ROOT as argv[1] so raw/ + derived_acc/ resolve from one argument.
set(STEPPE_AADR_ROOT "/workspace/data/aadr"
    CACHE PATH "AADR data root (contains raw/ TGENO triple + derived_acc/ oracle)")
steppe_add_test(NAME decode_equivalence COMMAND test_decode_equivalence "${STEPPE_AADR_ROOT}" LABELS "reference;gpu")

# --- Transpose primitive gate ------------------------------------------------
# Checks the SNP-major -> canonical individual-major transpose two ways: (A) a
# synthetic tile asserts GPU==CPU==expected output byte-for-byte (covering
# rlen-floor padding, reorder, and partial-last-byte edges), and (B) a PA slice
# transposes byte-identical to the raw TGENO slice of the same data. The real
# part SKIPs (exit 0) when the AADR root is absent. Self-checking executable.
add_executable(test_transpose_canonical reference/test_transpose_canonical.cu)
target_link_libraries(test_transpose_canonical
    PRIVATE steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_transpose_canonical PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME transpose_canonical COMMAND test_transpose_canonical "${STEPPE_AADR_ROOT}" LABELS "reference;gpu")

# --- PA-reader == TGENO-reader bit-exact gate --------------------------------
# Drives core::read_canonical_tile on a PA (SNP-major PACKEDANCESTRYMAP) prefix
# and on the raw TGENO prefix of the SAME data, and asserts the canonical tile,
# auto-detected sample_ploidy, and decoded Q/V/N match byte-for-byte on both the
# GPU backend and the CPU oracle. convertf is lossless, so a correct PA reader
# MUST reproduce the TGENO decode bit-for-bit. SKIPs (exit 0) when the AADR root
# is absent. Self-checking executable.
add_executable(test_pa_decode_equivalence reference/test_pa_decode_equivalence.cu)
target_link_libraries(test_pa_decode_equivalence
    PRIVATE steppe::core steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_pa_decode_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME pa_decode_equivalence COMMAND test_pa_decode_equivalence "${STEPPE_AADR_ROOT}" LABELS "reference;gpu")

# --- EIGENSTRAT-reader == PA-reader bit-exact gate ---------------------------
# Drives core::read_canonical_tile on an EIGENSTRAT (ASCII) .geno and on the
# convertf-PA .geno of the SAME 9-pop subset, and asserts the canonical tile,
# auto-detected sample_ploidy, and decoded Q/V/N match byte-for-byte on both the
# GPU backend and the CPU oracle. Both fixtures are convertf outputs of one PA
# source, so a correct EIGENSTRAT reader MUST reproduce the PA decode bit-for-bit.
# SKIPs (exit 0) when <root>/fixtures_eig is absent. Self-checking executable.
add_executable(test_eigenstrat_decode_equivalence reference/test_eigenstrat_decode_equivalence.cu)
target_link_libraries(test_eigenstrat_decode_equivalence
    PRIVATE steppe::core steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_eigenstrat_decode_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME eigenstrat_decode_equivalence COMMAND test_eigenstrat_decode_equivalence "${STEPPE_AADR_ROOT}" LABELS "reference;gpu")

# --- PLINK-reader == PA-reader bit-exact gate --------------------------------
# Drives core::read_canonical_tile on a PLINK .bed and on the convertf-PA .geno
# of the SAME 9-pop subset, and asserts the canonical tile, auto-detected
# sample_ploidy, and decoded Q/V/N match byte-for-byte on both the GPU backend and
# the CPU oracle. This pins the PLINK LUT, the LSB->MSB bit-order flip, and the
# ref:=A1 polarity — any of them wrong diverges immediately. SKIPs (exit 0) when
# <root>/fixtures_eig is absent. Self-checking executable.
add_executable(test_plink_decode_equivalence reference/test_plink_decode_equivalence.cu)
target_link_libraries(test_plink_decode_equivalence
    PRIVATE steppe::core steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_plink_decode_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME plink_decode_equivalence COMMAND test_plink_decode_equivalence "${STEPPE_AADR_ROOT}" LABELS "reference;gpu")

# --- ANCESTRYMAP-reader == PA-reader bit-exact gate --------------------------
# Drives core::read_canonical_tile on an ANCESTRYMAP .geno and on the convertf-PA
# .geno of the SAME 9-pop subset, and asserts the canonical tile, auto-detected
# sample_ploidy, and decoded Q/V/N match byte-for-byte on both the GPU backend and
# the CPU oracle. This pins the ANCESTRYMAP positional parse and its token->code
# map (0/1/2 copies, "-1"->missing) — a wrong parse/sentinel diverges immediately.
# SKIPs (exit 0) when <root>/fixtures_eig is absent. Self-checking executable.
add_executable(test_ancestrymap_decode_equivalence reference/test_ancestrymap_decode_equivalence.cu)
target_link_libraries(test_ancestrymap_decode_equivalence
    PRIVATE steppe::core steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_ancestrymap_decode_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME ancestrymap_decode_equivalence COMMAND test_ancestrymap_decode_equivalence "${STEPPE_AADR_ROOT}" LABELS "reference;gpu")

# --- Per-block f2 parity test ------------------------------------------------
# Loads real derived_acc Q/V/N + the .snp prefix, runs the shared assign_blocks,
# computes the per-block f2 tensor via core::compute_f2_blocks through CpuBackend
# and CudaBackend at Fp64 and EmuFp64{40}, and asserts: per-block tight tier, f2
# symmetry, Vpair == an independent recount, and single-block == compute_f2.
# Self-checking executable; CTest gates on the exit code.
add_executable(test_f2_blocks_equivalence reference/test_f2_blocks_equivalence.cu)
target_link_libraries(test_f2_blocks_equivalence
    PRIVATE steppe::io steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
# Separable compilation so the executable device-links steppe_device's RDC kernels.
set_target_properties(test_f2_blocks_equivalence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# The EmulatedFp64 arm only asserts when the fixed-slice Ozaki tuning is compiled
# in; define the macro when the option is on so this arm asserts.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_blocks_equivalence PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
steppe_add_test(NAME f2_blocks_equivalence COMMAND test_f2_blocks_equivalence "${STEPPE_AADR_ROOT}" LABELS "reference;gpu")

# --- Emulated-FP64 run-to-run determinism gate -------------------------------
# Computes f2 (compute_f2) and the per-block tensor (compute_f2_blocks) twice each
# through CudaBackend at EmulatedFp64{40} and asserts the two runs are
# bit-identical. This guards the run-to-run reproducibility of the emulated-FP64
# path. Self-checking executable; CTest gates on the exit code.
add_executable(test_f2_determinism reference/test_f2_determinism.cu)
target_link_libraries(test_f2_determinism
    PRIVATE steppe::io steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_determinism PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Engage the fixed-slice Ozaki tuning when the option is on so the emulated arm
# actually emulates; the bit-identity assertion holds either way.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_determinism PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
steppe_add_test(NAME f2_determinism COMMAND test_f2_determinism "${STEPPE_AADR_ROOT}" LABELS "reference;gpu")

# --- Grouped-batch wrapper guard + gather/scatter gate -----------------------
# Data-free. Drives the three device-private grouped-batch wrappers
# (launch_gather_group / run_f2_gemms_group / launch_assemble_blocks_group) on a
# tiny synthetic layout:
#   1. POSITIVE CONTROL — a valid, reordered 2-block layout gathered -> 3 GEMMs ->
#      scattered, with the resident [P x P x n_block] f2 + Vpair tensors diffed
#      block-for-block against a host oracle. Exercises the gather's pad-column
#      zero-fill and the scatter-to-arbitrary-id slab offset.
#   2. DEATH CASES (only when STEPPE_ASSERT is active; SKIPPED under NDEBUG): each
#      wrapper called with a degenerate extent (n_in_group <= 0, s_pad <= 0) must
#      abort in a forked child.
# Self-checking executable; CTest gates on the exit code.
add_executable(test_f2_blocks_group reference/test_f2_blocks_group.cu)
target_link_libraries(test_f2_blocks_group
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_blocks_group PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Match steppe_device's STEPPE_HAVE_EMU_TUNING when the option is on so any
# #if-guarded include agrees with the library (this test runs native Fp64).
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_blocks_group PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
steppe_add_test(NAME f2_blocks_group COMMAND test_f2_blocks_group LABELS "reference;gpu")

# --- Focused gather/scatter index-math gate ----------------------------------
# Data-free. Drives the gather + scatter wrappers on a tiny reordered 2-block
# layout and reads back the INTERMEDIATE buffers to pin the index math directly
# (coverage the final-tensor gates only reach transitively):
#   (i)   PAD-COLUMN ZERO — after the gather, every pad column c >= sz is exactly
#         0.0 (slabs pre-filled with a nonzero sentinel, so the zero-fill is a real
#         kernel write).
#   (ii)  GATHERED == FEEDER — every real column c < sz equals its feeder column
#         byte-for-byte (a pure copy).
#   (iii) SCATTER TO BLOCK-id SLAB — with a reordered {1,0} layout of two
#         different-size blocks, each resident slab matches ITS OWN block's oracle
#         and (negative control) NOT the other block's.
# Self-checking executable; CTest gates on the exit code.
add_executable(test_f2_blocks_gather reference/test_f2_blocks_gather.cu)
target_link_libraries(test_f2_blocks_gather
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_blocks_gather PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Match steppe_device's STEPPE_HAVE_EMU_TUNING when the option is on so any
# #if-guarded include agrees with the library (this test runs native Fp64).
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_blocks_gather PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
steppe_add_test(NAME f2_blocks_gather COMMAND test_f2_blocks_gather LABELS "reference;gpu")

# --- EmulatedFp64 honorability gate ------------------------------------------
# Data-free. Drives make_cuda_backend -> compute_f2 on synthetic Q/V/N to assert
# the engaged MODE (not a precision value): that f2_compute_type routes through the
# emulation_honorable predicate, and that an EmulatedFp64 request is EITHER the
# engaged fixed-slice Ozaki path (tuning on) OR an observably downgraded native
# Fp64 path with a logged capability tag (tuning off) — never silently dynamic.
# Self-checking executable; CTest gates on the exit code.
add_executable(test_emu_honorability reference/test_emu_honorability.cu)
target_link_libraries(test_emu_honorability
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_emu_honorability PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Match steppe_device's STEPPE_HAVE_EMU_TUNING so this TU's #if agrees with the
# library; the end-to-end arm trusts the library's compiled predicate regardless.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_emu_honorability PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
steppe_add_test(NAME emu_honorability COMMAND test_emu_honorability LABELS "reference;gpu")

# --- compute_f2 degenerate-extent guard gate ---------------------------------
# Data-free. Drives make_cuda_backend -> compute_f2 on degenerate extents:
# compute_f2 with P<=0 || M<=0 returns an EMPTY F2Result cleanly (no cuBLAS/CUDA
# throw from deep in the runtime), plus a positive control that a tiny valid block
# still produces a populated [P x P] result. Self-checking executable.
add_executable(test_f2_empty_guard reference/test_f2_empty_guard.cu)
target_link_libraries(test_f2_empty_guard
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_empty_guard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME f2_empty_guard COMMAND test_f2_empty_guard LABELS "reference;gpu")

# --- compute_f2 M > INT_MAX narrowing guard gate -----------------------------
# Data-free. Drives make_cuda_backend -> compute_f2 with an SNP count M > INT_MAX.
# The f2 GEMMs take the contraction extent `k` as a signed 32-bit int, so an
# unguarded static_cast<int>(M) would wrap for M > INT_MAX and silently contract
# over fewer SNPs. The guard: compute_f2 with M == INT_MAX+1 THROWS
# std::runtime_error naming INT_MAX, the bound is a tight `M > INT_MAX` (M ==
# INT_MAX is accepted), and it fires before any allocation (data == nullptr for the
# over-limit cases); a valid tiny block still computes. Self-checking executable.
add_executable(test_f2_int_max_guard reference/test_f2_int_max_guard.cu)
target_link_libraries(test_f2_int_max_guard
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_int_max_guard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME f2_int_max_guard COMMAND test_f2_int_max_guard LABELS "reference;gpu")

# --- DeviceBuffer<T> n*sizeof(T) overflow guard gate -------------------------
# Data-free. Exercises the device-private DeviceBuffer<T> RAII owner directly.
# DeviceBuffer<T>(n) with `n` near SIZE_MAX/sizeof(T) must THROW a typed CudaError
# rather than silently wrap the `n * sizeof(T)` multiply and under-allocate (the
# guard fires before cudaMalloc). Also pins the tight inequality (n==SIZE_MAX/
# sizeof(T) is not rejected here), the untouched happy path (small alloc non-null
# with exact size()/bytes(), zero-size is {nullptr,0,0}), and the move-only RAII
# shape (move nulls source, move-assign frees-then-steals, self-move no-op).
# Self-checking executable; CTest gates on the exit code.
add_executable(test_device_buffer reference/test_device_buffer.cu)
target_link_libraries(test_device_buffer
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_device_buffer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME device_buffer_unit COMMAND test_device_buffer LABELS "reference;gpu")

# --- CUDA check macro non-throwing tagged-degrade gate -----------------------
# Data-free, no GPU work. Drives the device-private check.cuh on chosen
# cudaError_t values: STEPPE_CUDA_WARN on the expected capability-degrade statuses
# (cudaErrorPeerAccessAlreadyEnabled / cudaErrorPeerAccessNotEnabled) does NOT
# throw, logs one line, and yields the status; STEPPE_CUDA_CHECK on the SAME status
# DOES throw a CudaError carrying it. Also pins the success quiet-pass-through and
# the source_location capture. Runs on every lane (host-side string lookups only).
# Self-checking executable; CTest gates on the exit code.
add_executable(test_cuda_check reference/test_cuda_check.cu)
target_link_libraries(test_cuda_check
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_cuda_check PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME cuda_check_unit COMMAND test_cuda_check LABELS "reference;gpu")

# --- OOM fault-taxonomy translator gate --------------------------------------
# Data-free, no GPU. Pins that a genuine device allocation fault
# (cudaErrorMemoryAllocation / CUBLAS_STATUS_ALLOC_FAILED /
# CUSOLVER_STATUS_ALLOC_FAILED) maps to Status::DeviceOom -> kExitDeviceOom (3)
# through both the device seam (device::device_fault_status) and the app helper
# (app::exit_code_for_caught), while every non-alloc fault keeps the catch-all
# kExitRuntimeError (5). Self-checking executable; CTest gates on the exit code.
add_executable(test_device_fault_status reference/test_device_fault_status.cu)
target_link_libraries(test_device_fault_status
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_device_fault_status PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME device_fault_status_unit COMMAND test_device_fault_status LABELS "reference;gpu")

# --- CublasHandle MathModeScope + device-ordinal gate ------------------------
# Data-free; NEEDS a live device (cublasCreate / cublasGetMathMode need a cuBLAS
# context). Exercises the device-private handles.hpp RAII wrapper:
#   (a) MathModeScope captures the live cuBLAS math mode on construction and
#       restores it at scope exit — proven across nesting (inner restores to outer)
#       and a moved-from scope (which restores nothing, so the mode is restored
#       exactly once).
#   (b) CublasHandle records the device ordinal current at construction
#       (cudaGetDevice — query only, never cudaSetDevice inside the wrapper) and, in
#       DEBUG, the context-mutating methods assert the current device still matches.
# Asserts handle STATE, not a precision value. Self-checking executable.
add_executable(test_handles reference/test_handles.cu)
target_link_libraries(test_handles
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_handles PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME handles_unit COMMAND test_handles LABELS "reference;gpu")

# --- Host-only unit test of the shared f2 estimator primitive ----------------
# Pure C++ TU (no CUDA): proves the __host__ __device__ primitive compiles and is
# correct on the host path.
add_executable(test_f2_estimator unit/test_f2.cpp)
target_link_libraries(test_f2_estimator
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_f2_estimator PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_f2_estimator PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME f2_estimator_unit COMMAND test_f2_estimator LABELS "unit")

# --- Host-only unit test of the shared decode/allele-frequency primitive ------
# Pure C++ TU (no CUDA, no data): proves the __host__ __device__ decode primitives
# in core/internal/decode_af.hpp (genotype_code, genotype_valid, finalize_af)
# compile and are correct on the host path. Pins finalize_af: ploidy=0 ->
# masked-out {0,0,0} (not NaN/Inf), ploidy=1 -> haploid (N == 1*AN), ploidy=2 ->
# diploid (N == 2*AN).
add_executable(test_decode unit/test_decode.cpp)
target_link_libraries(test_decode
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_decode PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_decode PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME decode_unit COMMAND test_decode LABELS "unit")

# --- Host-only unit test of the launch-config + host/device headers ----------
# Pure C++ TU (no CUDA): pins core/internal/launch_config.hpp (cdiv int+long,
# grid_for, the 32x8 decode block dims) and core/internal/host_device.hpp
# (STEPPE_HD, STEPPE_DEBUG_ONLY, STEPPE_ASSERT), reached directly through their
# headers.
add_executable(test_launch_config unit/test_launch_config.cpp)
target_link_libraries(test_launch_config
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_launch_config PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_launch_config PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME launch_config_unit COMMAND test_launch_config LABELS "unit")

# --- Host-only unit test of the public config contract -----------------------
# Pure C++ TU (no CUDA, no GPU): includes the CUDA-free include/steppe/config.hpp
# and pins the DeviceConfig defaults — above all that the `deterministic` field
# exists, is a bool, and defaults to TRUE. The checks are compile-time
# static_asserts plus run-time guards; the TU links only steppe::api.
add_executable(test_config unit/test_config.cpp)
target_link_libraries(test_config
    PRIVATE steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_config PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_config PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME config_unit COMMAND test_config LABELS "unit")

# --- Access-layer config-builder unit test -----------------------------------
# Pure C++ TU (no CUDA, no GPU, no data): drives the CUDA-free ConfigBuilder/
# RunConfig/CliArgs (src/core/config/) — the precedence merge (compiled defaults <
# TOML < env STEPPE_* < CLI), the validating build() that maps the raw flags onto
# the real structs and fail-fasts (InvalidConfig) on bad config, and the
# Status->exit-code map. Links steppe::core + steppe::api.
add_executable(test_config_builder unit/test_config_builder.cpp)
target_link_libraries(test_config_builder
    PRIVATE steppe::core steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_config_builder PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_config_builder PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME config_builder_unit COMMAND test_config_builder LABELS "unit")

# --- Output-escaping primitives unit test (result_emit) ----------------------
# Pure C++ host TU (no CUDA, no GPU, no data): compiles src/app/result_emit.cpp
# directly and asserts app::csv_field (RFC-4180) + app::json_quote keep a normal
# pop label byte-identical while escaping a pathological comma/quote/newline label.
# Self-checking executable; CTest gates on the exit code.
add_executable(test_result_emit
    unit/test_result_emit.cpp
    ${CMAKE_SOURCE_DIR}/src/app/result_emit.cpp)
target_link_libraries(test_result_emit
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
steppe_add_test(NAME result_emit_unit COMMAND test_result_emit LABELS "unit")

# --- qpAdm parity test (BOTH backends) ---------------------------------------
# Reads the committed 9-pop f2 fixture and runs the single-model qpAdm fit
# (steppe::run_qpadm) on both backends: the CpuBackend reference (bit-exact diff
# oracle + no-GPU fallback) AND the CUDA backend (uploads the fixture to VRAM,
# asserts f2 resident, runs the fit on the GPU). Asserts weights/chisq/dof/p vs the
# golden AND CpuBackend == CudaBackend to 1e-9, plus the S3/S4 X/Q slice
# cross-check. The GPU block SKIPs cleanly (still exits via the CpuBackend gate)
# when no CUDA device is visible. Self-checking executable.
add_executable(test_qpadm_parity reference/test_qpadm_parity.cu)
target_link_libraries(test_qpadm_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpadm_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME qpadm_parity
         COMMAND test_qpadm_parity "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "reference;gpu")

# --- qpGraph single-graph-fit parity test (BOTH backends) --------------------
# Reads the committed 9-pop f2 fixture + the golden topology and runs the
# single-graph qpGraph fit (steppe::run_qpgraph) on both backends: the CpuBackend
# oracle AND the CUDA backend (f3 basis resident in VRAM, the fleet on-device).
# Gates score, the admix weight keyed by parent name (pSteppe->aCW), and the
# restart spread vs the AT2 golden, plus CpuBackend == CudaBackend. The GPU block
# SKIPs cleanly when no device is visible.
add_executable(test_qpgraph_parity reference/test_qpgraph_parity.cu)
target_link_libraries(test_qpgraph_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpgraph_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME qpgraph_parity
         COMMAND test_qpgraph_parity "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "reference;gpu")

# --- qpGraph topology enumerator unit test (host-only) -----------------------
# Plain C++ TU (no CUDA): the host enumerator reproduces admixtools
# generate_all_trees / generate_all_graphs exactly — count (105 trees + 1485
# non-iso nadmix=1) AND the canonical graph_hash set == the committed AT2 fixture.
# Also checks every enumerated topology parses through parse_qpgraph and the
# neighbor move set is connected.
add_executable(test_qpgraph_enumerate reference/test_qpgraph_enumerate.cpp)
target_link_libraries(test_qpgraph_enumerate
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
steppe_add_test(NAME qpgraph_enumerate
         COMMAND test_qpgraph_enumerate "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "reference;gpu")

# --- qpGraph topology-search acceptance (BOTH backends) ----------------------
# The exhaustive bounded-space search (run_qpgraph_search) on the committed
# real-AADR 5-pop subset. Gates: (1) per-candidate fit == admixtools::qpgraph over
# steppe's own 1590 enumerated graphs, (2) the exhaustive count == AT2 (105/1485),
# (3) the deterministic global-best argmin == AT2's argmin (same canonical hash,
# CpuBackend == CudaBackend), and the heuristic recovers it.
add_executable(test_qpgraph_search reference/test_qpgraph_search.cu)
target_link_libraries(test_qpgraph_search
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpgraph_search PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME qpgraph_search
         COMMAND test_qpgraph_search "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "reference;gpu")

# --- qpGraph topology-search per-candidate parity (BOTH backends) ------------
# Runs the same exhaustive search, exposes the full per-candidate scored vector,
# then for every candidate looks steppe's score up by canonical graph_hash in the
# trusted oracle (at2_scores_5pop.csv; 1590 rows). Gates: (a) the global-best +
# well-identified candidates match AT2 at rtol 1e-6 / atol 1e-9; (b) the
# global-best argmin hash == the AT2 argmin hash; (c) the full 1590 lie within the
# documented ill-conditioning envelope (flat graphs differ ~0.3-0.5%).
add_executable(test_qpgraph_search_parity reference/test_qpgraph_search_parity.cu)
target_link_libraries(test_qpgraph_search_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpgraph_search_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME qpgraph_search_parity
         COMMAND test_qpgraph_search_parity "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "reference;gpu")

# --- qpfstats smoothing-solve parity (BOTH backends) -------------------------
# Numerical-equivalence test of the qpfstats_smooth solve primitive (A = x'x +
# ridge*I, Cholesky, the multi-column Dtrsm solve, the NaN-row downdate,
# all-NaN->b=0) over a small design/RHS exercising every branch (no-NaN,
# partial-NaN, all-NaN blocks). CpuBackend == CudaBackend to ~1e-9. The GPU block
# SKIPs cleanly when no device is visible.
add_executable(test_qpfstats_smooth_parity reference/test_qpfstats_smooth_parity.cu)
target_link_libraries(test_qpfstats_smooth_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpfstats_smooth_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME qpfstats_smooth_parity COMMAND test_qpfstats_smooth_parity LABELS "reference;gpu")

# --- qpfstats fused reduce->jackknife->smooth->recenter parity ---------------
# Diffs the on-device block-jackknife kernels (the per-comb y/ymat + the per-pair
# recenter) against the CpuBackend oracle on a synthetic genotype field, at rtol
# 1e-6. SKIPs/fails fast with no GPU. The real-AADR end-to-end statistic is gated
# by cli_qpfstats.
add_executable(test_qpfstats_fused_parity reference/test_qpfstats_fused_parity.cu)
target_link_libraries(test_qpfstats_fused_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpfstats_fused_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME qpfstats_fused_parity COMMAND test_qpfstats_fused_parity LABELS "reference;gpu")

# --- DATES cuFFT LD-engine parity test (BOTH backends) -----------------------
# Asserts the CudaBackend cuFFT autocorrelation path (ComputeBackend::dates_curve)
# matches the CpuBackend FFT-free direct-autocorrelation oracle on the moment
# statistics that drive the corr curve — sum_g W[g]*W[g+lag] == IFFT(|FFT(W)|^2)/n.
# Constructed small inputs. SKIPs/fails fast with no GPU.
add_executable(test_dates_parity reference/test_dates_parity.cu)
target_link_libraries(test_dates_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_dates_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME dates_parity COMMAND test_dates_parity LABELS "reference;gpu")

# --- f-stat sweep parity test ------------------------------------------------
# Uploads the real 9-pop golden f2 fixture resident to VRAM and runs the GPU-only
# run_f4_sweep / run_f3_sweep. Asserts: every sweep survivor's est/se/z is
# bit-exact to the explicit run_f4/run_f3 over that item (no compute fork); the
# |z|>=3 filter keeps exactly the explicit |z|>=3 items; and an over-cap sweep
# refuses (capped, no compute) without --sure. SKIPs cleanly if no GPU.
add_executable(test_fstat_sweep_parity reference/test_fstat_sweep_parity.cu)
target_link_libraries(test_fstat_sweep_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_fstat_sweep_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME fstat_sweep_parity
         COMMAND test_fstat_sweep_parity "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "reference;gpu;slow")

# --- qpAdm rotation model-search test ----------------------------------------
# Reads the real AT2 rotation golden + the co-matching f2 fixture, uploads f2
# resident to VRAM, and runs the batched, multi-GPU-sharded rotation
# steppe::run_qpadm_search on the CudaBackend. Gates: per-model parity vs the
# golden (weights rtol 1e-6, p/feasible loose, f4rank exact) for the 84-model set;
# G=1 vs G=2 bit-identical and identically-ordered; no regression on the single-
# model goldens via a 1-element search; plus reported 1-GPU + 2-GPU throughput
# (timing only, no per-model golden). The GPU blocks SKIP cleanly if <1 / <2
# devices visible. Self-checking executable.
add_executable(test_qpadm_rotation reference/test_qpadm_rotation.cu)
target_link_libraries(test_qpadm_rotation
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpadm_rotation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME qpadm_rotation
         COMMAND test_qpadm_rotation "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "reference;gpu;slow")

# --- qpAdm domain-outcome test (BOTH backends) -------------------------------
# A deliberately degenerate model returns a per-model domain-outcome status value
# (RankDeficient / NonSpdCovariance / ChisqUndefined), never a crash / exception /
# silently-Ok NaN, on both the CpuBackend oracle and the production CudaBackend.
# This covers the non-Ok taxonomy every other qpadm assertion (== Status::Ok) does
# not.
#
# NO SYNTHETIC DATA: every degenerate model is built from the committed real-AADR
# 9-pop f2 fixture, arranged degenerate by selecting real pops (a left set with a
# duplicated source -> RankDeficient / non-SPD; an over-parameterized dof<=0 model
# -> ChisqUndefined). The CpuBackend oracle runs under STEPPE_THOROUGH=1 or when no
# GPU is visible; the CudaBackend block runs whenever a device is visible.
# Self-checking executable.
add_executable(test_qpadm_domain reference/test_qpadm_domain.cu)
target_link_libraries(test_qpadm_domain
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpadm_domain PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME qpadm_domain
         COMMAND test_qpadm_domain "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "reference;gpu")

# --- qpAdm missing-block acceptance gate (BOTH backends) ---------------------
# The one qpAdm gate that exercises a MISSING block — a real-AADR f2 set extracted
# with maxmiss>0 in which one jackknife block has a population pair with Vpair==0.
# AT2 read_f2(remove_na=TRUE) DROPS that block entirely before the jackknife (not
# impute-0, which biases f4 toward 0); steppe reproduces the drop on both backends.
# Gates weights/chisq/dof/p/rankdrop + the f4 X over survivors vs the AT2 golden AND
# CpuBackend == CudaBackend. Without the drop the X/Q differ by ~1e-3 (above the
# tight tier), so the test fails the buggy impute-0 path.
#
# NO SYNTHETIC DATA: the missing block arises from a real sparse ancient pop as a
# right pop; the fixture is the raw (pre-drop) AT2 f2 tensor with the NA pair stored
# as NaN. CpuBackend oracle runs under STEPPE_THOROUGH=1 or no GPU; CudaBackend on
# default.
add_executable(test_qpadm_missing_block reference/test_qpadm_missing_block.cu)
target_link_libraries(test_qpadm_missing_block
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpadm_missing_block PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME qpadm_missing_block
         COMMAND test_qpadm_missing_block "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "reference;gpu")

# --- qpWave standalone acceptance gate (BOTH backends) -----------------------
# Calls the public run_qpwave entry directly on both the CpuBackend oracle and the
# production CudaBackend and gates est_rank / f4rank / the rankdrop table / chisq vs
# the AT2 qpwave() golden. Validates the distinguishing no-target-prepend /
# left[0]-is-reference semantic.
#
# NO SYNTHETIC DATA: both qpWave models reuse the committed real-AADR 9-pop f2
# fixture arranged as qpWave left/right splits over real pops (M1 3-left
# est_rank=1; M2 2-left clade est_rank=0). The CpuBackend oracle runs under
# STEPPE_THOROUGH=1 or when no GPU is visible; the CudaBackend block runs whenever a
# device is visible. Self-checking executable.
add_executable(test_qpwave_parity reference/test_qpwave_parity.cu)
target_link_libraries(test_qpwave_parity
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_qpwave_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME qpwave_parity
         COMMAND test_qpwave_parity "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "reference;gpu")

# --- Host-only unit test of the model-search shard planner -------------------
# Pure C++ TU (no CUDA, no GPU, no io, no data): drives
# steppe::core::qpadm::plan_model_shards (the model->device shard plan) + the
# pre-sized-slot re-sort discipline in isolation — the GPU-free coverage of the
# determinism mechanism run_qpadm_search relies on. Self-checking executable.
add_executable(test_model_search_core unit/test_model_search_core.cpp)
target_link_libraries(test_model_search_core
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
steppe_add_test(NAME model_search_core_unit COMMAND test_model_search_core LABELS "unit")

# --- Host-only unit test of the backend-factory header -----------------------
# Pure C++ TU (no CUDA, no GPU): includes the CUDA-free device/backend_factory.hpp
# and static_asserts that both factories — make_cpu_backend and make_cuda_backend —
# are declared in the one namespace steppe::device with the documented
# std::unique_ptr<ComputeBackend>() signature. The checks are decltype-only, so
# this TU compiles + links WITHOUT steppe::device — proof the header is CUDA-free
# (a CUDA leak would fail this host compile).
add_executable(test_backend_factory unit/test_backend_factory.cpp)
target_link_libraries(test_backend_factory
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_backend_factory PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_backend_factory PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME backend_factory_unit COMMAND test_backend_factory LABELS "unit")

# --- Host-only unit test of the BackendCapabilities POD ----------------------
# Pure C++ TU (no CUDA, no GPU): includes the CUDA-free device/backend.hpp and
# pins the shape + value-initialized "unknown" defaults of the BackendCapabilities
# POD: it is an aggregate / trivially-copyable / standard-layout POD whose every
# field defaults to zero/false, AND ComputeBackend::capabilities() has a usable base
# default returning that unknown (so CpuBackend need not override it). The real
# probe values are asserted by the CUDA probe gate, not here. Links only
# steppe::core_internal + steppe::api — a CUDA leak would fail this host compile.
add_executable(test_backend_capabilities unit/test_backend_capabilities.cpp)
target_link_libraries(test_backend_capabilities
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_backend_capabilities PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_backend_capabilities PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME backend_capabilities_unit COMMAND test_backend_capabilities LABELS "unit")

# --- CUDA capabilities() probe + device_id gate ------------------------------
# Data-free. Drives make_cuda_backend(device_id) + CudaBackend::capabilities() on a
# live device, pinning the real probe values the CUDA override reports:
#   EVERY LANE: compute_major >= 12 (sm_120), total_vram_bytes > 0, free <= total,
#     device_count == cudaGetDeviceCount, emulated_fp64_honorable == the library's
#     emulation_honorable predicate, and a backend constructs on device 0.
#   MULTI-GPU: device_count == 2, a backend constructs on device 1 too, and
#     can_access_peer == true. The strict peer assertion is KEYED to the tier
#     (device name contains "PRO"); on a single-peer box can_access_peer == false is
#     the expected tagged degrade to the host-staged combine (a PASS, not a fault).
# Self-checking executable; CTest gates on the exit code.
add_executable(test_backend_capabilities_probe reference/test_backend_capabilities_probe.cu)
target_link_libraries(test_backend_capabilities_probe
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_backend_capabilities_probe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Match steppe_device's STEPPE_HAVE_EMU_TUNING so this TU's #if-guarded include
# agrees with the library; the end-to-end check trusts the compiled predicate.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_backend_capabilities_probe PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
steppe_add_test(NAME backend_capabilities_probe COMMAND test_backend_capabilities_probe LABELS "reference;gpu")

# --- build_resources / Resources / PerGpuResources gate ----------------------
# Data-free. Drives steppe::device::build_resources on live devices, pinning that
# it binds one backend PER device in DeviceConfig::devices order (the fixed
# g=0..G-1 combine order) and stores each device's probe in PerGpuResources::caps:
#   EVERY LANE: devices={0} -> exactly 1 entry bound to device 0 with compute_major
#     >= 12 and total_vram > 0; devices={} (auto) -> all visible ordinals
#     0..count-1 in order.
#   MULTI-GPU (count >= 2): devices={0,1} -> exactly 2 entries bound to devices 0
#     and 1 in order, each compute_major >= 12, and — on the peer-capable tier
#     (device name contains "PRO") — each can_access_peer == true. On a single-peer
#     box can_access_peer == false is the expected tagged degrade (a PASS), so the
#     strict peer assertion is KEYED to the tier and the 2-device build still
#     succeeds.
# No real f2 compute. Self-checking executable; CTest gates on the exit code.
add_executable(test_resources_build reference/test_resources_build.cu)
target_link_libraries(test_resources_build
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_resources_build PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# No STEPPE_HAVE_EMU_TUNING-gated symbol is touched here, so no emulation-tuning
# define is needed for this TU.
steppe_add_test(NAME resources_build COMMAND test_resources_build LABELS "reference;gpu")

# --- Multi-GPU f2 parity gate ------------------------------------------------
# Loads real derived_acc (P=50, fast) AND derived_full (P=768, scale) Q/V/N + the
# .snp prefix, runs the shared assign_blocks, and asserts the multi-GPU combined
# f2_blocks AND Vpair are BIT-IDENTICAL (std::memcmp, not a tolerance) to the
# single-GPU reference (core::compute_f2_blocks), across G:
#   (1) G==1 multi-GPU == single-GPU reference (the bit-identity floor).
#   (2) G==2 host-staged combine (prefer_p2p_combine=false) == single-GPU reference.
#   (3) G==2 P2P device-resident combine (prefer_p2p_combine=true) == single-GPU
#       reference (the opt-in fast-path; runs on the peer-capable tier).
#   (4) G==2 P2P == G==2 host-staged (the two combine tiers bit-identical to each
#       other).
# Bit-identity holds because the sharding is block-aligned and both tiers sum in
# fixed g=0..G-1 order onto a zero tensor. Strict for EmulatedFp64{40} (the f2
# default); native Fp64 at G>=2 is the documented batched-GEMM shape-sensitivity
# cell (oracle-tolerance). On a single-GPU box the G==2 lane is skipped; an absent
# dataset directory is skipped. The which-path tag (Resources::last_combine_path) is
# asserted: prefer_p2p_combine=false => HostStaged; on the peer-capable tier the P2P
# candidate's tag == P2pDeviceResident (confirms (3) exercised cudaMemcpyPeer, not a
# silent fallback). Self-checking executable.
add_executable(test_f2_multigpu_parity reference/test_f2_multigpu_parity.cu)
target_link_libraries(test_f2_multigpu_parity
    PRIVATE steppe::io steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
# Separable compilation so the executable device-links steppe_device's RDC kernels.
set_target_properties(test_f2_multigpu_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Match steppe_device's STEPPE_HAVE_EMU_TUNING when the option is on so the
# EmulatedFp64 arm exercises the fixed-slice Ozaki path (which must produce
# identical bits sharded vs not) and this TU's #if-guarded includes agree.
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(test_f2_multigpu_parity PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()
steppe_add_test(NAME f2_multigpu_parity COMMAND test_f2_multigpu_parity "${STEPPE_AADR_ROOT}" LABELS "reference;gpu;slow")

# --- Multi-GPU wall-clock bench (manual, not a ctest gate) -------------------
# Times core::compute_f2_blocks_multigpu G==1 vs G==2 end-to-end on real AADR;
# reports the speedup and catches the single-GPU OOM at large P.
add_executable(bench_f2_multigpu reference/bench_f2_multigpu.cu)
target_link_libraries(bench_f2_multigpu
    PRIVATE steppe::io steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(bench_f2_multigpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(bench_f2_multigpu PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()

# Manual bench (not a ctest gate): times the batched run_qpadm_search rotation
# (CudaBackend::fit_models_batched, f2 resident in VRAM) single-GPU over a real
# 1240K f2_blocks dir.
add_executable(bench_rotation_1240k reference/bench_rotation_1240k.cu)
target_link_libraries(bench_rotation_1240k
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(bench_rotation_1240k PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- f-stats (f4/f3/f4-ratio) SINGLE-GPU batched throughput bench (manual, not a gate) --
add_executable(bench_fstats_1240k reference/bench_fstats_1240k.cu)
target_link_libraries(bench_fstats_1240k
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(bench_fstats_1240k PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- qpGraph GPU-optimizer spike bench (manual, not a ctest gate) ------------
# Compares two candidate qpGraph GPU optimizer shapes — a batched-sequential fleet
# (bounded projected-Newton/L-BFGS) vs a batched-population one (CMA-ES/DE,
# curand-sampled) — on a real-f2-grounded box-constrained graph-GLS objective from
# the committed 9-pop AADR fixture. Both run entirely on-device; the bench pins the
# optimum on the host oracle and reports wall-clock / obj-evals / convergence at
# increasing N. Self-contained (both optimizer kernels live in this TU). Links
# CUDA::curand for the device-side Gaussian sampling.
add_executable(bench_optimizers reference/bench_optimizers.cu)
target_link_libraries(bench_optimizers
    PRIVATE steppe::core_internal steppe::api steppe::warnings CUDA::curand)
set_target_properties(bench_optimizers PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- Multi-GPU combine gate-logic test ---------------------------------------
# Host-pure logic. Drives the capability-tier fork in
# core::compute_f2_blocks_multigpu through a CUDA-free recording FakeBackend over a
# hand-built Resources (no real GPU compute, no peer access), asserting the full
# combine-gate truth table for the four-term gate:
#   * enable_peer_access=false forces last_combine_path==HostStaged even if
#     can_access_peer were true (the objective gate — never call the forbidden
#     cudaDeviceEnablePeerAccess).
#   * prefer && enable && can_access && G>=2 is the ONLY combo that selects P2P;
#     prefer=false or no-peer degrade to HostStaged; the G==1 fast path never runs
#     the gate (tag stays None), so the `&& G >= 2` term bites at G==1.
# Asserts WHICH transport the gate picks, never a number.
#
# It is a .cu (not a host .cpp) because compute_f2_blocks_multigpu references the
# device symbols plan_block_shards + combine_f2_partials_p2p, so the executable must
# device-link steppe::device to resolve them (the P2P symbol is resolved, never
# called — every driven row resolves to the host-staged baseline).
add_executable(test_f2_multigpu_gate reference/test_f2_multigpu_gate.cu)
target_link_libraries(test_f2_multigpu_gate
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_f2_multigpu_gate PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME f2_multigpu_gate COMMAND test_f2_multigpu_gate LABELS "reference;gpu")

# --- Host-only unit test of the shared f2-combine validator ------------------
# Pure C++ TU (no CUDA, no GPU, no io, no data): exercises the single-home guard
# core::validate_f2_partials — the one validator both f2 combine tiers (host-staged
# combine_f2_partials_host and device-resident combine_f2_partials_p2p) call, so the
# two tiers reject malformed inputs identically. Every valid case passes (single/
# two-shard tiling, empty/trailing-empty degenerates, the n_block_full==0 corner)
# and every malformed case FAILS FAST (throws): count mismatch, negative
# P/n_block_full, n_block != shard span, P disagreement, the short-partial storage
# gap (f2/vpair/block_sizes under-sized), and shards that do not tile
# [0, n_block_full). Links only steppe::core_internal + steppe::api.
add_executable(test_f2_partials_validate unit/test_f2_partials_validate.cpp)
target_link_libraries(test_f2_partials_validate
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_f2_partials_validate PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_f2_partials_validate PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME f2_partials_validate_unit COMMAND test_f2_partials_validate LABELS "unit")

# --- Host-only unit test of the host-staged f2 combine -----------------------
# Pure C++ TU (no CUDA, no GPU, no io, no data): exercises
# core::combine_f2_partials_host — the host-staged fixed-order combine that is the
# only combine path on a no-peer box and the bit-identity reference the P2P sibling
# must match. The combine PLACES each device's compact partial at its block offset
# via a contiguous std::copy_n, which reproduces a -0.0 partial element byte-for-byte
# (the prior `+=`-onto-+0.0 flipped it to +0.0). The gate: (1) two disjoint shards
# place over the full cover bit-exact; (2) the -0.0 case lands as -0.0; (3) the G==1
# single-full-shard floor; (4) the degenerates (all-empty -> empty, trailing empty
# skipped); (5) fail-fast through the shared validate_f2_partials guard. Links only
# the CUDA-free F2BlockTensor + DeviceShard, so no GPU symbol is pulled.
add_executable(test_f2_combine unit/test_f2_combine.cpp)
target_link_libraries(test_f2_combine
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_f2_combine PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_f2_combine PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME f2_combine_unit COMMAND test_f2_combine LABELS "unit")

# --- Host-only unit test of the block-aligned shard planner ------------------
# Pure C++ TU (no CUDA, no GPU, no io, no data): exercises the host-pure
# deterministic planner steppe::device::plan_block_shards — the block->device
# mapping that is the structural parity floor (whole-block-on-one-device) under both
# combine tiers. Each block's SNP count is derived from ranges[b].size(). Every case
# asserts plan == an independent greedy reference + the documented exact layout, so a
# divergence (which would change the combine layout and break the locked
# f2_multigpu_parity bit-identity) fails fast here before the slow GPU run. Only
# shard_plan.cpp.o is pulled (no kernel object), so no GPU code links.
add_executable(test_shard_plan unit/test_shard_plan.cpp)
target_link_libraries(test_shard_plan
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
steppe_add_test(NAME shard_plan_unit COMMAND test_shard_plan LABELS "unit")

# --- Host-only unit test of the multi-GPU orchestrator core ------------------
# Pure C++ TU (no CUDA, no GPU, no io, no data): exercises the host-pure, P2P-free
# core of the multi-GPU precompute orchestrator — core::plan_multigpu_shards + the
# per-device concurrent std::jthread fan-out core::compute_multigpu_partials —
# factored out of the public entry point compute_f2_blocks_multigpu. The gate:
#   (1) SUB-VIEW + LOCAL-ID — each device g is handed exactly its shard's columns
#       (Q.data + P*s0, M == s1-s0) and a dense zero-based local block_id
#       (block_id[s0+k] - b0), recorded + asserted by a fake backend;
#   (2) FAN-OUT + COMBINE — the G partials, keyed on the global block id, recombine
#       via the host-staged combine into the full tensor bit-for-bit vs an
#       independent reference (over G=1/2/4 incl. an uneven 8-block/4-device case);
#   (3) G==1 IDENTITY — the single full shard with local==global ids;
#   (4) EMPTY / n_block<G — trailing empty shards place nothing, prefix still exact;
#   (5) EXCEPTION PROPAGATION — a faulting worker surfaces as a normal throw (the
#       exception_ptr rethrow), never std::terminate.
# The extracted core references only the CUDA-free seam, plan_block_shards, and the
# host-staged combine — never combine_f2_partials_p2p — so it links steppe::device to
# resolve plan_block_shards.o but pulls no kernel object and needs no device-link.
add_executable(test_f2_blocks_multigpu unit/test_f2_blocks_multigpu.cpp)
target_link_libraries(test_f2_blocks_multigpu
    PRIVATE steppe::core steppe::device steppe::core_internal steppe::api steppe::warnings)
steppe_add_test(NAME f2_blocks_multigpu_unit COMMAND test_f2_blocks_multigpu LABELS "unit")

# --- Host-only unit test of the device-ordinal validator ---------------------
# Pure C++ TU (no CUDA, no GPU, no io, no data): exercises the build()-validation
# predicate steppe::device::validate_device_order(span<const int>, int) — reject
# duplicate or out-of-range device ordinals fail-fast. The gate: every g=0..G-1
# order passes; every duplicate ({0,0}, {0,1,0}) and every out-of-range (>=visible,
# negative, a device on a zero-visible box) THROWS std::runtime_error; out-of-range
# is checked BEFORE the `seen` duplicate index (no OOB). validate_device_order is
# out-of-line in steppe_device (resources.cpp) but CUDA-free, so this TU links
# steppe::device to resolve the symbol yet needs no GPU at run time.
add_executable(test_validate_device_order unit/test_validate_device_order.cpp)
target_link_libraries(test_validate_device_order
    PRIVATE steppe::device steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_validate_device_order PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_validate_device_order PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME validate_device_order_unit COMMAND test_validate_device_order LABELS "unit")

# --- Host-only unit test of the VRAM budget helper ---------------------------
# Pure C++ TU (no CUDA, no io): exercises device/vram_budget.hpp — the host-pure
# chunk-budget arithmetic. Pins that the budget counts BOTH resident f2+vpair
# tensors (the 2x term), subtracts the cuBLAS workspace before applying
# kMaxVramUtilizationFraction, clamps in size_t before the int narrowing, and that
# one chunk's slabs never exceed the fraction-of-net budget.
add_executable(test_vram_budget unit/test_vram_budget.cpp)
target_link_libraries(test_vram_budget
    PRIVATE steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_vram_budget PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_vram_budget PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME vram_budget_unit COMMAND test_vram_budget LABELS "unit")

# --- Host-only unit test of the SNP->block partition rule --------------------
# Pure C++ TU (no CUDA, no io): exercises core/domain/block_partition_rule on
# synthetic chromosome/position layouts and the cM->Morgan conversion site. Links
# steppe::core for the out-of-line assign_blocks.
add_executable(test_block_partition unit/test_block_partition.cpp)
target_link_libraries(test_block_partition
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_block_partition PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_block_partition PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME block_partition_unit COMMAND test_block_partition LABELS "unit")

# --- Host-only unit test of the block-range inverse rule ---------------------
# Pure C++ TU (no CUDA, no io): exercises core::block_ranges — the single-source
# inverse of assign_blocks both backends call. Pins the valid layouts (empty,
# single, multi-block, round-trip vs assign_blocks) and that malformed partitions
# FAIL-FAST (throw): short/null block_id, out-of-range id, non-monotonic.
# block_ranges is header-inline; linking steppe::core also pulls assign_blocks for
# the round-trip case.
add_executable(test_block_ranges unit/test_block_ranges.cpp)
target_link_libraries(test_block_ranges
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_block_ranges PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_block_ranges PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME block_ranges_unit COMMAND test_block_ranges LABELS "unit")

# Real-AADR internal-consistency check: feed the real v66 .snp to assign_blocks and
# assert the measured block count + dense/ordering invariants. Path overridable via
# -DSTEPPE_AADR_SNP=<path>. Registered only on the non-gtest fallback build, where
# main() consumes argv[1]; under gtest the consistency body is compiled out.
set(STEPPE_AADR_SNP "/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB.snp"
    CACHE FILEPATH "Real AADR .snp file for the block-partition consistency check")
if(NOT STEPPE_HAVE_GTEST)
  steppe_add_test(NAME block_partition_aadr_consistency
           COMMAND test_block_partition "${STEPPE_AADR_SNP}" LABELS "unit")
endif()

# --- Host-only unit test of the f2 composition root --------------------------
# Pure C++ TU (no CUDA, no GPU, no io, no data): drives the f2 orchestration
# (core::compute_f2_block / core::compute_f2_blocks) through a CUDA-free recording
# MockBackend. The gate: (1) the orchestration forwards Q/V/N + precision + block_id
# .data()/n_block to the backend byte-for-byte and returns its result unchanged;
# (2) a malformed contract — Q/V/N disagreeing on P/M, a short/null block_id, an
# n_block out of range, or a non-decreasing block_id — FAILS FAST (STEPPE_ASSERT
# abort) before reaching the backend (death cases skipped under NDEBUG). This TU
# names only the abstract ComputeBackend, so no GPU symbol is pulled.
add_executable(test_f2_from_blocks unit/test_f2_from_blocks.cpp)
target_link_libraries(test_f2_from_blocks
    PRIVATE steppe::core steppe::core_internal steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_f2_from_blocks PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_f2_from_blocks PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME f2_from_blocks_unit COMMAND test_f2_from_blocks LABELS "unit")

# --- Host-only unit test of the filter predicates ----------------------------
# Pure C++ TU: exercises the host-pure filter_decision.hpp predicates + the
# include/exclude membership + the --mind per-sample resolution on hand-built
# inputs. Links steppe::io (the filter sources live in the io leaf).
add_executable(test_filters unit/test_filters.cpp)
target_link_libraries(test_filters
    PRIVATE steppe::io steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_filters PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_filters PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME filters_unit COMMAND test_filters LABELS "unit")

# --- Host-only unit test of io::GenoReader ------------------------------------
# Pure C++ TU (no CUDA, no real AADR data): writes tiny synthetic TGENO .geno
# fixtures to a temp path and exercises GenoReader's happy-path gather plus its
# fail-fast boundary checks. A hostile / oversized partition (row >=
# records_present_, or one that would wrap the n_ind*bytes_per_rec multiply) must
# THROW rather than corrupt the heap. Also pins a normal partition decoding
# correctly, empty-partition reject, SNP-range/partial-file edges, and constructor
# fail-fast. Links steppe::io only.
add_executable(test_geno_reader unit/test_geno_reader.cpp)
target_link_libraries(test_geno_reader
    PRIVATE steppe::io steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_geno_reader PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_geno_reader PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME geno_reader_unit COMMAND test_geno_reader LABELS "unit")

# --- Host-only unit test of io::read_snp -------------------------------------
# Pure C++ TU (no CUDA, no real AADR data): writes tiny synthetic .snp fixtures to
# a temp path and exercises read_snp's token-count column decision plus its
# fail-fast behavior on malformed records. A malformed record (too few fields, or a
# non-finite/garbage genpos) THROWS with the offending 1-based line number rather
# than silently dropping the row; a well-formed file yields the correct chrom codes
# (numeric pass-through + X->23) and the Morgan genpos as-read. An all-digit chrom
# code too large for int routes to the negative-sentinel path with no uncaught throw
# (parsed via std::from_chars, not std::stoi). Also pins the >=3-but-<6 field allele
# defaulting, max_snps truncation, and the interior-blank-throws policy. Links
# steppe::io only.
add_executable(test_snp_reader unit/test_snp_reader.cpp)
target_link_libraries(test_snp_reader
    PRIVATE steppe::io steppe::api steppe::warnings)
if(STEPPE_HAVE_GTEST)
  target_link_libraries(test_snp_reader PRIVATE GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_snp_reader PRIVATE STEPPE_TEST_WITH_GTEST=1)
endif()
steppe_add_test(NAME snp_reader_unit COMMAND test_snp_reader LABELS "unit")

# --- Filter oracle test (real AADR) ------------------------------------------
# Reads the raw TGENO triple, decodes via the CPU reference backend, builds the
# filter keep-mask via the io filters, and validates it against an independent
# scalar host oracle on real AADR: no-op-when-default, drop-equals-mask
# bit-identity, exact-mask match, --mind exact, monotonicity, autosomes_only block
# count, strand/tv class. Self-checking executable; takes the data ROOT as argv[1].
add_executable(test_filter_oracle reference/test_filter_oracle.cu)
target_link_libraries(test_filter_oracle
    PRIVATE steppe::io steppe::device steppe::core steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_filter_oracle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME filter_oracle COMMAND test_filter_oracle "${STEPPE_AADR_ROOT}" LABELS "reference;gpu")

# --- Regime-B on-device keep-set parity gate ---------------------------------
# The gate for the extract_f2 regime-B filter moved on-device. Decodes a real v66
# AADR tile, runs CudaBackend::decode_af_compact_filter (the keep-mask kernel + the
# N lockstep CUB compaction) and the verbatim host filter oracle, and asserts the
# on-device keep-set + the lockstep-compacted Q/V/N are bit-identical to the host
# (M_kept exact, chrom/genpos bit-identical, Q/V/N memcmp). Spans the integer
# filters (transversions / maxmiss / monomorphic / autosome) and the FP-fragile
# pooled-MAF boundary. SKIPs (exit 0) with no CUDA device / no raw AADR.
# Self-checking executable; takes the data ROOT as argv[1].
add_executable(test_extract_f2_regimeB_parity reference/test_extract_f2_regimeB_parity.cu)
target_link_libraries(test_extract_f2_regimeB_parity
    PRIVATE steppe::io steppe::device steppe::core_internal steppe::api steppe::warnings)
set_target_properties(test_extract_f2_regimeB_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
steppe_add_test(NAME extract_f2_regimeB_parity COMMAND test_extract_f2_regimeB_parity "${STEPPE_AADR_ROOT}" LABELS "reference;gpu")

# --- `steppe qpadm` CLI acceptance gate --------------------------------------
# Plain C++ host TU (no CUDA): writes the committed real-AADR golden f2 fixtures
# into a temp f2-dir (f2.bin STPF2BK1 + pops.txt + meta.json), runs the built
# `steppe qpadm` binary for the golden model, parses its CSV + JSON output, and
# asserts weights/se/z/p/chisq/dof/rankdrop/popdrop == golden_fit0 + golden_fit1_NRBIG
# (the large nr=39 case) within the two fit-parity tiers. Exercises the name->index
# resolution through the CLI's pops.txt. SKIPs (exit 0) when no CUDA device is
# visible. The host TU device-links nothing; the GPU work is in the spawned child.
# Depends on steppe_app; the binary path + golden dir are argv.
if(STEPPE_BUILD_CLI)
  add_executable(test_cli_qpadm cli/test_cli_qpadm.cpp)
  target_link_libraries(test_cli_qpadm
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_qpadm steppe_app)
  steppe_add_test(NAME cli_qpadm
           COMMAND test_cli_qpadm
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu;slow")

  # --- `steppe qpadm-rotate` golden gate ------------------------------------
  # Plain C++ host TU (no CUDA): runs the built `steppe qpadm-rotate` over the
  # committed real-AADR rotation fixture with golden_rot.json's exact inputs (84
  # models) and diffs every per-model row (weights/p/f4rank/feasible/left/status)
  # against golden_rot.json at the tiers the engine gate uses. The CLI counterpart
  # of test_qpadm_rotation. SKIPs (exit 0) when no CUDA device is visible. Depends on
  # steppe_app; the binary path + golden dir are argv.
  add_executable(test_cli_rotate cli/test_cli_rotate.cpp)
  target_link_libraries(test_cli_rotate
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_rotate steppe_app)
  steppe_add_test(NAME cli_rotate
           COMMAND test_cli_rotate
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu")

  # --- `steppe qpwave` golden gate ------------------------------------------
  # Plain C++ host TU (no CUDA): runs the built `steppe qpwave` over the committed
  # real-AADR 9-pop f2 fixture with golden_qpwave's exact inputs (NO --target — the
  # distinguishing qpWave invocation; left[0]=reference) and diffs the rank-sweep
  # output (est_rank/f4rank/rankdrop chisq/dof/p + the per_rank cross-check) against
  # the constants test_qpwave_parity gates. The CLI counterpart of qpwave_parity.
  # SKIPs (exit 0) when no CUDA device is visible. Depends on steppe_app; the binary
  # path + golden dir are argv.
  add_executable(test_cli_qpwave cli/test_cli_qpwave.cpp)
  target_link_libraries(test_cli_qpwave
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_qpwave steppe_app)
  steppe_add_test(NAME cli_qpwave
           COMMAND test_cli_qpwave
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu")

  # --- `steppe f4` standalone-f4 golden gate --------------------------------
  # Plain C++ host TU (no CUDA): runs the built `steppe f4` over the committed
  # real-AADR 9-pop f2 fixture for all 60 quartets of the fixture-matched golden and
  # diffs the est/se/z/p table row-for-row at rtol 1e-6. Also exercises the
  # single-quartet --pops convenience. SKIPs (exit 0) when no CUDA device is visible.
  # Depends on steppe_app; the binary path + golden dir are argv.
  add_executable(test_cli_f4 cli/test_cli_f4.cpp)
  target_link_libraries(test_cli_f4
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_f4 steppe_app)
  steppe_add_test(NAME cli_f4
           COMMAND test_cli_f4
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu")

  # --- I/O fault taxonomy gate: torn --out write returns kExitIoError(4) -----
  # Plain C++ host TU (no CUDA): runs the built `steppe f4 ... --out /dev/full`
  # (open succeeds, every write ENOSPCs) and asserts exit == 4 with a "write failed"
  # diagnostic — the shared emit_to_destination open->write->flush->good() contract.
  # A control `--out <tmp>/ok.csv` run asserts exit 0 + a non-empty, header-bearing
  # file. f4 is the representative emit site (every cmd_*.cpp routes through the same
  # helper). SKIPs (exit 0) when no CUDA device is visible.
  add_executable(test_cli_io_fault cli/test_cli_io_fault.cpp)
  target_link_libraries(test_cli_io_fault
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_io_fault steppe_app)
  steppe_add_test(NAME cli_io_fault
           COMMAND test_cli_io_fault
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu")

  # --- `steppe qpdstat` Part-A golden gate (D-stat/f4 over an f2-dir) --------
  # Plain C++ host TU (no CUDA): the qpDstat clone of the cli_f4 gate. Runs the
  # built `steppe qpdstat` over the committed real-AADR 9-pop f2 fixture for all 60
  # quadruples and diffs the est/se/z/p table row-for-row at rtol 1e-6. The qpdstat
  # f2-path == f4, so est is byte-identical to the f4 golden. Also exercises the
  # single-quadruple --pops convenience AND the --prefix (Part B) fail-fast guard.
  # SKIPs (exit 0) when no CUDA device is visible.
  add_executable(test_cli_qpdstat cli/test_cli_qpdstat.cpp)
  target_link_libraries(test_cli_qpdstat
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_qpdstat steppe_app)
  steppe_add_test(NAME cli_qpdstat
           COMMAND test_cli_qpdstat
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu")

  # --- `steppe qpdstat --prefix` Part-B golden gate (genotype-path D) --------
  # Plain C++ host TU (no CUDA): runs the built `steppe qpdstat --prefix <raw TGENO>`
  # over all 60 quadruples of the genotype-path golden and diffs the est/se/z table
  # row-for-row at rtol 1e-6 — through the per-SNP D kernel + num/den block-jackknife
  # (never the f2 cache). The est is the NORMALIZED D (~+-0.06), distinct from the
  # f2-path f4 (~10x smaller). The test takes <steppe-binary> <aadr-root>
  # <golden-dir> and derives the raw TGENO prefix internally. SKIPs (exit 0) when no
  # CUDA device is visible OR the TGENO prefix / golden is absent. Also exercises the
  # single-quadruple --pops convenience. NO SYNTHETIC DATA.
  add_executable(test_cli_dstat_geno cli/test_cli_dstat_geno.cpp)
  target_link_libraries(test_cli_dstat_geno PRIVATE steppe::warnings)
  add_dependencies(test_cli_dstat_geno steppe_app)
  steppe_add_test(NAME cli_dstat_geno
           COMMAND test_cli_dstat_geno
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu")

  # --- `steppe dates` admixture-dating golden gate --------------------------
  # Plain C++ host TU (no CUDA): runs the built `steppe dates --prefix <raw TGENO>
  # --target PUR --left CEU,YRI` and asserts the inferred admixture date reproduces
  # the DATES reference golden (9.742 gen, SE 0.317) — through the cuFFT
  # autocorrelation LD engine (never the f2 cache, never a host O(M^2) loop). The
  # date is gated at rtol 2%, the SE at atol 0.10. SKIPs (exit 0) when no CUDA device
  # is visible OR the raw AADR is absent. NO SYNTHETIC DATA.
  add_executable(test_cli_dates cli/test_cli_dates.cpp)
  target_link_libraries(test_cli_dates PRIVATE steppe::warnings)
  add_dependencies(test_cli_dates steppe_app)
  steppe_add_test(NAME cli_dates
           COMMAND test_cli_dates
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}" LABELS "cli;gpu;slow")

  # --- `steppe qpfstats` genotype-path smoothed-f2-tensor golden gate --------
  # Plain C++ host TU (no CUDA): runs the built `steppe qpfstats --prefix <raw TGENO>
  # --pops <9 sorted fit0> --out-dir <tmp>`, reads the smoothed f2.bin (STPF2BK1),
  # and diffs every (i,j,block) entry vs the AT2 smoothed-f2 tensor golden (9x9x711
  # upper-tri incl diag) at rtol 1e-6 — through the qpDstat-B numerator engine + the
  # on-device shared-factor smoothing solve. The all-NaN block is matched as 0. SKIPs
  # (exit 0) when no CUDA device is visible OR the prefix / golden is absent. NO
  # SYNTHETIC DATA.
  add_executable(test_cli_qpfstats cli/test_cli_qpfstats.cpp)
  target_link_libraries(test_cli_qpfstats PRIVATE steppe::warnings)
  add_dependencies(test_cli_qpfstats steppe_app)
  steppe_add_test(NAME cli_qpfstats
           COMMAND test_cli_qpfstats
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu")

  # --- `steppe f3` standalone-f3 golden gate --------------------------------
  # Plain C++ host TU (no CUDA): the three-slab clone of the cli_f4 gate. Runs the
  # built `steppe f3` over the committed real-AADR 9-pop f2 fixture for all 36 triples
  # and diffs the est/se/z/p table row-for-row at rtol 1e-6. Also exercises the
  # single-triple --pops convenience. SKIPs (exit 0) when no CUDA device is visible.
  add_executable(test_cli_f3 cli/test_cli_f3.cpp)
  target_link_libraries(test_cli_f3
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_f3 steppe_app)
  steppe_add_test(NAME cli_f3
           COMMAND test_cli_f3
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu")

  # --- `steppe qpgraph` single-graph-fit golden gate ------------------------
  # Plain C++ host TU (no CUDA): runs the built `steppe qpgraph` over the committed
  # real-AADR afprod=FALSE 9-pop f2 fixture for the golden topology and asserts the
  # CSV `summary` score + the `edges` admix weight (pSteppe->aCW) match the AT2
  # golden. SKIPs (exit 0) when no CUDA device is visible.
  add_executable(test_cli_qpgraph cli/test_cli_qpgraph.cpp)
  target_link_libraries(test_cli_qpgraph
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_qpgraph steppe_app)
  steppe_add_test(NAME cli_qpgraph
           COMMAND test_cli_qpgraph
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu")

  # --- `steppe f4-ratio` standalone-f4-ratio golden gate --------------------
  # Plain C++ host TU (no CUDA): the five-column clone of the cli_f4/cli_f3 gate.
  # Runs the built `steppe f4-ratio` over the committed real-AADR 9-pop f2 fixture for
  # all 10 5-tuples and diffs the alpha/se/z table row-for-row at rtol 1e-6. Also
  # exercises the single-tuple --pops convenience. SKIPs (exit 0) when no CUDA device
  # is visible.
  add_executable(test_cli_f4ratio cli/test_cli_f4ratio.cpp)
  target_link_libraries(test_cli_f4ratio
      PRIVATE steppe::core_internal steppe::api steppe::warnings)
  add_dependencies(test_cli_f4ratio steppe_app)
  steppe_add_test(NAME cli_f4ratio
           COMMAND test_cli_f4ratio
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu")

  # --- native `steppe cache` host-only inspector gate -----------------------
  # Plain C++ host TU (NO CUDA, NO GPU): drives the built `steppe cache ls/show/verify`
  # over the committed real 9-pop example cache and asserts the parity behaviors the
  # Python steppe-cache tool has (size / f2_cache_id / header-P cross-checks, report-only
  # when there is no stored id, the dtype gate, and missing-root -> exit 0). Because the
  # cache tool is host-only this runs everywhere (label "cli", not "gpu"); it SKIPs only
  # if the example cache is absent. argv = steppe binary + the example cache dir. NO
  # SYNTHETIC DATA (the fixture is the committed real-AADR example cache).
  add_executable(test_cli_cache cli/test_cli_cache.cpp)
  target_link_libraries(test_cli_cache PRIVATE steppe::warnings)
  add_dependencies(test_cli_cache steppe_app)
  steppe_add_test(NAME cli_cache
           COMMAND test_cli_cache
                   "$<TARGET_FILE:steppe_app>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/../docs/examples/f2_9pop" LABELS "cli")

  # --- READv2 concordance-validator host-only gate --------------------------
  # Plain C++ host TU (NO CUDA, NO GPU): the Phase-0 "ruler". Writes hand-authored
  # TOY READv2 tables (steppe schema) to temp files, runs the built
  # `steppe readv2-concord --a .. --b ..` and asserts the confusion matrix, degree
  # agreement, P0_norm within-tol counts, coverage, threshold PASS/FAIL, and exit
  # code against known-by-construction expectations. Tests the validator's OWN diff
  # arithmetic (NOT science). Host lane (label "cli", NOT "gpu") — needs no device.
  add_executable(test_cli_readv2_concord cli/test_cli_readv2_concord.cpp)
  target_link_libraries(test_cli_readv2_concord PRIVATE steppe::warnings)
  add_dependencies(test_cli_readv2_concord steppe_app)
  steppe_add_test(NAME cli_readv2_concord
           COMMAND test_cli_readv2_concord "$<TARGET_FILE:steppe_app>" LABELS "cli")

  # --- `steppe extract-f2 -> qpadm` end-to-end gate -------------------------
  # Plain C++ host TU (no CUDA): runs the built `steppe extract-f2` on the raw v66
  # AADR triple for the golden pop sets, then `steppe qpadm --f2-dir <extracted>`, and
  # asserts the result reproduces the committed AT2 goldens (golden_fit0 maxmiss=0 +
  # golden_fitNA maxmiss=0.99) within the established tiers. Proves the whole chain
  # end-to-end: io->decode->filter->assign_blocks->precompute AND the STPF2BK1 dir
  # writer with REAL vpair (golden_fitNA's dropped block is reproduced only because
  # vpair is real, not zeros). SKIPs (exit 0) when no CUDA device is visible OR the raw
  # AADR is absent. Depends on steppe_app; the binary path + AADR root + golden dir are
  # argv. NO SYNTHETIC DATA.
  add_executable(test_cli_extract_qpadm cli/test_cli_extract_qpadm.cpp)
  target_link_libraries(test_cli_extract_qpadm PRIVATE steppe::warnings)
  add_dependencies(test_cli_extract_qpadm steppe_app)
  steppe_add_test(NAME cli_extract_qpadm
           COMMAND test_cli_extract_qpadm
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/reference/goldens/at2" LABELS "cli;gpu;slow")

  # --- PA-prefix CLI auto-cover gate ----------------------------------------
  # Plain C++ host TU (no CUDA): runs the built steppe twice per tool — once with the
  # PA (SNP-major PACKEDANCESTRYMAP) --prefix and once with the raw TGENO --prefix —
  # and asserts the PA run WORKS (exit 0) AND matches the TGENO run: extract-f2's
  # f2.bin byte-identical, qpdstat est/se/z row-for-row (rtol 1e-6), qpfstats smoothed
  # f2.bin byte-identical. convertf is lossless, so a correct PA reader MUST reproduce
  # the TGENO result (no new golden). SKIPs (exit 0) when no CUDA device is visible OR
  # either prefix is absent. NO SYNTHETIC DATA.
  add_executable(test_cli_pa_prefix cli/test_cli_pa_prefix.cpp)
  target_link_libraries(test_cli_pa_prefix PRIVATE steppe::warnings)
  add_dependencies(test_cli_pa_prefix steppe_app)
  steppe_add_test(NAME cli_pa_prefix
           COMMAND test_cli_pa_prefix
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}" LABELS "cli;gpu")

  # --- EIGENSTRAT-prefix CLI auto-cover gate --------------------------------
  # Plain C++ host TU (no CUDA): runs the built steppe twice per tool — once with the
  # EIGENSTRAT (ASCII) --prefix and once with the convertf-PA (GENO) --prefix of the
  # SAME 9-pop subset — and asserts the EIGENSTRAT run WORKS (exit 0) AND matches the
  # PA run: extract-f2's f2.bin byte-identical, qpdstat est/se/z row-for-row (rtol
  # 1e-6). Both fixtures are convertf outputs of one PA source, so a correct
  # EIGENSTRAT reader MUST reproduce the PA result (no new golden). SKIPs (exit 0)
  # when no CUDA device is visible OR the fixtures_eig prefixes are absent. NO
  # SYNTHETIC DATA.
  add_executable(test_cli_eigenstrat_prefix cli/test_cli_eigenstrat_prefix.cpp)
  target_link_libraries(test_cli_eigenstrat_prefix PRIVATE steppe::warnings)
  add_dependencies(test_cli_eigenstrat_prefix steppe_app)
  steppe_add_test(NAME cli_eigenstrat_prefix
           COMMAND test_cli_eigenstrat_prefix
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}" LABELS "cli;gpu")

  # --- PLINK-prefix CLI auto-cover gate -------------------------------------
  # Plain C++ host TU (no CUDA): runs the built steppe twice per tool — once with the
  # PLINK (.bed/.bim/.fam) --prefix and once with the convertf-PA (GENO) --prefix of
  # the SAME 9-pop subset — and asserts the PLINK run WORKS (exit 0) AND matches the
  # PA run: extract-f2's f2.bin byte-identical, qpdstat est/se/z row-for-row (rtol
  # 1e-6). A correct PLINK reader (LUT + bit-flip + ref:=A1) MUST reproduce the PA
  # result. Also exercises the format-aware --prefix expansion (no .geno ->
  # resolve_genotype_triple picks .bed/.bim/.fam) + the read_bim/read_fam dispatch.
  # No new golden. SKIPs (exit 0) when no CUDA device is visible OR the fixtures_eig
  # prefixes are absent. NO SYNTHETIC DATA.
  add_executable(test_cli_plink_prefix cli/test_cli_plink_prefix.cpp)
  target_link_libraries(test_cli_plink_prefix PRIVATE steppe::warnings)
  add_dependencies(test_cli_plink_prefix steppe_app)
  steppe_add_test(NAME cli_plink_prefix
           COMMAND test_cli_plink_prefix
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}" LABELS "cli;gpu")

  # --- ANCESTRYMAP-prefix CLI auto-cover gate -------------------------------
  # Plain C++ host TU (no CUDA): runs the built steppe twice per tool — once with the
  # ANCESTRYMAP (unpacked text triple) --prefix and once with the convertf-PA (GENO)
  # --prefix of the SAME 9-pop subset — and asserts the ANCESTRYMAP run WORKS (exit 0)
  # AND matches the PA run: extract-f2's f2.bin byte-identical, qpdstat est/se/z
  # row-for-row (rtol 1e-6). Both fixtures are convertf outputs of one PA source, so a
  # correct ANCESTRYMAP reader MUST reproduce the PA result (no new golden). SKIPs
  # (exit 0) when no CUDA device is visible OR the fixtures_eig prefixes are absent.
  # NO SYNTHETIC DATA.
  add_executable(test_cli_ancestrymap_prefix cli/test_cli_ancestrymap_prefix.cpp)
  target_link_libraries(test_cli_ancestrymap_prefix PRIVATE steppe::warnings)
  add_dependencies(test_cli_ancestrymap_prefix steppe_app)
  steppe_add_test(NAME cli_ancestrymap_prefix
           COMMAND test_cli_ancestrymap_prefix
                   "$<TARGET_FILE:steppe_app>"
                   "${STEPPE_AADR_ROOT}" LABELS "cli;gpu")
endif()

# --- nanobind Python gate ----------------------------------------------------
# Reproduces the real-AADR golden_fit0 through the Python API (read_f2 -> qpadm ->
# the pandas-shaped result) + the f2 numpy round-trip + a qpwave/qpadm_search
# sanity, gated by STEPPE_BUILD_PYTHON. SKIPs (the pytest device-present guard) when
# no CUDA device is visible. STEPPE_BINDINGS_DIR points the conftest sys.path shim at
# the build-tree-staged steppe package. Depends on _core so the extension is built
# before the test runs. Stages the committed fixtures into a temp f2-dir.
if(STEPPE_BUILD_PYTHON)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter)
  steppe_add_test(NAME py_qpadm
           COMMAND "${Python_EXECUTABLE}" -m pytest -v
                   "${CMAKE_CURRENT_SOURCE_DIR}/python" LABELS "python;gpu")
  # STEPPE_AADR_ROOT is passed so test_py_dstat_geno.py finds the raw TGENO prefix;
  # that test SKIPs cleanly when the prefix is absent.
  set_tests_properties(py_qpadm PROPERTIES
      ENVIRONMENT "STEPPE_BINDINGS_DIR=${CMAKE_BINARY_DIR}/bindings;STEPPE_AADR_ROOT=${STEPPE_AADR_ROOT}")
  # CTest does not build, so the `cmake --build` step (which builds _core) must
  # precede `ctest`.
endif()
