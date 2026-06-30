# cmake/SteppeOptions.cmake
#
# All option()/cache vars in ONE place (architecture.md §4 "NO targets defined
# here", §6). Included once from the top-level CMakeLists.txt before any target
# is declared, so every subdirectory sees the same switches.
#
# M0 scope: the switches the structure-lift slice actually consumes. The wider
# set (sanitizers, NVTX, Python, compress-fatbin, device-LTO) is named here with
# its architecture.md home so later milestones add behavior, not new option
# vocabulary.

include_guard(GLOBAL)

# --- Build-surface toggles ---------------------------------------------------

# Tests are the M0 gate (CPU-reference-equivalence; ROADMAP §6 DoD). ON by
# default for a from-scratch dev build; CI presets set it explicitly.
option(STEPPE_BUILD_TESTS "Build steppe tests (CTest + GoogleTest)" ON)

# Python bindings / CLI are later phases (ROADMAP Phase 3); declared here so the
# top-level guards have a real variable, default OFF for the M0 slice.
option(STEPPE_BUILD_PYTHON "Build the nanobind Python bindings (Phase 3)" OFF)
option(STEPPE_BUILD_CLI    "Build the steppe CLI app (Phase 3)"          OFF)

# examples/ quick-starts (read_f2 -> qpadm -> inspect). OPT-IN, default OFF, mirroring
# STEPPE_BUILD_CLI above. The C++ quickstart reuses the SAME CUDA-free entry points the
# CLI `qpadm` command uses (read_f2_dir / PopResolver / build_resources /
# upload_f2_blocks_to_device / run_qpadm) via steppe::access + steppe::device, so it
# needs steppe::access — i.e. STEPPE_BUILD_CLI (or _PYTHON) must ALSO be ON, and it only
# builds on a CUDA box (the local RTX 2070 / wrong arch cannot). examples/CMakeLists.txt
# skips the C++ target with a clear message when steppe::access is absent. The Python
# quickstart is a plain script (no build).
option(STEPPE_BUILD_EXAMPLES "Build examples/ (read_f2 -> qpadm quick-starts)" OFF)

# Generated API reference (Doxygen over include/steppe + pdoc over bindings/steppe).
# OPT-IN, default OFF, mirroring the build-surface toggles above. cmake/Docs.cmake gates
# each target on find_package(Doxygen) / find_program(pdoc), so it is a SOFT NO-OP when
# the tools are absent and never breaks a normal build. DOC-ONLY: defines no library
# target and is NOT on the compute/precision path. The Doxygen `docs` target needs NO
# compiler and NO CUDA (it parses headers as text), so — unlike every other target — it
# builds on the LOCAL non-Blackwell box (iterate docs locally; architecture.md §6 lists
# Doxygen as the API-doc generator). kimiactions C5.
option(STEPPE_BUILD_DOCS "Build the generated API reference (Doxygen + pdoc; doc-only, opt-in)" OFF)

# --- Warning policy + build-iteration speedups (architecture.md §6, §8) -------
#
# Warnings-as-errors is the M0 contract (C++20 with warnings-as-errors), so ON is
# the DEFAULT and the shipping invariant — CI/release presets pin it ON. OFF is a
# dev-only escape hatch: a local build keeps -Wall -Wextra but proceeds THROUGH a
# warning instead of halting mid-experiment. cmake/SteppeWarnings.cmake consumes
# this to gate the three -Werror tokens (host -Werror / device --Werror;all-warnings
# / the -Xcompiler host-forward). NOT on the compute/precision path — it changes
# only whether a warning is fatal, never codegen.
option(STEPPE_WERROR "Treat warnings as errors (default ON; OFF for local iteration)" ON)

# ccache as the compiler launcher: a no-op when ccache is absent (the top-level
# CMakeLists.txt only wires CMAKE_{CXX,CUDA}_COMPILER_LAUNCHER when find_program
# locates it). Byte-identical objects — ccache hashes the full compile command +
# preprocessed source and is nvcc-aware — so it never touches codegen/parity; a
# pure rebuild-latency win on the ephemeral box's heavy .cu recompiles.
option(STEPPE_CCACHE "Use ccache as compiler launcher when found" ON)

# --- Precision emulation tuning (architecture.md §3, §12; ROADMAP §0) --------
#
# The fixed-slice-Ozaki engagement symbols
#   cublasSetEmulationStrategy / cublasSetFixedPointEmulationMantissaControl /
#   cublasSetFixedPointEmulationMaxMantissaBitCount
# are gated behind STEPPE_HAVE_EMU_TUNING in the device code. The MEASURED policy
# (ROADMAP §0: FIXED 40-bit default, never DYNAMIC) REQUIRES these symbols, so
# the production build turns the macro ON. It maps to -DSTEPPE_HAVE_EMU_TUNING=1
# on the device target (set in src/device, kept PRIVATE to steppe_device).
option(STEPPE_HAVE_EMU_TUNING
       "Enable cuBLAS fixed-slice Ozaki emulation tuning calls (REQUIRED for the \
measured precision policy; architecture.md §12, ROADMAP §0)" ON)

# --- Observability / sanitizers ----------------------------------------------

# NVTX phase-boundary ranges (architecture.md §10). OFF (shipping default) is
# structurally zero-overhead: STEPPE_NVTX_RANGE (src/core/internal/nvtx.hpp)
# expands to nothing and no NVTX header is included, so object code is byte-
# identical to a build without it. ON maps to -DSTEPPE_NVTX on steppe_device
# (set PRIVATE in src/device, mirroring STEPPE_HAVE_EMU_TUNING) and pulls in the
# header-only NVTX v3 C++ API from the CUDA toolkit (no library link). HOST-side
# annotation only — never touches the statistic stream (§12 parity untouched).
option(STEPPE_NVTX "Emit NVTX ranges (architecture.md §10; zero-overhead off)" OFF)

# Empty ⇒ no sanitizer. Accepts a list like "asan;ubsan" or "compute"
# (architecture.md §6 presets, §13). Consumed in cmake/SteppeSanitizers.cmake
# (included from the top-level CMakeLists.txt right after this module): it maps
# asan;ubsan → host-only -fsanitize flags and compute → the CI compute-sanitizer
# lane marker. Declared here so the preset cache vars resolve.
set(STEPPE_SANITIZER "" CACHE STRING
    "Sanitizer set: empty | asan;ubsan | compute (architecture.md §6, §13)")

# --- Release-only device-codegen opt-ins (architecture.md §6) ----------------

option(STEPPE_COMPRESS_FATBIN
       "Use nvcc --compress-mode to shrink device fatbins on Release (§6)" OFF)
