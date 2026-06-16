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

# --- Observability / sanitizers (named now, wired in later milestones) -------

option(STEPPE_NVTX "Emit NVTX ranges (architecture.md §10; zero-overhead off)" OFF)

# Empty ⇒ no sanitizer. Accepts a list like "asan;ubsan" or "compute"
# (architecture.md §6 presets, §13). Wired in cmake/SteppeSanitizers.cmake at a
# later milestone; declared here so the preset cache vars resolve.
set(STEPPE_SANITIZER "" CACHE STRING
    "Sanitizer set: empty | asan;ubsan | compute (architecture.md §6, §13)")

# --- Release-only device-codegen opt-ins (architecture.md §6) ----------------

option(STEPPE_COMPRESS_FATBIN
       "Use nvcc --compress-mode to shrink device fatbins on Release (§6)" OFF)
