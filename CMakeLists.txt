# CMakeLists.txt (top level)
#
# steppe — GPU/CUDA reimplementation of admixtools2/qpAdm.
# Milestone M0: lift the VALIDATED f2 kernel into the production structure.
#
# Top-level responsibility ONLY: project policy + add_subdirectory (architecture.md
# §6). Each src/ subdirectory owns exactly one target; layering is enforced by the
# compiler via link visibility (CUDA is PRIVATE to steppe_device, so core/api/tests
# physically cannot include a CUDA header — architecture.md §4).
#
# Build on the sm_120 remote box (RTX 5090, CUDA 13). The local dev box is an
# RTX 2070 / CUDA 11.8 — WRONG arch; do NOT build here (milestone brief).

cmake_minimum_required(VERSION 3.28)          # box ships cmake 3.28; CMAKE_CUDA_ARCHITECTURES=120
                                              # (a bare number passed to nvcc -arch) is handled fine.
                                              # architecture.md §6 pins 3.30 for the release matrix's
                                              # named-Blackwell handling; 3.28 suffices for the M0
                                              # single-arch (sm_120) structure-lift on the box.

project(steppe
        VERSION 0.1.0
        DESCRIPTION "GPU/CUDA qpAdm: f-statistics & model fitting"
        LANGUAGES CXX CUDA)

# --- Language policy (architecture.md §6) ------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)                    # do NOT trust nvcc's host-derived default
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)          # clang-tidy/IWYU/clangd (Ninja/Make only)

# Position the build/ tree off the source root by default; presets override this.
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# --- Reusable build logic (cmake/ — NO targets defined in these) -------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SteppeOptions)     # all option()/cache vars in one place
include(CUDAArch)          # CMAKE_CUDA_ARCHITECTURES policy (M0: sm_120)
include(SteppeWarnings)    # defines INTERFACE target steppe::warnings
include(SteppeSanitizers)  # consumes STEPPE_SANITIZER (host-only asan/ubsan flags;
                           # compute-lane marker). MUST precede add_subdirectory()
                           # so the directory-scoped sanitizer flags propagate; the
                           # empty default is a no-op (Release/CI parity stays clean).

# Position-independent code for the whole build when the Python bindings are on: the
# nanobind extension `_core` is a SHARED module, and it links the first-party STATIC libs
# (steppe_core / steppe_device / steppe_io / steppe_access). A shared object cannot embed
# non-PIC static objects (R_X86_64_PC32 relocations fail), so every static lib + the CUDA
# device-link must be PIC. PIC is negligible cost and standard for libs feeding a shared
# module; it does not change the CLI-only / tests-only build (the option is OFF there).
# Must follow include(SteppeOptions) so STEPPE_BUILD_PYTHON is defined.
if(STEPPE_BUILD_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# --- CUDA toolkit (cuBLAS for the f2 GEMMs; architecture.md §3, §8) ----------
find_package(CUDAToolkit REQUIRED)

# --- Threads (the SPMG per-device fan-out; architecture.md §11.4) ------------
# steppe_core drives the G per-device backends concurrently on G host threads
# (one std::jthread per device — fstats/f2_blocks_multigpu.cpp), so the host
# orchestration links the platform thread library. THREADS_PREFER_PTHREAD_FLAG
# makes the imported Threads::Threads target carry -pthread (compile + link) on
# POSIX, the CMake-documented way to request pthreads (FindThreads).
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# --- ccache compiler launcher (architecture.md §6; build-iteration only) ------
# Opt-in via STEPPE_CCACHE (default ON, SteppeOptions.cmake). A no-op when ccache
# is not installed, so the box's stock build is unaffected. ccache hashes the full
# compile command + preprocessed source and is nvcc-aware, so cached objects are
# byte-identical — zero codegen/parity impact, purely a rebuild-latency win on the
# heavy .cu recompiles. CMAKE_<LANG>_COMPILER_LAUNCHER is read as each target is
# created, so this MUST precede the add_subdirectory() calls below.
if(STEPPE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER  "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "steppe: ccache enabled (${CCACHE_PROGRAM})")
  endif()
endif()

# --- Targets (one per subdirectory) ------------------------------------------
# Order matters: api (INTERFACE) → core_internal (INTERFACE) → device → core.
# io is an ISOLATED LEAF (architecture.md §4): genotype decode front-end, depends
# on nothing in core/device. app/bindings/benchmarks arrive in later milestones.
add_subdirectory(include)        # steppe_api          (INTERFACE: public headers)
add_subdirectory(src/io)         # steppe_io           (LEAF: genotype readers; M1)
add_subdirectory(src/core)       # steppe_core_internal (INTERFACE) + steppe_core (host)
add_subdirectory(src/device)     # steppe_device        (CUDA, PRIVATE cuBLAS)

# access — steppe_access: the CUDA-FREE host access helpers (the f2-dir READER +
# the pop-name->index resolver) shared by BOTH the CLI app AND the Python bindings
# (cli-bindings.md M(py-1) §0 finding: those helpers were exe-private; this is their
# shared linkable home). A PLAIN CXX target; built whenever either consumer is on.
if(STEPPE_BUILD_CLI OR STEPPE_BUILD_PYTHON)
  add_subdirectory(src/access)
endif()

# extract — steppe_extract: the genotype->f2 EXTRACT library entry (run_extract_f2) + the
# STPF2BK1 dir WRITER, shared by BOTH the CLI app AND the Python bindings (M(py-2) §0
# finding: the extract chain was exe-private in src/app, which is CLI-only — a wheel build
# would have no extract entry; this is its shared linkable home). UNLIKE access it reaches
# the GPU through the CUDA-free seams (links steppe::{core,io,device}); still a PLAIN CXX
# target (NO LANGUAGES CUDA, NO .cu). Built whenever either consumer is on.
if(STEPPE_BUILD_CLI OR STEPPE_BUILD_PYTHON)
  add_subdirectory(src/extract)
endif()

# app — the steppe CLI (Phase 3 / step 2 productization). A PLAIN CXX target (NO
# LANGUAGES CUDA) behind the already-present STEPPE_BUILD_CLI option
# (SteppeOptions.cmake:23, default OFF), per cli-bindings.md §6.1 + architecture.md §4.
# It reaches the GPU ONLY through the CUDA-free seams; CLI11 + the CPM machinery it
# fetches stay PRIVATE to the src/app subtree (never reaching core/device).
if(STEPPE_BUILD_CLI)
  add_subdirectory(src/app)
endif()

# bindings — the nanobind Python module steppe._core (Phase 3 / M(py-1)). A PLAIN CXX
# target (NO LANGUAGES CUDA, NO .cu) behind STEPPE_BUILD_PYTHON (SteppeOptions.cmake:22,
# default OFF), per cli-bindings.md §6.1 + architecture.md §4/§15. It reaches the GPU
# ONLY through the CUDA-free seams (build_resources / run_qpadm* / run_qpwave /
# upload_f2_blocks_to_device) + steppe::access; nanobind + its CPM machinery stay PRIVATE
# to the bindings subtree (never reaching core/device). A leaked CUDA header hard-fails
# this host compile — the §4 layering / arch-grep gate is structural (cli-bindings.md §6.2).
if(STEPPE_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()

# examples — the read_f2 -> qpadm -> inspect quick-starts (C++ + Python), behind the
# opt-in STEPPE_BUILD_EXAMPLES (SteppeOptions.cmake, default OFF). The C++ quickstart
# reuses the CUDA-free seams the CLI `qpadm` command uses (no duplicated compute), so it
# links steppe::access — which only exists when STEPPE_BUILD_CLI (or _PYTHON) is also ON;
# examples/CMakeLists.txt degrades gracefully (skips the C++ target with a message) when
# that target is absent. A pure-host CXX add (NO LANGUAGES CUDA), like the app subtree.
# Placed after the access/app/bindings targets so they exist for the link + the
# `if(NOT TARGET steppe::access)` probe.
if(STEPPE_BUILD_EXAMPLES)
  add_subdirectory(docs/examples)
endif()

if(STEPPE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# docs — opt-in GENERATED API reference (Doxygen C++ over include/steppe + pdoc Python
# over bindings/steppe), behind STEPPE_BUILD_DOCS (SteppeOptions.cmake, default OFF).
# DOC-ONLY: cmake/Docs.cmake defines no library target and never touches the compute
# path; each custom target is a soft no-op when its tool is absent. The Doxygen `docs`
# target needs NO compiler and NO CUDA (parses headers as text), so it is the one target
# that builds on the LOCAL non-Blackwell box (architecture.md §6 names Doxygen as the
# API-doc generator). Placed last so PROJECT_VERSION (single-sourced into the Doxyfile's
# PROJECT_NUMBER) is set. kimiactions C5.
if(STEPPE_BUILD_DOCS)
  include(Docs)
endif()
