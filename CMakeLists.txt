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

# --- Targets (one per subdirectory) ------------------------------------------
# Order matters: api (INTERFACE) → core_internal (INTERFACE) → device → core.
# io is an ISOLATED LEAF (architecture.md §4): genotype decode front-end, depends
# on nothing in core/device. app/bindings/benchmarks arrive in later milestones.
add_subdirectory(include)        # steppe_api          (INTERFACE: public headers)
add_subdirectory(src/io)         # steppe_io           (LEAF: genotype readers; M1)
add_subdirectory(src/core)       # steppe_core_internal (INTERFACE) + steppe_core (host)
add_subdirectory(src/device)     # steppe_device        (CUDA, PRIVATE cuBLAS)

if(STEPPE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
