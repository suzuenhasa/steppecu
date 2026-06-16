# cmake/CUDAArch.cmake
#
# CMAKE_CUDA_ARCHITECTURES policy (architecture.md §6). NO targets defined here.
#
# M0 reality: the validated f2 kernel is benchmarked and shipped on the remote
# RTX 5090 box — consumer/workstation Blackwell == sm_120. The local dev box is
# an RTX 2070 / CUDA 11.8 (Turing, sm_75) which is the WRONG arch and must NOT be
# built on (per the milestone brief); building happens on the sm_120 box. So the
# M0 default is a single concrete target: 120.
#
# NOTE on the wider ship list. architecture.md §6 pins a shippable
# Turing→Blackwell base list ("75-real;80-real;86-real;89-real;90-real;
# 100-real;120-real;120-virtual") for the release artifact. That is the eventual
# release matrix; for the M0 structure-lift on the sm_120 validation box we pin
# 120 so the slice builds fast and runs where it was measured. The release preset
# (CMakePresets.json) carries the full list; flip STEPPE_CUDA_ARCH there.
#
# 100 (datacenter Blackwell) is NOT cubin-compatible with 120 despite both being
# "Blackwell"; 103/110/121 and the a/f accelerated family targets are
# deliberately omitted (narrow SKUs, not generic-fatbin-friendly) — see §6.

include_guard(GLOBAL)

# CUDA-arch resolution (CMake 3.28 + CUDA 13 wrinkle, handled here).
#
# With CMake 3.28 and CUDA 13, `project(... LANGUAGES CUDA)` auto-populates
# CMAKE_CUDA_ARCHITECTURES during compiler detection (to the toolkit's lowest
# baseline, sm_75) BEFORE this module runs — so a bare
#   cmake -S . -B build -GNinja
# (the task's documented configure command) would otherwise silently build for
# sm_75, NOT the sm_120 RTX-5090 validation box, and the FP64-emulation path would
# never exercise on Blackwell. A plain `if(NOT DEFINED ...)` cannot fix this
# because the variable is already defined by the time we get here.
#
# Resolution: a dedicated cache override `STEPPE_CUDA_ARCH` carries the project's
# intended arch (M0 default: 120). When the user/preset explicitly passes
# -DCMAKE_CUDA_ARCHITECTURES=<x>, that wins (we detect it via STEPPE_CUDA_ARCH
# being out of sync). Concretely: the preset and the bare command both end up at
# the M0 sm_120 default; an explicit -DCMAKE_CUDA_ARCHITECTURES on the command
# line still takes precedence because CMake records it as a cache entry the user
# set, which we honor below.
set(STEPPE_CUDA_ARCH "120" CACHE STRING
    "steppe target CUDA arch (M0 default: consumer/workstation Blackwell sm_120 \
== RTX 5090 validation box). Overridden by an explicit -DCMAKE_CUDA_ARCHITECTURES.")

# Honor an explicit user/preset -DCMAKE_CUDA_ARCHITECTURES (anything other than the
# bare toolkit auto-default). Otherwise pin the project default (sm_120). The
# toolkit auto-default on this box is "75"; treat that single bare value as
# "unset by the user" and replace it with the project arch.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES
   OR CMAKE_CUDA_ARCHITECTURES STREQUAL ""
   OR CMAKE_CUDA_ARCHITECTURES STREQUAL "75"
   OR CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF")
  set(CMAKE_CUDA_ARCHITECTURES "${STEPPE_CUDA_ARCH}" CACHE STRING
      "CUDA arch (M0 default: consumer/workstation Blackwell sm_120)" FORCE)
endif()

# Shippable base list for the release artifact (architecture.md §6). Not applied
# by default at M0; the release preset selects it. Exposed as a cache var so a
# release configure can do -DCMAKE_CUDA_ARCHITECTURES="${STEPPE_CUDA_ARCH_RELEASE}".
set(STEPPE_CUDA_ARCH_RELEASE
    "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
    CACHE STRING "Shippable base arch list (Turing→Blackwell + PTX fallback, §6)")

message(STATUS "steppe: CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
