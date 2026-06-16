# cmake/SteppeWarnings.cmake
#
# The ONE source of warning policy (architecture.md §6, §8). Defines the
# INTERFACE target steppe::warnings, linked PRIVATE by every first-party target
# so the flags never propagate to consumers — and so host vs device flags are
# separated by generator expressions (nvcc rejects raw GCC/Clang flags).
#
# Warnings-as-errors is a hard gate (the M0 contract: C++20 with
# warnings-as-errors, both the host-pure and __CUDACC__ device-qualifier paths).

include_guard(GLOBAL)

add_library(steppe_warnings INTERFACE)
add_library(steppe::warnings ALIAS steppe_warnings)

target_compile_options(steppe_warnings INTERFACE
  # Host C++ TUs (core, api consumers, host-side test units).
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>

  # Device TUs: nvcc's own warning surface, errors-on. all-warnings is an nvcc
  # warning *kind* (the argument to --Werror), so it is a SEPARATE token — passing
  # it as one "--Werror=all-warnings" string is rejected by nvcc. Matches
  # architecture.md §6.
  $<$<COMPILE_LANGUAGE:CUDA>:--Werror;all-warnings>

  # And forward the host-compiler warning flags through nvcc to the host backend.
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>

  # Nsight source mapping on profile-able builds; never with -G (architecture.md
  # §6 — -G makes kernels many× slower and timing unrepresentative).
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>)
