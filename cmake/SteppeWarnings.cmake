# cmake/SteppeWarnings.cmake
#
# The ONE source of warning policy (architecture.md §6, §8). Defines the
# INTERFACE target steppe::warnings, linked PRIVATE by every first-party target
# so the flags never propagate to consumers — and so host vs device flags are
# separated by generator expressions (nvcc rejects raw GCC/Clang flags).
#
# Warnings-as-errors is the M0 contract (C++20 with warnings-as-errors, both the
# host-pure and __CUDACC__ device-qualifier paths). It is now gated by the
# STEPPE_WERROR option (cmake/SteppeOptions.cmake) — but ON is the DEFAULT and the
# shipping contract (CI/release presets pin it ON); OFF is a dev-only escape hatch
# that keeps -Wall -Wextra yet lets a local build proceed THROUGH a warning. The
# option is NOT on the compute/precision path: it changes only whether a warning is
# fatal, never codegen.

include_guard(GLOBAL)

add_library(steppe_warnings INTERFACE)
add_library(steppe::warnings ALIAS steppe_warnings)

# Split the three -Werror tokens out of the always-on -Wall;-Wextra so STEPPE_WERROR
# can drop them for a relaxed local build. STEPPE_WERROR is known at configure time,
# so a plain if() selects the token sets; the COMPILE_LANGUAGE generator-expression
# split (nvcc rejects raw host flags) is preserved below.
if(STEPPE_WERROR)
  set(_steppe_host_warn    -Wall;-Wextra;-Werror)               # host C++ TUs
  set(_steppe_device_warn  --Werror;all-warnings)               # nvcc warning surface
  set(_steppe_hostfwd_warn -Xcompiler=-Wall,-Wextra,-Werror)    # host-forward through nvcc
else()
  set(_steppe_host_warn    -Wall;-Wextra)
  set(_steppe_device_warn  )                                    # --Werror dropped entirely
  set(_steppe_hostfwd_warn -Xcompiler=-Wall,-Wextra)
endif()

target_compile_options(steppe_warnings INTERFACE
  # Host C++ TUs (core, api consumers, host-side test units).
  $<$<COMPILE_LANGUAGE:CXX>:${_steppe_host_warn}>

  # Device TUs: nvcc's own warning surface, errors-on. all-warnings is an nvcc
  # warning *kind* (the argument to --Werror), so it is a SEPARATE token — passing
  # it as one "--Werror=all-warnings" string is rejected by nvcc. Matches
  # architecture.md §6. (Empty — and thus emits nothing — when STEPPE_WERROR=OFF.)
  $<$<COMPILE_LANGUAGE:CUDA>:${_steppe_device_warn}>

  # And forward the host-compiler warning flags through nvcc to the host backend.
  $<$<COMPILE_LANGUAGE:CUDA>:${_steppe_hostfwd_warn}>

  # Nsight source mapping on profile-able builds; never with -G (architecture.md
  # §6 — -G makes kernels many× slower and timing unrepresentative).
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>)
