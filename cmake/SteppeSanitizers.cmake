# cmake/SteppeSanitizers.cmake
#
# The ONE place that translates the STEPPE_SANITIZER cache var (declared in
# cmake/SteppeOptions.cmake) into actual behaviour. include()'d once from the
# top-level CMakeLists.txt AFTER SteppeOptions and BEFORE any add_subdirectory(),
# so the directory-scoped flags propagate to every first-party target created
# later (architecture.md §6 presets, §13 testing). NO targets are defined here.
#
# Contract (docs/archive/kimiactions/01-open-worth-doing.md §E2):
#   ""            no-op. The DEFAULT; the Release/CI parity build is NEVER
#                 sanitized (the perf/golden lane must stay clean FP64). The
#                 empty build is byte-for-byte the current build.
#   asan;ubsan    -fsanitize=address,undefined -fno-omit-frame-pointer on HOST
#                 C++ TUs ONLY. Gated by $<COMPILE_LANGUAGE:CXX> so device/.cu
#                 objects are EXCLUDED — host ASan does not instrument CUDA
#                 device code and mixing it with the CUDA runtime is fragile.
#                 Matching link flags are gated by $<LINK_LANGUAGE:CXX> (CMake
#                 >= 3.18) so only host-linked targets pull the ASan/UBSan
#                 runtime, not the CUDA device-linked ones. A dev/CI-only
#                 debug-build overlay.
#   compute       NO codegen flags. Selects the runtime compute-sanitizer GPU
#                 lane used by CI (`compute-sanitizer --tool memcheck ctest ...`)
#                 — this module only legalises + documents the spelling and sets
#                 the marker STEPPE_SANITIZER_COMPUTE that the CI lane reads.
#   <anything else>  hard configure error (fail loud).
#
# Parity-risk: NONE for the shipping build. The flags exist only under the debug
# sanitizer overlay, never on the Release/CI parity build, so no golden or perf
# number moves. The one care-point — keeping host ASan flags off .cu/device
# objects — is handled structurally by the $<COMPILE_LANGUAGE:CXX> gate.
#
# Lane invocations (asan/ubsan host ctest; compute-sanitizer GPU ctest) live in
# docs/archive/kimiactions/02-ci-plan.md: this module delivers the CMake seam, CI
# delivers the runner.

include_guard(GLOBAL)

if("${STEPPE_SANITIZER}" STREQUAL "")
  # Default: no sanitizer. Intentionally a no-op so the empty build is
  # byte-for-byte the current (Release/CI parity) build.

elseif("compute" IN_LIST STEPPE_SANITIZER)
  # `compute` is the runtime compute-sanitizer GPU lane — a no-codegen marker,
  # not a compile/link flag set. It cannot be combined with the host sanitizers.
  list(LENGTH STEPPE_SANITIZER _steppe_san_len)
  if(NOT _steppe_san_len EQUAL 1)
    message(FATAL_ERROR
      "STEPPE_SANITIZER='compute' is the runtime compute-sanitizer GPU lane and "
      "cannot be combined with asan/ubsan (got '${STEPPE_SANITIZER}'). Use it "
      "alone (cmake/SteppeSanitizers.cmake; docs/archive/kimiactions/01-open-worth-doing.md §E2).")
  endif()
  # Documented marker the CI GPU lane reads. NO codegen flags are added; the lane
  # wraps the normally-built binaries:
  #   compute-sanitizer --tool memcheck ctest --test-dir build-rel ...
  set(STEPPE_SANITIZER_COMPUTE ON CACHE INTERNAL
      "STEPPE_SANITIZER=compute: run ctest under compute-sanitizer (no codegen flags).")
  message(STATUS
    "steppe: STEPPE_SANITIZER=compute — no compile/link flags; the CI GPU lane "
    "wraps ctest with `compute-sanitizer --tool memcheck` on the built binaries.")

else()
  # Host compile/link sanitizer set (asan;ubsan, in any order/subset).
  set(_steppe_fsanitize "")        # the -fsanitize=<...> argument list
  foreach(_steppe_san_tok IN LISTS STEPPE_SANITIZER)
    if(_steppe_san_tok STREQUAL "asan")
      list(APPEND _steppe_fsanitize "address")
    elseif(_steppe_san_tok STREQUAL "ubsan")
      list(APPEND _steppe_fsanitize "undefined")
    else()
      message(FATAL_ERROR
        "Unknown STEPPE_SANITIZER value '${_steppe_san_tok}' (in "
        "'${STEPPE_SANITIZER}'). Valid: empty | asan;ubsan | compute "
        "(cmake/SteppeSanitizers.cmake; docs/archive/kimiactions/01-open-worth-doing.md §E2).")
    endif()
  endforeach()

  list(JOIN _steppe_fsanitize "," _steppe_fsanitize_arg)   # e.g. address,undefined

  # Host C++ TUs ONLY — $<COMPILE_LANGUAGE:CXX> excludes every .cu/device object,
  # which host ASan must not instrument. (Literal comma in the genex true_string
  # is literal text, not an argument separator — cmake-generator-expressions.7.)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${_steppe_fsanitize_arg}>"
    "$<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>")

  # Matching link flags on HOST-linked targets only ($<LINK_LANGUAGE:CXX>; CMake
  # >= 3.18, valid in link options). CUDA device-linked targets do not pull the
  # ASan/UBSan runtime.
  add_link_options(
    "$<$<LINK_LANGUAGE:CXX>:-fsanitize=${_steppe_fsanitize_arg}>")

  message(STATUS
    "steppe: STEPPE_SANITIZER='${STEPPE_SANITIZER}' — host C++ TUs built with "
    "-fsanitize=${_steppe_fsanitize_arg} -fno-omit-frame-pointer (device/.cu "
    "excluded). Dev/CI debug overlay; the Release/CI parity build stays clean.")
endif()
