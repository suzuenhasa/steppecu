# bindings/CMakeLists.txt — the nanobind Python module steppe._core (M(py-1)).
#
# A PLAIN C++20 CXX target (architecture.md §4, §6.1; cli-bindings.md §6.1/§6.2) — NOT a
# CUDA target. NO `LANGUAGES CUDA`, NO `.cu` sources, and it MUST NOT include any CUDA
# toolkit header. It reaches the GPU ONLY through the CUDA-FREE seams (build_resources /
# run_qpadm / run_qpwave / run_qpadm_search / upload_f2_blocks_to_device) — the arch-grep
# gate enforces this for free because a leaked CUDA runtime header would hard-fail this
# host compile (cli-bindings.md §6.2). It links steppe::device ONLY to resolve the CUDA-
# free factory / build_resources / handle symbols (the same path steppe_app uses):
# steppe_core sets CUDA_RESOLVE_DEVICE_SYMBOLS OFF, so a target linking BOTH steppe_core
# and steppe_device lets steppe_device device-link its RDC kernels; _core device-links
# nothing — it is a pure host target.
#
# nanobind (ADR-0002, architecture §15) is obtained the SAME no-network-first way the rest
# of the repo gets CLI11 / GoogleTest: find_package first (the pip-installed `nanobind`
# package ships its CMake config + the FindPython hint), CPM fetch as the offline fallback
# (cmake/CPM.cmake — the architecture-named home, §6). Confining nanobind + the CPM
# machinery to THIS subdir keeps the dependency PRIVATE to the bindings subtree and out of
# the core/device build entirely (the §4 layering rule).

# nanobind needs the Python development headers (the Module component is enough for an
# extension module; we do not embed the interpreter). FindPython is the CMake-blessed
# locator nanobind itself documents.
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
  # Ask the active interpreter where pip installed nanobind's CMake config (the common
  # case on the box: `pip install nanobind` provides find_package support).
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind, sys; sys.stdout.write(nanobind.cmake_dir())"
    OUTPUT_VARIABLE _nb_cmake_dir
    RESULT_VARIABLE _nb_rc
    ERROR_QUIET)
  if(_nb_rc EQUAL 0 AND EXISTS "${_nb_cmake_dir}")
    find_package(nanobind CONFIG REQUIRED PATHS "${_nb_cmake_dir}")
    message(STATUS "steppe bindings: using pip nanobind CMake config at ${_nb_cmake_dir}")
  else()
    message(STATUS "steppe bindings: nanobind not found — fetching v2.4.0 via CPM")
    include(CPM)
    CPMAddPackage(NAME nanobind GITHUB_REPOSITORY wjakob/nanobind GIT_TAG v2.4.0)
    list(APPEND CMAKE_MODULE_PATH "${nanobind_SOURCE_DIR}/cmake")
    find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_SOURCE_DIR}/cmake")
  endif()
else()
  message(STATUS "steppe bindings: using nanobind via find_package")
endif()

# The compiled extension is `steppe._core` (the user imports the pure-Python `steppe`
# package over it). NB_STATIC links nanobind's runtime statically into the module so the
# wheel ships one self-contained _core*.so.
nanobind_add_module(_core NB_STATIC module.cpp)
add_library(steppe::pycore ALIAS _core)

# CUDA-FREE seams only: steppe::access (the shared f2-dir reader + pop->index resolver),
# steppe::extract (the shared genotype->f2 extract chain run_extract_f2 + the STPF2BK1 dir
# WRITER — M(py-2)), steppe::{core,io,api,device} (the run_qpadm*/build_resources/upload/
# decode seams). steppe::io is linked for the genotype->f2 readers the M(py-2) extract
# binding builds the P-axis from; M(py-1) consumed a pre-built f2-dir and did not call io.
target_link_libraries(_core
    PRIVATE steppe::core steppe::io steppe::api steppe::device
            steppe::access steppe::extract steppe::warnings)
target_compile_features(_core PRIVATE cxx_std_20)

# Install the compiled extension next to the pure-Python facade (scikit-build-core lays
# the wheel out as steppe/_core*.so + steppe/__init__.py). For a plain CMake build the
# DESTINATION is relative to CMAKE_INSTALL_PREFIX; the pytest gate imports it from the
# build tree directly (no install needed), so this is for the wheel path.
install(TARGETS _core LIBRARY DESTINATION steppe)

# Stage an importable `steppe` package in the BUILD tree so the pytest gate can `import
# steppe` without an install: copy the pure-Python facade next to the freshly-built
# _core*.so under <build>/bindings/steppe/. STEPPE_BINDINGS_DIR (= <build>/bindings) is
# handed to the CTest env (tests/CMakeLists.txt) and the conftest sys.path shim.
set(STEPPE_PY_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/steppe")
add_custom_command(TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${STEPPE_PY_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/steppe/__init__.py" "${STEPPE_PY_STAGE_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy
            "$<TARGET_FILE:_core>" "${STEPPE_PY_STAGE_DIR}/"
    COMMENT "Staging the importable steppe package (__init__.py + _core) in the build tree"
    VERBATIM)
