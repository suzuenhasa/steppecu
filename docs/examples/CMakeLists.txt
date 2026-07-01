# examples/CMakeLists.txt — the read_f2 -> qpadm -> inspect quick-starts.
#
# Added from the top-level CMakeLists.txt ONLY when STEPPE_BUILD_EXAMPLES is ON
# (SteppeOptions.cmake, default OFF). The C++ quickstart is a LIVING API CANARY: it
# reuses the EXACT CUDA-free entry points the CLI `qpadm` command uses
# (src/app/cmd_qpadm.cpp run_qpadm_command) — read_f2_dir + PopResolver (steppe::access),
# build_resources + upload_f2_blocks_to_device (steppe::device CUDA-free seams), run_qpadm
# (steppe::api) — so it adds NO new compute and fails to compile/run if the public surface
# drifts. It is a PLAIN C++20 CXX target (NO LANGUAGES CUDA, NO .cu), mirroring src/app:
# it reaches the GPU only through the CUDA-free seams; linking BOTH steppe::core AND
# steppe::device lets steppe_device device-link its RDC kernels (the same path the CLI app
# and the CUDA reference tests use).

# steppe::access (the shared f2-dir reader + pop->index resolver) only exists when
# STEPPE_BUILD_CLI or STEPPE_BUILD_PYTHON is ON (top-level CMakeLists.txt guards
# add_subdirectory(src/access)). Without it the C++ quickstart cannot resolve the same
# seams the CLI uses, so skip it with a clear message rather than hard-failing the build —
# the Python quickstart + README still ship.
if(NOT TARGET steppe::access)
  message(STATUS
    "steppe examples: the C++ quickstart needs steppe::access — enable "
    "-DSTEPPE_BUILD_CLI=ON (or -DSTEPPE_BUILD_PYTHON=ON) alongside "
    "-DSTEPPE_BUILD_EXAMPLES=ON. Skipping the C++ example; the Python quickstart "
    "(examples/python/quickstart_qpadm.py) and README are unaffected.")
  return()
endif()

add_executable(quickstart_qpadm cpp/quickstart_qpadm.cpp)

# The SAME link set src/app/CMakeLists.txt gives steppe_app (minus CLI11 + steppe::extract,
# which the qpadm-from-f2dir path does not touch): the CUDA-free library seams + warnings.
# steppe_device is linked so the executable resolves the build_resources / handle symbols
# and device-links steppe_device's RDC kernels; quickstart_qpadm itself device-links
# nothing (a pure host target, like the CLI app).
target_link_libraries(quickstart_qpadm
    PRIVATE steppe::core steppe::io steppe::api steppe::device
            steppe::access steppe::warnings)
target_compile_features(quickstart_qpadm PRIVATE cxx_std_20)
