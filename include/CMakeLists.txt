# include/CMakeLists.txt
#
# steppe_api — the PUBLIC, header-only API contract (architecture.md §4, §17 step 5).
#
# INTERFACE target: no compiled sources, just the include directory. It is the
# C-ABI-shaped surface in include/steppe/ (config.hpp, error.hpp; M0 subset). It
# is deliberately CUDA-FREE and standard-C++ only, so it compiles into every
# layer — core, device, tests, and later the CLI/bindings — without dragging in
# the device toolkit (architecture.md §4 layering rule).

add_library(steppe_api INTERFACE)
add_library(steppe::api ALIAS steppe_api)

target_include_directories(steppe_api INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}")   # so consumers do #include "steppe/config.hpp"

target_compile_features(steppe_api INTERFACE cxx_std_20)
