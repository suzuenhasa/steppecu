# src/access/CMakeLists.txt — builds steppe_access, a small CUDA-free static lib
# holding the two host helpers (the f2-dir reader + the pop-name -> index resolver)
# that both the CLI app and the Python bindings link, so neither keeps its own copy.
#
# Reference: docs/reference/src_access_CMakeLists.txt.md

# Source files compiled (result_emit.cpp deliberately excluded) — reference §2
add_library(steppe_access STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../app/f2_dir_io.cpp"     
    "${CMAKE_CURRENT_SOURCE_DIR}/../app/pop_resolver.cpp") 
add_library(steppe::access ALIAS steppe_access)

# CUDA-free link set (no CUDA, no CLI11) — reference §3
target_link_libraries(steppe_access
    PUBLIC  steppe::api steppe::core_internal
    PRIVATE steppe::warnings)
# C++20 language standard — reference §4
target_compile_features(steppe_access PUBLIC cxx_std_20)
