# src/extract/CMakeLists.txt — builds steppe_extract, the static library that holds the
# genotype->f2 extract chain and the f2-blocks directory writer. It exists so both the CLI
# app and the Python bindings can link one copy of that chain instead of duplicating it.
#
# Reference: docs/reference/src_extract_CMakeLists.txt.md

# The static library and its sources — reference §3
add_library(steppe_extract STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../app/extract_f2_core.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../app/f2_dir_writer.cpp")
add_library(steppe::extract ALIAS steppe_extract)

# Link dependencies (CUDA-free host target) — reference §4
target_link_libraries(steppe_extract
    PUBLIC  steppe::api
    PRIVATE steppe::core steppe::io steppe::device steppe::warnings)
target_compile_features(steppe_extract PUBLIC cxx_std_20)

# Stamped version string — reference §5
target_compile_definitions(steppe_extract PRIVATE STEPPE_VERSION="${PROJECT_VERSION}")

# Device-symbol resolution invariant — reference §6
set_target_properties(steppe_extract PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
