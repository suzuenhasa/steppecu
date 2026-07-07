# src/io/CMakeLists.txt
#
# Builds steppe_io, the isolated genotype-decode I/O leaf: it reads the input
# files into plain data structs and links only the CUDA-free public surface, so
# an io translation unit cannot reach core/device or a CUDA header. The app layer
# is the only place that wires io output into compute.
#
# Reference: docs/reference/src_io_CMakeLists.txt.md

# Source files — reference §2
add_library(steppe_io STATIC
    eigenstrat_format.cpp
    ind_reader.cpp
    individual_partition.cpp
    snp_reader.cpp
    plink_reader.cpp
    genotype_source.cpp
    geno_reader.cpp
    ploidy_detect.cpp
    filter/include_exclude.cpp
    filter/snp_filter.cpp
    filter/mind_prepass.cpp)
add_library(steppe::io ALIAS steppe_io)

# Include root — reference §4
target_include_directories(steppe_io PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Isolation: links only the CUDA-free surface, never core/device — reference §3
target_link_libraries(steppe_io
    PUBLIC  steppe::api
    PRIVATE steppe::warnings)

# Language standard, propagated — reference §4
target_compile_features(steppe_io PUBLIC cxx_std_20)
