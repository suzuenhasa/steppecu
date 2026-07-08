# src/core/CMakeLists.txt — the core layer: two targets. steppe_core_internal is an
# INTERFACE library of shared host-pure helpers; steppe_core is a static host-
# orchestration library that reaches the GPU only through the CUDA-free ComputeBackend
# seam (CUDA stays private to steppe_device, so core cannot include a CUDA header).
#
# Reference: docs/reference/src_core_CMakeLists.txt.md

# steppe_core_internal — the shared host-pure helpers — reference §2
add_library(steppe_core_internal INTERFACE)
add_library(steppe::core_internal ALIAS steppe_core_internal)

target_include_directories(steppe_core_internal INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/..")

target_link_libraries(steppe_core_internal INTERFACE steppe::api)
target_compile_features(steppe_core_internal INTERFACE cxx_std_20)

# steppe_core — the host orchestration; the source files — reference §3, §5
add_library(steppe_core STATIC
    fstats/f2_from_blocks.cpp
    fstats/f2_combine.cpp
    fstats/f2_blocks_multigpu.cpp
    fstats/f2_blocks_multigpu_core.cpp
    domain/block_partition_rule.cpp
    qpadm/qpadm_fit.cpp
    qpadm/ranktest.cpp
    qpadm/nested_models.cpp
    qpadm/f4.cpp
    qpadm/f3.cpp
    qpadm/qpgraph_model.cpp
    qpadm/qpgraph_fit.cpp
    qpadm/qpgraph_enumerate.cpp
    qpadm/qpgraph_search.cpp
    qpadm/f4ratio.cpp
    qpadm/fstat_sweep.cpp
    qpadm/model_search.cpp
    qpadm/model_search_core.cpp
    stats/dstat.cpp
    stats/fst.cpp
    stats/sfs.cpp
    stats/pca.cpp
    stats/qpfstats.cpp
    stats/dates.cpp
    stats/li_stephens.cpp
    stats/li_stephens_validate.cpp
    stats/ancibd_segments.cpp
    stats/roh_segments.cpp
    stats/decode_keep_autosomes.cpp
    stats/read_canonical_tile.cpp
    stats/genotype_front_end.cpp
    readv2/readv2.cpp
    readv2/readv2_classify.cpp
    config/config_builder.cpp)
add_library(steppe::core ALIAS steppe_core)

# Link dependencies — reference §6
target_link_libraries(steppe_core
    PUBLIC  steppe::api steppe::core_internal
    PRIVATE steppe::device steppe::warnings
            steppe::io
            Threads::Threads)

target_compile_features(steppe_core PUBLIC cxx_std_20)

# Device-symbol resolution (CUDA_RESOLVE_DEVICE_SYMBOLS OFF) — reference §7
set_target_properties(steppe_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
