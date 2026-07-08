# src/app/CMakeLists.txt — build recipe for the steppe command-line executable.
#
# The one structural fact: this is a plain C++20 HOST target, NOT a CUDA target
# (no CUDA language, no .cu sources). It reaches the GPU only through CUDA-free
# seams, so a leaked CUDA header would hard-fail this compile (reference §2).
#
# Reference: docs/reference/src_app_CMakeLists.txt.md

# The CLI11 dependency — find_package first, CPM fetch fallback — reference §3

find_package(CLI11 2.4 CONFIG QUIET)
if(NOT CLI11_FOUND)
  message(STATUS "steppe CLI: CLI11 not found via find_package — fetching v2.4.2 via CPM")
  include(CPM)
  CPMAddPackage(
      NAME CLI11
      GITHUB_REPOSITORY CLIUtils/CLI11
      VERSION 2.4.2
      GIT_TAG 6c7b07a878ad834957b98d0f9ce1dbe0cb204fc9)
else()
  message(STATUS "steppe CLI: using CLI11 ${CLI11_VERSION} via find_package")
endif()

# The executable and its sources — reference §4
add_executable(steppe_app
    main.cpp
    cli_parse.cpp
    cmd_qpadm.cpp
    cmd_qpgraph.cpp
    cmd_qpwave.cpp
    cmd_f4.cpp
    cmd_f3.cpp
    cmd_f4ratio.cpp
    cmd_fstat_sweep.cpp
    cmd_qpdstat.cpp
    cmd_fst.cpp
    cmd_sfs.cpp
    cmd_pca.cpp
    pca_html_writer.cpp
    cmd_qpfstats.cpp
    cmd_dates.cpp
    cmd_paint.cpp
    cmd_rotate.cpp
    cmd_scan.cpp
    cmd_extract_f2.cpp
    cmd_cache.cpp
    cmd_ingest.cpp
    cmd_ibd.cpp
    cmd_roh.cpp
    cmd_pcangsd.cpp
    cmd_ingest_concord.cpp
    ingest_concord.cpp
    cmd_readv2_concord.cpp
    readv2_concord.cpp
    cmd_readv2.cpp
    readv2_emit.cpp
    readv2_shard_writer.cpp
    result_emit.cpp)
add_executable(steppe::app ALIAS steppe_app)
set_target_properties(steppe_app PROPERTIES OUTPUT_NAME steppe)

# Library links (the device-link arrangement + two shared helpers) — reference §5
target_link_libraries(steppe_app
    PRIVATE steppe::core steppe::io steppe::api steppe::device
            steppe::access
            steppe::extract
            CLI11::CLI11 steppe::warnings)
# Compile features and the version string — reference §6
target_compile_features(steppe_app PRIVATE cxx_std_20)

target_compile_definitions(steppe_app PRIVATE STEPPE_VERSION="${PROJECT_VERSION}")

# scikit-build-core wheel build: install the CLI into the wheel's scripts dir so `pip install
# steppe` ALSO puts the full `steppe` command (every subcommand, including scan and the f4/f3
# sweeps that have no Python binding) on PATH — the wheel becomes a complete install, not just the
# Python API. Skipped for the standalone CLI build (SKBUILD is unset there). — reference §7
if(SKBUILD)
  install(TARGETS steppe_app RUNTIME DESTINATION "${SKBUILD_SCRIPTS_DIR}")
endif()
