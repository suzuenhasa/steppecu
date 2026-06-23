# src/app/CMakeLists.txt — the steppe CLI app (M(cli-0) scaffold).
#
# A PLAIN C++20 CXX target (architecture.md §4, §6.1; cli-bindings.md §6.1) — NOT a
# CUDA target. NO `LANGUAGES CUDA`, NO `.cu` sources, NO CUDA_SEPARABLE_COMPILATION,
# and it MUST NOT include any CUDA toolkit header. It reaches the GPU ONLY through the
# CUDA-free seams (build_resources / run_qpadm / upload_f2_blocks_to_device /
# f2_disk_format) — the arch-grep gate enforces this for free because a leaked CUDA
# runtime header would hard-fail this host compile (cli-bindings.md §6.2).
#
# CLI11 (architecture.md §15 line 94: "CLI11 2.4.x"; line 392 pin "gh:CLIUtils/
# CLI11@2.4.2"; cli-bindings.md §6.1 line 462) is obtained the SAME no-network-first
# way the rest of the repo gets GoogleTest: find_package first, CPM fetch as the
# fallback (cmake/CPM.cmake — the architecture-named home, §6 / line 384). Confining
# CLI11 + the CPM machinery to THIS subdir keeps the dependency PRIVATE to the app
# subtree and out of the core/device build entirely (the §4 layering rule).

find_package(CLI11 2.4 CONFIG QUIET)
if(NOT CLI11_FOUND)
  message(STATUS "steppe CLI: CLI11 not found via find_package — fetching v2.4.2 via CPM")
  include(CPM)
  CPMAddPackage("gh:CLIUtils/CLI11@2.4.2")
else()
  message(STATUS "steppe CLI: using CLI11 ${CLI11_VERSION} via find_package")
endif()

# f2_dir_io.cpp (the f2-dir READER) + pop_resolver.cpp (name -> P-axis index) moved to
# the shared steppe::access lib (src/access) so the Python bindings reuse them too — the
# cli-bindings.md M(py-1) §0 DRY move. The .cpp files still live here in src/app/ (next to
# the CLI command flows that grew them); steppe::access compiles them and steppe_app links
# it (below) rather than compiling them a second time.
add_executable(steppe_app
    main.cpp
    cli_parse.cpp
    cmd_qpadm.cpp      # M(cli-1): the qpAdm GPU fit command
    cmd_qpwave.cpp     # M(cli-2): the qpWave rank-sweep command (no target; left[0]=ref)
    cmd_f4.cpp         # Standalone f4 statistic (run_f4): est/se/z/p per quartet
    cmd_f3.cpp         # Standalone f3 statistic (run_f3): est/se/z/p per triple
    cmd_rotate.cpp     # M(cli-3): the qpAdm ROTATION (pool-subset enumeration; batched)
    cmd_extract_f2.cpp # M(cli-4): the genotype -> f2_blocks-dir precompute command
    f2_dir_writer.cpp  # M(cli-4): the f2_blocks dir WRITER (STPF2BK1 REAL vpair + sidecars)
    result_emit.cpp)   # M(cli-1): QpAdmResult -> tidy CSV / TSV / JSON
add_executable(steppe::app ALIAS steppe_app)
set_target_properties(steppe_app PROPERTIES OUTPUT_NAME steppe)

# Links the CUDA-free library seams (core/io/api/device) + CLI11 + warnings. It links
# steppe::device only to resolve the CUDA-free factory / build_resources / handle
# symbols (the same path steppe_core uses); steppe_core sets CUDA_RESOLVE_DEVICE_SYMBOLS
# OFF, so an executable linking BOTH steppe_core and steppe_device lets steppe_device
# device-link its RDC kernels (the exact path the CUDA reference tests use). steppe_app
# itself device-links nothing — it is a pure host target (cli-bindings.md §6.1).
target_link_libraries(steppe_app
    PRIVATE steppe::core steppe::io steppe::api steppe::device
            steppe::access            # the shared f2-dir reader + pop->index resolver (DRY, §0)
            CLI11::CLI11 steppe::warnings)
target_compile_features(steppe_app PRIVATE cxx_std_20)

# The version string surfaced by `steppe --version` — the project VERSION, single-homed
# in the top-level project() declaration (no second literal in the app source).
target_compile_definitions(steppe_app PRIVATE STEPPE_VERSION="${PROJECT_VERSION}")
