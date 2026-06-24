# src/extract/CMakeLists.txt — steppe_extract: the genotype->f2 EXTRACT library entry
# + the f2_blocks dir WRITER, shared by BOTH the CLI app AND the Python bindings (M(py-2)).
#
# THE §0-STYLE FINDING (mirrors src/access): the genotype->f2 extract CHAIN lived ONLY
# inside the steppe_app executable (cmd_extract_f2.cpp's body), so there was NO shared
# linkable home for it. The Python binding needs the SAME chain to build an f2-dir from a
# genotype prefix, and src/app is compiled ONLY when STEPPE_BUILD_CLI is ON — so a wheel
# build (CLI=OFF, PYTHON=ON) would have no extract entry. This lib is that shared home: the
# decode->filter->assign_blocks->compute_f2_blocks_multigpu_tiered->to_host chain
# (extract_f2_core.cpp implementing include/steppe/extract.hpp) + the STPF2BK1 dir writer
# (f2_dir_writer.cpp), linked by BOTH steppe_app (the CLI command is a thin wrapper) AND
# bindings/_core (the run_extract_f2 binding). ONE copy of the chain — the §4-layering-
# correct DRY move.
#
# UNLIKE steppe::access (a CUDA-FREE reader linking only api+core_internal), this lib
# REACHES the GPU through the CUDA-FREE seams: it links steppe::{core,io,api,device} — the
# SAME set steppe_app uses (build_resources, the decode backend via Resources,
# compute_f2_blocks_multigpu_tiered). It is STILL a PLAIN C++20 CXX target (architecture.md
# §4): NO `LANGUAGES CUDA`, NO `.cu` sources, NO CUDA toolkit header. A leaked CUDA header
# would hard-fail this host compile — the arch-grep gate is structural. The .cpp files live
# in src/app/ next to the CLI command flows that grew them (include paths "app/..." are
# unchanged); this lib compiles them and steppe_app links it rather than compiling twice.

add_library(steppe_extract STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../app/extract_f2_core.cpp"  # the genotype->f2 chain (run_extract_f2)
    "${CMAKE_CURRENT_SOURCE_DIR}/../app/f2_dir_writer.cpp")   # the STPF2BK1 dir WRITER (REAL vpair + sidecars)
add_library(steppe::extract ALIAS steppe_extract)

# The CUDA-FREE seams the chain reaches: steppe::core (compute_f2_blocks_multigpu_tiered),
# steppe::io (geno/ind/snp readers + ploidy detect + the snp filter), steppe::api (the
# public config/extract/fstats headers), steppe::device (build_resources / Resources /
# F2BlocksOut / the decode backend handle). It links steppe::device ONLY to resolve the
# CUDA-free factory/handle symbols (the same path steppe_app/_core use); it device-links
# nothing (steppe_core sets CUDA_RESOLVE_DEVICE_SYMBOLS OFF), so it is a pure host target.
target_link_libraries(steppe_extract
    PUBLIC  steppe::api
    PRIVATE steppe::core steppe::io steppe::device steppe::warnings)
target_compile_features(steppe_extract PUBLIC cxx_std_20)

# The version string the writer stamps into meta.json (the same STEPPE_VERSION the CLI
# command used to pass via STEPPE_VERSION; single-homed in the top-level project VERSION).
target_compile_definitions(steppe_extract PRIVATE STEPPE_VERSION="${PROJECT_VERSION}")

# It links steppe::device (CUDA_SEPARABLE_COMPILATION ON) PRIVATE, the SAME edge
# steppe_core has, so it MUST mirror steppe_core's CUDA_RESOLVE_DEVICE_SYMBOLS OFF
# (src/core/CMakeLists.txt:134). The OFF property does NOT propagate from steppe_core
# to this sibling target, so without this line CMake archives a cmake_device_link.o
# into libsteppe_extract.a that registers cuda_backend.cu's fatbin; the final CUDA
# executable (bin/steppe) then double-handles that registration and the host link
# fails with `undefined reference to __fatbinwrap_..._cuda_backend_cu_...`. With OFF,
# this stays a pure host archive and the consuming executable device-links
# steppe::device once — the same path steppe_core and the CUDA tests use.
set_target_properties(steppe_extract PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
