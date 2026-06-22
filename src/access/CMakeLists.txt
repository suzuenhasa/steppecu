# src/access/CMakeLists.txt — steppe_access: the CUDA-FREE host access helpers.
#
# THE §0 FINDING (cli-bindings.md M(py-1) design §0): read_f2_dir + the pop-name->index
# PopResolver were compiled STRAIGHT INTO the steppe_app executable (src/app sources of
# add_executable), so there was NO shared linkable home for them. Both the CLI app AND
# the nanobind bindings module need them (the f2-dir reader + the name<->index map the
# INDEX-ONLY compute seam lacks). This tiny static lib is that shared home — the §4-
# layering-correct DRY move: ONE copy of the STPF2BK1 reader + the pops.txt resolver,
# linked by BOTH steppe_app and bindings/_core (no duplicated sources).
#
# It owns the TWO compute-free, CUDA-free helpers (the .cpp files still live in src/app/
# next to the CLI command flows that grew them — the include paths "app/..." are
# unchanged): f2_dir_io.cpp (the STPF2BK1 f2.bin + pops.txt READER) and pop_resolver.cpp
# (name -> P-axis index). result_emit.cpp is NOT here: it is CLI-text serialization the
# bindings do NOT reuse (the binding emits Python objects, not CSV/TSV/JSON).
#
# PLAIN C++20 CXX target (architecture.md §4): NO `LANGUAGES CUDA`, NO `.cu` sources, and
# it MUST NOT include a CUDA toolkit header. It reaches only the CUDA-FREE seams
# (fstats.hpp via steppe::api, f2_disk_format.hpp via steppe::core_internal which exposes
# the src/ include root). A leaked CUDA header would hard-fail this host compile — the
# arch-grep gate is structural (cli-bindings.md §6.2). It links NO CUDA, NO CLI11.

add_library(steppe_access STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../app/f2_dir_io.cpp"     # STPF2BK1 f2.bin + pops.txt reader
    "${CMAKE_CURRENT_SOURCE_DIR}/../app/pop_resolver.cpp") # name -> P-axis index resolver
add_library(steppe::access ALIAS steppe_access)

# steppe::core_internal carries the src/ include root (so "app/f2_dir_io.hpp" and
# "device/f2_disk_format.hpp" resolve) + steppe::api (steppe/error.hpp, steppe/fstats.hpp).
# Both are CUDA-FREE INTERFACE/seam targets — no CUDA toolkit reaches this host TU.
target_link_libraries(steppe_access
    PUBLIC  steppe::api steppe::core_internal
    PRIVATE steppe::warnings)
target_compile_features(steppe_access PUBLIC cxx_std_20)
