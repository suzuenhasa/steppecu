# src/io/CMakeLists.txt
#
# steppe_io — the genotype-decode I/O FRONT-END (architecture.md §4 "io is a
# sibling leaf"; ROADMAP M1). An ISOLATED LEAF: it produces plain data structs
# (GenotypeTile + per-SNP metadata) and depends on NOTHING in core/device — no
# CUDA, no ComputeBackend, no upward edge (architecture.md §4 dependency-direction
# rule). The `app` layer is the only place that wires `io` output into compute.
#
# THE LAYERING MECHANISM (architecture.md §4): steppe_io links ONLY steppe::api
# (the CUDA-free public config/constants) and steppe::warnings. It does NOT link
# steppe::core, steppe::core_internal, or steppe::device — so an `io` TU cannot
# reach a core domain rule or a CUDA header, and the leaf stays a leaf. (The one
# shared domain rule it WILL consume later — block_partition_rule — is host-pure
# in core and is read by `app`, not by io re-deriving it; architecture.md §2, §8.)

add_library(steppe_io STATIC
    eigenstrat_format.cpp        # packed-.geno header parse + DERIVED record stride
    ind_reader.cpp               # .ind → pop labels + selection (oracle-faithful)
    snp_reader.cpp               # .snp → id / chrom / genpos (Morgans) / ref / alt
    geno_reader.cpp              # .geno tiled raw-byte gather (NO decode)
    # M2 on-the-fly filters (host-pure; part of the SAME io leaf). filter_decision/
    # filter_plan are header-only predicates/structs (no .cpp). architecture.md §1,
    # §5 S-1/S0'; ROADMAP M2.
    filter/include_exclude.cpp   # resolve include/exclude + external prune.in (READ, never computed)
    filter/snp_filter.cpp        # per-SNP keep-mask: pooled MAF / geno / class / membership / flags
    filter/mind_prepass.cpp)     # conditional S-1 per-sample --mind pre-pass
add_library(steppe::io ALIAS steppe_io)

# Headers under src/io/ are included as "io/..."; the include root is src/ (one
# level above this dir), matching the "device/...", "core/..." convention.
target_include_directories(steppe_io PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/..")

# LEAF: only the CUDA-free public surface + warnings. Deliberately NO core/device.
target_link_libraries(steppe_io
    PUBLIC  steppe::api
    PRIVATE steppe::warnings)

target_compile_features(steppe_io PUBLIC cxx_std_20)
