// src/app/cmd_ingest.hpp
//
// Reference: docs/reference/src_app_cmd_ingest.hpp.md
//
// The `steppe ingest` command entry point: steppe's native gVCF-block-aware
// hardcall VCF reader (the sixth reader arm). Genotypes one sample of a .vcf.gz
// at a GRCh38 target-site table and emits (a) the per-site report TSV in the
// Stage-0 oracle schema (the primary Stage-1 artifact) and, optionally, the raw
// canonical 2-bit tile bytes (for the bit-exact check — the only path that needs
// a device, for the shared transpose). Self-contained args (no shared RunConfig
// pollution); reaches the GPU only through the CUDA-free transpose seam.
#ifndef STEPPE_APP_CMD_INGEST_HPP
#define STEPPE_APP_CMD_INGEST_HPP

#include <string>

namespace steppe::app {

struct IngestArgs {
    std::string vcf;          // the .vcf.gz / .vcf (required to genotype)
    // Target-site source — exactly one of:
    //   (A) Stage-1 back-compat: a pre-built GRCh38 target-site table.
    std::string targets;      // the GRCh38 target-site table
    //   (B) Stage-2 native: build the table in-process from panel + FASTA (+ lift).
    std::string panel;        // AADR EIGENSTRAT .snp (GRCh37) — anchors native mode
    std::string fasta;        // build-matched .fa (GRCh38 for a GRCh38 VCF, GRCh37 for
                              // a GRCh37 VCF); expects a sibling .fai -> ref38
    std::string lift;         // rsID->pos38 map: REQUIRED for a cross-build (GRCh38/other)
                              // VCF; auto-identity (unused) for a same-build GRCh37 VCF
    std::string assembly;     // OPTIONAL override of VCF build detection: "GRCh37"|"GRCh38"
    std::string emit_targets; // OPTIONAL (native only): dump the built 7-col table
    std::string sample;       // OPTIONAL: sample id (default = the sole sample)
    std::string report;       // OPTIONAL: per-site report TSV (primary artifact)
    std::string emit_tile;    // OPTIONAL: raw canonical 2-bit tile bytes (needs a device)
    //   Stage 3 (merge): append nikki as a size-1 population into an existing panel.
    std::string merge_into;   // OPTIONAL: source panel PREFIX (.geno/.snp/.ind) to append into
    std::string emit_merged;  // OPTIONAL: output merged panel PREFIX (needs --merge-into)
    std::string device;       // OPTIONAL: CUDA device ordinal(s), e.g. "0" (default auto)
    int min_dp = 8;           // ref-block MinDP / variant DP floor (frozen default)
    int min_gq = 20;          // variant GQ floor (frozen default)
};

[[nodiscard]] int run_ingest(const IngestArgs& args);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_INGEST_HPP
