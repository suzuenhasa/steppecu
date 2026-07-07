// src/app/cmd_ingest.hpp
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
    std::string vcf;          // REQUIRED: the .vcf.gz / .vcf
    std::string targets;      // REQUIRED: the GRCh38 target-site table
    std::string sample;       // OPTIONAL: sample id (default = the sole sample)
    std::string report;       // OPTIONAL: per-site report TSV (primary artifact)
    std::string emit_tile;    // OPTIONAL: raw canonical 2-bit tile bytes (needs a device)
    std::string device;       // OPTIONAL: CUDA device ordinal(s), e.g. "0" (default auto)
    int min_dp = 8;           // ref-block MinDP / variant DP floor (frozen default)
    int min_gq = 20;          // variant GQ floor (frozen default)
};

[[nodiscard]] int run_ingest(const IngestArgs& args);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_INGEST_HPP
