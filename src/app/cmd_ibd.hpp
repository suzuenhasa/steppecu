// src/app/cmd_ibd.hpp
//
// The `steppe ibd` command entry point: ancIBD (Ringbauer et al. 2023) IBD-segment
// detection between pairs of ancient individuals from imputed genotype posteriors
// (GP) at 1240K sites, via a per-pair 5-state (non-IBD + 4 haplotype-sharing) forward-
// backward HMM run block-per-pair on the GPU. Rides the shipped GL/GP likelihood-
// tensor reader (with the phased-GT increment) and the FB scan substrate.
//
// Self-contained args (like `ingest`; no shared RunConfig). Reference:
// docs/planning/ancibd-face-spec.md.
#ifndef STEPPE_APP_CMD_IBD_HPP
#define STEPPE_APP_CMD_IBD_HPP

#include <string>

namespace steppe::app {

struct IbdArgs {
    std::string vcf;        // imputed VCF with phased GT + GP FORMAT (GLIMPSE output) [required]
    // Target-site source — exactly one of:
    std::string targets;    // pre-built target-site table (rsID chrom [pos37] pos38 A1 A2 ref)
    std::string panel;      // native: AADR EIGENSTRAT .snp panel (GRCh37)
    std::string fasta;      // native: build-matched .fa (+ .fai)
    std::string lift;       // native: rsID->pos38 map (cross-build only)
    std::string assembly;   // native: override VCF build detection (GRCh37|GRCh38)

    std::string map;        // per-site genetic map FILE ("rsID<ws>pos"); REQUIRED
    std::string map_unit = "cm";  // map file unit: cm | morgan (default cm)
    std::string af;         // per-site derived (ALT) allele-freq FILE ("rsID<ws>freq")
    std::string af_mode = "panel";  // panel (--af file) | sample (in-sample AF) | half (p=0.5)

    std::string samples;    // OPTIONAL: sample-subset FILE (one IID/line); default = all
    std::string pairs;      // OPTIONAL: explicit pair FILE ("iid1<ws>iid2"/line); default = all C(n,2)

    // ancIBD parameters (defaults = the locked run_ancIBD.py values).
    double ibd_in = 1.0;
    double ibd_out = 10.0;
    double ibd_jump = 400.0;
    double in_val = 1e-4;
    double min_error = 1e-3;
    double p_min = 1e-3;
    double post_cutoff = 0.99;
    double max_gap_cm = 0.75;   // merge gap (cM); 0.0075 Morgan
    double min_cm = 8.0;        // called-segment length floor (cM)

    std::string device;     // CUDA device ordinal(s) (default auto)
    std::string out;        // per-segment table path (default stdout)
    std::string summary;    // per-pair summary path (default: <out>.ibd_summary, or stderr note)
    std::string format = "tsv";  // tsv | csv
};

[[nodiscard]] int run_ibd(const IbdArgs& args);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_IBD_HPP
