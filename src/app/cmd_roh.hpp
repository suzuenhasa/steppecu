// src/app/cmd_roh.hpp
//
// The `steppe roh` command entry point: hapROH (Ringbauer et al. 2021, the `hapsburg`
// package) runs-of-homozygosity detection in a single (pseudo-haploid) ancient genome by
// copying it against a phased reference-haplotype panel with a (K+1)-state Li-Stephens-
// style forward-backward HMM, then calling ROH segments. Rides the shipped phased-panel
// EIGENSTRAT reader (target = one ancient triple, panel = the reference haplotypes) and
// the FB scan substrate.
//
// Self-contained args (like `ibd`; no shared RunConfig — the ~15 method-specific hapROH
// knobs stay off the shared config surface, mirroring the ibd segment-face precedent).
// Reference: docs/planning/haproh-face-spec.md.
#ifndef STEPPE_APP_CMD_ROH_HPP
#define STEPPE_APP_CMD_ROH_HPP

#include <string>

namespace steppe::app {

struct RohArgs {
    std::string prefix;     // TARGET ancient genotype triple PREFIX.{geno,snp,ind} (pseudo-haploid) [required]
    std::string ref_panel;  // phased reference-haplotype panel triple PREFIX (staged from HDF5) [required]
    // The cM map is the panel .snp Morgan column (genpos_morgans), read by the front-end.
    std::string samples;    // OPTIONAL target subset FILE (one pop label/line); default = all target individuals
    std::string exclude_pops;  // OPTIONAL comma-separated panel population labels to drop (hapROH exclude_pops)

    // hapROH parameters (defaults = the locked hapsb_ind values).
    double e_rate = 0.01;    // genotype/miscopy error (haploid emission)
    double roh_in = 1.0;     // rate of jumping INTO a ROH copying state (per Morgan)
    double roh_out = 20.0;   // rate of jumping OUT of a ROH state (per Morgan)
    double roh_jump = 300.0; // rate of jumping within ROH to another reference haplotype (per Morgan)
    double in_val = 1e-4;    // initial per-state probability (fwd col 0 prior)
    double cutoff_post = 0.999;   // per-SNP ROH-posterior calling threshold
    std::string roh_bins = "4,8,12,20";  // summary length bins (cM)
    int n_ref = 0;           // reference INDIVIDUALS to use (K = 2*n_ref haplotypes); 0 = all in the panel

    std::string device;     // CUDA device ordinal(s) (default auto)
    std::string out;        // per-segment table path (default stdout)
    std::string summary;    // per-individual summary path (default: <out>.summary, or stderr note)
    std::string format = "tsv";  // tsv | csv
};

[[nodiscard]] int run_roh(const RohArgs& args);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_ROH_HPP
