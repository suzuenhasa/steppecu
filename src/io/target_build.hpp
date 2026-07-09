// src/io/target_build.hpp
//
// Reference: docs/reference/src_io_target_build.hpp.md
//
// target_build — the Stage-2 NATIVE panel harmonizer: builds a TargetSites in
// memory from the AADR .snp panel (GRCh37), the orchestrated rsID->pos38 lift
// map, and a FaidxReader over the GRCh38 FASTA. This is the native replacement
// for the Stage-0/Stage-1 orchestrated prep (oracle.load_panel + lift_panel +
// emit_target_table.py). It reproduces, line-for-line, the oracle's:
//   - autosome (chrom 1..22) + rsID (id starts "rs") filter (M1, Fork #1)
//   - strict 1:1 panel de-dup: an rsID with multiplicity>1 (counted over the
//     autosomal-rs rows only) is dropped (H3)
//   - palindrome flag (A/T, C/G) — palindromes are KEPT in the table, flag set
//   - lift join to pos38 (the surviving orchestrated input; absent -> dropped)
//   - native GRCh38 ref38 base fetch for EVERY lifted site (incl. palindromes)
//
// The GRCh37->GRCh38 lift itself stays orchestrated (plan Fork #1: no chain
// engine) and is consumed here as a plain `rsID<TAB>pos38` map. Everything else
// is native. Pure host C++20 io-leaf (stdlib only).
#ifndef STEPPE_IO_TARGET_BUILD_HPP
#define STEPPE_IO_TARGET_BUILD_HPP

#include <string>

#include "io/faidx_reader.hpp"
#include "io/target_sites.hpp"

namespace steppe::io {

struct TargetBuildOptions {
    bool autosomes_only = true;  // chrom 1..22 (M1); the only supported v1 mode
    // GRCh37-direct (same-build) mode: the VCF is on the panel's own GRCh37
    // coordinates, so the rsID/pos join is DIRECT — pos38 := pos37 (identity, no
    // lift file read) and ref38 is fetched from a GRCh37 fasta at the panel
    // physpos. lift_no_lift is then definitionally 0 and lift_ok == the surviving
    // autosomal-rs non-dup rows. The cross-build (GRCh38/other) path is unchanged.
    bool identity_lift = false;
};

// Mirrors the oracle load_panel/lift_panel counters (gate-1 counts assertion).
struct TargetBuildCounts {
    long long panel_total = 0;        // panel rows with >= 6 fields (oracle n_total)
    long long panel_autosomal = 0;    // rows on chrom 1..22
    long long panel_non_rsid = 0;     // autosomal rows whose id is not "rs..."
    long long panel_palindromic = 0;  // autosomal-rs rows that are palindromic (PRE-dedup)
    long long panel_dup_rsids = 0;    // DISTINCT rsIDs with multiplicity > 1 (len(dup))
    long long lift_dropped_dup = 0;   // ROWS dropped because their rsID is a dup (n_dropdup)
    long long lift_ok = 0;            // rows lifted to a pos38 (== emitted)
    long long lift_no_lift = 0;       // autosomal-rs, non-dup rows absent from the lift map
    long long emitted = 0;            // TargetSites.sites.size() (== lift_ok)
};

// Build the native target-site set.
//   panel_snp : AADR EIGENSTRAT .snp (GRCh37): id chrom genpos physpos A1 A2
//   lift_map  : orchestrated rsID<TAB>pos38 map (leading header row tolerated);
//               IGNORED when opts.identity_lift (same-build GRCh37, pos38:=pos37)
//   fasta     : FaidxReader supplying ref38 (GRCh38, or GRCh37 in identity mode)
// Palindromic sites are KEPT (flag set); only no-lift / dup rows are dropped.
// Throws std::runtime_error on open/format failure or an out-of-range lift pos38.
[[nodiscard]] TargetSites build_target_sites(const std::string& panel_snp,
                                             const std::string& lift_map, FaidxReader& fasta,
                                             const TargetBuildOptions& opts,
                                             TargetBuildCounts& counts);

// Build the native target-site set WITHOUT a reference FASTA — the consumer-raw
// (23andMe / AncestryDNA / MyHeritage) path. GRCh37 identity join (pos38 := pos37,
// no lift file read), ref38 left '.' (the consumer reader reconciles the two
// observed alleles against the panel A1/A2 directly and never consults ref38).
// Same autosome/rsID filter, strict 1:1 de-dup, and palindrome flag as the fasta
// path. Throws std::runtime_error on open/format failure.
[[nodiscard]] TargetSites build_target_sites_noref(const std::string& panel_snp,
                                                   const TargetBuildOptions& opts,
                                                   TargetBuildCounts& counts);

// Dump a TargetSites as the 7-col table (rsID chrom pos37 pos38 A1 A2 ref38) for
// the gate-1 diff against the orchestrated emit_target_table.py output.
void write_target_table(const std::string& path, const TargetSites& ts);

}  // namespace steppe::io

#endif  // STEPPE_IO_TARGET_BUILD_HPP
