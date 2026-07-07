// src/io/target_sites.hpp
//
// TargetSites — the GRCh38 target-site table the native VCF reader genotypes
// nikki against. This is the Stage-1 <-> Stage-2 boundary artifact: the panel
// already rsID-joined, lifted GRCh37->GRCh38, palindromes still IN, with nikki's
// GRCh38 REF base pre-fetched (ref38, upper-cased — soft-mask normalized). The
// reader does NOT do liftover / dbSNP-pos cross-check / strict-1:1 dedup; those
// stay orchestrated (shared with the Stage-0 oracle) for Stage 1.
//
// Columns (whitespace/tab separated, optional header line):
//     rsID  chrom  pos37  pos38  A1  A2  ref38      (7-col, canonical)
//     rsID  chrom  pos38  A1  A2  ref38             (6-col, pos37 defaults to 0)
//
// Per-chrom the reader needs a sorted pos38 array (duplicates KEPT) plus a
// last-wins (chrom,pos38)->slot map, built ONLY over non-palindromic sites —
// exactly the oracle's `ppos`/`slot` construction (oracle.py:143,236-237) so the
// interval-join coverage and slot lookup match at colliding lifted positions.
//
// Pure host C++20 io-leaf.
#ifndef STEPPE_IO_TARGET_SITES_HPP
#define STEPPE_IO_TARGET_SITES_HPP

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace steppe::io {

struct TargetSite {
    std::string rsid;
    int chrom = 0;
    long long pos37 = 0;
    long long pos38 = 0;
    char a1 = 'N';
    char a2 = 'N';
    char ref38 = '.';   // upper-cased GRCh38 REF base, or '.' when unavailable
    bool palindrome = false;
};

// Per-chromosome sorted pos38 index over the NON-palindromic sites.
struct ChromIndex {
    std::vector<long long> pos;                        // sorted ASC, duplicates kept
    std::unordered_map<long long, std::size_t> slot;   // pos38 -> index in `pos` (last-wins)
};

struct TargetSites {
    std::vector<TargetSite> sites;                    // panel order (as read)
    std::unordered_map<int, ChromIndex> by_chrom;     // only non-palindromic sites

    [[nodiscard]] bool has_chrom(int c) const { return by_chrom.find(c) != by_chrom.end(); }
};

// Parse the target-site table. Throws std::runtime_error on open/format failure.
[[nodiscard]] TargetSites read_target_sites(const std::string& path);

// Strand-ambiguous palindrome test: {A,T} or {C,G} (case-insensitive). Shared by
// read_target_sites and the Stage-2 native builder (target_build) so the drop
// policy is single-homed.
[[nodiscard]] bool is_palindrome(char a1, char a2);

// Build the per-chrom sorted pos38 index over the NON-palindromic sites, then the
// last-wins (chrom,pos38)->slot map — the interval-join contract shared by
// read_target_sites and build_target_sites. Overwrites ts.by_chrom.
void build_chrom_index(TargetSites& ts);

}  // namespace steppe::io

#endif  // STEPPE_IO_TARGET_SITES_HPP
