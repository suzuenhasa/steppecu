// src/io/panel_merge_writer.hpp
//
// panel_merge_writer — the first genotype WRITER in the tree (Stage 3 of the
// "place nikki among the ancients" build). It appends ONE new trailing
// individual (a size-1 population) to an existing EIGENSTRAT-family panel,
// format-preserving, so every pre-existing individual's genotype is left
// byte-for-byte unchanged and only the new column/record is added.
//
// Three on-disk branches, auto-detected from the source .geno magic/geometry:
//   - TGENO (individual-major, the AADR 1240K .PUB panel): the CLEANEST case —
//     each individual is its own record, so the entire n_ind-record data region
//     is copied verbatim and nikki's single packed-SNP record is appended; the
//     header's n_ind is bumped by one. (The spec assumed PACKEDANCESTRYMAP and
//     said "refuse TGENO"; the real panel IS TGENO and steppe decodes TGENO
//     correctly, so the native append is both correct and avoids a 7 GB offline
//     convertf TGENO->PA conversion. See docs/planning/nikki-stage3-4-spec.md.)
//   - GENO / PACKEDANCESTRYMAP (SNP-major): format-preserving PREFIX-append —
//     each SNP record's ancient slots are copied and nikki's 2-bit code is
//     mask-cleared-then-OR'd into the new trailing slot; the record stride grows
//     when n_ind crosses a 4-slot byte boundary.
//   - EIGENSTRAT (ASCII): one appended char per genotype line.
// TGENO source with a TGENO append is exact; the other two are covered by the
// by-construction unit test. Pure host C++20 io-leaf — no CUDA.
//
// Reference: docs/reference/src_io_panel_merge_writer.hpp.md
#ifndef STEPPE_IO_PANEL_MERGE_WRITER_HPP
#define STEPPE_IO_PANEL_MERGE_WRITER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "io/eigenstrat_format.hpp"

namespace steppe::io {

// The per-run accounting the caller prints and the gate checks.
struct MergeCounts {
    GenoFormat format = GenoFormat::Unknown;
    long long n_snp = 0;       // panel SNP rows (== nikki_codes.size())
    long long n_ind_src = 0;   // source individuals (pre-merge)
    long long n_ind_out = 0;   // n_ind_src + 1
    long long n_called = 0;    // nikki codes in {0,1,2}
    long long n_missing = 0;   // nikki codes == kMissingCode (3)
};

// Append `nikki_codes` (one 2-bit code per panel SNP row, in {0,1,2,3} where 3 ==
// kMissingCode) to the panel at `panel_prefix.{geno,snp,ind}`, writing the merged
// triple to `out_prefix.{geno,snp,ind}`. The new individual is labelled `label`
// for BOTH its .ind id and its population (a size-1 pop). `.snp` is copied
// byte-identically; `.ind` is copied and one row appended.
//
// Throws std::runtime_error on any open/format/size failure, and specifically if
// nikki_codes.size() != the source n_snp (the code vector must be panel-aligned).
[[nodiscard]] MergeCounts write_merged_panel(const std::string& panel_prefix,
                                             const std::string& out_prefix,
                                             const std::vector<std::uint8_t>& nikki_codes,
                                             const std::string& label);

}  // namespace steppe::io

#endif  // STEPPE_IO_PANEL_MERGE_WRITER_HPP
