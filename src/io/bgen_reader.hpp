// src/io/bgen_reader.hpp
//
// BGEN v1.2 dosage reader (the `steppe pca --bgen` ingest arm). Reads a BGEN v1.2
// file — the plink2 / UK Biobank imputed-dosage shape — into a real-valued FP32
// DosageTile: one ALT-allele dosage in [0, 2] per (variant, sample), with a NaN
// sentinel for a missing call. This is steppe's FIRST real-valued genotype
// representation (the rest of the toolset is 2-bit hardcalls or genotype
// likelihoods); it feeds the Patterson PCA standardize a fractional dosage in
// place of a decoded {0,1,2} code.
//
// FORMAT (biallelic diploid; layout 2). Header: uint32 offset, uint32 Lh, uint32
// M (variants), uint32 N (samples), "bgen" magic, uint32 flags (bits0-1 =
// compression 0=none/1=zlib/2=zstd, bits2-5 = layout, bit31 = sample-ids present).
// Genotypes begin at byte (4 + offset). Each variant = an identifying block
// (uint16 Lid,id; uint16 Lrsid,rsid; uint16 Lchr,chr; uint32 pos; uint16 K;
// K*(uint32 La,allele)) followed by a genotype block (uint32 C total length; when
// compressed, uint32 D uncompressed length + C-4 zlib bytes -> D bytes). The
// uncompressed layout-2 payload is: uint32 N, uint16 K, uint8 Pmin, uint8 Pmax, N
// ploidy/missing bytes (bit7 missing, bits0-5 ploidy), uint8 phased, uint8 B
// (bits/prob), then the B-bit LSB-first probability bitstream (2 probs/sample for
// biallelic diploid, both phased and unphased). Each stored value is v/(2^B - 1).
//   UNPHASED: v0=P(AA), v1=P(AB); ALT dosage = P(AB) + 2*P(BB) = 2 - 2*v0 - v1.
//   PHASED:   v0,v1 = P(REF) on hap0,hap1; ALT dosage = (1-v0) + (1-v1) = 2 - v0 - v1.
//
// SCOPE (v1): biallelic (K==2), diploid (every ploidy byte == 2, Pmin==Pmax==2),
// layout 2, compression none/zlib. Multiallelic, non-diploid, phased-with-mixed-
// ploidy, layout 1, and zstd (compression 2) raise a clear std::runtime_error.
// The bit-unpack is on the CPU for the MVP (correctness first); a GPU inflate +
// unpack is a biobank-scale follow-on.
//
// Pure host C++20 io-leaf (links zlib for the inflate only); failures surface as
// std::runtime_error.
#ifndef STEPPE_IO_BGEN_READER_HPP
#define STEPPE_IO_BGEN_READER_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace steppe::io {

// Per-variant identifying metadata parsed from the BGEN variant block. allele0 is
// the REFERENCE allele (index 0); allele1 is the ALT allele (index 1) — the one the
// emitted dosage counts, matching plink2 `--export A` (which counts the ALT allele).
struct BgenSnpMeta {
    std::string rsid;
    std::string chrom;
    std::uint32_t pos = 0;
    std::string allele0;  // REF (allele index 0)
    std::string allele1;  // ALT (allele index 1); the dosage-counted allele
};

// DosageTile — the real-valued (FP32) genotype tile. dosage is INDIVIDUAL-MAJOR:
// dosage[i * n_snp + s] is sample i's ALT dosage at variant s, in [0, 2], or a
// quiet NaN when that call is missing (mirrors the 2-bit tile's missing == code 3).
// This is exactly the access pattern the PCA standardize kernels consume (individual
// the fast axis when building the column-major Z operand).
struct DosageTile {
    std::vector<float> dosage;  // n_individuals * n_snp, individual-major; NaN = missing
    std::size_t n_snp = 0;
    std::size_t n_individuals = 0;
    std::vector<std::string> sample_ids;  // length n_individuals (synthesized if absent)
    std::vector<BgenSnpMeta> snps;        // length n_snp
};

// read_bgen_dosages — parse a BGEN v1.2 biallelic-diploid file at `path` into a
// DosageTile. Throws std::runtime_error on any I/O failure, a malformed/truncated
// file, or an out-of-scope shape (see the SCOPE note above).
[[nodiscard]] DosageTile read_bgen_dosages(const std::string& path);

}  // namespace steppe::io

#endif  // STEPPE_IO_BGEN_READER_HPP
