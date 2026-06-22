// src/io/eigenstrat_format.hpp
//
// EIGENSTRAT / packed-genotype FORMAT CONSTANTS + header parse (architecture.md
// §4 `io` LEAF, §5 S0; ROADMAP §4 "TGENO header 48 / ceil(nsnp/4) belong in io
// format constants", M1).
//
// This is the single home for the packed-genotype format literals the §4
// magic-number inventory flags: the 48-byte header record and the
// `ceil(n_snp/4)` packed-record stride are DERIVED from the parsed header here,
// never hardcoded at a decode call site. Two on-disk packings are recognized:
//
//   * TGENO  (magic "TGENO") — INDIVIDUAL-major: one record per INDIVIDUAL,
//     `ceil(n_snp/4)` bytes, 4 SNPs/byte. This is the real AADR v66 layout
//     (`TGENO 27594 584131` = n_ind, n_snp). The genotype code for SNP k in a
//     byte is `(byte >> (6 - 2*(k mod 4))) & 3` (MSB-first); SNP k lives in byte
//     `k/4` of the individual's record.
//   * GENO   (magic "GENO")  — SNP-major PACKEDANCESTRYMAP: one record per SNP,
//     `max(48, ceil(n_ind/4))` bytes, 4 individuals/byte. Recognized so the
//     reader can tell the two apart and refuse to mis-decode; the M1 decode path
//     targets TGENO (the real data), and a GENO file is reported, not silently
//     read with the wrong axis.
//
// The 2-bit code → genotype mapping is the RAW VALUE (verified bit-for-bit
// against the on-box oracle build_tgeno_matrix.py): 0→0 ref-allele copies, 1→1,
// 2→2, 3→MISSING. This is NOT the binary mapping (00→0,10→1,11→2,01→missing),
// which mis-decodes; the raw-value mapping reproduces the oracle exactly.
//
// LAYERING: this is an `io`-leaf header (architecture.md §4) — pure host C++20,
// depends on NOTHING in core/device, includes only the standard library. No
// CUDA, no upward dependency.
#ifndef STEPPE_IO_EIGENSTRAT_FORMAT_HPP
#define STEPPE_IO_EIGENSTRAT_FORMAT_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace steppe::io {

// ---------------------------------------------------------------------------
// Packed-format constants — the promoted §4 literals (ROADMAP §4). No decode
// site may re-spell these numbers; they live here once and the header parse
// derives the record stride from them.
// ---------------------------------------------------------------------------

/// Bytes in the leading ASCII header record of a packed EIGENSTRAT .geno file
/// (`TGENO`/`GENO`). The header begins with the magic, then the two decimal
/// counts and dataset hashes, NUL-padded to this fixed width. Verified against
/// the real AADR v66 TGENO file (the 48-byte `TGENO 27594 584131 ...` header).
inline constexpr std::size_t kGenoHeaderBytes = 48;

/// Format magic for the TGENO (individual-major) packing — the first whitespace-
/// delimited token of the .geno header. Single-homed here (the "single home for
/// format literals" policy) so the parse site references the constant rather than
/// re-spelling the literal (cleanup eigenstrat_format 5.4; NAMING-STYLE-STANDARD
/// §2.5 single-source, §4 "Unnamed literal"). Value unchanged — parity-neutral.
inline constexpr std::string_view kMagicTgeno = "TGENO";

/// Format magic for the GENO (SNP-major PACKEDANCESTRYMAP) packing — the first
/// header token. Single-homed beside kMagicTgeno for the same reason; recognized
/// so the reader can tell the two axes apart and refuse to mis-decode (cleanup
/// eigenstrat_format 5.4; NAMING-STYLE-STANDARD §2.5, §4). Value unchanged.
inline constexpr std::string_view kMagicGeno = "GENO";

/// Genotype codes packed per byte (4 SNPs/byte for TGENO, 4 individuals/byte for
/// GENO) — 2 bits each. A true structural constant of the 2-bit packing.
inline constexpr int kCodesPerByte = 4;

/// Bits per packed genotype code (2-bit packing).
inline constexpr int kBitsPerCode = 2;

/// Low-bit mask for a single packed code, derived from kBitsPerCode so the mask
/// (0x3 for 2-bit codes) can never desync from the packing width. Used by
/// code_in_byte to isolate the extracted code.
inline constexpr std::uint8_t kCodeMask = (1u << kBitsPerCode) - 1u;

/// The 2-bit code that denotes a MISSING genotype. Codes 0/1/2 are reference-
/// allele copy counts; code 3 is missing (RAW-VALUE mapping — verified against
/// the oracle). Excluded from BOTH numerator and denominator of the allele freq.
inline constexpr std::uint8_t kMissingCode = 3;

/// The 2-bit code that denotes a HETEROZYGOUS genotype (1 reference-allele copy).
/// Used by the AT2 pseudo-haploid auto-detection (adjust_pseudohaploid=TRUE): a
/// sample is DIPLOID iff it has at least one het call in the detection prefix
/// (a haploid genome can never be heterozygous), else PSEUDO-HAPLOID. Mirrors
/// core::kHeterozygousGenotypeCode (kept here so the io leaf does not depend on core).
inline constexpr std::uint8_t kHetCode = 1;

/// AT2 pseudo-haploid auto-detection window: the number of leading SNPs scanned
/// per sample for a het call (admixtools cpp_*_ploidy default ntest = 1000;
/// verified against admixtools 2.0.10). Mirrors core::kPloidyDetectSnps. A record
/// shorter than this just scans fewer SNPs (the whole prefix).
inline constexpr std::size_t kPloidyDetectSnps = 1000;

// ---------------------------------------------------------------------------
// EIGENSTRAT .snp non-autosomal chromosome codes — the single home for the
// sex/mitochondrial label→code convention (ROADMAP §4 "io format constants";
// architecture.md §8 single-home). These are FORMAT conventions (the EIGENSOFT
// integer codes), NOT mathematical constants, so they live here once rather than
// as bare literals at the `.snp` parse site (cleanup snp_reader F12 / B16, X-8).
//
// CORRECTNESS COUPLING — the M2 `autosomes_only` filter depends on these EXACT
// codes: AT2 parity = autosomes 1..22 (config.hpp `kAutosomeChromMin/Max`), so
// the sex chromosomes X→23 and Y→24 (and MT/other non-autosomal codes) are the
// ones the inclusive 1..22 range drops. The `.snp` reader EMITS these codes and
// the autosome filter EXCLUDES them by definition; both must agree on the X=23 /
// Y=24 / MT=90 mapping, so it is single-sourced here. See the cross-reference in
// `include/steppe/config.hpp` (`kAutosomeChromMax`).
// ---------------------------------------------------------------------------

/// EIGENSTRAT chromosome code for the X (sex) chromosome. Dropped by the
/// autosomes-only filter (outside the 1..22 autosome range).
inline constexpr int kChromCodeX = 23;

/// EIGENSTRAT chromosome code for the Y (sex) chromosome. Dropped by the
/// autosomes-only filter (outside the 1..22 autosome range).
inline constexpr int kChromCodeY = 24;

/// EIGENSTRAT chromosome code for the mitochondrial (MT) "chromosome". Dropped by
/// the autosomes-only filter (outside the 1..22 autosome range).
inline constexpr int kChromCodeMt = 90;

// ---------------------------------------------------------------------------
// EIGENSTRAT .snp TEXT-record format constants — the column layout
//   <id> <chrom> <genpos> [<physpos> <ref> <alt>]
// single-homed here (snp_reader.cpp consumes them) so the field-count gates, the
// allele column indices, and the prose cannot drift (cleanup snp_reader F12/B14).
// ---------------------------------------------------------------------------

/// Minimum well-formed .snp field count: <id> <chrom> <genpos>. Fewer ⇒ malformed.
inline constexpr std::size_t kMinSnpFields = 3;
/// Full .snp record field count (the 6-column form carrying explicit ref/alt).
inline constexpr std::size_t kFullSnpFields = 6;
/// 0-based column index of the reference allele in a full 6-column .snp record.
inline constexpr std::size_t kRefAlleleCol = 4;
/// 0-based column index of the alternate allele in a full 6-column .snp record.
inline constexpr std::size_t kAltAlleleCol = 5;
/// EIGENSTRAT "missing/unknown base": ref/alt default for a <6-column record.
inline constexpr char kMissingAllele = 'N';
/// First synthetic code for a non-numeric/non-X/Y/MT chromosome label; subsequent
/// distinct labels decrement (codes start at -1 and go more negative). Outside the
/// 1..22 autosome range, so such labels are dropped by the autosomes-only filter.
inline constexpr int kFirstOtherChromCode = -1;

/// Number of bytes needed to pack `n_codes` 2-bit codes, 4 per byte: ceil(n/4).
/// This is the `ceil(nsnp/4)` (TGENO) / `ceil(nind/4)` (GENO) record-stride
/// formula the §4 inventory flags — computed here, never open-coded elsewhere.
[[nodiscard]] constexpr std::size_t packed_bytes(std::size_t n_codes) noexcept {
    // Widen the int format constant once; it appears in both the numerator and the
    // divisor of the ceil-divide (cleanup eigenstrat_format 7.3).
    constexpr std::size_t cpb = static_cast<std::size_t>(kCodesPerByte);
    return (n_codes + cpb - 1) / cpb;
}

/// Extract the 2-bit code for position `k` (0-based) within a packed byte,
/// MSB-first: position 0 in bits 7-6, 1 in 5-4, 2 in 3-2, 3 in 1-0. This is the
/// `(byte >> (6 - 2*(k mod 4))) & 3` rule, the SINGLE bit-order site shared by
/// the host reader and (via the decode primitive) the GPU/CPU decoders.
[[nodiscard]] constexpr std::uint8_t code_in_byte(std::uint8_t byte, int k) noexcept {
    const int shift = (kCodesPerByte - 1 - (k % kCodesPerByte)) * kBitsPerCode;  // 6,4,2,0
    return static_cast<std::uint8_t>((byte >> shift) & kCodeMask);
}

// ---------------------------------------------------------------------------
// Header parse.
// ---------------------------------------------------------------------------

/// Which packed layout a .geno file uses (its row axis). TGENO records are
/// individual-major; GENO (PACKEDANCESTRYMAP) records are SNP-major. The M1
/// decode targets TGENO; GENO is recognized so the reader does not mis-decode.
enum class GenoFormat { Unknown, Tgeno, Geno };

/// Parsed packed-.geno header: the format, the two dataset counts, and the
/// DERIVED record stride. For TGENO `n_records == n_ind` and `bytes_per_record
/// == packed_bytes(n_snp)`; for GENO `n_records == n_snp` and `bytes_per_record
/// == max(kGenoHeaderBytes, packed_bytes(n_ind))` (the PACKEDANCESTRYMAP rlen
/// floor). All record offsets are `kGenoHeaderBytes + record * bytes_per_record`.
struct GenoHeader {
    GenoFormat format = GenoFormat::Unknown;
    std::size_t n_ind = 0;             ///< number of individuals (samples)
    std::size_t n_snp = 0;            ///< number of SNPs
    std::size_t n_records = 0;        ///< records on disk (n_ind for TGENO, n_snp for GENO)
    std::size_t bytes_per_record = 0; ///< stride between records (DERIVED, not hardcoded)
    std::size_t header_bytes = kGenoHeaderBytes;  ///< leading header record width
};

/// Parse the leading `kGenoHeaderBytes` of a .geno header buffer. `head` must
/// point to at least `kGenoHeaderBytes` bytes. Recognizes the "TGENO"/"GENO"
/// magic and the two decimal counts that follow it (n_ind, n_snp for TGENO;
/// n_ind, n_snp for GENO — both store the same two numbers, the difference is
/// the record axis). Returns a header with `format == Unknown` on a bad magic or
/// unparsable counts (the caller decides how to fail; the `io` leaf never throws
/// across the layer boundary on a format probe).
[[nodiscard]] GenoHeader parse_geno_header(const std::array<char, kGenoHeaderBytes>& head) noexcept;

}  // namespace steppe::io

#endif  // STEPPE_IO_EIGENSTRAT_FORMAT_HPP
