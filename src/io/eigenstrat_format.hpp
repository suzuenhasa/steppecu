// src/io/eigenstrat_format.hpp
//
// The single home for the on-disk genotype format literals and helper functions
// used to read the EIGENSTRAT family (packed TGENO/GENO, ASCII EIGENSTRAT, legacy
// ANCESTRYMAP) and PLINK .bed files. Pure host C++20, standard library only — no
// CUDA and no core/device dependency, so the file readers can include it freely.
//
// Reference: docs/reference/src_io_eigenstrat_format.hpp.md
#ifndef STEPPE_IO_EIGENSTRAT_FORMAT_HPP
#define STEPPE_IO_EIGENSTRAT_FORMAT_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace steppe::io {

// Packed-format constants — reference §4
inline constexpr std::size_t kGenoHeaderBytes = 48;
inline constexpr std::string_view kMagicTgeno = "TGENO";
inline constexpr std::string_view kMagicGeno = "GENO";
inline constexpr int kCodesPerByte = 4;
inline constexpr int kBitsPerCode = 2;
inline constexpr std::uint8_t kCodeMask = (1u << kBitsPerCode) - 1u;
inline constexpr std::uint8_t kMissingCode = 3;
inline constexpr std::uint8_t kHetCode = 1;
inline constexpr std::size_t kPloidyDetectSnps = 1000;

// Chromosome codes for the .snp file — reference §5
inline constexpr int kChromCodeX = 23;
inline constexpr int kChromCodeY = 24;
inline constexpr int kChromCodeMt = 90;

// The .snp text-record columns — reference §6
inline constexpr std::size_t kMinSnpFields = 3;
inline constexpr std::size_t kFullSnpFields = 6;
inline constexpr std::size_t kPhysposCol = 3;
inline constexpr std::size_t kRefAlleleCol = 4;
inline constexpr std::size_t kAltAlleleCol = 5;
inline constexpr char kMissingAllele = 'N';
inline constexpr int kFirstOtherChromCode = -1;

// Bit-packing helpers — reference §7
[[nodiscard]] constexpr std::size_t packed_bytes(std::size_t n_codes) noexcept {
    constexpr std::size_t cpb = static_cast<std::size_t>(kCodesPerByte);
    return (n_codes + cpb - 1) / cpb;
}

[[nodiscard]] constexpr std::uint8_t code_in_byte(std::uint8_t byte, int k) noexcept {
    const int shift = (kCodesPerByte - 1 - (k % kCodesPerByte)) * kBitsPerCode;
    return static_cast<std::uint8_t>((byte >> shift) & kCodeMask);
}

// GenoFormat: the on-disk layouts — reference §8
enum class GenoFormat { Unknown, Tgeno, Geno, Eigenstrat, Plink, Ancestrymap };

// PLINK .bed constants — reference §9
inline constexpr unsigned char kBedMagic0 = 0x6c;
inline constexpr unsigned char kBedMagic1 = 0x1b;
inline constexpr unsigned char kBedModeSnpMajor = 0x01;
inline constexpr std::size_t kBedMagicBytes = 3;

inline constexpr std::array<std::uint8_t, 4> kBedToCanon = {2, kMissingCode, kHetCode, 0};

[[nodiscard]] constexpr std::uint8_t bed_code_in_byte(std::uint8_t byte, int k) noexcept {
    const int shift = (k % kCodesPerByte) * kBitsPerCode;
    return static_cast<std::uint8_t>((byte >> shift) & kCodeMask);
}

// EIGENSTRAT ASCII .geno constants — reference §10
inline constexpr char kEigenstratMissingChar = '9';
inline constexpr std::uint8_t kEigenstratMissingCode = kMissingCode;
inline constexpr char kEigenstratMaxCopiesChar = '2';

[[nodiscard]] constexpr bool eigenstrat_char_to_code(char c, std::uint8_t& out) noexcept {
    if (c >= '0' && c <= kEigenstratMaxCopiesChar) {
        out = static_cast<std::uint8_t>(c - '0');
        return true;
    }
    if (c == kEigenstratMissingChar) {
        out = kEigenstratMissingCode;
        return true;
    }
    return false;
}

// ANCESTRYMAP text constants — reference §11
inline constexpr std::size_t kAncestrymapFields = 3;

inline constexpr std::string_view kAncestrymapMissingToken = "-1";

[[nodiscard]] constexpr bool ancestrymap_token_to_code(std::string_view tok,
                                                       std::uint8_t& out) noexcept {
    if (tok.size() == 1 && tok[0] >= '0' && tok[0] <= kEigenstratMaxCopiesChar) {
        out = static_cast<std::uint8_t>(tok[0] - '0');
        return true;
    }
    if (tok == kAncestrymapMissingToken) {
        out = kMissingCode;
        return true;
    }
    return false;
}

// GenoHeader and parse_geno_header — reference §12
struct GenoHeader {
    GenoFormat format = GenoFormat::Unknown;
    std::size_t n_ind = 0;
    std::size_t n_snp = 0;
    std::size_t n_records = 0;
    std::size_t bytes_per_record = 0;
    std::size_t header_bytes = kGenoHeaderBytes;
};

[[nodiscard]] GenoHeader parse_geno_header(const std::array<char, kGenoHeaderBytes>& head) noexcept;

}  // namespace steppe::io

#endif  // STEPPE_IO_EIGENSTRAT_FORMAT_HPP
