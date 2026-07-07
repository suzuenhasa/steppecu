// src/io/panel_merge_writer.cpp
//
// Implementation of the format-preserving single-individual panel merge (Stage 3).
// See the header for the branch overview. Every failure surfaces as
// std::runtime_error; no CUDA, no core/device dependency.
#include "io/panel_merge_writer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "io/eigenstrat_format.hpp"

namespace steppe::io {

namespace {

[[nodiscard]] std::string geno_path(const std::string& prefix) { return prefix + ".geno"; }
[[nodiscard]] std::string snp_path(const std::string& prefix) { return prefix + ".snp"; }
[[nodiscard]] std::string ind_path(const std::string& prefix) { return prefix + ".ind"; }

// nikki's canonical 2-bit code -> the EIGENSTRAT ASCII char.
[[nodiscard]] char code_to_eigenstrat_char(std::uint8_t code) {
    switch (code) {
        case 0: return '0';
        case 1: return '1';
        case 2: return '2';
        default: return kEigenstratMissingChar;  // kMissingCode / anything else -> '9'
    }
}

// Byte-copy src -> dst (both whole files). Throws on any I/O failure.
void copy_file_bytes(const std::string& src, const std::string& dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in) throw std::runtime_error("panel_merge_writer: cannot open source: " + src);
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("panel_merge_writer: cannot open output: " + dst);
    constexpr std::size_t kChunk = 1u << 22;  // 4 MiB
    std::vector<char> buf(kChunk);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const std::streamsize got = in.gcount();
        if (got > 0) out.write(buf.data(), got);
        if (!out) throw std::runtime_error("panel_merge_writer: failed writing: " + dst);
    }
    if (in.bad()) throw std::runtime_error("panel_merge_writer: read error on: " + src);
}

// Copy the source .ind verbatim, then append one "<label>\tU\t<label>" row (its
// own size-1 population). Guarantees a newline separates the appended row.
void write_merged_ind(const std::string& src_ind, const std::string& out_ind,
                      const std::string& label) {
    std::ifstream in(src_ind, std::ios::binary);
    if (!in) throw std::runtime_error("panel_merge_writer: cannot open .ind: " + src_ind);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) throw std::runtime_error("panel_merge_writer: read error on .ind: " + src_ind);

    std::ofstream out(out_ind, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("panel_merge_writer: cannot open output .ind: " + out_ind);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!bytes.empty() && bytes.back() != '\n') out << '\n';
    out << label << '\t' << 'U' << '\t' << label << '\n';
    if (!out) throw std::runtime_error("panel_merge_writer: failed writing .ind: " + out_ind);
}

// Rebuild a packed-format 48-byte header text (TGENO/GENO) with n_ind replaced by
// `new_n_ind`, preserving the original trailing tokens (hashes) when present. The
// text is written into a zero-filled buffer of `region_bytes` length (the on-disk
// header stride: 48 for TGENO, max(48, packed_bytes(n_ind)) for GENO).
[[nodiscard]] std::vector<char> rebuild_packed_header(const std::array<char, kGenoHeaderBytes>& head,
                                                      std::string_view magic,
                                                      std::size_t new_n_ind, std::size_t n_snp,
                                                      std::size_t region_bytes) {
    // Recover the original tokens after the magic (n_ind n_snp [ihash shash]).
    std::string text;
    for (char c : head) {
        if (c == '\0') break;
        text.push_back(c);
    }
    std::istringstream ss(text);
    std::vector<std::string> tok;
    std::string t;
    while (ss >> t) tok.push_back(t);

    // Rebuild: magic new_n_ind n_snp [orig tok[3] tok[4] ...]. The trailing hash
    // tokens are kept verbatim (shash over the unchanged .snp stays valid; ihash
    // is stale after adding an individual but steppe's reader ignores both —
    // parse_geno_header reads only magic + n_ind + n_snp).
    std::ostringstream out;
    out << magic << ' ' << new_n_ind << ' ' << n_snp;
    for (std::size_t i = 3; i < tok.size(); ++i) out << ' ' << tok[i];
    const std::string hdr = out.str();
    if (hdr.size() > kGenoHeaderBytes) {
        throw std::runtime_error("panel_merge_writer: rebuilt header exceeds " +
                                 std::to_string(kGenoHeaderBytes) + " bytes");
    }
    std::vector<char> region(region_bytes, '\0');
    std::copy(hdr.begin(), hdr.end(), region.begin());
    return region;
}

// --- TGENO (individual-major) append: the primary AADR-1240K path -------------
// Each individual is its own record of packed_bytes(n_snp) bytes. Copy the whole
// n_ind-record data region verbatim, then append nikki's single packed record.
MergeCounts merge_tgeno(const GenoHeader& h, const std::array<char, kGenoHeaderBytes>& head,
                        const std::string& src_geno, const std::string& out_geno,
                        const std::vector<std::uint8_t>& nikki_codes) {
    const std::size_t n_ind = h.n_ind;
    const std::size_t n_snp = h.n_snp;
    if (nikki_codes.size() != n_snp) {
        throw std::runtime_error("panel_merge_writer(TGENO): nikki_codes has " +
                                 std::to_string(nikki_codes.size()) + " entries but the panel has " +
                                 std::to_string(n_snp) + " SNP rows");
    }
    const std::size_t rec_bytes = packed_bytes(n_snp);  // bytes per individual record

    std::ifstream in(src_geno, std::ios::binary);
    if (!in) throw std::runtime_error("panel_merge_writer: cannot open .geno: " + src_geno);
    std::ofstream out(out_geno, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("panel_merge_writer: cannot open output .geno: " + out_geno);

    // Header (48 bytes) with n_ind -> n_ind+1.
    const std::vector<char> hdr =
        rebuild_packed_header(head, kMagicTgeno, n_ind + 1, n_snp, kGenoHeaderBytes);
    out.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));

    // Byte-copy the entire n_ind-record data region (existing individuals are
    // untouched — this is the structural proof of Stage-3 gate (a)).
    in.seekg(static_cast<std::streamoff>(h.header_bytes), std::ios::beg);
    std::size_t remaining = n_ind * rec_bytes;
    constexpr std::size_t kChunk = 1u << 22;  // 4 MiB
    std::vector<char> buf(kChunk);
    while (remaining > 0) {
        const std::size_t want = std::min(kChunk, remaining);
        in.read(buf.data(), static_cast<std::streamsize>(want));
        if (in.gcount() != static_cast<std::streamsize>(want)) {
            throw std::runtime_error("panel_merge_writer(TGENO): short read of the data region "
                                     "(truncated .geno?) in " + src_geno);
        }
        out.write(buf.data(), static_cast<std::streamsize>(want));
        remaining -= want;
    }

    // Append nikki's single packed record (fresh buffer -> plain OR is safe).
    std::vector<std::uint8_t> rec(rec_bytes, 0u);
    MergeCounts mc;
    for (std::size_t s = 0; s < n_snp; ++s) {
        const std::uint8_t code = nikki_codes[s] > kMissingCode ? kMissingCode : nikki_codes[s];
        rec[s / static_cast<std::size_t>(kCodesPerByte)] =
            pack_code_into_byte(rec[s / static_cast<std::size_t>(kCodesPerByte)],
                                static_cast<int>(s), code);
        (code == kMissingCode) ? ++mc.n_missing : ++mc.n_called;
    }
    out.write(reinterpret_cast<const char*>(rec.data()), static_cast<std::streamsize>(rec.size()));
    if (!out) throw std::runtime_error("panel_merge_writer: failed writing .geno: " + out_geno);

    mc.format = GenoFormat::Tgeno;
    mc.n_snp = static_cast<long long>(n_snp);
    mc.n_ind_src = static_cast<long long>(n_ind);
    mc.n_ind_out = static_cast<long long>(n_ind + 1);
    return mc;
}

// --- GENO / PACKEDANCESTRYMAP (SNP-major) prefix-append -----------------------
// nikki is the new LAST slot n_ind in every SNP record: copy the ancient prefix
// bytes, then mask-clear-and-OR her 2-bit code into the (possibly previously
// partial) new slot's byte. Record stride grows when n_ind%4 == 0.
MergeCounts merge_geno(const GenoHeader& h, const std::array<char, kGenoHeaderBytes>& head,
                       const std::string& src_geno, const std::string& out_geno,
                       const std::vector<std::uint8_t>& nikki_codes) {
    const std::size_t n_ind = h.n_ind;
    const std::size_t n_snp = h.n_snp;
    if (nikki_codes.size() != n_snp) {
        throw std::runtime_error("panel_merge_writer(GENO): nikki_codes has " +
                                 std::to_string(nikki_codes.size()) + " entries but the panel has " +
                                 std::to_string(n_snp) + " SNP rows");
    }
    const std::size_t meaningful = packed_bytes(n_ind);               // ancient prefix bytes
    const std::size_t src_stride = std::max(kGenoHeaderBytes, meaningful);
    const std::size_t out_stride = std::max(kGenoHeaderBytes, packed_bytes(n_ind + 1));

    // nikki's slot within her record.
    const std::size_t nikki_byte = n_ind / static_cast<std::size_t>(kCodesPerByte);
    const int nikki_shift =
        (kCodesPerByte - 1 - static_cast<int>(n_ind % static_cast<std::size_t>(kCodesPerByte))) *
        kBitsPerCode;

    std::ifstream in(src_geno, std::ios::binary);
    if (!in) throw std::runtime_error("panel_merge_writer: cannot open .geno: " + src_geno);
    std::ofstream out(out_geno, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("panel_merge_writer: cannot open output .geno: " + out_geno);

    const std::vector<char> hdr =
        rebuild_packed_header(head, kMagicGeno, n_ind + 1, n_snp, out_stride);
    out.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));

    in.seekg(static_cast<std::streamoff>(h.header_bytes), std::ios::beg);
    std::vector<std::uint8_t> src_rec(src_stride);
    std::vector<std::uint8_t> out_rec(out_stride);
    MergeCounts mc;
    for (std::size_t r = 0; r < n_snp; ++r) {
        in.read(reinterpret_cast<char*>(src_rec.data()), static_cast<std::streamsize>(src_stride));
        if (in.gcount() != static_cast<std::streamsize>(src_stride)) {
            throw std::runtime_error("panel_merge_writer(GENO): short read of SNP record " +
                                     std::to_string(r) + " in " + src_geno);
        }
        std::fill(out_rec.begin(), out_rec.end(), std::uint8_t{0});
        std::copy_n(src_rec.begin(), meaningful, out_rec.begin());  // ancient prefix
        const std::uint8_t code = nikki_codes[r] > kMissingCode ? kMissingCode : nikki_codes[r];
        // Mask-clear nikki's 2-bit slot before OR: robust against nonzero padding
        // bits in a previously-partial last byte (critic fix).
        out_rec[nikki_byte] = static_cast<std::uint8_t>(
            out_rec[nikki_byte] & ~static_cast<std::uint8_t>(kCodeMask << nikki_shift));
        out_rec[nikki_byte] = static_cast<std::uint8_t>(
            out_rec[nikki_byte] | static_cast<std::uint8_t>(code << nikki_shift));
        out.write(reinterpret_cast<const char*>(out_rec.data()),
                  static_cast<std::streamsize>(out_stride));
        (code == kMissingCode) ? ++mc.n_missing : ++mc.n_called;
    }
    if (!out) throw std::runtime_error("panel_merge_writer: failed writing .geno: " + out_geno);

    mc.format = GenoFormat::Geno;
    mc.n_snp = static_cast<long long>(n_snp);
    mc.n_ind_src = static_cast<long long>(n_ind);
    mc.n_ind_out = static_cast<long long>(n_ind + 1);
    return mc;
}

// --- EIGENSTRAT (ASCII) append: one char per genotype line --------------------
MergeCounts merge_eigenstrat(const std::string& src_geno, const std::string& out_geno,
                             const std::vector<std::uint8_t>& nikki_codes) {
    std::ifstream in(src_geno, std::ios::binary);
    if (!in) throw std::runtime_error("panel_merge_writer: cannot open .geno: " + src_geno);
    std::ofstream out(out_geno, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("panel_merge_writer: cannot open output .geno: " + out_geno);

    MergeCounts mc;
    std::string line;
    std::size_t n_ind = 0;
    std::size_t r = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // tolerate CRLF
        if (line.empty()) {
            if (in.peek() == std::char_traits<char>::eof()) break;
            throw std::runtime_error("panel_merge_writer(EIGENSTRAT): interior blank .geno line");
        }
        if (r == 0) n_ind = line.size();
        else if (line.size() != n_ind) {
            throw std::runtime_error("panel_merge_writer(EIGENSTRAT): ragged .geno line at row " +
                                     std::to_string(r));
        }
        if (r >= nikki_codes.size()) {
            throw std::runtime_error("panel_merge_writer(EIGENSTRAT): more .geno rows than "
                                     "nikki_codes entries (" + std::to_string(nikki_codes.size()) + ")");
        }
        const std::uint8_t code = nikki_codes[r] > kMissingCode ? kMissingCode : nikki_codes[r];
        out << line << code_to_eigenstrat_char(code) << '\n';
        (code == kMissingCode) ? ++mc.n_missing : ++mc.n_called;
        ++r;
    }
    if (r != nikki_codes.size()) {
        throw std::runtime_error("panel_merge_writer(EIGENSTRAT): panel has " + std::to_string(r) +
                                 " SNP rows but nikki_codes has " +
                                 std::to_string(nikki_codes.size()));
    }
    if (!out) throw std::runtime_error("panel_merge_writer: failed writing .geno: " + out_geno);

    mc.format = GenoFormat::Eigenstrat;
    mc.n_snp = static_cast<long long>(r);
    mc.n_ind_src = static_cast<long long>(n_ind);
    mc.n_ind_out = static_cast<long long>(n_ind + 1);
    return mc;
}

// Detect the source .geno format: parse the 48-byte header for TGENO/GENO magic;
// fall back to an EIGENSTRAT ASCII geometry probe (first-line length = n_ind).
[[nodiscard]] GenoHeader detect_geno(const std::string& src_geno,
                                     std::array<char, kGenoHeaderBytes>& head) {
    std::ifstream in(src_geno, std::ios::binary);
    if (!in) throw std::runtime_error("panel_merge_writer: cannot open .geno: " + src_geno);
    head.fill('\0');
    in.read(head.data(), static_cast<std::streamsize>(head.size()));
    GenoHeader h = parse_geno_header(head);
    if (h.format == GenoFormat::Tgeno || h.format == GenoFormat::Geno) return h;

    // EIGENSTRAT probe: first non-empty line length is n_ind; count lines for n_snp.
    std::ifstream tin(src_geno, std::ios::binary);
    std::string line;
    std::size_t n_ind = 0, n_snp = 0;
    bool first = true;
    while (std::getline(tin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) {
            if (tin.peek() == std::char_traits<char>::eof()) break;
            return GenoHeader{};  // interior blank -> Unknown
        }
        for (char c : line) {
            std::uint8_t code = 0;
            if (!eigenstrat_char_to_code(c, code)) return GenoHeader{};  // not ASCII geno
        }
        if (first) { n_ind = line.size(); first = false; }
        else if (line.size() != n_ind) return GenoHeader{};
        ++n_snp;
    }
    GenoHeader e;
    if (n_ind > 0 && n_snp > 0) {
        e.format = GenoFormat::Eigenstrat;
        e.n_ind = n_ind;
        e.n_snp = n_snp;
        e.n_records = n_snp;
    }
    return e;
}

}  // namespace

MergeCounts write_merged_panel(const std::string& panel_prefix, const std::string& out_prefix,
                               const std::vector<std::uint8_t>& nikki_codes,
                               const std::string& label) {
    if (label.empty()) throw std::runtime_error("panel_merge_writer: empty individual label");

    // .snp is unchanged (byte-copy); .ind gains one row.
    copy_file_bytes(snp_path(panel_prefix), snp_path(out_prefix));
    write_merged_ind(ind_path(panel_prefix), ind_path(out_prefix), label);

    std::array<char, kGenoHeaderBytes> head{};
    const GenoHeader h = detect_geno(geno_path(panel_prefix), head);

    MergeCounts mc;
    switch (h.format) {
        case GenoFormat::Tgeno:
            mc = merge_tgeno(h, head, geno_path(panel_prefix), geno_path(out_prefix), nikki_codes);
            break;
        case GenoFormat::Geno:
            mc = merge_geno(h, head, geno_path(panel_prefix), geno_path(out_prefix), nikki_codes);
            break;
        case GenoFormat::Eigenstrat:
            mc = merge_eigenstrat(geno_path(panel_prefix), geno_path(out_prefix), nikki_codes);
            break;
        case GenoFormat::Ancestrymap:
        case GenoFormat::Plink:
        case GenoFormat::Unknown:
        default:
            throw std::runtime_error(
                "panel_merge_writer: unsupported source .geno format (need TGENO, GENO/"
                "PACKEDANCESTRYMAP, or EIGENSTRAT ASCII): " + geno_path(panel_prefix));
    }
    return mc;
}

}  // namespace steppe::io
