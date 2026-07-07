// tests/unit/test_panel_merge_writer.cpp
//
// Host-only by-construction unit test of io::write_merged_panel (Stage 3 of the
// "place nikki among the ancients" build). Pure C++ TU, NO GPU, NO real AADR
// data: it hand-authors tiny SYNTHETIC panels in all three supported .geno
// layouts (TGENO individual-major, GENO/PACKEDANCESTRYMAP SNP-major, and
// EIGENSTRAT ASCII), appends a known nikki code vector, and asserts on the
// merged output that
//   (a) every PRE-EXISTING individual's decoded genotype is byte/decode-identical
//       to the source (adding nikki perturbs only her column) — Stage-3 gate (a);
//   (b) nikki's appended column decodes back to the exact input codes, missing
//       included — Stage-3 gate (b);
//   (c) the rewritten packed header parses (steppe's reader accepts it) with
//       n_ind bumped by one, and the .ind gains exactly one trailing row.
// The GENO case deliberately uses n_ind==3 (nikki lands in a previously-PARTIAL
// last byte) AND seeds that byte's unused low bits with garbage, pinning the
// mask-clear-before-OR fix (a plain OR would corrupt nikki's slot). Layout
// fixtures, not a precision claim (the numeric gate stays the real-AADR path).
//
// Self-checking main(): returns non-zero on the first failure. Links steppe::io.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "io/eigenstrat_format.hpp"
#include "io/panel_merge_writer.hpp"

namespace {

using steppe::io::code_in_byte;
using steppe::io::eigenstrat_char_to_code;
using steppe::io::GenoFormat;
using steppe::io::GenoHeader;
using steppe::io::kCodesPerByte;
using steppe::io::kGenoHeaderBytes;
using steppe::io::kMissingCode;
using steppe::io::MergeCounts;
using steppe::io::pack_code_into_byte;
using steppe::io::packed_bytes;
using steppe::io::parse_geno_header;

int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("  [FAIL] %s\n", what.c_str()); ++g_fail; }
}

// A tiny panel: n_ind individuals x n_snp SNPs, source code (i,s) = (i + s) % 3.
constexpr std::uint8_t src_code(std::size_t i, std::size_t s) {
    return static_cast<std::uint8_t>((i + s) % 3u);
}

void write_text(const std::filesystem::path& p, const std::string& s) {
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(s.data(), static_cast<std::streamsize>(s.size()));
}

void write_snp_ind(const std::string& prefix, std::size_t n_ind, std::size_t n_snp) {
    std::string snp;
    for (std::size_t s = 0; s < n_snp; ++s) {
        snp += "rs" + std::to_string(s) + "\t1\t0.0\t" + std::to_string(1000 + s) + "\tA\tG\n";
    }
    write_text(prefix + ".snp", snp);
    std::string ind;
    for (std::size_t i = 0; i < n_ind; ++i) {
        ind += "IND" + std::to_string(i) + "\tU\tPOP" + std::to_string(i % 2) + "\n";
    }
    write_text(prefix + ".ind", ind);
}

std::string packed_header(std::string_view magic, std::size_t n_ind, std::size_t n_snp,
                          std::size_t region_bytes) {
    std::string h(magic);
    h += " " + std::to_string(n_ind) + " " + std::to_string(n_snp) + " 0 0";
    std::string region(region_bytes, '\0');
    std::copy(h.begin(), h.end(), region.begin());
    return region;
}

// ---- TGENO source (individual-major) ----------------------------------------
void write_tgeno(const std::string& prefix, std::size_t n_ind, std::size_t n_snp) {
    const std::size_t rec = packed_bytes(n_snp);
    std::string out = packed_header("TGENO", n_ind, n_snp, kGenoHeaderBytes);
    for (std::size_t i = 0; i < n_ind; ++i) {
        std::vector<std::uint8_t> r(rec, 0u);
        for (std::size_t s = 0; s < n_snp; ++s)
            r[s / kCodesPerByte] = pack_code_into_byte(r[s / kCodesPerByte], static_cast<int>(s),
                                                       src_code(i, s));
        out.append(reinterpret_cast<const char*>(r.data()), r.size());
    }
    write_text(prefix + ".geno", out);
    write_snp_ind(prefix, n_ind, n_snp);
}

// ---- GENO / PACKEDANCESTRYMAP source (SNP-major), with garbage padding bits ---
void write_geno(const std::string& prefix, std::size_t n_ind, std::size_t n_snp) {
    const std::size_t stride = std::max(kGenoHeaderBytes, packed_bytes(n_ind));
    std::string out = packed_header("GENO", n_ind, n_snp, stride);
    for (std::size_t s = 0; s < n_snp; ++s) {
        std::vector<std::uint8_t> r(stride, 0u);
        for (std::size_t i = 0; i < n_ind; ++i)
            r[i / kCodesPerByte] = pack_code_into_byte(r[i / kCodesPerByte], static_cast<int>(i),
                                                       src_code(i, s));
        // Seed the unused trailing bits of the last meaningful byte with garbage
        // (nikki's slot lands here when n_ind % 4 != 0) to pin the mask-clear fix.
        if (n_ind % kCodesPerByte != 0) {
            const std::size_t last = (n_ind - 1) / kCodesPerByte;
            for (std::size_t k = n_ind; k < (last + 1) * kCodesPerByte; ++k) {
                const int sh = (kCodesPerByte - 1 - static_cast<int>(k % kCodesPerByte)) * 2;
                r[last] = static_cast<std::uint8_t>(r[last] | static_cast<std::uint8_t>(0b11 << sh));
            }
        }
        out.append(reinterpret_cast<const char*>(r.data()), r.size());
    }
    write_text(prefix + ".geno", out);
    write_snp_ind(prefix, n_ind, n_snp);
}

// ---- EIGENSTRAT ASCII source -------------------------------------------------
void write_eigenstrat(const std::string& prefix, std::size_t n_ind, std::size_t n_snp) {
    std::string out;
    for (std::size_t s = 0; s < n_snp; ++s) {
        for (std::size_t i = 0; i < n_ind; ++i)
            out += static_cast<char>('0' + src_code(i, s));
        out += '\n';
    }
    write_text(prefix + ".geno", out);
    write_snp_ind(prefix, n_ind, n_snp);
}

// Read the whole merged .geno and decode code (individual, snp).
std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Decode a merged packed panel (TGENO or GENO) into an (n_ind_out x n_snp) grid.
void decode_and_check(GenoFormat fmt, const std::string& out_prefix, std::size_t n_ind,
                      std::size_t n_snp, const std::vector<std::uint8_t>& nikki,
                      const std::string& tag) {
    const std::size_t n_out = n_ind + 1;
    std::array<char, kGenoHeaderBytes> head{};

    auto code_at = [&](const std::vector<std::uint8_t>& g, std::size_t i, std::size_t s) -> std::uint8_t {
        if (fmt == GenoFormat::Tgeno) {
            const std::size_t rec = packed_bytes(n_snp);
            const std::uint8_t* r = g.data() + kGenoHeaderBytes + i * rec;
            return code_in_byte(r[s / kCodesPerByte], static_cast<int>(s));
        }
        if (fmt == GenoFormat::Geno) {
            const std::size_t stride = std::max(kGenoHeaderBytes, packed_bytes(n_out));
            const std::uint8_t* r = g.data() + stride + s * stride;  // header region == 1 stride
            return code_in_byte(r[i / kCodesPerByte], static_cast<int>(i));
        }
        return 0;  // handled separately for EIGENSTRAT
    };

    if (fmt == GenoFormat::Eigenstrat) {
        std::ifstream in(out_prefix + ".geno", std::ios::binary);
        std::string line;
        std::size_t s = 0;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) break;
            check(line.size() == n_out, tag + ": eigenstrat row width == n_ind+1");
            for (std::size_t i = 0; i < n_ind; ++i) {
                std::uint8_t c = 0;
                check(eigenstrat_char_to_code(line[i], c), tag + ": ancient char decodes");
                check(c == src_code(i, s), tag + ": ancient col unchanged");
            }
            std::uint8_t nc = 0;
            check(eigenstrat_char_to_code(line[n_ind], nc), tag + ": nikki char decodes");
            const std::uint8_t want = nikki[s] > kMissingCode ? kMissingCode : nikki[s];
            check(nc == want, tag + ": nikki col == input code, snp " + std::to_string(s));
            ++s;
        }
        check(s == n_snp, tag + ": all snp rows present");
        return;
    }

    const std::vector<std::uint8_t> g = read_bytes(out_prefix + ".geno");
    std::copy_n(g.begin(), kGenoHeaderBytes, head.begin());
    const GenoHeader h = parse_geno_header(head);
    check(h.format == fmt, tag + ": header format round-trips");
    check(h.n_ind == n_out, tag + ": header n_ind bumped to n_ind+1");
    check(h.n_snp == n_snp, tag + ": header n_snp unchanged");
    for (std::size_t i = 0; i < n_ind; ++i)
        for (std::size_t s = 0; s < n_snp; ++s)
            check(code_at(g, i, s) == src_code(i, s), tag + ": ancient code unchanged");
    for (std::size_t s = 0; s < n_snp; ++s) {
        const std::uint8_t want = nikki[s] > kMissingCode ? kMissingCode : nikki[s];
        check(code_at(g, n_ind, s) == want, tag + ": nikki code, snp " + std::to_string(s));
    }
}

void check_ind(const std::string& out_prefix, std::size_t n_ind, const std::string& label,
               const std::string& tag) {
    std::ifstream in(out_prefix + ".ind");
    std::string line, last;
    std::size_t rows = 0, first_tok_ok = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        ++rows;
        last = line;
    }
    check(rows == n_ind + 1, tag + ": .ind gained exactly one row");
    check(last.rfind(label, 0) == 0, tag + ": appended .ind row id == label");
    check(last.find('\t' + label) != std::string::npos || last.find(' ' + label) != std::string::npos,
          tag + ": appended .ind pop == label (size-1 pop)");
    (void)first_tok_ok;
}

void run_case(GenoFormat fmt, const std::string& tag, const std::filesystem::path& dir,
              std::size_t n_ind, std::size_t n_snp) {
    const std::string src = (dir / ("src_" + tag)).string();
    const std::string out = (dir / ("out_" + tag)).string();
    if (fmt == GenoFormat::Tgeno) write_tgeno(src, n_ind, n_snp);
    else if (fmt == GenoFormat::Geno) write_geno(src, n_ind, n_snp);
    else write_eigenstrat(src, n_ind, n_snp);

    // nikki codes: a mix of 0/1/2 and missing (kMissingCode), length n_snp.
    std::vector<std::uint8_t> nikki(n_snp);
    long long want_called = 0, want_missing = 0;
    for (std::size_t s = 0; s < n_snp; ++s) {
        nikki[s] = (s % 4 == 3) ? kMissingCode : static_cast<std::uint8_t>(s % 3);
        (nikki[s] == kMissingCode) ? ++want_missing : ++want_called;
    }

    const MergeCounts mc = steppe::io::write_merged_panel(src, out, nikki, "SQBC6428");
    check(mc.format == fmt, tag + ": reported format");
    check(mc.n_snp == static_cast<long long>(n_snp), tag + ": counts n_snp");
    check(mc.n_ind_src == static_cast<long long>(n_ind), tag + ": counts n_ind_src");
    check(mc.n_ind_out == static_cast<long long>(n_ind + 1), tag + ": counts n_ind_out");
    check(mc.n_called == want_called, tag + ": counts n_called");
    check(mc.n_missing == want_missing, tag + ": counts n_missing");

    decode_and_check(fmt, out, n_ind, n_snp, nikki, tag);
    check_ind(out, n_ind, "SQBC6428", tag);

    // .snp byte-identical
    check(read_bytes(src + ".snp") == read_bytes(out + ".snp"), tag + ": .snp byte-identical");
}

}  // namespace

int main() {
    std::printf("=== panel_merge_writer by-construction unit test ===\n");
    std::error_code ec;
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_merge_test_" + std::to_string(static_cast<long long>(
            std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(dir, ec);

    // TGENO: n_ind large enough that packed_bytes(n_snp) spans >1 byte.
    run_case(GenoFormat::Tgeno, "tgeno", dir, /*n_ind=*/5, /*n_snp=*/9);
    // GENO/PA: n_ind==3 -> nikki lands in the previously-partial last byte (with
    // seeded garbage padding) -> pins mask-clear. Also n_ind==4 -> new byte.
    run_case(GenoFormat::Geno, "geno_partial", dir, /*n_ind=*/3, /*n_snp=*/7);
    run_case(GenoFormat::Geno, "geno_newbyte", dir, /*n_ind=*/4, /*n_snp=*/7);
    // EIGENSTRAT ASCII.
    run_case(GenoFormat::Eigenstrat, "eig", dir, /*n_ind=*/6, /*n_snp=*/8);

    std::filesystem::remove_all(dir, ec);
    if (g_fail == 0) {
        std::printf("\nRESULT: PASS (TGENO/GENO/EIGENSTRAT append: ancients unchanged, nikki "
                    "column exact, header + .ind + .snp correct)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_fail);
    return 1;
}
