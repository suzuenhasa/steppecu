// tests/unit/test_snp_reader.cpp
//
// Host-only unit test of io::read_snp (architecture.md §13 "Unit tests";
// ROADMAP M1, §5). Pure C++ TU, NO GPU, NO real AADR data: it writes tiny
// SYNTHETIC .snp files to a temp path and exercises the parser's happy-path
// column decision plus its FAIL-FAST behavior on malformed records — control
// flow / boundary logic, not a precision claim (ROADMAP §0: the no-synthetic
// rule is for precision claims; a hand-written .snp line is a layout fixture).
// The numerical-parity gate stays the real-AADR equivalence tests
// (tests/reference/test_f2_blocks_equivalence.cu).
//
// PRIMARY PURPOSE (cleanup snp_reader B14 / C1+N1 verdict gate): the old parser
// used an extraction-failure FALL-THROUGH — on a 6-field `operator>>` failure it
// retried a 3-field read and silently `continue`d (dropped the row) on a short
// line, shifting every later SNP's metadata relative to its genotype (silent
// SNP-axis misalignment), and it accepted "inf"/"nan"/overflowing genpos tokens
// that become static_cast<int>(NaN/Inf) UB in core::block_of. The fix:
//   (1) a TOKEN-COUNT-based column decision (>=3 fields well-formed; full 6 carry
//       alleles; <3 is a malformed record),
//   (2) genpos parsed with std::from_chars (locale-free; rejects inf/nan/garbage),
//   (3) FAIL-FAST with the 1-based LINE NUMBER on any malformed record.
// This test pins (the verdict gates): a malformed record THROWS and the message
// carries the offending line number; a well-formed file yields the correct chrom
// codes and the Morgan genpos as-read.
//
// Dual harness (identical to tests/unit/test_geno_reader.cpp): with
// -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest; otherwise a self-checking main()
// returning non-zero on the first failure — all CTest needs to gate. No CUDA.
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "io/snp_reader.hpp"  // read_snp, SnpTable

namespace {

using steppe::io::SnpTable;
using steppe::io::read_snp;

// A unique temp path per fixture so parallel ctest invocations do not collide.
[[nodiscard]] std::filesystem::path temp_snp(const char* tag) {
    static int counter = 0;
    auto p = std::filesystem::temp_directory_path();
    p /= ("steppe_snp_reader_test_" + std::string(tag) + "_" +
          std::to_string(++counter) + ".snp");
    return p;
}

struct TempFile {
    std::filesystem::path path;
    explicit TempFile(const char* tag) : path(temp_snp(tag)) {}
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

void write_text(const std::filesystem::path& path, const std::string& body) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
    out.close();
}

// Capture whether `fn` throws std::runtime_error AND whether the message contains
// `needle` (used to pin that the diagnostic carries the offending line number).
struct ThrowResult {
    bool threw_runtime_error = false;
    bool message_contains = false;
};

[[nodiscard]] ThrowResult run_expecting_throw(const std::function<void()>& fn,
                                              const std::string& needle) {
    ThrowResult r;
    try {
        fn();
    } catch (const std::runtime_error& e) {
        r.threw_runtime_error = true;
        r.message_contains = std::string(e.what()).find(needle) != std::string::npos;
    } catch (...) {
        // wrong exception type fails the contract (stays false)
    }
    return r;
}

// ---- (1) WELL-FORMED file → correct chrom + Morgan genpos (verdict gate) ------
// Three records: a numeric chromosome, a sex label (X→23), and a full 6-field
// record with explicit alleles. genpos is surfaced unchanged in Morgans.
[[nodiscard]] bool test_wellformed_chrom_and_genpos() {
    TempFile f("wellformed");
    write_text(f.path,
               "rs1 1 0.020130 752566 G A\n"   // numeric chrom, full 6 fields
               "rs2 X 0.500000 999999 C T\n"   // sex label X -> 23
               "rs3 22 1.250000 12345 A G\n");  // numeric chrom 22

    SnpTable t = read_snp(f.path.string(), SIZE_MAX);

    if (t.count != 3) return false;
    if (t.id.size() != 3 || t.chrom.size() != 3 || t.genpos_morgans.size() != 3) {
        return false;
    }
    // Chromosome codes: numeric pass-through and the X->23 EIGENSTRAT mapping.
    if (t.chrom[0] != 1) return false;
    if (t.chrom[1] != 23) return false;
    if (t.chrom[2] != 22) return false;
    // genpos surfaced as-read in Morgans (correctly-rounded decimal->double).
    if (t.genpos_morgans[0] != 0.020130) return false;
    if (t.genpos_morgans[1] != 0.500000) return false;
    if (t.genpos_morgans[2] != 1.250000) return false;
    // Alleles from cols 5,6 (single-char).
    if (t.ref[0] != 'G' || t.alt[0] != 'A') return false;
    if (t.ref[1] != 'C' || t.alt[1] != 'T') return false;
    // ids preserved in file order.
    if (t.id[0] != "rs1" || t.id[1] != "rs2" || t.id[2] != "rs3") return false;
    return true;
}

// ---- (2) MALFORMED record (too few fields) THROWS with the line number --------
// The verdict gate: a record with < 3 whitespace-separated fields is malformed;
// the parser must fail-fast (not silently drop the row) and name the offending
// 1-based line number. The bad record is on line 2.
[[nodiscard]] bool test_too_few_fields_throws_with_line() {
    TempFile f("toofew");
    write_text(f.path,
               "rs1 1 0.10 1000 G A\n"  // line 1: ok
               "rs2 2\n"                // line 2: only 2 fields -> malformed
               "rs3 3 0.30 3000 C T\n");

    const ThrowResult r = run_expecting_throw(
        [&] { (void)read_snp(f.path.string(), SIZE_MAX); }, "line 2");
    return r.threw_runtime_error && r.message_contains;
}

// ---- (3) NON-FINITE / garbage genpos THROWS with the line number --------------
// parse_genpos rejects a non-finite ("nan"/"inf") or unparseable ("0.5x") genetic
// position via std::from_chars (whole-token + overflow) PLUS an explicit
// std::isfinite guard — load-bearing because libstdc++'s from_chars ACCEPTS
// "inf"/"nan", so the finite check (not the parser grammar) is what closes the
// static_cast<int>(NaN/Inf) UB in core::block_of. Each bad token is on line 1.
[[nodiscard]] bool test_nonfinite_genpos_throws_with_line() {
    // "nan"
    {
        TempFile f("nan");
        write_text(f.path, "rs1 1 nan 1000 G A\n");
        const ThrowResult r = run_expecting_throw(
            [&] { (void)read_snp(f.path.string(), SIZE_MAX); }, "line 1");
        if (!(r.threw_runtime_error && r.message_contains)) return false;
    }
    // "inf"
    {
        TempFile f("inf");
        write_text(f.path, "rs1 1 inf 1000 G A\n");
        const ThrowResult r = run_expecting_throw(
            [&] { (void)read_snp(f.path.string(), SIZE_MAX); }, "line 1");
        if (!(r.threw_runtime_error && r.message_contains)) return false;
    }
    // trailing garbage in the genpos token ("0.5x")
    {
        TempFile f("garbage");
        write_text(f.path, "rs1 1 0.5x 1000 G A\n");
        const ThrowResult r = run_expecting_throw(
            [&] { (void)read_snp(f.path.string(), SIZE_MAX); }, "line 1");
        if (!(r.threw_runtime_error && r.message_contains)) return false;
    }
    return true;
}

// ---- (4) INTERIOR blank line THROWS; a TRAILING blank line is tolerated -------
// The .snp row index IS the SNP index: a blank line followed by more records would
// desync the SNP axis from the .geno, so it is a format error (names line 2). A
// single trailing blank line at EOF (a common trailing newline) is tolerated.
[[nodiscard]] bool test_blank_line_policy() {
    // Interior blank -> throw, line 2.
    {
        TempFile f("interiorblank");
        write_text(f.path,
                   "rs1 1 0.10 1000 G A\n"
                   "\n"  // line 2: interior blank
                   "rs2 2 0.20 2000 C T\n");
        const ThrowResult r = run_expecting_throw(
            [&] { (void)read_snp(f.path.string(), SIZE_MAX); }, "line 2");
        if (!(r.threw_runtime_error && r.message_contains)) return false;
    }
    // Trailing blank at EOF -> tolerated (2 records parsed cleanly).
    {
        TempFile f("trailingblank");
        write_text(f.path,
                   "rs1 1 0.10 1000 G A\n"
                   "rs2 2 0.20 2000 C T\n"
                   "\n");  // trailing blank line at EOF
        SnpTable t = read_snp(f.path.string(), SIZE_MAX);
        if (t.count != 2) return false;
    }
    return true;
}

// ---- (5) TOKEN-COUNT column decision: >=3-but-<6 record -> alleles default 'N'
// A 3-field record (no physpos/alleles) is well-formed; ref/alt default to the
// EIGENSTRAT "missing/unknown base" 'N'. This is the legitimate column decision
// the old fall-through tried to express, now driven by the token count.
[[nodiscard]] bool test_three_field_record_defaults_alleles() {
    TempFile f("threefield");
    write_text(f.path,
               "rs1 1 0.10\n"            // 3 fields: no alleles -> 'N'
               "rs2 2 0.20 2000 C T\n");  // full 6 fields
    SnpTable t = read_snp(f.path.string(), SIZE_MAX);
    if (t.count != 2) return false;
    if (t.ref[0] != 'N' || t.alt[0] != 'N') return false;  // defaulted
    if (t.ref[1] != 'C' || t.alt[1] != 'T') return false;  // explicit
    if (t.chrom[0] != 1 || t.chrom[1] != 2) return false;
    if (t.genpos_morgans[0] != 0.10 || t.genpos_morgans[1] != 0.20) return false;
    return true;
}

// ---- (6) max_snps truncation: read only the first N records in file order -----
[[nodiscard]] bool test_max_snps_truncation() {
    TempFile f("trunc");
    write_text(f.path,
               "rs1 1 0.10 1 G A\n"
               "rs2 1 0.20 2 C T\n"
               "rs3 1 0.30 3 A G\n"
               "rs4 1 0.40 4 T C\n");
    SnpTable t = read_snp(f.path.string(), 2);  // first 2 only
    if (t.count != 2) return false;
    if (t.id[0] != "rs1" || t.id[1] != "rs2") return false;
    return true;
}

struct Case { const char* name; bool (*fn)(); };

constexpr Case kCases[] = {
    {"well-formed -> correct chrom + Morgan genpos (verdict gate)",
     test_wellformed_chrom_and_genpos},
    {"malformed record (too few fields) THROWS with line number (verdict gate)",
     test_too_few_fields_throws_with_line},
    {"non-finite/garbage genpos THROWS with line number (B14 from_chars gate)",
     test_nonfinite_genpos_throws_with_line},
    {"interior blank THROWS / trailing blank tolerated", test_blank_line_policy},
    {"3-field record -> alleles default 'N' (token-count column decision)",
     test_three_field_record_defaults_alleles},
    {"max_snps truncation (first N in file order)", test_max_snps_truncation},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(SnpReader, AllCases) {
    for (const auto& c : kCases) {
        EXPECT_TRUE(c.fn()) << "failed: " << c.name;
    }
}
#else
int main() {
    int failures = 0;
    for (const auto& c : kCases) {
        const bool ok = c.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "test_snp_reader: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_snp_reader: all %zu cases PASS\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}
#endif
