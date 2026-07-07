// tests/unit/test_faidx_reader.cpp
//
// Host-only by-construction unit test of io::FaidxReader (Stage-2 Component A).
// Pure C++ TU, NO GPU: it writes tiny hand-computed FASTA + .fai fixtures to a
// temp path and checks the faidx byte-offset arithmetic against answers computed
// by hand — multi-line contigs, a line boundary, soft-mask lowercase -> upper, a
// literal 'N' returned as 'N', the last base of a contig, '\r\n' line endings,
// the 'chr'-prefix tolerance (both directions), and the four hard-error paths
// (missing .fai, unknown contig, pos>length, pos<1). Reports NO science number
// (synthetic layout fixture, per the no-synthetic rule which is for precision
// claims only). Self-checking main(); returns non-zero on the first failure.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "io/faidx_reader.hpp"

namespace {

int g_failures = 0;

void check_char(const char* what, char got, char want) {
    if (got != want) {
        std::printf("  [FAIL] %-28s got '%c' want '%c'\n", what, got, want);
        ++g_failures;
    }
}

void check_true(const char* what, bool cond) {
    if (!cond) {
        std::printf("  [FAIL] %-28s expected true\n", what);
        ++g_failures;
    }
}

// Assert that `fn` throws std::runtime_error.
template <class Fn>
void check_throws(const char* what, Fn fn) {
    try {
        fn();
        std::printf("  [FAIL] %-28s expected throw, none\n", what);
        ++g_failures;
    } catch (const std::runtime_error&) {
        // expected
    } catch (...) {
        std::printf("  [FAIL] %-28s threw wrong type\n", what);
        ++g_failures;
    }
}

void write_file(const std::filesystem::path& p, const std::string& bytes) {
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

int main() {
    using steppe::io::FaidxReader;
    std::printf("=== FaidxReader by-construction unit test (no GPU) ===\n");

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_faidx_test_" +
         std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    // ---- fixture 1: '\n' FASTA, LINEBASES=4 ---------------------------------
    // >1                header  bytes 0..2   ('>','1','\n')
    // ACGT\n            base off 3   (pos1..4 = A C G T)
    // acgt\n                         (pos5..8 = a c g t, soft-mask)
    // NCGT\n                         (pos9..12 = N C G T)
    // >chr2             header
    // AAAA\n            (pos1..4)
    // CC\n              (pos5..6)
    const std::filesystem::path fa = tmp / "tiny.fa";
    write_file(fa, ">1\nACGT\nacgt\nNCGT\n>chr2\nAAAA\nCC\n");
    write_file(tmp / "tiny.fa.fai",
               "1\t12\t3\t4\t5\n"
               "chr2\t6\t24\t4\t5\n");

    {
        FaidxReader r(fa.string());
        // contig "1"
        check_char("pos1 (first base)", r.base_at("1", 1), 'A');
        check_char("pos3 (mid line 1)", r.base_at("1", 3), 'G');
        check_char("pos5 (soft-mask, line 2)", r.base_at("1", 5), 'A');  // 'a' -> 'A'
        check_char("pos8 (last of line 2)", r.base_at("1", 8), 'T');     // 't' -> 'T'
        check_char("pos9 (literal N, line 3)", r.base_at("1", 9), 'N');
        check_char("pos12 (last base)", r.base_at("1", 12), 'T');
        // contig prefix tolerance
        check_char("chr-prefix add (\"2\")", r.base_at("2", 1), 'A');     // "2" -> "chr2"
        check_char("chr-prefix direct", r.base_at("chr2", 5), 'C');
        check_char("chr2 last base", r.base_at("2", 6), 'C');
        check_true("has_contig(\"1\")", r.has_contig("1"));
        check_true("has_contig(\"2\")->chr2", r.has_contig("2"));
        check_true("has_contig(\"chr1\")->1", r.has_contig("chr1"));  // strip-chr direction
        check_char("chr-prefix strip (\"chr1\")", r.base_at("chr1", 1), 'A');
        check_true("!has_contig(\"9\")", !r.has_contig("9"));

        // hard errors
        check_throws("unknown contig", [&] { (void)r.base_at("9", 1); });
        check_throws("pos > length", [&] { (void)r.base_at("1", 13); });
        check_throws("pos < 1", [&] { (void)r.base_at("1", 0); });
    }

    // ---- fixture 2: '\r\n' FASTA, LINEBASES=2, LINEWIDTH=4 -------------------
    // >c\r\n            header bytes 0..3
    // AC\r\n            base off 4  (pos1..2 = A C)
    // GT\r\n                        (pos3..4 = G T)
    const std::filesystem::path fac = tmp / "crlf.fa";
    write_file(fac, ">c\r\nAC\r\nGT\r\n");
    write_file(tmp / "crlf.fa.fai", "c\t4\t4\t2\t4\n");
    {
        FaidxReader r(fac.string());
        check_char("crlf pos1", r.base_at("c", 1), 'A');
        check_char("crlf pos2 (last of line)", r.base_at("c", 2), 'C');
        check_char("crlf pos3 (line 2 first)", r.base_at("c", 3), 'G');
        check_char("crlf pos4 (last base)", r.base_at("c", 4), 'T');
    }

    // ---- fixture 3: missing .fai throws at construction ---------------------
    const std::filesystem::path bare = tmp / "noindex.fa";
    write_file(bare, ">x\nACGT\n");
    check_throws("missing .fai at ctor", [&] { FaidxReader r(bare.string()); });

    std::filesystem::remove_all(tmp, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (FaidxReader offset math + tolerance + errors)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
