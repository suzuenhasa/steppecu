// tests/unit/test_target_build.cpp
//
// Host-only by-construction unit test of io::build_target_sites (Stage-2
// Component B — the native panel harmonizer). Pure C++ TU, NO GPU: it authors a
// tiny EIGENSTRAT .snp panel, an orchestrated rsID->pos38 lift map, and a FASTA
// (+ hand-computed .fai), then asserts the built TargetSites + TargetBuildCounts
// reproduce the Stage-0 oracle's load_panel/lift_panel bookkeeping exactly. It
// pins the correctness traps the critic flagged:
//   #1 rs_count computed over autosomal-rs rows ONLY  (same rsID on chr1 + chrX
//      stays count 1 -> kept, not a dup)
//   #2 ref38 fetched for EVERY lifted site incl. palindromes (kept+flagged)
//   #3 panel_dup_rsids = DISTINCT dup rsIDs; lift_dropped_dup = dropped ROWS
//   #7 palindromes counted PRE-dedup; short (<6-field) rows skipped + not counted
// plus soft-mask lowercase -> upper in the fetched ref38, and shared-index parity
// with read_target_sites. Reports NO science number. Self-checking main().
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "io/faidx_reader.hpp"
#include "io/target_build.hpp"
#include "io/target_sites.hpp"

namespace {

int g_failures = 0;

void eq_ll(const char* what, long long got, long long want) {
    if (got != want) {
        std::printf("  [FAIL] %-22s got %lld want %lld\n", what, got, want);
        ++g_failures;
    }
}

void eq_str(const char* what, const std::string& got, const std::string& want) {
    if (got != want) {
        std::printf("  [FAIL] %-22s got '%s' want '%s'\n", what, got.c_str(), want.c_str());
        ++g_failures;
    }
}

void eq_char(const char* what, char got, char want) {
    if (got != want) {
        std::printf("  [FAIL] %-22s got '%c' want '%c'\n", what, got, want);
        ++g_failures;
    }
}

void want_true(const char* what, bool cond) {
    if (!cond) {
        std::printf("  [FAIL] %-22s expected true\n", what);
        ++g_failures;
    }
}

void write_file(const std::filesystem::path& p, const std::string& s) {
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(s.data(), static_cast<std::streamsize>(s.size()));
}

}  // namespace

int main() {
    using namespace steppe::io;
    std::printf("=== build_target_sites by-construction unit test (no GPU) ===\n");

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_target_build_test_" +
         std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    // ---- panel .snp: id chrom genpos physpos A1 A2 ---------------------------
    // (12 six-field rows + 1 short row; comments give the fate)
    const std::filesystem::path snp = tmp / "panel.snp";
    write_file(snp,
               "rs1\t1\t0.0\t1000\tA\tG\n"        // autosomal rs -> lifted
               "rs2\t1\t0.0\t1100\tA\tT\n"        // palindrome (A/T) -> lifted, KEPT+flagged
               "rsX\t23\t0.0\t1200\tA\tG\n"       // chrom 23 -> not autosomal, dropped
               "snp_novel\t2\t0.0\t1300\tC\tG\n"  // non-rs -> panel_non_rsid
               "rsdup\t2\t0.0\t1400\tC\tT\n"      // dup rsID x3 -> dropped
               "rsdup\t2\t0.0\t1500\tC\tT\n"
               "rsdup\t2\t0.0\t1600\tC\tT\n"
               "rspaldup\t3\t0.0\t1700\tG\tC\n"   // palindrome AND dup x2 -> counted pre-dedup, dropped
               "rspaldup\t3\t0.0\t1800\tG\tC\n"
               "rsnolift\t4\t0.0\t1900\tA\tC\n"   // absent from lift map -> no_lift
               "rscross\t5\t0.0\t2000\tA\tG\n"    // count-1 (its X twin filtered pre-count) -> kept
               "rscross\t23\t0.0\t2100\tA\tG\n"   // chrom 23 -> dropped BEFORE rs_count (fix #1)
               "rsshort\t6\t0.0\t2200\n");        // <6 fields -> skipped, NOT counted in panel_total

    // ---- lift map: rsID<TAB>pos38 (leading header row tolerated) -------------
    const std::filesystem::path lift = tmp / "lift.tsv";
    write_file(lift,
               "rsID\tpos38\n"     // header (fix #5: must be skipped)
               "rs1\t1\n"          // -> contig "1" pos1 = 'a' (soft-mask -> 'A')
               "rs2\t4\n"          // -> contig "1" pos4 = 'T' (palindrome ref38)
               "rscross\t3\n");    // -> contig "5" pos3 = 'G'

    // ---- FASTA (+ hand-computed .fai), single-line contigs -------------------
    // >1\n   bytes 0..2 ;  aCGTACGT\n bytes 3..11 (off 3, len 8, width 9)
    // >5\n   bytes 12..14; TTGCAA\n   bytes 15..21 (off 15, len 6, width 7)
    const std::filesystem::path fa = tmp / "ref.fa";
    write_file(fa, ">1\naCGTACGT\n>5\nTTGCAA\n");
    write_file(tmp / "ref.fa.fai", "1\t8\t3\t8\t9\n5\t6\t15\t6\t7\n");

    FaidxReader fasta(fa.string());
    TargetBuildCounts c;
    const TargetSites ts = build_target_sites(snp.string(), lift.string(), fasta, {}, c);

    // ---- counts (mirror oracle load_panel/lift_panel) -----------------------
    eq_ll("panel_total", c.panel_total, 12);
    eq_ll("panel_autosomal", c.panel_autosomal, 10);
    eq_ll("panel_non_rsid", c.panel_non_rsid, 1);
    eq_ll("panel_palindromic", c.panel_palindromic, 3);  // rs2 + rspaldup x2 (pre-dedup)
    eq_ll("panel_dup_rsids", c.panel_dup_rsids, 2);       // {rsdup, rspaldup} distinct
    eq_ll("lift_dropped_dup", c.lift_dropped_dup, 5);     // 3 + 2 rows
    eq_ll("lift_ok", c.lift_ok, 3);
    eq_ll("lift_no_lift", c.lift_no_lift, 1);
    eq_ll("emitted", c.emitted, 3);

    // ---- emitted sites (panel order: rs1, rs2, rscross) ---------------------
    if (ts.sites.size() != 3) {
        std::printf("  [FAIL] sites.size()=%zu want 3\n", ts.sites.size());
        ++g_failures;
    } else {
        const TargetSite& s0 = ts.sites[0];
        const TargetSite& s1 = ts.sites[1];
        const TargetSite& s2 = ts.sites[2];
        eq_str("site0 rsid", s0.rsid, "rs1");
        eq_ll("site0 pos38", s0.pos38, 1);
        eq_char("site0 ref38 (soft-mask)", s0.ref38, 'A');  // 'a' -> 'A'
        want_true("site0 !palindrome", !s0.palindrome);

        eq_str("site1 rsid", s1.rsid, "rs2");
        want_true("site1 palindrome kept", s1.palindrome);
        eq_char("site1 palindrome ref38", s1.ref38, 'T');   // fix #2: real base on a palindrome
        eq_ll("site1 pos38", s1.pos38, 4);

        eq_str("site2 rsid", s2.rsid, "rscross");           // fix #1: count-1, not dup-dropped
        eq_ll("site2 chrom", s2.chrom, 5);
        eq_char("site2 ref38", s2.ref38, 'G');
    }

    // ---- shared-index parity: build == read_target_sites(emitted table) -----
    const std::filesystem::path emitted = tmp / "emitted.tsv";
    write_target_table(emitted.string(), ts);
    const TargetSites rt = read_target_sites(emitted.string());
    want_true("by_chrom same #chroms", rt.by_chrom.size() == ts.by_chrom.size());
    for (const auto& [chrom, ci] : ts.by_chrom) {
        const auto it = rt.by_chrom.find(chrom);
        if (it == rt.by_chrom.end()) {
            std::printf("  [FAIL] chrom %d missing from read-back index\n", chrom);
            ++g_failures;
            continue;
        }
        if (it->second.pos != ci.pos) {
            std::printf("  [FAIL] chrom %d pos index differs\n", chrom);
            ++g_failures;
        }
        for (const auto& [pos, slot] : ci.slot) {
            const auto sit = it->second.slot.find(pos);
            if (sit == it->second.slot.end() || sit->second != slot) {
                std::printf("  [FAIL] chrom %d slot[%lld] differs\n", chrom, pos);
                ++g_failures;
            }
        }
    }
    // rs2 (palindrome) must be EXCLUDED from the interval-join index on chrom 1,
    // leaving only rs1 -> exactly one chrom-1 position.
    {
        const auto it = ts.by_chrom.find(1);
        want_true("chrom1 index excludes palindrome",
                  it != ts.by_chrom.end() && it->second.pos.size() == 1 && it->second.pos[0] == 1);
    }

    std::filesystem::remove_all(tmp, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (native harmonizer reproduces oracle join/dedup/ref38)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
