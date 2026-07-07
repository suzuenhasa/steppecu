// tests/cli/test_cli_ingest.cpp
//
// Host-only by-construction gate for the native gVCF-block-aware VCF reader
// (`steppe ingest`). PLAIN C++ TU (NO CUDA header, NO GPU): it authors a tiny
// PLAIN .vcf + a GRCh38 target table whose every reader branch has a KNOWN
// answer, runs `steppe ingest --report` (the report path needs no device), and
// asserts the emitted per-site report equals the by-construction truth. This
// exercises the reader's OWN machinery — the gVCF interval join, the H4 floor,
// H1 hom-ref->A1 reconciliation, M4 multiallelic/half-call handling, palindrome
// drop, strand-flip, and the H3 rsID-mismap drop — and reports NO science
// (synthetic is fine here, the readv2-concord precedent). argv[1] = steppe binary.
//
// Matches the test_cli_readv2_concord.cpp harness idiom.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#  include <sys/wait.h>
#endif

namespace {

int g_failures = 0;

struct Cell {
    std::string call, dosage, source, drop_reason, flip, pos37, pos38;
};

void expect(const std::map<std::string, Cell>& rep, const std::string& rs, const std::string& call,
            const std::string& dosage, const std::string& source, const std::string& drop_reason) {
    const auto it = rep.find(rs);
    if (it == rep.end()) {
        std::printf("  [FAIL] %-16s missing from report\n", rs.c_str());
        ++g_failures;
        return;
    }
    const Cell& c = it->second;
    const bool ok = (c.call == call) && (c.dosage == dosage) && (c.source == source) &&
                    (c.drop_reason == drop_reason);
    if (!ok) {
        std::printf("  [FAIL] %-16s got{call=%s,dos=%s,src=%s,dr=%s} want{call=%s,dos=%s,src=%s,dr=%s}\n",
                    rs.c_str(), c.call.c_str(), c.dosage.c_str(), c.source.c_str(),
                    c.drop_reason.c_str(), call.c_str(), dosage.c_str(), source.c_str(),
                    drop_reason.c_str());
        ++g_failures;
    }
}

int run_steppe(const std::string& bin, const std::vector<std::string>& args,
               const std::filesystem::path& logf) {
    std::string cmd = "\"" + bin + "\"";
    for (const std::string& a : args) cmd += " \"" + a + "\"";
    cmd += " > \"" + logf.string() + "\" 2>&1";
    const int sys = std::system(cmd.c_str());
#ifdef WIFEXITED
    return (sys == -1) ? -1 : (WIFEXITED(sys) ? WEXITSTATUS(sys) : -1);
#else
    return sys;
#endif
}

std::map<std::string, Cell> parse_report(const std::filesystem::path& p) {
    std::map<std::string, Cell> out;
    std::ifstream f(p);
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
        if (header) { header = false; continue; }
        if (line.empty()) continue;
        std::vector<std::string> c;
        std::string cur;
        std::istringstream ss(line);
        while (std::getline(ss, cur, '\t')) c.push_back(cur);
        // rsID chrom pos37 pos38 A1 A2 call dosage source flip drop_reason
        if (c.size() < 10) continue;
        Cell cell;
        cell.pos37 = c[2];
        cell.pos38 = c[3];
        cell.call = c[6];
        cell.dosage = c[7];
        cell.source = c[8];
        cell.flip = c[9];
        cell.drop_reason = (c.size() >= 11) ? c[10] : "";
        out[c[0]] = cell;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <steppe-binary>\n", argv[0]);
        return 2;
    }
    const std::string bin = argv[1];
    std::printf("=== steppe ingest by-construction reader gate (no GPU) ===\n");
    if (!std::filesystem::exists(bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", bin.c_str());
        return 1;
    }

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_ingest_test_" + std::to_string(static_cast<long long>(
            std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    // ---- author the target table (rsID chrom pos37 pos38 A1 A2 ref38) --------
    const std::filesystem::path tpath = tmp / "targets.tsv";
    {
        std::ofstream t(tpath, std::ios::trunc);
        t << "rsID\tchrom\tpos37\tpos38\tA1\tA2\tref38\n";
        t << "rs_hrA\t1\t1\t1200\tA\tG\tA\n";       // passing block, REF==A1 -> homref 2
        t << "rs_hrG\t1\t2\t1300\tG\tA\tA\n";       // passing block, REF==A2 -> homref 0
        t << "rs_bound\t1\t3\t2000\tC\tT\tC\n";     // == block END (L1) -> homref 2
        t << "rs_flip\t1\t4\t1400\tT\tC\tA\n";      // comp(A)=T==A1 -> homref 2 flip
        t << "rs_low\t1\t5\t3200\tA\tG\tA\n";       // below-floor block -> missing below_floor
        t << "rs_none\t1\t6\t9000\tA\tG\tA\n";      // no coverage -> missing no_coverage
        t << "rs_het\t1\t7\t5000\tA\tG\t.\n";       // variant 0/1 -> het 1
        t << "rs_homalt\t1\t8\t5100\tA\tG\t.\n";    // variant 1/1 -> homalt 0
        t << "rs_multi\t1\t9\t5200\tA\tC\t.\n";     // multiallelic 0/1 (A/C) -> het 1
        t << "rs_nonpanel\t1\t10\t5300\tA\tG\t.\n"; // indel ALT -> non_panel_allele
        t << "rs_half\t1\t11\t5400\tA\tG\t.\n";     // 0/. -> half_or_missing_gt
        t << "rs_pal\t1\t12\t5500\tA\tT\t.\n";      // palindrome -> dropped
        t << "rs_mismap\t1\t13\t5600\tA\tG\t.\n";   // record ID != panel rsID -> dropped
        t << "rs_belowdp\t1\t14\t5700\tA\tG\t.\n";  // variant DP 5 -> below_floor
        t << "rs_notpass\t1\t15\t5800\tA\tG\t.\n";  // FILTER != PASS -> not_pass
        // --- Stage-2 fold-in fixes (a) ref-block N / '.' ref-base -----------
        t << "rs_Nref\t1\t16\t1500\tA\tG\tN\n";     // passing block, ref38 'N' -> dropped ref_change
        t << "rs_dotref\t1\t17\t1600\tA\tG\t.\n";   // passing block, ref38 '.' -> missing no_refbase
        // --- fix (b) GQ==0 tie-break (asserted only in the --min-gq 0 run) ---
        t << "rs_gq0\t1\t18\t6000\tA\tG\t.\n";      // dup-POS variant: no-GQ vs GQ==0 tie-break
    }

    // ---- author the plain .vcf ----------------------------------------------
    const std::filesystem::path vpath = tmp / "toy.vcf";
    {
        std::ofstream v(vpath, std::ios::trunc);
        v << "##fileformat=VCFv4.2\n";
        v << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT1\n";
        // passing ref block [1000,2000] MinDP=15
        v << "1\t1000\t.\tA\t.\t.\tPASS\tEND=2000;MinDP=15\tGT:DP\t0/0:20\n";
        // failing ref block [3000,3500] MinDP=6 (< floor)
        v << "1\t3000\t.\tA\t.\t.\tPASS\tEND=3500;MinDP=6\tGT:DP\t0/0:6\n";
        // explicit variants
        v << "1\t5000\trs_het\tA\tG\t.\tPASS\t.\tGT:DP:GQ\t0/1:30:99\n";
        v << "1\t5100\trs_homalt\tA\tG\t.\tPASS\t.\tGT:DP:GQ\t1/1:30:99\n";
        v << "1\t5200\trs_multi\tA\tC,G\t.\tPASS\t.\tGT:DP:GQ\t0/1:30:99\n";
        v << "1\t5300\trs_nonpanel\tA\tAT\t.\tPASS\t.\tGT:DP:GQ\t0/1:30:99\n";
        v << "1\t5400\trs_half\tA\tG\t.\tPASS\t.\tGT:DP:GQ\t0/.:30:99\n";
        v << "1\t5600\trs_OTHER\tA\tG\t.\tPASS\t.\tGT:DP:GQ\t0/1:30:99\n";  // rsID mismap
        v << "1\t5700\trs_belowdp\tA\tG\t.\tPASS\t.\tGT:DP:GQ\t0/1:5:99\n"; // DP < 8
        v << "1\t5800\trs_notpass\tA\tG\t.\tLowGQ\t.\tGT:DP:GQ\t0/1:30:15\n";
        // fix (b): two duplicate-POS variant records at 6000 — record #1 carries
        // NO GQ field, record #2 has GQ==0. With the GQ==0-as-falsy tie-break the
        // no-GQ record (first-seen) is kept; without the fix the GQ==0 record wins.
        // Distinguishable only when GQ==0 clears the floor -> the --min-gq 0 run.
        v << "1\t6000\trs_gq0\tA\tG\t.\tPASS\t.\tGT:DP\t0/1:30\n";        // no GQ, first
        v << "1\t6000\trs_gq0\tA\tG\t.\tPASS\t.\tGT:DP:GQ\t1/1:30:0\n";  // GQ==0, second
    }

    const std::filesystem::path rpath = tmp / "report.tsv";
    const std::filesystem::path logf = tmp / "ingest.log";
    const int rc = run_steppe(bin, {"ingest", "--vcf", vpath.string(), "--targets", tpath.string(),
                                    "--report", rpath.string()}, logf);
    if (rc != 0) {
        std::printf("  [FAIL] steppe ingest exit=%d (log follows)\n", rc);
        std::ifstream lf(logf);
        std::stringstream ss; ss << lf.rdbuf();
        std::printf("%s\n", ss.str().c_str());
        ++g_failures;
    }

    const auto rep = parse_report(rpath);

    //          rsID          call       dosage source     drop_reason
    expect(rep, "rs_hrA",     "homref",  "2",   "refblock", "");
    expect(rep, "rs_hrG",     "homref",  "0",   "refblock", "");
    expect(rep, "rs_bound",   "homref",  "2",   "refblock", "");
    expect(rep, "rs_flip",    "homref",  "2",   "refblock", "");
    expect(rep, "rs_low",     "missing", "NA",  "refblock", "below_floor");
    expect(rep, "rs_none",    "missing", "NA",  "none",     "no_coverage");
    expect(rep, "rs_het",     "het",     "1",   "variant",  "");
    expect(rep, "rs_homalt",  "homalt",  "0",   "variant",  "");
    expect(rep, "rs_multi",   "het",     "1",   "variant",  "");
    expect(rep, "rs_nonpanel","missing", "NA",  "variant",  "non_panel_allele");
    expect(rep, "rs_half",    "missing", "NA",  "variant",  "half_or_missing_gt");
    expect(rep, "rs_pal",     "dropped", "NA",  "none",     "palindrome");
    expect(rep, "rs_mismap",  "dropped", "NA",  "none",     "rsid_mismap");
    expect(rep, "rs_belowdp", "missing", "NA",  "variant",  "below_floor");
    expect(rep, "rs_notpass", "missing", "NA",  "variant",  "not_pass");
    // fix (a): a passing ref block whose GRCh38 REF base is 'N' drops as
    // ref_change (N flows into reconcile, matches neither allele); a '.' base is
    // genuinely unavailable -> missing/no_refbase (source stays "none").
    expect(rep, "rs_Nref",    "dropped", "NA",  "refblock", "ref_change");
    expect(rep, "rs_dotref",  "missing", "NA",  "none",     "no_refbase");

    // ---- fix (b): the GQ==0 tie-break, exercised with --min-gq 0 --min-dp 1 --
    // A floor of 0 now means "field not required" (the GT-only hardcall relaxation),
    // so at --min-gq 0 --min-dp 1 the GQ gate is skipped and DP=30 clears --min-dp 1.
    // The better() tie-break still DECIDES which duplicate-POS record is kept, and
    // this assertion still distinguishes fixed-vs-not: WITH the fix the no-GQ record
    // (first-seen, GT 0/1) is kept -> het/1; WITHOUT it the GQ==0 record (GT 1/1)
    // wins -> homalt/0. (Under the pre-relaxation code the no-GQ record instead
    // resolved to missing/below_floor via the !has_gq branch — that path is gone.)
    {
        const std::filesystem::path rpath0 = tmp / "report_gq0.tsv";
        const std::filesystem::path logf0 = tmp / "ingest_gq0.log";
        const int rc0 = run_steppe(
            bin, {"ingest", "--vcf", vpath.string(), "--targets", tpath.string(), "--report",
                  rpath0.string(), "--min-gq", "0", "--min-dp", "1"},
            logf0);
        if (rc0 != 0) {
            std::printf("  [FAIL] steppe ingest (min-gq 0) exit=%d\n", rc0);
            ++g_failures;
        }
        const auto rep0 = parse_report(rpath0);
        expect(rep0, "rs_gq0", "het", "1", "variant", "");
    }

    // The strand-flip site must carry flip=1.
    {
        const auto it = rep.find("rs_flip");
        if (it == rep.end() || it->second.flip != "1") {
            std::printf("  [FAIL] rs_flip flip flag != 1\n");
            ++g_failures;
        }
    }

    // ---- GRCh37 same-build DIRECT path (auto-detect, no --lift) --------------
    // A GRCh37 VCF against the fixed-GRCh37 panel joins directly: pos38 := pos37
    // (identity), ref38 from a GRCh37 fasta at the panel physpos, no lift file.
    // The fixture is a phased GT-only hardcall VCF (FORMAT=GT; no DP/GQ) genotyped
    // at --min-dp 0 --min-gq 0 — this also exercises the floor relaxation on BOTH
    // the variant path and the ref-confidence block (no depth field, --min-dp 0).
    {
        const std::filesystem::path g37dir = tmp / "g37";
        std::filesystem::create_directories(g37dir, ec);
        // panel .snp (GRCh37): id chrom genpos physpos A1 A2
        const std::filesystem::path snp = g37dir / "panel.snp";
        {
            std::ofstream s(snp, std::ios::trunc);
            s << "rsA\t1\t0.0\t5\tA\tG\n";    // variant target at physpos 5
            s << "rsB\t1\t0.0\t10\tC\tT\n";   // ref-block homref target at physpos 10
        }
        // GRCh37 fasta: contig "1" = NNNNANNNNC (pos5='A', pos10='C') + .fai.
        const std::filesystem::path fa = g37dir / "ref.fa";
        {
            std::ofstream f(fa, std::ios::binary | std::ios::trunc);
            f << ">1\nNNNNANNNNC\n";
        }
        {
            std::ofstream fi(g37dir / "ref.fa.fai", std::ios::binary | std::ios::trunc);
            fi << "1\t10\t3\t10\t11\n";
        }
        // GRCh37 VCF (chr1 contig length 249250621), GT-only hardcalls.
        const std::filesystem::path v37 = g37dir / "g37.vcf";
        {
            std::ofstream v(v37, std::ios::trunc);
            v << "##fileformat=VCFv4.2\n";
            v << "##contig=<ID=1,length=249250621>\n";
            v << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n";
            v << "1\t5\trsA\tA\tG\t.\tPASS\t.\tGT\t0/1\n";           // -> het 1
            v << "1\t8\t.\tA\t.\t.\tPASS\tEND=12\tGT\t0/0\n";        // passing block over pos10
        }
        const std::filesystem::path r37 = g37dir / "report.tsv";
        const std::filesystem::path log37 = g37dir / "ingest.log";
        const int rc37 = run_steppe(bin, {"ingest", "--vcf", v37.string(), "--panel", snp.string(),
                                          "--fasta", fa.string(), "--report", r37.string(),
                                          "--min-dp", "0", "--min-gq", "0"},
                                    log37);
        if (rc37 != 0) {
            std::printf("  [FAIL] GRCh37 direct-path ingest exit=%d (log follows)\n", rc37);
            std::ifstream lf(log37);
            std::stringstream ss; ss << lf.rdbuf();
            std::printf("%s\n", ss.str().c_str());
            ++g_failures;
        }
        // stderr must report the detected GRCh37 assembly.
        {
            std::ifstream lf(log37);
            std::stringstream ss; ss << lf.rdbuf();
            if (ss.str().find("detected assembly GRCh37") == std::string::npos) {
                std::printf("  [FAIL] GRCh37 direct-path: 'detected assembly GRCh37' not in stderr\n");
                ++g_failures;
            }
        }
        const auto rep37 = parse_report(r37);
        expect(rep37, "rsA", "het", "1", "variant", "");
        expect(rep37, "rsB", "homref", "2", "refblock", "");
        // identity join: pos38 == pos37 on every row.
        for (const char* rs : {"rsA", "rsB"}) {
            const auto it = rep37.find(rs);
            if (it == rep37.end() || it->second.pos37 != it->second.pos38) {
                std::printf("  [FAIL] GRCh37 direct-path %s: pos37 != pos38 (identity join)\n", rs);
                ++g_failures;
            }
        }

        // ---- negative: a GRCh38 VCF against the GRCh37 panel needs --lift -----
        const std::filesystem::path v38 = g37dir / "g38.vcf";
        {
            std::ofstream v(v38, std::ios::trunc);
            v << "##fileformat=VCFv4.2\n";
            v << "##contig=<ID=1,length=248956422>\n";  // GRCh38 chr1 length
            v << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n";
            v << "1\t5\trsA\tA\tG\t.\tPASS\t.\tGT\t0/1\n";
        }
        const std::filesystem::path log38 = g37dir / "ingest38.log";
        const int rc38 = run_steppe(bin, {"ingest", "--vcf", v38.string(), "--panel", snp.string(),
                                          "--fasta", fa.string(), "--report",
                                          (g37dir / "r38.tsv").string()},
                                    log38);
        if (rc38 != 2) {  // cfg::kExitInvalidConfig == 2
            std::printf("  [FAIL] GRCh38-no-lift should exit InvalidConfig(2), got %d\n", rc38);
            ++g_failures;
        }
    }

    std::filesystem::remove_all(tmp, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (native VCF reader reproduces all by-construction branches, "
                    "incl. the Stage-2 N-ref / '.'-ref / GQ==0 fixes)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
