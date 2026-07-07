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
    std::string call, dosage, source, drop_reason, flip;
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

    // The strand-flip site must carry flip=1.
    {
        const auto it = rep.find("rs_flip");
        if (it == rep.end() || it->second.flip != "1") {
            std::printf("  [FAIL] rs_flip flip flag != 1\n");
            ++g_failures;
        }
    }

    std::filesystem::remove_all(tmp, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (native VCF reader reproduces all 15 by-construction branches)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
