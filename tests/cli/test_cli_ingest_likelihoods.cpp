// tests/cli/test_cli_ingest_likelihoods.cpp
//
// Host-only by-construction gate for the GL/PL/GP likelihood path of `steppe
// ingest --likelihoods`. PLAIN C++ TU (NO CUDA, NO GPU): the --emit-pl-raw dump is
// pure host (only --emit-likelihoods needs a device), so this authors a tiny plain
// .vcf carrying PL + a target table and asserts (a) the raw triplet dump reproduces
// the VCF-native tokens verbatim, self-keyed by rsID/chrom/pos38/sample, and (b)
// the flag validation (--emit-likelihoods without --likelihoods; --likelihoods
// without an output). Mirrors the test_cli_ingest harness. argv[1] = steppe binary.
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

struct Raw {
    std::string chrom, pos38, sample, v0, v1, v2;
};

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

std::map<std::string, Raw> parse_raw(const std::filesystem::path& p) {
    std::map<std::string, Raw> out;
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
        if (c.size() < 7) continue;  // rsID chrom pos38 sample v0 v1 v2
        Raw r;
        r.chrom = c[1]; r.pos38 = c[2]; r.sample = c[3];
        r.v0 = c[4]; r.v1 = c[5]; r.v2 = c[6];
        out[c[0]] = r;  // keyed by rsID
    }
    return out;
}

void expect_raw(const std::map<std::string, Raw>& m, const std::string& rs, const std::string& v0,
                const std::string& v1, const std::string& v2) {
    const auto it = m.find(rs);
    if (it == m.end()) {
        std::printf("  [FAIL] %-12s missing from raw dump\n", rs.c_str());
        ++g_failures;
        return;
    }
    const Raw& r = it->second;
    if (r.v0 != v0 || r.v1 != v1 || r.v2 != v2) {
        std::printf("  [FAIL] %-12s got{%s,%s,%s} want{%s,%s,%s}\n", rs.c_str(), r.v0.c_str(),
                    r.v1.c_str(), r.v2.c_str(), v0.c_str(), v1.c_str(), v2.c_str());
        ++g_failures;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <steppe-binary>\n", argv[0]);
        return 2;
    }
    const std::string bin = argv[1];
    std::printf("=== steppe ingest --likelihoods raw-dump + validation gate (no GPU) ===\n");
    if (!std::filesystem::exists(bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", bin.c_str());
        return 1;
    }

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_gl_cli_" + std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const std::filesystem::path tpath = tmp / "targets.tsv";
    {
        std::ofstream t(tpath, std::ios::trunc);
        t << "rsID\tchrom\tpos37\tpos38\tA1\tA2\tref38\n";
        t << "rs_a\t1\t100\t100\tA\tG\tA\n";       // variant with PL
        t << "rs_b\t1\t200\t200\tG\tA\tA\n";       // variant with PL (swap-side; raw is pre-swap)
        t << "rs_multi\t1\t300\t300\tA\tG\tA\n";   // multiallelic -> not in raw dump
        t << "rs_none\t1\t400\t400\tA\tG\tA\n";    // no record -> not in raw dump
    }
    const std::filesystem::path vpath = tmp / "gl.vcf";
    {
        std::ofstream v(vpath, std::ios::trunc);
        v << "##fileformat=VCFv4.2\n";
        v << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMP1\n";
        v << "1\t100\trs_a\tA\tG\t.\tPASS\t.\tGT:PL:DP\t0/1:40,0,136:14\n";
        v << "1\t200\trs_b\tA\tG\t.\tLowQUAL\t.\tGT:PL:DP\t1/1:214,11,0:18\n";  // non-PASS still emits
        v << "1\t300\trs_multi\tA\tG,C\t.\tPASS\t.\tGT:PL\t0/1:20,0,20,30,30,40\n";
    }

    // ---- --emit-pl-raw (host-only) --------------------------------------------
    const std::filesystem::path rawp = tmp / "raw.tsv";
    const std::filesystem::path logf = tmp / "gl.log";
    const int rc = run_steppe(bin,
                              {"ingest", "--vcf", vpath.string(), "--targets", tpath.string(),
                               "--likelihoods", "--gl-field", "PL", "--emit-pl-raw", rawp.string()},
                              logf);
    if (rc != 0) {
        std::printf("  [FAIL] steppe ingest --likelihoods exit=%d (log follows)\n", rc);
        std::ifstream lf(logf);
        std::stringstream ss; ss << lf.rdbuf();
        std::printf("%s\n", ss.str().c_str());
        ++g_failures;
    }
    const auto raw = parse_raw(rawp);
    // Raw dump is VCF-native order, verbatim tokens, pre-polarity-swap.
    expect_raw(raw, "rs_a", "40", "0", "136");
    expect_raw(raw, "rs_b", "214", "11", "0");  // non-PASS record STILL emitted (soft info)
    if (raw.find("rs_multi") != raw.end()) {
        std::printf("  [FAIL] rs_multi (multiallelic) must NOT appear in the raw dump\n");
        ++g_failures;
    }
    if (raw.find("rs_none") != raw.end()) {
        std::printf("  [FAIL] rs_none (no record) must NOT appear in the raw dump\n");
        ++g_failures;
    }

    // ---- validation: --emit-likelihoods without --likelihoods -> InvalidConfig(2)
    {
        const std::filesystem::path l2 = tmp / "v2.log";
        const int rc2 = run_steppe(bin,
                                   {"ingest", "--vcf", vpath.string(), "--targets", tpath.string(),
                                    "--emit-likelihoods", (tmp / "x.stpgl").string()},
                                   l2);
        if (rc2 != 2) {
            std::printf("  [FAIL] --emit-likelihoods without --likelihoods should exit 2, got %d\n",
                        rc2);
            ++g_failures;
        }
    }
    // ---- validation: --likelihoods with no output -> InvalidConfig(2) ---------
    {
        const std::filesystem::path l3 = tmp / "v3.log";
        const int rc3 = run_steppe(bin,
                                   {"ingest", "--vcf", vpath.string(), "--targets", tpath.string(),
                                    "--likelihoods"},
                                   l3);
        if (rc3 != 2) {
            std::printf("  [FAIL] --likelihoods with no output should exit 2, got %d\n", rc3);
            ++g_failures;
        }
    }
    // ---- validation: bad --gl-field -> InvalidConfig(2) -----------------------
    {
        const std::filesystem::path l4 = tmp / "v4.log";
        const int rc4 = run_steppe(bin,
                                   {"ingest", "--vcf", vpath.string(), "--targets", tpath.string(),
                                    "--likelihoods", "--gl-field", "XY", "--emit-pl-raw",
                                    (tmp / "y.tsv").string()},
                                   l4);
        if (rc4 != 2) {
            std::printf("  [FAIL] bad --gl-field should exit 2, got %d\n", rc4);
            ++g_failures;
        }
    }

    std::filesystem::remove_all(tmp, ec);
    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (raw GL dump verbatim + flag validation correct)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
