// tests/cli/test_cli_ibd.cpp
//
// Host-only wiring gate for `steppe ibd` (ancIBD). No GPU: the test env exposes no
// CUDA device, so the command falls back to the CpuBackend reference oracle for the
// 5-state forward-backward — this exercises the FULL path (target build, phased GT +
// GP read via the ancIBD-native reader increment, map/AF join, per-chromosome FB,
// segment calling, summary emit) end-to-end on the CPU. It authors a tiny phased
// imputed VCF + a target table + a map + an AF file and asserts (a) a clean run with
// the reference output columns, and (b) the AF-mode flag validation. Segment CONTENT
// (the concordance numbers) is the real-data box gate, not this plumbing test.
// argv[1] = steppe binary.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#  include <sys/wait.h>
#endif

namespace {
int g_fail = 0;

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

std::string first_line(const std::filesystem::path& p) {
    std::ifstream f(p);
    std::string line;
    std::getline(f, line);
    return line;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <steppe-binary>\n", argv[0]);
        return 2;
    }
    const std::string bin = argv[1];
    std::printf("=== steppe ibd wiring gate (CPU-backend FB, no GPU) ===\n");
    if (!std::filesystem::exists(bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", bin.c_str());
        return 1;
    }

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_ibd_cli_" + std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    // Targets: 6 biallelic 1240K-style sites on chr1 (GRCh37 identity build).
    const int NS = 6;
    const std::filesystem::path tpath = tmp / "targets.tsv";
    {
        std::ofstream t(tpath, std::ios::trunc);
        t << "rsID\tchrom\tpos37\tpos38\tA1\tA2\tref38\n";
        for (int i = 0; i < NS; ++i) {
            const long long pos = 100 + i * 100;
            t << "rs" << i << "\t1\t" << pos << "\t" << pos << "\tA\tG\tA\n";
        }
    }
    // Map (cM) + AF (derived freq), keyed by rsID.
    const std::filesystem::path mpath = tmp / "map.tsv";
    const std::filesystem::path apath = tmp / "af.tsv";
    {
        std::ofstream m(mpath, std::ios::trunc), a(apath, std::ios::trunc);
        for (int i = 0; i < NS; ++i) {
            m << "rs" << i << "\t" << (i * 2.0) << "\n";   // cM
            a << "rs" << i << "\t" << 0.3 << "\n";         // derived AF
        }
    }
    // Imputed VCF: 3 samples, phased GT (|) + GP, header ##reference GRCh37 (identity).
    const std::filesystem::path vpath = tmp / "imp.vcf";
    {
        std::ofstream v(vpath, std::ios::trunc);
        v << "##fileformat=VCFv4.2\n";
        v << "##reference=GRCh37\n";
        v << "##contig=<ID=1,length=249250621>\n";
        v << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n";
        for (int i = 0; i < NS; ++i) {
            const long long pos = 100 + i * 100;
            v << "1\t" << pos << "\trs" << i << "\tA\tG\t.\tPASS\t.\tGT:GP";
            // S1 & S2 share a haplotype (0|1, high GP); S3 differs.
            v << "\t0|1:0.05,0.9,0.05";
            v << "\t0|1:0.05,0.9,0.05";
            v << "\t1|0:0.05,0.9,0.05";
            v << "\n";
        }
    }

    // ---- clean run -----------------------------------------------------------
    const std::filesystem::path seg = tmp / "seg.tsv";
    const std::filesystem::path sum = tmp / "sum.tsv";
    const std::filesystem::path logf = tmp / "run.log";
    const int rc = run_steppe(
        bin, {"ibd", "--gp-vcf", vpath.string(), "--targets", tpath.string(), "--map",
              mpath.string(), "--af", apath.string(), "--out", seg.string(), "--summary",
              sum.string()},
        logf);
    if (rc != 0) {
        std::printf("  [FAIL] steppe ibd exit=%d (log follows)\n", rc);
        std::ifstream lf(logf);
        std::stringstream ss; ss << lf.rdbuf();
        std::printf("%s\n", ss.str().c_str());
        ++g_fail;
    } else {
        const std::string sh = first_line(seg);
        if (sh.rfind("iid1", 0) != 0 || sh.find("lengthCM") == std::string::npos) {
            std::printf("  [FAIL] segment header wrong: '%s'\n", sh.c_str());
            ++g_fail;
        }
        const std::string uh = first_line(sum);
        if (uh.rfind("iid1", 0) != 0 || uh.find("sum_IBD>8") == std::string::npos) {
            std::printf("  [FAIL] summary header wrong: '%s'\n", uh.c_str());
            ++g_fail;
        }
    }

    // ---- validation: bad --af-mode -> InvalidConfig(2) -----------------------
    {
        const std::filesystem::path l2 = tmp / "v2.log";
        const int rc2 = run_steppe(
            bin, {"ibd", "--gp-vcf", vpath.string(), "--targets", tpath.string(), "--map",
                  mpath.string(), "--af-mode", "bogus", "--out", (tmp / "x.tsv").string()},
            l2);
        if (rc2 != 2) {
            std::printf("  [FAIL] bad --af-mode should exit 2, got %d\n", rc2);
            ++g_fail;
        }
    }
    // ---- validation: --af-mode panel without --af -> InvalidConfig(2) --------
    {
        const std::filesystem::path l3 = tmp / "v3.log";
        const int rc3 = run_steppe(
            bin, {"ibd", "--gp-vcf", vpath.string(), "--targets", tpath.string(), "--map",
                  mpath.string(), "--af-mode", "panel", "--out", (tmp / "y.tsv").string()},
            l3);
        if (rc3 != 2) {
            std::printf("  [FAIL] --af-mode panel without --af should exit 2, got %d\n", rc3);
            ++g_fail;
        }
    }

    std::filesystem::remove_all(tmp, ec);
    if (g_fail == 0) {
        std::printf("\nRESULT: PASS (steppe ibd wiring + CPU FB + flag validation correct)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_fail);
    return 1;
}
