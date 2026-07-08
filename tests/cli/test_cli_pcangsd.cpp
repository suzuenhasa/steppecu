// tests/cli/test_cli_pcangsd.cpp
//
// Host-only wiring gate for `steppe pcangsd` (PCAngsd GL-PCA). No GPU: the test env
// exposes no CUDA device, so the command falls back to the CpuBackend reference
// oracle for the IAF EM — exercising the FULL path (beagle parse -> tile -> EM ->
// covariance/PCA -> emit) end-to-end on the CPU. It authors a tiny plain-text beagle
// file (4 individuals, certain genotypes) and asserts a clean run producing
// PREFIX.{cov,eigenvec,eigenval} with the right shapes (+ .freq under --emit-freq),
// plus flag validation. The numeric concordance vs the pcangsd package (a frozen
// golden) is the real-data box gate, not this plumbing test. argv[1] = steppe binary.
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

// Count lines and the field count of the first line of a whitespace table.
bool table_shape(const std::filesystem::path& p, int& rows, int& first_cols) {
    std::ifstream f(p);
    if (!f) return false;
    rows = 0;
    first_cols = -1;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (first_cols < 0) {
            std::istringstream ss(line);
            std::string tok;
            int c = 0;
            while (ss >> tok) ++c;
            first_cols = c;
        }
        ++rows;
    }
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <steppe-binary>\n", argv[0]);
        return 2;
    }
    const std::string bin = argv[1];
    std::printf("=== steppe pcangsd wiring gate (CPU-backend EM, no GPU) ===\n");
    if (!std::filesystem::exists(bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", bin.c_str());
        return 1;
    }

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_pcangsd_cli_" + std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    // Beagle: 4 individuals, 30 certain-genotype sites (allele-2 counts cycle so every
    // site is polymorphic and survives the MAF filter).
    const int NIND = 4;
    const long M = 30;
    const std::filesystem::path bpath = tmp / "toy.beagle";
    {
        std::ofstream b(bpath, std::ios::trunc);
        b << "marker\tallele1\tallele2";
        for (int i = 0; i < NIND; ++i) b << "\tInd" << i << "\tInd" << i << "\tInd" << i;
        b << "\n";
        for (long j = 0; j < M; ++j) {
            b << (j / 1000 + 1) << "_" << (100 + j) << "\tA\tG";
            const int g[NIND] = {static_cast<int>(j % 3), static_cast<int>((j + 1) % 3),
                                 static_cast<int>((j + 2) % 3), static_cast<int>((j + 1) % 3)};
            for (int i = 0; i < NIND; ++i) {
                const double aa = (g[i] == 0) ? 1.0 : 0.0;   // P(A1A1)
                const double het = (g[i] == 1) ? 1.0 : 0.0;  // P(Aa)
                const double bb = (g[i] == 2) ? 1.0 : 0.0;   // P(A2A2)
                b << "\t" << aa << "\t" << het << "\t" << bb;
            }
            b << "\n";
        }
    }

    // ---- clean run -----------------------------------------------------------
    const std::filesystem::path pref = tmp / "out";
    const std::filesystem::path logf = tmp / "run.log";
    const int rc = run_steppe(bin,
                              {"pcangsd", "--beagle", bpath.string(), "-e", "2", "--emit-freq",
                               "--out", pref.string()},
                              logf);
    if (rc != 0) {
        std::printf("  [FAIL] steppe pcangsd exit=%d (log follows)\n", rc);
        std::ifstream lf(logf);
        std::stringstream ss;
        ss << lf.rdbuf();
        std::printf("%s\n", ss.str().c_str());
        ++g_fail;
    } else {
        int rows = 0, cols = 0;
        if (!table_shape(pref.string() + ".cov", rows, cols) || rows != NIND || cols != NIND) {
            std::printf("  [FAIL] .cov shape rows=%d cols=%d (want %dx%d)\n", rows, cols, NIND, NIND);
            ++g_fail;
        }
        if (!table_shape(pref.string() + ".eigenvec", rows, cols) || rows != NIND || cols != 2) {
            std::printf("  [FAIL] .eigenvec shape rows=%d cols=%d (want %dx2)\n", rows, cols, NIND);
            ++g_fail;
        }
        if (!table_shape(pref.string() + ".eigenval", rows, cols) || rows != 2) {
            std::printf("  [FAIL] .eigenval rows=%d (want 2)\n", rows);
            ++g_fail;
        }
        if (!table_shape(pref.string() + ".freq", rows, cols) || rows != static_cast<int>(M)) {
            std::printf("  [FAIL] .freq rows=%d (want %ld kept sites)\n", rows, M);
            ++g_fail;
        }
    }

    // ---- validation: bad --precision -> InvalidConfig(2) ---------------------
    {
        const std::filesystem::path l2 = tmp / "v2.log";
        const int rc2 = run_steppe(bin,
                                   {"pcangsd", "--beagle", bpath.string(), "-e", "2", "--precision",
                                    "bogus", "--out", (tmp / "x").string()},
                                   l2);
        if (rc2 != 2) {
            std::printf("  [FAIL] bad --precision should exit 2, got %d\n", rc2);
            ++g_fail;
        }
    }
    // ---- validation: -e 0 -> InvalidConfig(2) --------------------------------
    {
        const std::filesystem::path l3 = tmp / "v3.log";
        const int rc3 = run_steppe(
            bin, {"pcangsd", "--beagle", bpath.string(), "-e", "0", "--out", (tmp / "y").string()},
            l3);
        if (rc3 != 2) {
            std::printf("  [FAIL] -e 0 should exit 2, got %d\n", rc3);
            ++g_fail;
        }
    }

    std::filesystem::remove_all(tmp, ec);
    if (g_fail == 0) {
        std::printf("\nRESULT: PASS (steppe pcangsd wiring + CPU EM + output shapes + validation)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_fail);
    return 1;
}
