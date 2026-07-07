// tests/cli/test_cli_paint.cpp
//
// `steppe paint` CLI + validator smoke gate (Phase 0). Plain C++ host TU (no CUDA):
// it writes tiny EIGENSTRAT triples to a temp dir and drives the built `steppe paint`
// binary, asserting the host-pure up-front validator's contracts THROUGH the CLI:
//   1. a well-formed PHASED HAPLOID panel (self-painted, leave-one-out) VALIDATES
//      (exit 0, reports a plan) — no GPU compute in Phase 0;
//   2. a DIPLOID recipient triple (a heterozygous call) is REJECTED with a
//      "phase first" error (kExitInvalidConfig);
//   3. a missing cM map is REJECTED unless --bp-fallback is opted in;
//   4. a missing --donors is REJECTED.
// These are small hand-written fixtures for a VALIDATOR smoke test (not a scientific
// result). Self-checking main(); CTest gates on the exit code. Needs NO CUDA device
// (Phase 0 paint is host-only). argv[1] = the built steppe binary.

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

struct RunResult {
    int exit_code = -1;
    std::string text;
};

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "paint_stdout.txt";
    std::string cmd = "\"" + bin + "\"";
    for (const std::string& a : args) cmd += " \"" + a + "\"";
    cmd += " > \"" + outf.string() + "\" 2>&1";
    const int sys = std::system(cmd.c_str());
    rr.exit_code = (sys == -1) ? -1 :
#ifdef WIFEXITED
                   (WIFEXITED(sys) ? WEXITSTATUS(sys) : -1);
#else
                   sys;
#endif
    std::ifstream f(outf, std::ios::binary);
    if (f) { std::ostringstream ss; ss << f.rdbuf(); rr.text = ss.str(); }
    return rr;
}

void write_file(const std::filesystem::path& p, const std::string& s) {
    std::ofstream f(p);
    f << s;
}

// Write an EIGENSTRAT triple. geno_rows = SNP-major ASCII rows (one char/indiv);
// snp lines carry the cM map in genpos_morgans (col 3). n_ind individuals, pop PANEL.
void write_triple(const std::filesystem::path& dir, const std::string& stem, int n_ind,
                  const std::vector<std::string>& geno_rows,
                  const std::vector<double>& genpos_morgans) {
    std::string ind;
    for (int i = 0; i < n_ind; ++i)
        ind += "h" + std::to_string(i) + " U PANEL\n";
    write_file(dir / (stem + ".ind"), ind);

    std::string snp;
    for (std::size_t s = 0; s < geno_rows.size(); ++s) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "rs%zu 1 %.6f %zu A C\n", s, genpos_morgans[s],
                      1000 * (s + 1));
        snp += buf;
    }
    write_file(dir / (stem + ".snp"), snp);

    std::string geno;
    for (const std::string& r : geno_rows) geno += r + "\n";
    write_file(dir / (stem + ".geno"), geno);
}

void check(bool cond, const char* what, const RunResult& r) {
    if (!cond) {
        std::printf("  [FAIL] %s (exit=%d)\n%s\n", what, r.exit_code, r.text.c_str());
        ++g_fail;
    } else {
        std::printf("  [ ok ] %s (exit=%d)\n", what, r.exit_code);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <steppe-binary>\n", argv[0]);
        return 2;
    }
    const std::string bin = argv[1];
    std::printf("=== steppe paint CLI + validator smoke gate (Phase 0, host-only) ===\n");
    if (!std::filesystem::exists(bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", bin.c_str());
        return 1;
    }

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_cli_paint_" +
         std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const int N = 6;
    const std::vector<double> gp = {0.01, 0.02, 0.03, 0.04, 0.05};
    // A phased HAPLOID panel: pseudo-haploid codes {0,2}, no heterozygotes.
    write_triple(tmp, "hap", N,
                 {"020020", "200200", "022002", "002200", "220022"}, gp);
    // A DIPLOID triple: individual 0 carries a heterozygous call (code 1) at SNP 0.
    write_triple(tmp, "dip", N,
                 {"120020", "200200", "022002", "002200", "220022"}, gp);
    // The haploid geno/ind with an ABSENT map (all genpos 0).
    write_triple(tmp, "nomap", N,
                 {"020020", "200200", "022002", "002200", "220022"},
                 {0.0, 0.0, 0.0, 0.0, 0.0});

    const std::string hap = (tmp / "hap").string();
    const std::string dip = (tmp / "dip").string();
    const std::string nomap = (tmp / "nomap").string();

    // 1. Well-formed self-painted haploid panel validates (leave-one-out on).
    {
        RunResult r = run_steppe(bin, {"paint", "--prefix", hap, "--donors", hap}, tmp);
        check(r.exit_code == 0 && r.text.find("validated") != std::string::npos,
              "phased haploid panel validates (plan reported)", r);
    }

    // 2. Diploid recipient triple is rejected with a phase-first error.
    {
        RunResult r = run_steppe(bin, {"paint", "--prefix", dip, "--donors", hap}, tmp);
        check(r.exit_code != 0 && r.text.find("PRE-PHASED") != std::string::npos,
              "diploid input rejected (phase first)", r);
    }

    // 3. Absent cM map is rejected by default, accepted with --bp-fallback.
    {
        RunResult r = run_steppe(bin, {"paint", "--prefix", nomap, "--donors", hap}, tmp);
        check(r.exit_code != 0 && r.text.find("cM map") != std::string::npos,
              "absent cM map rejected by default", r);
        RunResult r2 =
            run_steppe(bin, {"paint", "--prefix", nomap, "--donors", hap, "--bp-fallback"}, tmp);
        check(r2.exit_code == 0, "absent map + --bp-fallback accepted", r2);
    }

    // 4. Missing --donors is rejected.
    {
        RunResult r = run_steppe(bin, {"paint", "--prefix", hap}, tmp);
        check(r.exit_code != 0 && r.text.find("--donors") != std::string::npos,
              "missing --donors rejected", r);
    }

    std::filesystem::remove_all(tmp, ec);
    if (g_fail == 0) {
        std::printf("\nRESULT: PASS\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s))\n", g_fail);
    return 1;
}
