// tests/cli/test_cli_roh.cpp
//
// Host-only wiring gate for `steppe roh` (hapROH). No GPU: the test env exposes no CUDA
// device, so the command falls back to the CpuBackend reference oracle for the (K+1)-
// state forward-backward — exercising the FULL path (target + phased panel read, site
// intersection + polarity, per-chromosome FB, ROH segment calling, summary emit) end-to-
// end on the CPU. It authors a tiny pseudo-haploid target that PERFECTLY copies one panel
// haplotype over a long dense genetic run (a clean synthetic ROH) plus a 4-haplotype
// panel, and asserts (a) a clean run with the reference output columns, (b) a called ROH
// (sum_roh>4 cM > 0 for the copy target), and (c) the required-flag validation. Segment
// CONTENT concordance vs pip-hapROH is the real-data box gate, not this plumbing test.
// argv[1] = the built steppe binary.
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

void write_file(const std::filesystem::path& p, const std::string& s) {
    std::ofstream f(p);
    f << s;
}

// Write an EIGENSTRAT triple. geno_rows = SNP-major ASCII (one char/indiv, {0,2} codes);
// snp col 3 carries the cM map in Morgan. All individuals labelled `pop`.
void write_triple(const std::filesystem::path& dir, const std::string& stem, int n_ind,
                  const std::vector<std::string>& geno_rows,
                  const std::vector<double>& genpos_morgans, const std::string& pop) {
    std::string ind;
    for (int i = 0; i < n_ind; ++i) ind += stem + "_h" + std::to_string(i) + " U " + pop + "\n";
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
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <steppe-binary>\n", argv[0]);
        return 2;
    }
    const std::string bin = argv[1];
    std::printf("=== steppe roh wiring gate (CPU-backend FB, no GPU) ===\n");
    if (!std::filesystem::exists(bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", bin.c_str());
        return 1;
    }

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_roh_cli_" + std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    // 600 dense SNPs at 0.01 cM (0.0001 Morgan) spacing on chr1 -> a 6 cM span. The TARGET
    // perfectly copies panel haplotype 0 across the whole span (a clean synthetic ROH).
    // The spacing sets the SNP density: density = 1/(100*spacing) = 100 SNPs/cM here, well
    // above hapROH's post_process_roh_df snp_cm=50 SNPs/cM summary-density floor (a genuine
    // hapROH default the summary applies per bin), so the >4 cM ROH survives the density
    // filter. Real 1240K data far exceeds 50 SNPs/cM; a sparse fixture would be filtered out.
    const int NS = 600;
    std::vector<double> gp(NS);
    std::vector<std::string> panel_rows(NS), tgt_rows(NS);
    for (int s = 0; s < NS; ++s) {
        gp[static_cast<std::size_t>(s)] = 0.0001 * s;  // Morgan
        const int a = s % 2;                            // the target/hap0 allele
        auto code = [](int allele) -> char { return allele ? '2' : '0'; };  // {0,2}
        // panel: hap0 = a (matches target), hap1 = 1-a, hap2 = (s/2)%2, hap3 = (s/3)%2.
        std::string prow;
        prow += code(a);
        prow += code(1 - a);
        prow += code((s / 2) % 2);
        prow += code((s / 3) % 2);
        panel_rows[static_cast<std::size_t>(s)] = prow;
        tgt_rows[static_cast<std::size_t>(s)] = std::string(1, code(a));
    }
    write_triple(tmp, "panel", 4, panel_rows, gp, "REF");
    write_triple(tmp, "target", 1, tgt_rows, gp, "ANC");

    const std::string panel = (tmp / "panel").string();
    const std::string target = (tmp / "target").string();

    // ---- clean run -----------------------------------------------------------
    const std::filesystem::path seg = tmp / "seg.tsv";
    const std::filesystem::path sum = tmp / "sum.tsv";
    const std::filesystem::path logf = tmp / "run.log";
    const int rc = run_steppe(
        bin, {"roh", "--prefix", target, "--ref-panel", panel, "--out", seg.string(), "--summary",
              sum.string()},
        logf);
    if (rc != 0) {
        std::printf("  [FAIL] steppe roh exit=%d (log follows)\n", rc);
        std::ifstream lf(logf);
        std::stringstream ss; ss << lf.rdbuf();
        std::printf("%s\n", ss.str().c_str());
        ++g_fail;
    } else {
        const std::string sh = first_line(seg);
        if (sh.rfind("iid", 0) != 0 || sh.find("lengthCM") == std::string::npos) {
            std::printf("  [FAIL] segment header wrong: '%s'\n", sh.c_str());
            ++g_fail;
        }
        const std::string uh = first_line(sum);
        if (uh.rfind("iid", 0) != 0 || uh.find("sum_roh>4") == std::string::npos) {
            std::printf("  [FAIL] summary header wrong: '%s'\n", uh.c_str());
            ++g_fail;
        }
        // The perfect-copy target must show a called ROH: sum_roh>4 (2nd summary column) > 0.
        std::ifstream sf(sum);
        std::string line;
        std::getline(sf, line);  // header
        bool found_roh = false;
        while (std::getline(sf, line)) {
            if (line.empty()) continue;
            std::vector<std::string> f;
            std::string tok; std::istringstream ls(line);
            while (ls >> tok) f.push_back(tok);
            // columns: iid max_roh sum_roh>4 n_roh>4 ...
            if (f.size() >= 3 && std::strtod(f[2].c_str(), nullptr) > 0.0) found_roh = true;
        }
        if (!found_roh) {
            std::printf("  [FAIL] no ROH detected on the perfect-copy target (sum_roh>4 == 0)\n");
            std::ifstream lf(logf);
            std::stringstream ss; ss << lf.rdbuf();
            std::printf("%s\n", ss.str().c_str());
            ++g_fail;
        }
    }

    // ---- validation: missing --ref-panel -> InvalidConfig(2) -----------------
    {
        const std::filesystem::path l2 = tmp / "v2.log";
        const int rc2 =
            run_steppe(bin, {"roh", "--prefix", target, "--out", (tmp / "x.tsv").string()}, l2);
        if (rc2 == 0) {
            std::printf("  [FAIL] missing --ref-panel should be rejected, got exit 0\n");
            ++g_fail;
        }
    }

    std::filesystem::remove_all(tmp, ec);
    if (g_fail == 0) {
        std::printf("\nRESULT: PASS (steppe roh wiring + CPU FB + ROH call + validation correct)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_fail);
    return 1;
}
