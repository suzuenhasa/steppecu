// tests/cli/test_cli_ancestrymap_prefix.cpp
//
// M-FR-AM CLI AUTO-COVER GATE (docs/design/format-readers.md §5.3, the EIGENSTRAT/
// PLINK precedent): the FULL ANCESTRYMAP (unpacked legacy EIGENSOFT, TEXT triple)
// reader works THROUGH the production CLI and matches the convertf-PA (GENO, SNP-
// major) reader, ON THE GPU, on the REAL AADR — with NO new golden.
//
// The mechanism: run the SAME tool TWICE — once with `--prefix <fixtures_eig/v66_fit9_am>`
// (the ANCESTRYMAP prefix that USED TO THROW at the GenoReader ctor's "unrecognized
// format") and once with `--prefix <fixtures_eig/v66_fit9_pa>` (the convertf-PA prefix
// of the SAME 9-pop subset) — and assert the ANCESTRYMAP run (1) WORKS (exit 0, no
// throw) and (2) produces the SAME result as the PA run. Both fixtures are convertf
// outputs from the SAME PA source with the SAME poplist, so they are the SAME genotypes
// two ways (text triple vs packed) with identical .ind/.snp; a correct ANCESTRYMAP
// reader MUST reproduce the PA result. Because Tier-1 (test_ancestrymap_decode_equivalence)
// already proves the canonical tiles + decoded Q/V/N are bit-identical, EVERY downstream
// statistic is auto-covered — this gate proves that holds end-to-end through the REAL
// CLI binary. Since PA==TGENO==goldens (test_cli_pa_prefix), ANCESTRYMAP is transitively
// pinned to the goldens.
//
// TWO TOOLS, EACH ANCESTRYMAP-vs-PA:
//   (1) extract-f2 --prefix: assert AM exit==0 (the throw is GONE) AND the written
//       f2.bin is BYTE-FOR-BYTE identical to the PA run's f2.bin (STPF2BK1 is
//       deterministic from shape+values, no timestamps).
//   (2) qpdstat --prefix: assert AM exit==0 AND est/se/z match the PA run row-for-row
//       (rtol 1e-6) — the genotype-path normalized-D.
//
// PLAIN C++ host TU (NO CUDA header): it spawns the steppe binary; the GPU work runs in
// the child. SKIPs cleanly (exit 0) when no CUDA device is visible OR the fixtures_eig
// prefixes are absent (they live only on box5090). Self-checking main(); CTest gates on
// the exit code. NO SYNTHETIC DATA.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>   // std::memcmp (the f2.bin bit-identity gate)
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

void check_true(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
    else std::printf("  [ok]   %s\n", what);
}
void check_eq_int(const char* what, long long got, long long want) {
    const bool ok = (got == want);
    if (!ok) std::printf("  [FAIL] %-44s got=%lld want=%lld\n", what, got, want);
    else std::printf("  [ok]   %s\n", what);
    if (!ok) ++g_failures;
}
void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const double diff = std::fabs(got - want);
    const bool ok = (std::isnan(got) && std::isnan(want)) || diff <= tol;
    if (!ok)
        std::printf("  [FAIL] %-40s am=% .12e pa=% .12e |d|=% .3e tol=% .3e\n",
                    what, got, want, diff, tol);
    if (!ok) ++g_failures;
}

struct RunResult { int exit_code = -1; std::string text; };

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp, const std::string& tag) {
    RunResult rr;
    const std::filesystem::path outf = tmp / ("out_" + tag + ".txt");
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

bool looks_like_no_gpu(const std::string& t) {
    return t.find("no CUDA device") != std::string::npos ||
           t.find("device error") != std::string::npos ||
           t.find("No CUDA") != std::string::npos;
}

std::vector<char> read_file_bytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss; ss << f.rdbuf();
    const std::string s = ss.str();
    return std::vector<char>(s.begin(), s.end());
}

// ---- Minimal CSV parser for qpdstat output ("pop1..pop4,est,se,z,p"). -------------
std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cells; std::string cur; bool in_q = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_q) { if (c == '"') in_q = false; else cur += c; }
        else { if (c == '"') in_q = true; else if (c == ',') { cells.push_back(cur); cur.clear(); } else cur += c; }
    }
    cells.push_back(cur);
    return cells;
}
double cell_d(const std::string& s) {
    if (s == "NA" || s == "null" || s.empty()) return std::nan("");
    return std::strtod(s.c_str(), nullptr);
}
struct DRow { std::string key; double est = 0, se = 0, z = 0; };
std::map<std::string, DRow> parse_dstat(const std::string& text) {
    std::map<std::string, DRow> out;
    std::istringstream ss(text); std::string line; bool header = true;
    std::map<std::string, std::size_t> col;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> c = split_csv_line(line);
        if (header) { for (std::size_t i = 0; i < c.size(); ++i) col[c[i]] = i; header = false; continue; }
        if (!col.count("pop1") || !col.count("est")) continue;
        DRow r;
        r.key = c[col["pop1"]] + "|" + c[col["pop2"]] + "|" + c[col["pop3"]] + "|" + c[col["pop4"]];
        r.est = cell_d(c[col["est"]]);
        r.se = col.count("se") ? cell_d(c[col["se"]]) : std::nan("");
        r.z = col.count("z") ? cell_d(c[col["z"]]) : std::nan("");
        out[r.key] = r;
    }
    return out;
}
std::string join(const std::vector<std::string>& v) {
    std::string s; for (std::size_t i = 0; i < v.size(); ++i) { if (i) s += ","; s += v[i]; } return s;
}

}  // namespace

int main(int argc, char** argv) {
    // argv[1] = built steppe binary; argv[2] = AADR root.
    if (argc < 3) {
        std::printf("usage: %s <steppe-binary> <aadr-root>\n", argv[0]);
        return 2;
    }
    const std::string bin = argv[1];
    const std::string root = argv[2];
    const std::string am_prefix = root + "/fixtures_eig/v66_fit9_am";   // ANCESTRYMAP (text) — the new path
    const std::string pa_prefix = root + "/fixtures_eig/v66_fit9_pa";   // convertf PA (GENO) — the oracle

    std::printf("=== M-FR-AM CLI auto-cover: ANCESTRYMAP --prefix == PA(GENO) --prefix (ON THE GPU) ===\n");
    std::printf("steppe:    %s\nAM prefix: %s\nPA prefix: %s\n", bin.c_str(),
                am_prefix.c_str(), pa_prefix.c_str());

    if (!std::filesystem::exists(bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", bin.c_str());
        return 1;
    }
    auto have = [](const std::string& pfx) {
        return std::filesystem::exists(pfx + ".geno") && std::filesystem::exists(pfx + ".snp") &&
               std::filesystem::exists(pfx + ".ind");
    };
    if (!have(am_prefix) || !have(pa_prefix)) {
        std::printf("\nRESULT: SKIP (ANCESTRYMAP or PA fixture absent: %s.{geno,snp,ind} / %s.{geno,snp,ind})\n",
                    am_prefix.c_str(), pa_prefix.c_str());
        return 0;
    }

    std::error_code ec;
    const std::filesystem::path tmp = std::filesystem::temp_directory_path(ec) /
        ("steppe_cli_am_prefix_" + std::to_string(static_cast<long long>(
            std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    // The 9 fit0 pops (sorted ASC == steppe's internal sort) — the f2 set.
    const std::vector<std::string> pops9 = {
        "Czechia_EBA_CordedWare", "England_BellBeaker", "Han", "Iran_GanjDareh_N",
        "Israel_Natufian", "Karitiana", "Mbuti", "Papuan", "Turkey_N"};
    const std::string pops9_arg = join(pops9);
    const std::string blgsize = "0.05";  // Morgans (AT2 convention)

    // ====================================================================== (1) extract-f2
    // The ANCESTRYMAP prefix USED TO THROW at the GenoReader ctor (unrecognized format).
    // Assert it now WORKS (exit 0) AND the f2.bin is byte-identical to the PA run's.
    std::printf("\n--- (1) extract-f2 --prefix: ANCESTRYMAP WORKS + f2.bin == PA -------------\n");
    {
        const std::filesystem::path am_dir = tmp / "f2_am";
        const std::filesystem::path pa_dir = tmp / "f2_pa";
        std::vector<std::string> am_args = {
            "extract-f2", "--prefix", am_prefix, "--pops", pops9_arg,
            "--blgsize", blgsize, "--maxmiss", "0", "--out", am_dir.string()};
        std::vector<std::string> pa_args = {
            "extract-f2", "--prefix", pa_prefix, "--pops", pops9_arg,
            "--blgsize", blgsize, "--maxmiss", "0", "--out", pa_dir.string()};
        const RunResult am = run_steppe(bin, am_args, tmp, "ef2_am");
        if (looks_like_no_gpu(am.text)) {
            std::printf("RESULT: SKIP (no CUDA device — extract-f2 GPU path cannot run)\n");
            return 0;
        }
        const RunResult pa = run_steppe(bin, pa_args, tmp, "ef2_pa");
        check_eq_int("extract-f2 ANCESTRYMAP prefix exit == 0 (the throw is GONE)", am.exit_code, 0);
        check_eq_int("extract-f2 PA prefix exit == 0", pa.exit_code, 0);
        if (am.exit_code != 0) std::printf("  AM stderr:\n%s\n", am.text.c_str());
        const std::vector<char> ab = read_file_bytes(am_dir / "f2.bin");
        const std::vector<char> pb = read_file_bytes(pa_dir / "f2.bin");
        check_true("ANCESTRYMAP f2.bin non-empty", !ab.empty());
        check_true("PA f2.bin non-empty", !pb.empty());
        check_true("ANCESTRYMAP f2.bin size == PA f2.bin size", ab.size() == pb.size());
        const bool eq = !ab.empty() && ab.size() == pb.size() &&
                        std::memcmp(ab.data(), pb.data(), ab.size()) == 0;
        check_true("extract-f2 ANCESTRYMAP f2.bin BIT-IDENTICAL to PA f2.bin", eq);
        std::printf("  f2.bin: AM %zu B, PA %zu B\n", ab.size(), pb.size());
    }

    // ====================================================================== (2) qpdstat
    // Genotype-path normalized-D: a few quadruples over the 9-pop union. AM == PA
    // row-for-row (rtol 1e-6).
    std::printf("\n--- (2) qpdstat --prefix: ANCESTRYMAP est/se/z == PA ----------------------\n");
    {
        const std::vector<std::array<std::string, 4>> quads = {
            {"Mbuti", "Han", "Karitiana", "Papuan"},
            {"Mbuti", "Turkey_N", "Iran_GanjDareh_N", "Israel_Natufian"},
            {"Mbuti", "England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N"}};
        std::vector<std::string> p1, p2, p3, p4;
        for (const auto& q : quads) { p1.push_back(q[0]); p2.push_back(q[1]); p3.push_back(q[2]); p4.push_back(q[3]); }
        auto dstat_args = [&](const std::string& prefix) {
            return std::vector<std::string>{
                "qpdstat", "--prefix", prefix,
                "--pop1", join(p1), "--pop2", join(p2), "--pop3", join(p3), "--pop4", join(p4),
                "--format", "csv", "--device", "0"};
        };
        const RunResult am = run_steppe(bin, dstat_args(am_prefix), tmp, "dstat_am");
        const RunResult pa = run_steppe(bin, dstat_args(pa_prefix), tmp, "dstat_pa");
        check_eq_int("qpdstat ANCESTRYMAP prefix exit == 0", am.exit_code, 0);
        check_eq_int("qpdstat PA prefix exit == 0", pa.exit_code, 0);
        if (am.exit_code != 0) std::printf("  AM stderr:\n%s\n", am.text.c_str());
        const std::map<std::string, DRow> amap = parse_dstat(am.text);
        const std::map<std::string, DRow> pm = parse_dstat(pa.text);
        check_true("qpdstat parsed >=1 AM row", !amap.empty());
        check_true("qpdstat AM row count == PA row count", amap.size() == pm.size());
        for (const auto& [k, pr] : pm) {
            auto it = amap.find(k);
            if (it == amap.end()) { check_true(("qpdstat AM has row " + k).c_str(), false); continue; }
            check_close((k + " est").c_str(), it->second.est, pr.est, 1e-6, 1e-9);
            check_close((k + " se").c_str(), it->second.se, pr.se, 1e-6, 1e-9);
            check_close((k + " z").c_str(), it->second.z, pr.z, 1e-6, 1e-9);
        }
    }

    std::filesystem::remove_all(tmp, ec);
    std::printf("\nRESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
