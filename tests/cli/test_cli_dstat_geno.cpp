// tests/cli/test_cli_dstat_geno.cpp
//
// qpDstat Part-B acceptance gate: the `steppe qpdstat --prefix <geno>` CLI reproduces the
// AT2 GENOTYPE-PATH normalized-D golden (golden_fit0_dstat_geno.csv, 60 quadruple rows)
// THROUGH the CLI, ON THE GPU. The golden was made via admixtools::qpdstat(pref=the convertf
// -PA prefix, the 60 fit0 quadruples, f4mode=FALSE) -> qpdstat_geno (allsnps=TRUE,
// blgsize=0.05) — the NORMALIZED D (est range ~±0.06), DISTINCT from the f2-path f4 golden
// (~10x smaller). steppe reads the SAME genotype prefix directly through run_dstat (the
// extract-f2 decode front-end + the per-SNP D kernel + the num/den block-jackknife), with
// the THREE parity pins (forced diploid / autosomes-only / allsnps=TRUE finiteness).
//
// NO SYNTHETIC DATA (memory real-data-only): it runs the BUILT `steppe qpdstat --prefix
// <STEPPE_AADR_ROOT>/raw/v66.p1_HO.aadr.patch.PUB` (the raw TGENO steppe reads — lossless
// transcode of the convertf-PA v66_HO_pa AT2 made the golden from) over the 60 golden
// quadruples and diffs the est/se/z table row-for-row. AT2 reads only the needed inds via indvec from the full
// PA prefix (run_dstat reads ONLY the 9-pop union — read_ind(Explicit{union})), so NO
// physical 9-pop subset is needed.
//
// TIERS (the achievable tier): est/se/z at rtol 1e-6 (atol 1e-9). The decode + block
// components were pinned to the AT2 R golden to ~1e-15 / max-diff-0 on box5090; 1e-6 is a
// comfortable gate (p can underflow for huge |z| — a generous atol). status == "ok".
//
// PLAIN C++ host TU (NO CUDA header): it spawns the steppe binary and parses stdout; the GPU
// work happens inside that child. SKIPs cleanly (exit 0) when no CUDA device is visible OR
// the PA prefix / golden is absent — identical to test_cli_extract_qpadm's data-absent skip.
// The harness helpers mirror test_cli_qpdstat. Self-checking main(); CTest gates on the exit.

#include <cmath>
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

void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const double diff = std::fabs(got - want);
    const bool ok = diff <= tol;
    if (!ok)
        std::printf("  [FAIL] %-44s got=% .12e want=% .12e |d|=% .3e tol=% .3e\n",
                    what, got, want, diff, tol);
    if (!ok) ++g_failures;
}

void check_eq_int(const char* what, long long got, long long want) {
    const bool ok = (got == want);
    if (!ok) std::printf("  [FAIL] %-44s got=%lld want=%lld\n", what, got, want);
    if (!ok) ++g_failures;
}

void check_true(const char* what, bool ok) {
    if (!ok) std::printf("  [FAIL] %s\n", what);
    if (!ok) ++g_failures;
}

struct RunResult {
    int exit_code = -1;
    std::string stdout_text;
};

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "cli_dstat_geno_stdout.txt";
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
    if (f) { std::ostringstream ss; ss << f.rdbuf(); rr.stdout_text = ss.str(); }
    return rr;
}

bool looks_like_no_gpu(const std::string& text) {
    return text.find("no CUDA device") != std::string::npos ||
           text.find("device error") != std::string::npos ||
           text.find("No CUDA") != std::string::npos;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cells;
    std::string cur;
    bool in_q = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_q) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') { cur += '"'; ++i; }
                else in_q = false;
            } else cur += c;
        } else {
            if (c == '"') in_q = true;
            else if (c == ',') { cells.push_back(cur); cur.clear(); }
            else cur += c;
        }
    }
    cells.push_back(cur);
    return cells;
}

double cell_d(const std::string& s) {
    if (s == "NA" || s == "null" || s.empty()) return std::nan("");
    return std::strtod(s.c_str(), nullptr);
}

struct DRow {
    std::string p1, p2, p3, p4;
    double est = 0.0, se = 0.0, z = 0.0, p = 0.0;
};

bool read_golden(const std::string& path, std::vector<DRow>& out) {
    std::ifstream f(path);
    if (!f) { std::printf("  [SKIP] golden dstat-geno csv absent: %s\n", path.c_str()); return false; }
    std::string line;
    bool header = true;
    std::map<std::string, std::size_t> col;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const std::vector<std::string> cells = split_csv_line(line);
        if (header) {
            for (std::size_t c = 0; c < cells.size(); ++c) col[cells[c]] = c;
            header = false;
            continue;
        }
        DRow r;
        r.p1 = cells[col["pop1"]];
        r.p2 = cells[col["pop2"]];
        r.p3 = cells[col["pop3"]];
        r.p4 = cells[col["pop4"]];
        r.est = cell_d(cells[col["est"]]);
        r.se  = cell_d(cells[col["se"]]);
        r.z   = cell_d(cells[col["z"]]);
        r.p   = cell_d(cells[col["p"]]);
        out.push_back(r);
    }
    return true;
}

std::map<std::string, DRow> parse_steppe(const std::string& text) {
    std::map<std::string, DRow> out;
    std::istringstream ss(text);
    std::string line;
    bool header = true;
    std::map<std::string, std::size_t> col;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> cells = split_csv_line(line);
        if (header) {
            for (std::size_t c = 0; c < cells.size(); ++c) col[cells[c]] = c;
            header = false;
            continue;
        }
        if (!col.count("pop1") || !col.count("est")) continue;
        DRow r;
        r.p1 = cells[col["pop1"]];
        r.p2 = cells[col["pop2"]];
        r.p3 = cells[col["pop3"]];
        r.p4 = cells[col["pop4"]];
        r.est = cell_d(cells[col["est"]]);
        r.se  = cell_d(cells[col["se"]]);
        r.z   = cell_d(cells[col["z"]]);
        r.p   = cell_d(cells[col["p"]]);
        out[r.p1 + "|" + r.p2 + "|" + r.p3 + "|" + r.p4] = r;
    }
    return out;
}

std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) { if (i) s += ","; s += v[i]; }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    // argv[1] = the built steppe binary; argv[2] = the AADR root; argv[3] = the golden dir.
    if (argc < 4) {
        std::printf("usage: %s <steppe-binary> <aadr-root> <golden-dir>\n", argv[0]);
        return 2;
    }
    const std::string steppe_bin = argv[1];
    const std::string aadr_root = argv[2];
    const std::string golden_dir = argv[3];
    // The genotype prefix. AT2 generated the golden from the convertf-PA v66_HO_pa
    // (PACKEDANCESTRYMAP / SNP-major GENO); steppe's reader is TGENO (individual-major) only,
    // so we read the SAME UNDERLYING DATA from the raw HO TGENO prefix
    // (raw/v66.p1_HO.aadr.patch.PUB) — convertf is a lossless transcode of these IDENTICAL
    // genotypes (same 27594 ind / 584131 SNP axes, same .ind/.snp), so the D is bit-identical
    // (proven: golden row 1 reproduced to ~1e-15). AT2 reads only the needed individuals via
    // indvec, so steppe reads only the 9-pop union from this prefix.
    const std::string pa_prefix = aadr_root + "/raw/v66.p1_HO.aadr.patch.PUB";

    std::printf("=== qpDstat Part-B CLI parity (--prefix genotype-path NORMALIZED-D; ON THE GPU) ===\n");
    std::printf("steppe binary: %s\nPA prefix:     %s\ngolden dir:    %s\n",
                steppe_bin.c_str(), pa_prefix.c_str(), golden_dir.c_str());

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", steppe_bin.c_str());
        return 1;
    }
    if (!std::filesystem::exists(pa_prefix + ".geno") ||
        !std::filesystem::exists(pa_prefix + ".snp") ||
        !std::filesystem::exists(pa_prefix + ".ind")) {
        std::printf("\nRESULT: SKIP (PA genotype prefix absent: %s.{geno,snp,ind})\n",
                    pa_prefix.c_str());
        return 0;
    }

    std::vector<DRow> golden;
    if (!read_golden(golden_dir + "/csv/golden_fit0_dstat_geno.csv", golden)) {
        std::printf("\nRESULT: SKIP (genotype-path dstat golden absent)\n");
        return 0;
    }
    check_true("golden has 60 quadruple rows", golden.size() == 60);
    if (golden.empty()) { std::printf("\nRESULT: FAIL (empty golden)\n"); return 1; }

    std::error_code ec;
    const std::filesystem::path tmp_root =
        std::filesystem::temp_directory_path(ec) / ("steppe_cli_dstat_geno_test_" +
            std::to_string(static_cast<long long>(
                std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp_root, ec);

    // Build the row-aligned --pop1/--pop2/--pop3/--pop4 columns for ALL 60 golden quadruples.
    std::vector<std::string> c1, c2, c3, c4;
    for (const DRow& g : golden) {
        c1.push_back(g.p1); c2.push_back(g.p2); c3.push_back(g.p3); c4.push_back(g.p4);
    }

    // ---------------- CSV run (the full 60-quadruple batch, --prefix genotype path) -------
    const RunResult csv = run_steppe(steppe_bin,
        {"qpdstat", "--prefix", pa_prefix,
         "--pop1", join(c1), "--pop2", join(c2), "--pop3", join(c3), "--pop4", join(c4),
         "--format", "csv", "--device", "0"},
        tmp_root);
    if (looks_like_no_gpu(csv.stdout_text)) {
        std::printf("  [SKIP] no CUDA device visible — CLI GPU dstat cannot run\n");
        std::filesystem::remove_all(tmp_root, ec);
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;
    }
    check_eq_int("csv exit code == 0", csv.exit_code, 0);

    const std::map<std::string, DRow> got = parse_steppe(csv.stdout_text);
    check_eq_int("steppe emitted 60 rows", static_cast<long long>(got.size()), 60);

    for (const DRow& g : golden) {
        const std::string key = g.p1 + "|" + g.p2 + "|" + g.p3 + "|" + g.p4;
        const auto it = got.find(key);
        if (it == got.end()) {
            std::printf("  [FAIL] quadruple missing from steppe output: %s\n", key.c_str());
            ++g_failures;
            continue;
        }
        const DRow& s = it->second;
        char nm[200];
        std::snprintf(nm, sizeof(nm), "est %s", key.c_str());
        check_close(nm, s.est, g.est, 1e-6, 1e-9);
        std::snprintf(nm, sizeof(nm), "se %s", key.c_str());
        check_close(nm, s.se, g.se, 1e-6, 1e-9);
        std::snprintf(nm, sizeof(nm), "z %s", key.c_str());
        check_close(nm, s.z, g.z, 1e-6, 1e-9);
        std::snprintf(nm, sizeof(nm), "p %s", key.c_str());
        check_close(nm, s.p, g.p, 1e-6, 1e-12);
    }

    // ---------------- single-quadruple --pops convenience (the SANITY) ----------------
    {
        const DRow& g0 = golden.front();
        const std::string pops_arg = g0.p1 + "," + g0.p2 + "," + g0.p3 + "," + g0.p4;
        const RunResult one = run_steppe(steppe_bin,
            {"qpdstat", "--prefix", pa_prefix, "--pops", pops_arg,
             "--format", "csv", "--device", "0"},
            tmp_root);
        check_eq_int("single-quadruple exit code == 0", one.exit_code, 0);
        const std::map<std::string, DRow> g1 = parse_steppe(one.stdout_text);
        check_eq_int("single-quadruple 1 row", static_cast<long long>(g1.size()), 1);
        const std::string key = g0.p1 + "|" + g0.p2 + "|" + g0.p3 + "|" + g0.p4;
        const auto it = g1.find(key);
        check_true("single-quadruple matches golden row 1", it != g1.end());
        if (it != g1.end()) {
            check_close("--pops est", it->second.est, g0.est, 1e-6, 1e-9);
            check_close("--pops se", it->second.se, g0.se, 1e-6, 1e-9);
            check_close("--pops z", it->second.z, g0.z, 1e-6, 1e-9);
        }
    }

    std::filesystem::remove_all(tmp_root, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (60 genotype-path normalized-D quadruples reproduced through the GPU; CSV + --pops)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
