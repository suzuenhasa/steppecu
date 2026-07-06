// tests/cli/test_cli_readv2_concord.cpp
//
// Host-only self-checking gate for the READv2 concordance validator (the Phase-0 "ruler").
// PLAIN C++ TU (NO CUDA header, NO GPU, needs no device): it writes hand-authored TOY
// READv2 tables (the frozen steppe schema) to a temp dir, runs the built
// `steppe readv2-concord --a .. --b ..`, parses the stable `concord_*` machine trailer +
// the process exit code, and asserts the confusion matrix, degree agreement, P0_norm
// within-tol counts, coverage, threshold-driven PASS/FAIL, and the fail-fast exit codes
// against expectations that are KNOWN BY CONSTRUCTION. This tests the validator's OWN diff
// arithmetic — it is NOT presenting synthetic science (the real READv2 oracle fixture comes
// later, in the supervised real-data step). argv[1] = the steppe binary; the test authors
// its own tables (no golden dir). CTest gates on the exit code.
//
// Matches the test_cli_f4ratio.cpp harness idiom (run_steppe + a self-checking main()).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#  include <sys/wait.h>  // WIFEXITED / WEXITSTATUS for std::system's status word
#endif

namespace {

int g_failures = 0;

void check_eq_int(const char* what, long long got, long long want) {
    const bool ok = (got == want);
    if (!ok) std::printf("  [FAIL] %-32s got=%lld want=%lld\n", what, got, want);
    if (!ok) ++g_failures;
}

void check_close(const char* what, double got, double want) {
    const double tol = 1e-6 + 1e-4 * std::fabs(want);
    const bool ok = std::fabs(got - want) <= tol;
    if (!ok)
        std::printf("  [FAIL] %-32s got=% .8g want=% .8g\n", what, got, want);
    if (!ok) ++g_failures;
}

void check_true(const char* what, bool ok) {
    if (!ok) std::printf("  [FAIL] %s\n", what);
    if (!ok) ++g_failures;
}

// --- table authoring --------------------------------------------------------
// Only the four columns the validator reads carry meaning; the rest are valid placeholders.
struct Row {
    std::string a, b, degree, p0;  // p0 is a literal cell (may be "NA")
};

const std::string kHeader = "sampleA,sampleB,n_windows,n_overlap_sites,P0_mean,P0_norm,degree,z";

void write_table(const std::filesystem::path& path, const std::vector<Row>& rows) {
    std::ofstream o(path, std::ios::trunc);
    o << kHeader << "\n";
    for (const Row& r : rows)
        o << r.a << "," << r.b << ",100,5000," << r.p0 << "," << r.p0 << "," << r.degree << ",0.0\n";
    o.close();
}

// The base ORACLE table B (scope §D), 6 pairs.
std::vector<Row> base_b() {
    return {
        {"S1", "S2", "identical", "0.30"},
        {"S3", "S4", "first", "0.72"},
        {"S5", "S6", "first", "0.70"},
        {"S7", "S8", "second", "0.86"},
        {"S9", "SA", "unrelated", "1.00"},
        {"SB", "SC", "unrelated", "0.98"},
    };
}

// --- run + parse ------------------------------------------------------------
struct RunResult {
    int exit_code = -1;
    std::string text;
};

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "concord_stdout.txt";
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

// Parse the `concord_<key>: <value...>` trailer lines into a map (value = the rest of line).
std::map<std::string, std::string> parse_trailer(const std::string& text) {
    std::map<std::string, std::string> m;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string pfx = "concord_";
        if (line.rfind(pfx, 0) != 0) continue;
        const std::size_t colon = line.find(": ");
        if (colon == std::string::npos) continue;
        m[line.substr(0, colon)] = line.substr(colon + 2);
    }
    return m;
}

long long ti(const std::map<std::string, std::string>& m, const std::string& k) {
    const auto it = m.find(k);
    return (it == m.end()) ? -999999 : std::strtoll(it->second.c_str(), nullptr, 10);
}
double td(const std::map<std::string, std::string>& m, const std::string& k) {
    const auto it = m.find(k);
    return (it == m.end()) ? std::nan("") : std::strtod(it->second.c_str(), nullptr);
}
std::vector<int> confusion(const std::map<std::string, std::string>& m) {
    std::vector<int> v;
    const auto it = m.find("concord_confusion");
    if (it == m.end()) return v;
    std::istringstream ss(it->second);
    int x;
    while (ss >> x) v.push_back(x);
    return v;
}
bool result_pass(const std::string& text) {
    return text.find("RESULT: PASS") != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <steppe-binary>\n", argv[0]);
        return 2;
    }
    const std::string bin = argv[1];
    std::printf("=== READv2 concordance-validator host-only gate (no GPU) ===\n");
    std::printf("steppe binary: %s\n", bin.c_str());
    if (!std::filesystem::exists(bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", bin.c_str());
        return 1;
    }

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_concord_test_" + std::to_string(static_cast<long long>(
            std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const std::filesystem::path bpath = tmp / "oracle.csv";
    write_table(bpath, base_b());

    // ---- Case 1: perfect agreement (A == B, with one pair written reversed) --------------
    {
        std::printf("\n-- Case 1: perfect agreement + reversed-key canonicalization --\n");
        std::vector<Row> a = base_b();
        a[0] = {"S2", "S1", "identical", "0.30"};  // reversed order must still match S1,S2
        const std::filesystem::path apath = tmp / "case1_a.csv";
        write_table(apath, a);
        const RunResult r = run_steppe(bin, {"readv2-concord", "--a", apath.string(),
                                             "--b", bpath.string()}, tmp);
        const auto m = parse_trailer(r.text);
        check_eq_int("c1 exit==0 (PASS)", r.exit_code, 0);
        check_true("c1 RESULT PASS", result_pass(r.text));
        check_eq_int("c1 common", ti(m, "concord_common_pairs"), 6);
        check_eq_int("c1 a_only", ti(m, "concord_a_only"), 0);
        check_eq_int("c1 b_only", ti(m, "concord_b_only"), 0);
        check_close("c1 coverage", td(m, "concord_coverage"), 1.0);
        const std::vector<int> want = {1,0,0,0, 0,2,0,0, 0,0,1,0, 0,0,0,2};
        check_true("c1 confusion == expected", confusion(m) == want);
        check_eq_int("c1 degree_agree_num", ti(m, "concord_degree_agree_num"), 6);
        check_close("c1 degree_agreement", td(m, "concord_degree_agreement"), 1.0);
        check_close("c1 p0_max_abs_dev", td(m, "concord_p0_max_abs_dev"), 0.0);
        check_eq_int("c1 within_num", ti(m, "concord_p0_within_tol_num"), 6);
        check_close("c1 within_frac", td(m, "concord_p0_within_tol_frac"), 1.0);
    }

    // ---- Case 2: one degree flip + one P0 out-of-tol -> FAIL at defaults ------------------
    // Reused by Case 3 (relaxed thresholds).
    std::vector<Row> a2 = base_b();
    a2[1] = {"S3", "S4", "second", "0.72"};  // degree flip first->second, P0 in tol
    a2[2] = {"S5", "S6", "first", "0.78"};   // P0 out of tol (0.78 vs 0.70), degree unchanged
    const std::filesystem::path a2path = tmp / "case2_a.csv";
    write_table(a2path, a2);
    {
        std::printf("\n-- Case 2: degree flip + P0 out-of-tol (defaults -> FAIL) --\n");
        const RunResult r = run_steppe(bin, {"readv2-concord", "--a", a2path.string(),
                                             "--b", bpath.string()}, tmp);
        const auto m = parse_trailer(r.text);
        check_eq_int("c2 exit==1 (concord FAIL)", r.exit_code, 1);
        check_true("c2 RESULT FAIL", !result_pass(r.text));
        check_close("c2 coverage", td(m, "concord_coverage"), 1.0);
        const std::vector<int> want = {1,0,0,0, 0,1,1,0, 0,0,1,0, 0,0,0,2};
        const std::vector<int> got = confusion(m);
        check_true("c2 confusion == expected", got == want);
        // Confusion cell [first][second] (row 1, col 2) == 1.
        check_true("c2 confusion[first][second]==1", got.size() == 16 && got[1 * 4 + 2] == 1);
        check_eq_int("c2 degree_agree_num", ti(m, "concord_degree_agree_num"), 5);
        check_close("c2 degree_agreement", td(m, "concord_degree_agreement"), 5.0 / 6.0);
        check_eq_int("c2 within_num", ti(m, "concord_p0_within_tol_num"), 5);
        check_close("c2 within_frac", td(m, "concord_p0_within_tol_frac"), 5.0 / 6.0);
        check_close("c2 p0_max_abs_dev", td(m, "concord_p0_max_abs_dev"), 0.08);
        check_close("c2 p0_max_rel_dev", td(m, "concord_p0_max_rel_dev"), 0.08 / 0.70);
    }

    // ---- Case 3: same tables, relaxed thresholds -> PASS ---------------------------------
    {
        std::printf("\n-- Case 3: same tables, relaxed thresholds -> PASS --\n");
        const RunResult r = run_steppe(bin, {"readv2-concord", "--a", a2path.string(),
                                             "--b", bpath.string(),
                                             "--degree-agreement-min", "0.80",
                                             "--p0-within-tol-min", "0.80"}, tmp);
        check_eq_int("c3 exit==0 (PASS)", r.exit_code, 0);
        check_true("c3 RESULT PASS", result_pass(r.text));
    }

    // ---- Case 4: coverage miss + extra pair ----------------------------------------------
    {
        std::printf("\n-- Case 4: coverage miss + extra pair --\n");
        std::vector<Row> a = base_b();
        a.pop_back();                                 // remove SB,SC
        a.push_back({"SX", "SY", "unrelated", "1.01"});  // extra pair not in the oracle
        const std::filesystem::path apath = tmp / "case4_a.csv";
        write_table(apath, a);
        const RunResult r = run_steppe(bin, {"readv2-concord", "--a", apath.string(),
                                             "--b", bpath.string()}, tmp);
        const auto m = parse_trailer(r.text);
        check_eq_int("c4 exit==1 (FAIL at defaults)", r.exit_code, 1);
        check_eq_int("c4 common", ti(m, "concord_common_pairs"), 5);
        check_eq_int("c4 a_only", ti(m, "concord_a_only"), 1);
        check_eq_int("c4 b_only", ti(m, "concord_b_only"), 1);
        check_close("c4 coverage", td(m, "concord_coverage"), 5.0 / 6.0);
        check_close("c4 degree_agreement", td(m, "concord_degree_agreement"), 1.0);
        // Relax the coverage floor -> PASS.
        const RunResult r2 = run_steppe(bin, {"readv2-concord", "--a", apath.string(),
                                              "--b", bpath.string(),
                                              "--coverage-min", "0.80"}, tmp);
        check_eq_int("c4 relaxed exit==0 (PASS)", r2.exit_code, 0);
        check_true("c4 relaxed RESULT PASS", result_pass(r2.text));
    }

    // ---- Case 5: NaN guard ---------------------------------------------------------------
    {
        std::printf("\n-- Case 5: NaN guard (P0_norm = NA) --\n");
        std::vector<Row> a = base_b();
        a[3] = {"S7", "S8", "second", "NA"};  // NaN P0_norm; degree column intact
        const std::filesystem::path apath = tmp / "case5_a.csv";
        write_table(apath, a);
        const RunResult r = run_steppe(bin, {"readv2-concord", "--a", apath.string(),
                                             "--b", bpath.string()}, tmp);
        const auto m = parse_trailer(r.text);
        check_eq_int("c5 exit==1 (FAIL, NaN not within tol)", r.exit_code, 1);
        check_eq_int("c5 within_num", ti(m, "concord_p0_within_tol_num"), 5);
        check_eq_int("c5 within_den", ti(m, "concord_p0_within_tol_den"), 6);
        check_close("c5 degree_agreement", td(m, "concord_degree_agreement"), 1.0);
        check_true("c5 did not crash (trailer present)", m.count("concord_common_pairs") == 1);
    }

    // ---- Case 6: fail-fast input errors --------------------------------------------------
    {
        std::printf("\n-- Case 6: fail-fast input errors --\n");
        // (i) an oracle with a degree token outside the enum -> exit 2.
        std::vector<Row> bad = base_b();
        bad[1].degree = "cousin";
        const std::filesystem::path badb = tmp / "case6_badenum.csv";
        write_table(badb, bad);
        const RunResult r_enum = run_steppe(bin, {"readv2-concord", "--a", bpath.string(),
                                                  "--b", badb.string()}, tmp);
        check_eq_int("c6(i) bad enum -> exit 2", r_enum.exit_code, 2);

        // (ii) a table with a duplicate pair key -> exit 2.
        std::vector<Row> dup = base_b();
        dup.push_back({"S1", "S2", "identical", "0.30"});  // duplicate of row 0
        const std::filesystem::path dupa = tmp / "case6_dup.csv";
        write_table(dupa, dup);
        const RunResult r_dup = run_steppe(bin, {"readv2-concord", "--a", dupa.string(),
                                                 "--b", bpath.string()}, tmp);
        check_eq_int("c6(ii) duplicate key -> exit 2", r_dup.exit_code, 2);

        // (iii) a missing --a file -> exit 4.
        const RunResult r_io = run_steppe(bin, {"readv2-concord",
                                                "--a", (tmp / "nonexistent.csv").string(),
                                                "--b", bpath.string()}, tmp);
        check_eq_int("c6(iii) missing file -> exit 4", r_io.exit_code, 4);
    }

    std::filesystem::remove_all(tmp, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (validator diff arithmetic verified on 6 toy cases)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
