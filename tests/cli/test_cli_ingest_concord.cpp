// tests/cli/test_cli_ingest_concord.cpp
//
// Host-only self-checking gate for the VCF-ingest concordance validator (the
// Stage-1 block-correctness gate arithmetic). PLAIN C++ TU (NO CUDA, NO GPU): it
// writes hand-authored TOY ingest-report TSVs (the oracle schema) to a temp dir,
// runs the built `steppe ingest-concord --a .. --b ..`, parses the stable
// `iconcord_*` machine trailer + the exit code, and asserts the overall /
// ref-block-hom-ref / explicit-variant exact-match counts, coverage, the
// null-pos38 oracle-row exclusion, threshold PASS/FAIL, and the fail-fast exit
// codes against expectations KNOWN BY CONSTRUCTION. Tests the validator's OWN diff
// arithmetic — NOT science. argv[1] = the steppe binary.

#include <cstdlib>
#include <cstdio>
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

void check_eq_int(const char* what, long long got, long long want) {
    if (got != want) { std::printf("  [FAIL] %-28s got=%lld want=%lld\n", what, got, want); ++g_failures; }
}
void check_true(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

struct Row {
    std::string rs, pos38, call, dosage, source, drop_reason;
};

const std::string kHeader = "rsID\tchrom\tpos37\tpos38\tA1\tA2\tcall\tdosage\tsource\tflip\tdrop_reason";

void write_table(const std::filesystem::path& p, const std::vector<Row>& rows) {
    std::ofstream o(p, std::ios::trunc);
    o << kHeader << "\n";
    for (const Row& r : rows)
        o << r.rs << "\t1\t0\t" << r.pos38 << "\tA\tG\t" << r.call << "\t" << r.dosage << "\t"
          << r.source << "\t0\t" << r.drop_reason << "\n";
}

// Base oracle B: 6 valid-pos38 rows. refblock-homref subset = {rs1,rs2};
// variant subset = {rs3,rs4}.
std::vector<Row> base_b() {
    return {
        {"rs1", "1000", "homref", "2", "refblock", ""},
        {"rs2", "1100", "homref", "0", "refblock", ""},
        {"rs3", "1200", "het", "1", "variant", ""},
        {"rs4", "1300", "homalt", "0", "variant", ""},
        {"rs5", "1400", "missing", "NA", "refblock", "below_floor"},
        {"rs6", "1500", "dropped", "NA", "none", "palindrome"},
    };
}

struct RunResult { int exit_code = -1; std::string text; };

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "ic_stdout.txt";
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

std::map<std::string, std::string> parse_trailer(const std::string& text) {
    std::map<std::string, std::string> m;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("iconcord_", 0) != 0) continue;
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
bool result_pass(const std::string& text) { return text.find("RESULT: PASS") != std::string::npos; }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <steppe-binary>\n", argv[0]); return 2; }
    const std::string bin = argv[1];
    std::printf("=== VCF-ingest concordance-validator host-only gate (no GPU) ===\n");
    if (!std::filesystem::exists(bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", bin.c_str());
        return 1;
    }

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_iconcord_test_" + std::to_string(static_cast<long long>(
            std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const std::filesystem::path bpath = tmp / "oracle.tsv";
    write_table(bpath, base_b());

    // ---- Case 1: perfect agreement -> PASS ----------------------------------
    {
        std::printf("\n-- Case 1: perfect agreement --\n");
        const std::filesystem::path apath = tmp / "c1_a.tsv";
        write_table(apath, base_b());
        const RunResult r = run_steppe(bin, {"ingest-concord", "--a", apath.string(), "--b", bpath.string()}, tmp);
        const auto m = parse_trailer(r.text);
        check_eq_int("c1 exit==0 (PASS)", r.exit_code, 0);
        check_true("c1 RESULT PASS", result_pass(r.text));
        check_eq_int("c1 common", ti(m, "iconcord_common"), 6);
        check_eq_int("c1 match_all", ti(m, "iconcord_match_all"), 6);
        check_eq_int("c1 refblock_total", ti(m, "iconcord_refblock_total"), 2);
        check_eq_int("c1 refblock_match", ti(m, "iconcord_refblock_match"), 2);
        check_eq_int("c1 variant_total", ti(m, "iconcord_variant_total"), 2);
        check_eq_int("c1 variant_match", ti(m, "iconcord_variant_match"), 2);
    }

    // ---- Case 2: a variant-subset mismatch -> FAIL --------------------------
    {
        std::printf("\n-- Case 2: variant-subset mismatch -> FAIL --\n");
        std::vector<Row> a = base_b();
        a[2] = {"rs3", "1200", "homalt", "0", "variant", ""};  // was het/1
        const std::filesystem::path apath = tmp / "c2_a.tsv";
        write_table(apath, a);
        const RunResult r = run_steppe(bin, {"ingest-concord", "--a", apath.string(), "--b", bpath.string()}, tmp);
        const auto m = parse_trailer(r.text);
        check_eq_int("c2 exit==1 (FAIL)", r.exit_code, 1);
        check_true("c2 RESULT FAIL", !result_pass(r.text));
        check_eq_int("c2 match_all", ti(m, "iconcord_match_all"), 5);
        check_eq_int("c2 variant_match", ti(m, "iconcord_variant_match"), 1);
        check_eq_int("c2 refblock_match", ti(m, "iconcord_refblock_match"), 2);
    }

    // ---- Case 3: a ref-block hom-ref mismatch -> FAIL (novel-surface line) ---
    {
        std::printf("\n-- Case 3: ref-block hom-ref mismatch -> FAIL --\n");
        std::vector<Row> a = base_b();
        a[0] = {"rs1", "1000", "homref", "0", "refblock", ""};  // dosage 2 -> 0
        const std::filesystem::path apath = tmp / "c3_a.tsv";
        write_table(apath, a);
        const RunResult r = run_steppe(bin, {"ingest-concord", "--a", apath.string(), "--b", bpath.string()}, tmp);
        const auto m = parse_trailer(r.text);
        check_eq_int("c3 exit==1 (FAIL)", r.exit_code, 1);
        check_eq_int("c3 refblock_match", ti(m, "iconcord_refblock_match"), 1);
        check_eq_int("c3 refblock_total", ti(m, "iconcord_refblock_total"), 2);
    }

    // ---- Case 4: null-pos38 oracle row is excluded from the join ------------
    {
        std::printf("\n-- Case 4: null-pos38 oracle row excluded --\n");
        std::vector<Row> b = base_b();
        b.push_back({"rs7", "None", "dropped", "NA", "none", "no_lift"});  // lift-stage drop
        const std::filesystem::path bpath4 = tmp / "c4_b.tsv";
        write_table(bpath4, b);
        const std::filesystem::path apath = tmp / "c4_a.tsv";
        write_table(apath, base_b());  // steppe never emits rs7
        const RunResult r = run_steppe(bin, {"ingest-concord", "--a", apath.string(), "--b", bpath4.string()}, tmp);
        const auto m = parse_trailer(r.text);
        check_eq_int("c4 exit==0 (PASS)", r.exit_code, 0);
        check_eq_int("c4 b_rows (valid)", ti(m, "iconcord_b_rows"), 6);
        check_eq_int("c4 b_rows_raw", ti(m, "iconcord_b_rows_raw"), 7);
        check_eq_int("c4 common", ti(m, "iconcord_common"), 6);
        check_true("c4 coverage == 1.0", r.text.find("iconcord_coverage: 1\n") != std::string::npos);
    }

    // ---- Case 5: coverage miss (FAIL at default; PASS relaxed) --------------
    {
        std::printf("\n-- Case 5: coverage miss --\n");
        std::vector<Row> a = base_b();
        a.pop_back();  // drop rs6
        const std::filesystem::path apath = tmp / "c5_a.tsv";
        write_table(apath, a);
        const RunResult r = run_steppe(bin, {"ingest-concord", "--a", apath.string(), "--b", bpath.string()}, tmp);
        check_eq_int("c5 exit==1 (FAIL at defaults)", r.exit_code, 1);
        const RunResult r2 = run_steppe(bin, {"ingest-concord", "--a", apath.string(), "--b", bpath.string(),
                                              "--coverage-min", "0.8"}, tmp);
        check_eq_int("c5 relaxed exit==0 (PASS)", r2.exit_code, 0);
        check_true("c5 relaxed RESULT PASS", result_pass(r2.text));
    }

    // ---- Case 6: fail-fast input errors ------------------------------------
    {
        std::printf("\n-- Case 6: fail-fast input errors --\n");
        // duplicate rsID -> exit 2
        std::vector<Row> dup = base_b();
        dup.push_back({"rs1", "1000", "homref", "2", "refblock", ""});
        const std::filesystem::path dupa = tmp / "c6_dup.tsv";
        write_table(dupa, dup);
        const RunResult r_dup = run_steppe(bin, {"ingest-concord", "--a", dupa.string(), "--b", bpath.string()}, tmp);
        check_eq_int("c6 duplicate rsID -> exit 2", r_dup.exit_code, 2);
        // missing --a file -> exit 4
        const RunResult r_io = run_steppe(bin, {"ingest-concord", "--a", (tmp / "nope.tsv").string(),
                                                "--b", bpath.string()}, tmp);
        check_eq_int("c6 missing file -> exit 4", r_io.exit_code, 4);
    }

    std::filesystem::remove_all(tmp, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (validator diff arithmetic verified on 6 toy cases)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
