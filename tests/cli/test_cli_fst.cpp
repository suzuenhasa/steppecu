// tests/cli/test_cli_fst.cpp
//
// `steppe fst` per-SNP Weir & Cockerham 1984 FST acceptance gate: the built
// `steppe fst --prefix <bed> --pops A,B --method wc --per-snp` reproduces plink2's
// `--fst method=wc report-variants` per-variant table AND the genome-wide summary, ON
// THE GPU, on REAL AADR 1240K data.
//
// Alignment is on a SHARED PLINK .bed (steppe reads .bed natively; plink2 reads .bed), so
// samples/SNPs/genotypes are byte-identical by construction and REF/ALT polarity is a
// non-issue for WC. The gate is DRIVEN BY A FROZEN plink2 GOLDEN produced once on the box
// (docs: the design §7 recipe): a meta file names the bed prefix + the two pop labels
// (written into the .fam phenotype column steppe reads AND plink2's --pheno so BOTH sides
// see the identical A/B partition — critic fix #1), and the plink2 outputs are frozen as
// plink2.fst.var + plink2.fst.summary.
//
// SKIPs cleanly (exit 0) when no CUDA device is visible OR the fixture/golden is absent —
// identical to the other cli data-absent skips. NO SYNTHETIC DATA. Self-checking main().

#include <algorithm>
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

void check_true(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

void check_close(const char* what, double got, double want, double atol) {
    const double d = std::fabs(got - want);
    if (!(d <= atol)) {
        std::printf("  [FAIL] %-40s got=% .12e want=% .12e |d|=% .3e\n", what, got, want, d);
        ++g_failures;
    }
}

struct RunResult { int exit_code = -1; std::string text; };

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "cli_fst_stdout.txt";
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

std::string slurp(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool looks_like_no_gpu(const std::string& t) {
    return t.find("no CUDA device") != std::string::npos ||
           t.find("device error") != std::string::npos ||
           t.find("No CUDA") != std::string::npos;
}

std::vector<std::string> split(const std::string& line, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == sep) { out.push_back(cur); cur.clear(); }
        else if (c != '\r') cur += c;
    }
    out.push_back(cur);
    return out;
}

double cell_d(const std::string& s) {
    if (s == "NA" || s == "null" || s == "nan" || s == "-nan" || s.empty()) return std::nan("");
    return std::strtod(s.c_str(), nullptr);
}

// Detect a delimiter: tab if present, else comma, else whitespace-collapsed.
std::vector<std::string> split_auto(const std::string& line) {
    if (line.find('\t') != std::string::npos) return split(line, '\t');
    if (line.find(',') != std::string::npos) return split(line, ',');
    std::vector<std::string> out;
    std::istringstream ss(line);
    std::string tok;
    while (ss >> tok) out.push_back(tok);
    return out;
}

struct PlinkRow { double num = std::nan(""), den = std::nan(""), fst = std::nan(""); };

// Parse a plink2 .fst.var (header line starts with '#'): pull ID + WC_FST (or FST) and,
// when present, FST_NUMER / FST_DENOM. Robust to column presence/order across versions.
bool read_plink_var(const std::string& path, std::map<std::string, PlinkRow>& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    std::map<std::string, std::size_t> col;
    bool have_header = false;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (!have_header && line[0] == '#') {
            std::string h = line.substr(1);  // drop leading '#'
            const std::vector<std::string> cells = split_auto(h);
            for (std::size_t c = 0; c < cells.size(); ++c) col[cells[c]] = c;
            have_header = true;
            continue;
        }
        if (!have_header) continue;
        const std::vector<std::string> cells = split_auto(line);
        auto at = [&](const char* name) -> std::string {
            const auto it = col.find(name);
            return (it != col.end() && it->second < cells.size()) ? cells[it->second]
                                                                  : std::string();
        };
        const std::string id = at("ID");
        if (id.empty()) continue;
        PlinkRow r;
        std::string fst = at("WC_FST");
        if (fst.empty()) fst = at("FST");
        r.fst = cell_d(fst);
        r.num = cell_d(at("FST_NUMER"));
        r.den = cell_d(at("FST_DENOM"));
        out[id] = r;
    }
    return have_header;
}

// Pull the single WC FST value out of a plink2 .fst.summary (find the numeric token on the
// pop-pair data line).
bool read_plink_summary(const std::string& path, double& fst_out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    std::map<std::string, std::size_t> col;
    bool have_header = false;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (!have_header && (line[0] == '#')) {
            const std::vector<std::string> cells = split_auto(line.substr(1));
            for (std::size_t c = 0; c < cells.size(); ++c) col[cells[c]] = c;
            have_header = true;
            continue;
        }
        const std::vector<std::string> cells = split_auto(line);
        if (have_header) {
            const auto it = col.find("HUDSON_FST");
            const auto it2 = col.find("WC_FST");
            std::size_t idx = (it2 != col.end()) ? it2->second
                              : (it != col.end()) ? it->second : cells.size();
            if (idx < cells.size()) { fst_out = cell_d(cells[idx]); return !std::isnan(fst_out); }
        }
        // Fallback: last finite numeric token on the line.
        for (auto rit = cells.rbegin(); rit != cells.rend(); ++rit) {
            const double v = cell_d(*rit);
            if (!std::isnan(v)) { fst_out = v; return true; }
        }
    }
    return false;
}

std::map<std::string, PlinkRow> parse_steppe_per_snp(const std::string& text) {
    std::map<std::string, PlinkRow> out;
    std::istringstream ss(text);
    std::string line;
    std::map<std::string, std::size_t> col;
    bool header = true;
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> cells = split_auto(line);
        if (header) {
            for (std::size_t c = 0; c < cells.size(); ++c) col[cells[c]] = c;
            header = false;
            continue;
        }
        if (!col.count("snp_id") || !col.count("fst")) continue;
        auto at = [&](const char* n) -> std::string {
            const auto it = col.find(n);
            return (it != col.end() && it->second < cells.size()) ? cells[it->second]
                                                                  : std::string();
        };
        PlinkRow r;
        r.num = cell_d(at("fst_num"));
        r.den = cell_d(at("fst_den"));
        r.fst = cell_d(at("fst"));
        out[at("snp_id")] = r;
    }
    return out;
}

double parse_steppe_summary_ratio(const std::string& text) {
    std::istringstream ss(text);
    std::string line;
    std::map<std::string, std::size_t> col;
    bool header = true;
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> cells = split_auto(line);
        if (header) {
            for (std::size_t c = 0; c < cells.size(); ++c) col[cells[c]] = c;
            header = false;
            continue;
        }
        const auto it = col.find("fst_ratio");
        if (it != col.end() && it->second < cells.size()) return cell_d(cells[it->second]);
    }
    return std::nan("");
}

}  // namespace

int main(int argc, char** argv) {
    // argv[1] = steppe binary; argv[2] = golden root dir (contains fst/...).
    if (argc < 3) {
        std::printf("usage: %s <steppe-binary> <golden-dir>\n", argv[0]);
        return 2;
    }
    const std::string steppe_bin = argv[1];
    const std::string golden_dir = std::string(argv[2]) + "/fst";
    const std::string meta = golden_dir + "/aadr_fst_meta.txt";

    std::printf("=== steppe fst per-SNP WC parity vs plink2 (--fst method=wc; ON THE GPU) ===\n");
    std::printf("steppe binary: %s\ngolden dir:    %s\n", steppe_bin.c_str(), golden_dir.c_str());

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", steppe_bin.c_str());
        return 1;
    }
    // The frozen-golden gate needs a meta file (prefix + pop labels) and the plink2 outputs.
    std::ifstream mf(meta);
    if (!mf) {
        std::printf("\nRESULT: SKIP (plink2 FST golden meta absent: %s — produce it on the box "
                    "via the design §7 recipe)\n", meta.c_str());
        return 0;
    }
    std::string bed_prefix, popA, popB;
    std::getline(mf, bed_prefix);
    std::getline(mf, popA);
    std::getline(mf, popB);
    auto trim = [](std::string& s) {
        while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\n')) s.pop_back();
    };
    trim(bed_prefix); trim(popA); trim(popB);
    if (bed_prefix.empty() || popA.empty() || popB.empty()) {
        std::printf("\nRESULT: SKIP (meta incomplete: need prefix / popA / popB)\n");
        return 0;
    }
    if (!std::filesystem::exists(bed_prefix + ".bed")) {
        std::printf("\nRESULT: SKIP (bed fixture absent: %s.bed)\n", bed_prefix.c_str());
        return 0;
    }
    std::map<std::string, PlinkRow> plink;
    if (!read_plink_var(golden_dir + "/plink2.fst.var", plink) || plink.empty()) {
        std::printf("\nRESULT: SKIP (plink2 .fst.var golden absent/empty)\n");
        return 0;
    }
    double plink_summary = std::nan("");
    const bool have_summary =
        read_plink_summary(golden_dir + "/plink2.fst.summary", plink_summary);

    std::error_code ec;
    const std::filesystem::path tmp = std::filesystem::temp_directory_path(ec) /
        ("steppe_cli_fst_" + std::to_string(static_cast<long long>(
            std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const std::string pops = popA + "," + popB;

    // ---- per-SNP table (TSV) ----  (write to --out so the stderr summary line does not
    // pollute the parsed table).
    const std::filesystem::path ps_out = tmp / "steppe_fst.tsv";
    const RunResult ps = run_steppe(steppe_bin,
        {"fst", "--prefix", bed_prefix, "--pops", pops, "--method", "wc", "--per-snp",
         "--format", "tsv", "--device", "0", "--out", ps_out.string()}, tmp);
    if (looks_like_no_gpu(ps.text)) {
        std::filesystem::remove_all(tmp, ec);
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;
    }
    check_true("per-snp exit 0", ps.exit_code == 0);
    const std::map<std::string, PlinkRow> got = parse_steppe_per_snp(slurp(ps_out));
    check_true("steppe emitted per-SNP rows", !got.empty());

    long compared = 0, num_cmp = 0;
    double max_abs_fst = 0.0, max_abs_num = 0.0, max_abs_den = 0.0;
    for (const auto& [id, pr] : plink) {
        const auto it = got.find(id);
        if (it == got.end()) continue;
        const PlinkRow& sr = it->second;
        // Compare only where BOTH sides have a finite WC FST (both mark the site valid).
        if (std::isnan(pr.fst) || std::isnan(sr.fst)) continue;
        ++compared;
        max_abs_fst = std::max(max_abs_fst, std::fabs(pr.fst - sr.fst));
        if (!std::isnan(pr.num) && !std::isnan(sr.num) && !std::isnan(pr.den) &&
            !std::isnan(sr.den)) {
            ++num_cmp;
            max_abs_num = std::max(max_abs_num, std::fabs(pr.num - sr.num));
            max_abs_den = std::max(max_abs_den, std::fabs(pr.den - sr.den));
        }
    }
    std::printf("joined valid sites: %ld  (num/den compared: %ld)\n", compared, num_cmp);
    std::printf("max|d| WC_FST=%.3e  FST_NUMER=%.3e  FST_DENOM=%.3e\n",
                max_abs_fst, max_abs_num, max_abs_den);
    check_true("at least 100 sites compared", compared >= 100);
    check_close("max abs diff WC_FST < 1e-6", max_abs_fst, 0.0, 1e-6);
    if (num_cmp > 0) {
        check_close("max abs diff FST_NUMER < 1e-6", max_abs_num, 0.0, 1e-6);
        check_close("max abs diff FST_DENOM < 1e-6", max_abs_den, 0.0, 1e-6);
    }

    // ---- genome-wide summary ----
    const std::filesystem::path sm_out = tmp / "steppe_sum.tsv";
    const RunResult sm = run_steppe(steppe_bin,
        {"fst", "--prefix", bed_prefix, "--pops", pops, "--method", "wc",
         "--format", "tsv", "--device", "0", "--out", sm_out.string()}, tmp);
    check_true("summary exit 0", sm.exit_code == 0);
    const double steppe_ratio = parse_steppe_summary_ratio(slurp(sm_out));
    check_true("steppe summary fst_ratio finite", !std::isnan(steppe_ratio));
    if (have_summary) {
        std::printf("summary: steppe=%.9f  plink2=%.9f\n", steppe_ratio, plink_summary);
        check_close("summary WC FST vs plink2 .fst.summary < 1e-6", steppe_ratio, plink_summary,
                    1e-6);
    }

    std::filesystem::remove_all(tmp, ec);
    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (per-SNP WC FST + summary reproduce plink2 within 1e-6)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
