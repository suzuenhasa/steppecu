// tests/cli/test_cli_fst_all_pairs.cpp
//
// `steppe fst --all-pairs` acceptance gate: the P x P WC FST matrix INHERITS the single-pair
// path's plink2 parity. The built `steppe fst --all-pairs --pops A,B ...` matrix cell (A,B)
// must equal the plink2-gated single-pair `steppe fst --pops A,B` genome-wide fst_ratio on
// REAL AADR data, ON THE GPU — so the matrix carries the same plink2-bit-exact WC parity the
// single-pair path is separately gated on (test_cli_fst.cpp). Also checks the matrix is
// symmetric with a zero diagonal.
//
// Reuses the frozen fixture/meta (aadr_fst_meta.txt: bed prefix + two pop labels, plus an
// OPTIONAL 3rd label on line 4). With the 3rd pop the matrix is 3x3, so EVERY off-diagonal
// cell is checked against its single-pair ratio -- this exercises the on-device
// readv2_unrank_pair for pair ranks r=0,1,2 and the flat-rank -> (row,col) placement, not just
// the r=0 corner a 2-pop matrix would give. Runs wherever test_cli_fst runs. The per-pair
// per-SNP num/den are bit-exact BY CONSTRUCTION (the matrix path and the single-pair path call
// the SAME shared wc_finalize on integer-exact {n,ac,het}); the genome-wide ratio agrees to
// FP64 round-off (the two reductions differ in summation order), so each cell is gated on an
// ABSOLUTE FST tolerance (critic fix: |cell - single| < 1e-9), not a relative tolerance on the
// near-zero ratio.
//
// SKIPs cleanly (exit 0) when no CUDA device is visible OR the fixture/meta is absent. Self-
// checking main(). NO SYNTHETIC DATA.

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

struct RunResult { int exit_code = -1; std::string text; };

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args) {
    RunResult rr;
    std::string cmd = "\"" + bin + "\"";
    for (const std::string& a : args) cmd += " \"" + a + "\"";
    cmd += " 2>&1";
    std::string out;
    if (FILE* p = popen(cmd.c_str(), "r")) {
        char buf[4096];
        std::size_t n = 0;
        while ((n = std::fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, n);
        const int st = pclose(p);
        rr.exit_code =
#ifdef WIFEXITED
            WIFEXITED(st) ? WEXITSTATUS(st) : -1;
#else
            st;
#endif
    }
    rr.text = out;
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

std::vector<std::string> split_auto(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    const char sep = (line.find('\t') != std::string::npos) ? '\t'
                     : (line.find(',') != std::string::npos) ? ',' : '\0';
    if (sep == '\0') {
        std::istringstream ss(line);
        std::string tok;
        while (ss >> tok) out.push_back(tok);
        return out;
    }
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

// Parse the emitted matrix (header: pop <sep> L0 <sep> L1 ...; then one labeled row per pop)
// into label -> row-index and a dense P*P value array.
struct Matrix {
    std::vector<std::string> labels;
    std::map<std::string, std::size_t> idx;
    std::vector<double> v;  // P*P row-major
    std::size_t P = 0;
    double at(const std::string& a, const std::string& b) const {
        return v[idx.at(a) * P + idx.at(b)];
    }
};

bool parse_matrix(const std::string& text, Matrix& m) {
    std::istringstream ss(text);
    std::string line;
    bool header = true;
    std::vector<std::vector<double>> rows;
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> cells = split_auto(line);
        if (cells.size() < 2) continue;
        if (header) {
            for (std::size_t c = 1; c < cells.size(); ++c) {
                m.idx[cells[c]] = m.labels.size();
                m.labels.push_back(cells[c]);
            }
            header = false;
            continue;
        }
        std::vector<double> row;
        for (std::size_t c = 1; c < cells.size(); ++c) row.push_back(cell_d(cells[c]));
        rows.push_back(row);
    }
    m.P = m.labels.size();
    if (m.P == 0 || rows.size() != m.P) return false;
    m.v.assign(m.P * m.P, std::nan(""));
    for (std::size_t i = 0; i < m.P; ++i) {
        if (rows[i].size() != m.P) return false;
        for (std::size_t j = 0; j < m.P; ++j) m.v[i * m.P + j] = rows[i][j];
    }
    return true;
}

double parse_single_ratio(const std::string& text) {
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
    if (argc < 3) {
        std::printf("usage: %s <steppe-binary> <golden-dir>\n", argv[0]);
        return 2;
    }
    const std::string steppe_bin = argv[1];
    const std::string golden_dir = std::string(argv[2]) + "/fst";
    const std::string meta = golden_dir + "/aadr_fst_meta.txt";

    std::printf("=== steppe fst --all-pairs matrix inherits single-pair parity (ON THE GPU) ===\n");

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", steppe_bin.c_str());
        return 1;
    }
    std::ifstream mf(meta);
    if (!mf) {
        std::printf("\nRESULT: SKIP (fst meta absent: %s)\n", meta.c_str());
        return 0;
    }
    std::string bed_prefix, popA, popB, popC;
    std::getline(mf, bed_prefix);
    std::getline(mf, popA);
    std::getline(mf, popB);
    std::getline(mf, popC);  // OPTIONAL 3rd pop: lifts the matrix to 3x3 so the on-device
                             // readv2_unrank_pair fires for pair ranks r=0,1,2 and the flat->
                             // (row,col) cell placement (not just the r=0 corner) is gated.
    auto trim = [](std::string& s) {
        while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\n')) s.pop_back();
    };
    trim(bed_prefix); trim(popA); trim(popB); trim(popC);
    if (bed_prefix.empty() || popA.empty() || popB.empty() || popA == popB) {
        std::printf("\nRESULT: SKIP (meta incomplete: need prefix + two distinct pops)\n");
        return 0;
    }
    // Build the pop selection (2 or 3 distinct pops). A 3-pop matrix is the meaningful gate:
    // it exercises the multi-pair on-device unrank + nontrivial off-diagonal placement.
    std::vector<std::string> sel = {popA, popB};
    if (!popC.empty() && popC != popA && popC != popB) sel.push_back(popC);
    std::string pops;
    for (std::size_t i = 0; i < sel.size(); ++i) { if (i) pops += ","; pops += sel[i]; }
    if (!std::filesystem::exists(bed_prefix + ".bed")) {
        std::printf("\nRESULT: SKIP (bed fixture absent: %s.bed)\n", bed_prefix.c_str());
        return 0;
    }

    std::error_code ec;
    const std::filesystem::path tmp = std::filesystem::temp_directory_path(ec) /
        ("steppe_cli_fst_ap_" + std::to_string(static_cast<long long>(
            std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const std::filesystem::path mat_out = tmp / "matrix.tsv";
    const RunResult mr = run_steppe(steppe_bin,
        {"fst", "--all-pairs", "--prefix", bed_prefix, "--pops", pops, "--method", "wc",
         "--maf", "0", "--format", "tsv", "--device", "0", "--out", mat_out.string()});
    if (looks_like_no_gpu(mr.text)) {
        std::filesystem::remove_all(tmp, ec);
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;
    }
    check_true("all-pairs exit 0", mr.exit_code == 0);
    std::printf("matrix over %zu pops: %s\n", sel.size(), pops.c_str());

    Matrix m;
    check_true("matrix parsed", parse_matrix(slurp(mat_out), m));
    bool have_all = m.P >= sel.size();
    for (const std::string& s : sel) have_all = have_all && m.idx.count(s) != 0;
    if (have_all) {
        // Symmetry + zero diagonal over the whole emitted matrix.
        for (std::size_t i = 0; i < m.P; ++i) {
            check_true("zero diagonal", m.v[i * m.P + i] == 0.0);
            for (std::size_t j = 0; j < m.P; ++j) {
                const double a = m.v[i * m.P + j], b = m.v[j * m.P + i];
                check_true("symmetric", a == b || (std::isnan(a) && std::isnan(b)));
            }
        }
        // Parity: EVERY selected pair's matrix cell == its plink2-gated single-pair ratio.
        // With 3 pops this covers all 3 off-diagonal cells, so a mis-placed on-device unrank
        // (cell(A,C) carrying pair(B,C)'s accumulation, say) would mismatch here.
        for (std::size_t a = 0; a < sel.size(); ++a) {
            for (std::size_t b = a + 1; b < sel.size(); ++b) {
                const double cell = m.at(sel[a], sel[b]);
                const std::string pr = sel[a] + "," + sel[b];
                const std::filesystem::path sp_out =
                    tmp / (std::string("single_") + std::to_string(a) + "_" + std::to_string(b) + ".tsv");
                const RunResult sp = run_steppe(steppe_bin,
                    {"fst", "--prefix", bed_prefix, "--pops", pr, "--method", "wc",
                     "--maf", "0", "--format", "tsv", "--device", "0", "--out", sp_out.string()});
                check_true("single-pair exit 0", sp.exit_code == 0);
                const double single = parse_single_ratio(slurp(sp_out));
                std::printf("  cell(%s,%s)=%.12f  single-pair=%.12f  |d|=%.3e\n",
                            sel[a].c_str(), sel[b].c_str(), cell, single, std::fabs(cell - single));
                check_true("cell finite", !std::isnan(cell));
                check_true("single finite", !std::isnan(single));
                // Absolute FST tolerance (order-of-summation only; per-SNP num/den bit-exact).
                check_true("matrix cell == single-pair ratio (|d| < 1e-9)",
                           std::fabs(cell - single) < 1e-9);
            }
        }
    } else {
        std::printf("  [note] %zux%zu matrix missing a selected pop; parity check needs all present\n",
                    m.P, m.P);
        check_true("matrix has all selected pops", have_all);
    }

    std::filesystem::remove_all(tmp, ec);
    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (all-pairs matrix cell == single-pair within 1e-9; symmetric)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
