// tests/cli/test_cli_pca.cpp
//
// `steppe pca` acceptance gate: the built `steppe pca --prefix <bed> --pops ... -k K` on the
// GPU reproduces a FROZEN scikit-allel `allel.pca(scaler='patterson')` golden on REAL AADR
// data, AND emits a self-contained interactive HTML artifact.
//
// PCA eigenvectors carry an arbitrary sign and ROTATE arbitrarily within near-degenerate
// eigenvalue subspaces, so the PRIMARY gate is a rotation- AND sign-invariant subspace check:
// the sample x sample Gram matrix S*S^T of the top-K coordinates (== the rank-K covariance
// reconstruction, coords = U*S) must match the oracle's A*A^T. As a secondary, data-driven
// check the leading well-separated PCs (PC1/PC2, and any PC whose eigenvalue gap is large)
// must agree per-PC sign-aligned (|Pearson r| > 0.999); the full |r| vector + var_explained
// ratios are reported. NOT a naive all-PCs equality (critic fix #1).
//
// The golden + meta (bed prefix + pops + K) are frozen once on the box by
// tests/reference/pca_allel_gate.py. SKIPs cleanly (exit 0) when no CUDA device is visible OR
// the fixture / golden / meta is absent. NO SYNTHETIC DATA. Self-checking main().
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

struct RunResult { int exit_code = -1; std::string text; };

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "cli_pca_stdout.txt";
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
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}
bool looks_like_no_gpu(const std::string& t) {
    return t.find("no CUDA device") != std::string::npos ||
           t.find("device error") != std::string::npos;
}
std::vector<std::string> split(const std::string& line, char sep) {
    std::vector<std::string> out; std::string cur;
    for (char c : line) { if (c == sep) { out.push_back(cur); cur.clear(); }
                          else if (c != '\r') cur += c; }
    out.push_back(cur); return out;
}
std::vector<std::string> split_auto(const std::string& line) {
    if (line.find('\t') != std::string::npos) return split(line, '\t');
    if (line.find(',') != std::string::npos) return split(line, ',');
    std::vector<std::string> out; std::istringstream ss(line); std::string tok;
    while (ss >> tok) out.push_back(tok);
    return out;
}
double cell_d(const std::string& s) {
    if (s.empty() || s == "NA" || s == "null" || s == "nan" || s == "-nan") return std::nan("");
    return std::strtod(s.c_str(), nullptr);
}

// Parse a coord table (header: sample [pop] PC1..PCK) -> sample -> coord vector (K long).
std::map<std::string, std::vector<double>> parse_coords(const std::string& text, int& K_out) {
    std::map<std::string, std::vector<double>> out;
    std::istringstream ss(text); std::string line;
    std::vector<std::size_t> pc_cols; std::size_t sample_col = 0;
    bool header = true; K_out = 0;
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> c = split_auto(line);
        if (header) {
            for (std::size_t i = 0; i < c.size(); ++i) {
                if (c[i] == "sample" || c[i] == "IID" || c[i] == "id") sample_col = i;
                if (c[i].rfind("PC", 0) == 0) pc_cols.push_back(i);
            }
            K_out = static_cast<int>(pc_cols.size());
            header = false; continue;
        }
        if (sample_col >= c.size()) continue;
        std::vector<double> v; v.reserve(pc_cols.size());
        for (std::size_t col : pc_cols) v.push_back(col < c.size() ? cell_d(c[col]) : std::nan(""));
        out[c[sample_col]] = std::move(v);
    }
    return out;
}

double abs_pearson(const std::vector<double>& a, const std::vector<double>& b) {
    const std::size_t n = a.size();
    double ma = 0, mb = 0;
    for (std::size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double sab = 0, saa = 0, sbb = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        sab += da * db; saa += da * da; sbb += db * db;
    }
    if (saa <= 0 || sbb <= 0) return 1.0;
    return std::fabs(sab / std::sqrt(saa * sbb));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: %s <steppe-binary> <golden-dir>\n", argv[0]); return 2; }
    const std::string steppe_bin = argv[1];
    const std::string golden_dir = std::string(argv[2]) + "/pca";
    const std::string meta = golden_dir + "/aadr_pca_meta.txt";

    std::printf("=== steppe pca vs scikit-allel (Patterson PCA; ON THE GPU) ===\n");
    std::printf("steppe binary: %s\ngolden dir:    %s\n", steppe_bin.c_str(), golden_dir.c_str());

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found)\n"); return 1;
    }
    std::ifstream mf(meta);
    if (!mf) {
        std::printf("\nRESULT: SKIP (allel PCA golden meta absent: %s — freeze it on the box "
                    "via tests/reference/pca_allel_gate.py)\n", meta.c_str());
        return 0;
    }
    std::string bed_prefix, pops_line, k_line;
    std::getline(mf, bed_prefix); std::getline(mf, pops_line); std::getline(mf, k_line);
    auto trim = [](std::string& s) {
        while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\n')) s.pop_back();
    };
    trim(bed_prefix); trim(pops_line); trim(k_line);
    const int K_meta = k_line.empty() ? 10 : std::atoi(k_line.c_str());
    if (bed_prefix.empty() || !std::filesystem::exists(bed_prefix + ".bed")) {
        std::printf("\nRESULT: SKIP (bed fixture absent: %s.bed)\n", bed_prefix.c_str());
        return 0;
    }
    int K_allel = 0;
    const auto allel = parse_coords(slurp(golden_dir + "/allel_coords.tsv"), K_allel);
    if (allel.empty() || K_allel <= 0) {
        std::printf("\nRESULT: SKIP (allel_coords.tsv golden absent/empty)\n"); return 0;
    }
    // Frozen allel var_explained (scree table: pc_index / eigenvalue / var_explained).
    std::vector<double> allel_ve;
    {
        std::istringstream ss(slurp(golden_dir + "/allel_eigenvalues.tsv"));
        std::string line; bool header = true; std::size_t ve_col = 2;
        while (std::getline(ss, line)) {
            if (line.empty() || line[0] == '#') continue;
            const auto c = split_auto(line);
            if (header) { for (std::size_t i = 0; i < c.size(); ++i)
                              if (c[i] == "var_explained") ve_col = i;
                          header = false; continue; }
            if (ve_col < c.size()) allel_ve.push_back(cell_d(c[ve_col]));
        }
    }

    std::error_code ec;
    const std::filesystem::path tmp = std::filesystem::temp_directory_path(ec) /
        ("steppe_cli_pca_" + std::to_string(static_cast<long long>(
            std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const int K = std::min(K_meta, K_allel);
    const std::filesystem::path co = tmp / "steppe_pca.tsv";
    const std::filesystem::path ho = tmp / "steppe_pca.html";
    std::vector<std::string> args = {"pca", "--prefix", bed_prefix, "-k", std::to_string(K),
                                     "--maf", "0", "--format", "tsv", "--device", "0",
                                     "--out", co.string(), "--emit-html", ho.string()};
    if (!pops_line.empty() && pops_line != "ALL") { args.push_back("--pops"); args.push_back(pops_line); }
    const RunResult rr = run_steppe(steppe_bin, args, tmp);
    if (looks_like_no_gpu(rr.text)) {
        std::filesystem::remove_all(tmp, ec);
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n"); return 0;
    }
    check_true("pca exit 0", rr.exit_code == 0);

    int K_steppe = 0;
    const auto steppe = parse_coords(slurp(co), K_steppe);
    check_true("steppe emitted coord rows", !steppe.empty());
    check_true("steppe has K PC columns", K_steppe >= K);

    // Join on sample id, build the common-sample coordinate matrices (K components).
    std::vector<std::vector<double>> S, A;  // per-sample K-vectors
    for (const auto& [id, sv] : steppe) {
        const auto it = allel.find(id);
        if (it == allel.end()) continue;
        std::vector<double> s(K), a(K); bool ok = true;
        for (int k = 0; k < K; ++k) {
            if (k >= static_cast<int>(sv.size()) || k >= static_cast<int>(it->second.size()) ||
                std::isnan(sv[k]) || std::isnan(it->second[k])) { ok = false; break; }
            s[k] = sv[k]; a[k] = it->second[k];
        }
        if (ok) { S.push_back(std::move(s)); A.push_back(std::move(a)); }
    }
    const std::size_t n = S.size();
    std::printf("joined samples: %zu (K=%d)\n", n, K);
    check_true("joined >= 20 samples", n >= 20);
    if (n < 2) {
        std::filesystem::remove_all(tmp, ec);
        std::printf("\nRESULT: FAIL (too few joined samples)\n"); return 1;
    }

    // PRIMARY: rotation- & sign-invariant subspace check — the sample x sample Gram of the
    // top-K coords (== rank-K covariance reconstruction) must match the oracle's.
    double max_g = 0.0, sq_g = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double gs = 0, ga = 0;
            for (int k = 0; k < K; ++k) { gs += S[i][k] * S[j][k]; ga += A[i][k] * A[j][k]; }
            max_g = std::max(max_g, std::fabs(gs - ga));
            sq_g += ga * ga;
        }
    const double rms_g = std::sqrt(sq_g / (static_cast<double>(n) * n));
    std::printf("subspace Gram: max|dG|=%.3e  rms(G_allel)=%.3e  rel=%.3e\n",
                max_g, rms_g, max_g / (rms_g + 1e-30));
    check_true("top-K subspace matches oracle (Gram rel < 5e-3)", max_g < 5e-3 * (rms_g + 1e-30));

    // SECONDARY: per-PC sign-aligned |r| (report all; hard-gate the well-separated leaders).
    std::vector<double> eg;  // relative eigenvalue gaps from the frozen allel var_explained
    for (int k = 0; k < K; ++k) {
        std::vector<double> sc(n), ac(n);
        for (std::size_t i = 0; i < n; ++i) { sc[i] = S[i][k]; ac[i] = A[i][k]; }
        const double r = abs_pearson(sc, ac);
        const double ve = (k < static_cast<int>(allel_ve.size())) ? allel_ve[k] : 0.0;
        const double ve_next = (k + 1 < static_cast<int>(allel_ve.size())) ? allel_ve[k + 1] : 0.0;
        const double gap = (ve > 0) ? (ve - ve_next) / ve : 0.0;
        const bool well_sep = (k < 2) || (gap > 0.05);
        std::printf("  PC%-2d |r|=%.6f  var_explained=%.5f  gap=%.3f  %s\n",
                    k + 1, r, ve, gap, well_sep ? "[gated>0.999]" : "[report-only]");
        if (well_sep) {
            char lbl[48]; std::snprintf(lbl, sizeof(lbl), "PC%d |r|>0.999 (well-separated)", k + 1);
            check_true(lbl, r > 0.999);
        }
    }

    // HTML self-containment (critic fix #3): the artifact must have the scatter + legend +
    // embedded coord blob AND make NO network request. The writer is canvas-2D only (no inline
    // SVG), so it emits no scheme/URL at all — we assert zero network-egress vectors.
    const std::string html = slurp(ho);
    check_true("html non-empty", !html.empty());
    check_true("html has <canvas scatter", html.find("<canvas") != std::string::npos);
    check_true("html has legend", html.find("id=\"legend\"") != std::string::npos);
    check_true("html embeds coord blob (const DATA)", html.find("const DATA") != std::string::npos);
    const char* egress[] = {"http://", "https://", "//cdn", "<link", "@import",
                            "fetch(", "XMLHttpRequest", "WebSocket", "src=\"http"};
    for (const char* v : egress) {
        char lbl[64]; std::snprintf(lbl, sizeof(lbl), "html has NO network egress: %s", v);
        check_true(lbl, html.find(v) == std::string::npos);
    }

    std::filesystem::remove_all(tmp, ec);
    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (steppe PCA reproduces scikit-allel; self-contained HTML)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
