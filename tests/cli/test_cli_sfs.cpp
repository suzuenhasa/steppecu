// tests/cli/test_cli_sfs.cpp
//
// `steppe sfs` 2D joint site-frequency spectrum acceptance gate: the built
// `steppe sfs --prefix <bed> --pops A,B [--fold|--unfold]` reproduces scikit-allel's
// joint_sfs / joint_sfs_folded CELL-BY-CELL BIT-EXACT, ON THE GPU, on REAL AADR 1240K data.
//
// A pure INTEGER-count stat, so the gate is exact integer equality (not a float tolerance).
// It is DRIVEN BY FROZEN scikit-allel GOLDENS produced once on the box (design §9 recipe):
// a meta file (aadr_sfs_meta.txt) names the shared 2-pop .bed prefix + the two pop labels,
// and the two oracle matrices are frozen as joint_sfs_unfolded.tsv / joint_sfs_folded.tsv
// (row-major, row axis = popA). The bridge builds both matrices on the SAME individuals +
// SNPs + complete-data restriction + A1-aligned counted allele, so cell (i,j) of steppe row
// i (== popA) matches scikit axis0 i with no transpose.
//
// SKIPs cleanly (exit 0) when no CUDA device is visible OR the fixture/golden is absent —
// identical to the other cli data-absent skips. NO SYNTHETIC DATA. Self-checking main().

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

int g_failures = 0;

void check_true(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

struct RunResult { int exit_code = -1; std::string text; };

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "cli_sfs_stdout.txt";
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

// Parse a whitespace/comma/tab-separated integer matrix, skipping blank and '#'-comment
// lines. Each non-comment line is one matrix row.
struct Matrix {
    std::vector<std::vector<long long>> rows;
    bool ok = false;
    long ncol() const { return rows.empty() ? 0 : static_cast<long>(rows.front().size()); }
    long nrow() const { return static_cast<long>(rows.size()); }
};

Matrix parse_matrix(const std::string& text) {
    Matrix m;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        // strip CR
        while (!line.empty() && (line.back() == '\r')) line.pop_back();
        if (line.empty()) continue;
        // leading-whitespace check for a comment
        std::size_t p = line.find_first_not_of(" \t");
        if (p == std::string::npos) continue;
        if (line[p] == '#') continue;
        // tokenize on tab/comma/space
        std::vector<long long> row;
        std::string tok;
        for (char c : line) {
            if (c == '\t' || c == ',' || c == ' ') {
                if (!tok.empty()) { row.push_back(std::strtoll(tok.c_str(), nullptr, 10)); tok.clear(); }
            } else {
                tok += c;
            }
        }
        if (!tok.empty()) row.push_back(std::strtoll(tok.c_str(), nullptr, 10));
        if (!row.empty()) m.rows.push_back(std::move(row));
    }
    m.ok = !m.rows.empty();
    return m;
}

Matrix parse_matrix_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return Matrix{};
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_matrix(ss.str());
}

std::string slurp(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Compare two integer matrices cell-by-cell; report dims + first mismatch. Returns true iff
// identical shape AND every cell equal.
bool matrices_bit_exact(const char* tag, const Matrix& got, const Matrix& want) {
    if (got.nrow() != want.nrow() || got.ncol() != want.ncol()) {
        std::printf("  [FAIL] %s dims: steppe=%ldx%ld golden=%ldx%ld\n", tag, got.nrow(),
                    got.ncol(), want.nrow(), want.ncol());
        return false;
    }
    long long mism = 0;
    long fi = -1, fj = -1;
    long long fg = 0, fw = 0;
    for (long i = 0; i < got.nrow(); ++i) {
        for (long j = 0; j < got.ncol(); ++j) {
            if (got.rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] !=
                want.rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]) {
                if (mism == 0) {
                    fi = i; fj = j;
                    fg = got.rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                    fw = want.rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                }
                ++mism;
            }
        }
    }
    if (mism != 0) {
        std::printf("  [FAIL] %s: %lld mismatched cells; first (%ld,%ld) steppe=%lld golden=%lld\n",
                    tag, mism, fi, fj, fg, fw);
        return false;
    }
    std::printf("  [OK]   %s bit-exact: %ldx%ld, all cells equal\n", tag, got.nrow(), got.ncol());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // argv[1] = steppe binary; argv[2] = golden root dir (contains sfs/...).
    if (argc < 3) {
        std::printf("usage: %s <steppe-binary> <golden-dir>\n", argv[0]);
        return 2;
    }
    const std::string steppe_bin = argv[1];
    const std::string golden_dir = std::string(argv[2]) + "/sfs";
    const std::string meta = golden_dir + "/aadr_sfs_meta.txt";

    std::printf("=== steppe sfs 2D joint SFS bit-exact vs scikit-allel (ON THE GPU) ===\n");
    std::printf("steppe binary: %s\ngolden dir:    %s\n", steppe_bin.c_str(), golden_dir.c_str());

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", steppe_bin.c_str());
        return 1;
    }
    std::ifstream mf(meta);
    if (!mf) {
        std::printf("\nRESULT: SKIP (scikit-allel SFS golden meta absent: %s — produce it on the "
                    "box via the design §9 recipe)\n", meta.c_str());
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
    if (!std::filesystem::exists(bed_prefix + ".bed") &&
        !std::filesystem::exists(bed_prefix + ".geno")) {
        std::printf("\nRESULT: SKIP (genotype fixture absent: %s.{bed,geno})\n", bed_prefix.c_str());
        return 0;
    }
    const Matrix gold_unf = parse_matrix_file(golden_dir + "/joint_sfs_unfolded.tsv");
    const Matrix gold_fold = parse_matrix_file(golden_dir + "/joint_sfs_folded.tsv");
    if (!gold_unf.ok || !gold_fold.ok) {
        std::printf("\nRESULT: SKIP (scikit-allel joint_sfs golden matrices absent/empty)\n");
        return 0;
    }

    std::error_code ec;
    const std::filesystem::path tmp = std::filesystem::temp_directory_path(ec) /
        ("steppe_cli_sfs_" + std::to_string(static_cast<long long>(
            std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const std::string pops = popA + "," + popB;

    // ---- unfolded ----
    const std::filesystem::path unf_out = tmp / "steppe_sfs_unfolded.tsv";
    const RunResult ru = run_steppe(steppe_bin,
        {"sfs", "--prefix", bed_prefix, "--pops", pops, "--unfold", "--format", "tsv",
         "--device", "0", "--out", unf_out.string()}, tmp);
    if (looks_like_no_gpu(ru.text)) {
        std::filesystem::remove_all(tmp, ec);
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;
    }
    check_true("unfolded exit 0", ru.exit_code == 0);
    const Matrix got_unf = parse_matrix(slurp(unf_out));
    check_true("steppe emitted unfolded matrix", got_unf.ok);
    check_true("unfolded bit-exact vs scikit-allel joint_sfs",
               matrices_bit_exact("unfolded", got_unf, gold_unf));

    // ---- folded ----
    const std::filesystem::path fold_out = tmp / "steppe_sfs_folded.tsv";
    const RunResult rf = run_steppe(steppe_bin,
        {"sfs", "--prefix", bed_prefix, "--pops", pops, "--fold", "--format", "tsv",
         "--device", "0", "--out", fold_out.string()}, tmp);
    check_true("folded exit 0", rf.exit_code == 0);
    const Matrix got_fold = parse_matrix(slurp(fold_out));
    check_true("steppe emitted folded matrix", got_fold.ok);
    check_true("folded bit-exact vs scikit-allel joint_sfs_folded",
               matrices_bit_exact("folded", got_fold, gold_fold));

    std::filesystem::remove_all(tmp, ec);
    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (2D joint SFS folded + unfolded reproduce scikit-allel "
                    "cell-by-cell)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
