// tests/cli/test_cli_qpfstats.cpp
//
// qpfstats acceptance gate: the `steppe qpfstats --prefix <geno> --pops <9> --out-dir <dir>`
// CLI reproduces the AT2 GENOTYPE-PATH smoothed-f2 TENSOR golden
// (golden_qpfstats_geno.csv, 9×9×711 upper-tri incl diag = 31995 rows) THROUGH the CLI, ON
// THE GPU. The golden was made via admixtools::qpfstats(pref = the raw HO TGENO prefix, the 9
// fit0 pops, include_f2/f3/f4=TRUE) -> a 9×9×711 R array (the smoothed per-block f2 tensor,
// symmetric, diag=0). steppe reads the SAME genotype prefix directly through run_qpfstats (the
// qpDstat-B decode front-end + the dstat-numerator engine over the full f2/f3/f4 popcomb set +
// the on-device shared-factor smoothing solve + scatter/recenter), writes a smoothed f2 dir,
// and this test reads its f2.bin and diffs every (i,j,block) entry vs the golden.
//
// TIER: rtol 1e-6 (atol 1e-9). The numerator path (dstat kernel) matches the qpDstat-B
// genotype golden to ~1e-15; the smoothing solve adds a 36×36 SPD Cholesky + a 711-col Dtrsm
// (EmulatedFp64 syrk/gemm + native potrf/trsm carve-out) — comfortably inside 1e-6.
//
// PLAIN C++ host TU (NO CUDA header): it spawns the steppe binary; the GPU work happens inside
// that child. SKIPs cleanly (exit 0) when no CUDA device is visible OR the prefix / golden is
// absent. NO SYNTHETIC DATA (memory real-data-only). Self-checking main(); CTest gates on exit.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    if (diff > tol) {
        if (g_failures < 30)
            std::printf("  [FAIL] %-30s got=% .10e want=% .10e |d|=% .3e tol=% .3e\n",
                        what, got, want, diff, tol);
        ++g_failures;
    }
}

void check_true(const char* what, bool ok) {
    if (!ok) std::printf("  [FAIL] %s\n", what);
    if (!ok) ++g_failures;
}

struct RunResult { int exit_code = -1; std::string text; };

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "cli_qpfstats_stdout.txt";
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

// One golden row: (i,j,block) -> smoothed f2.
struct GRow { int i, j, block; double f2; };

bool read_golden(const std::string& path, std::vector<GRow>& out, int& max_block, int& max_pop) {
    std::ifstream f(path);
    if (!f) { std::printf("  [SKIP] golden qpfstats csv absent: %s\n", path.c_str()); return false; }
    std::string line;
    bool header = true;
    max_block = -1; max_pop = -1;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (header) { header = false; continue; }  // "i","j","block","f2"
        // strip quotes
        std::string s;
        for (char c : line) if (c != '"') s += c;
        std::stringstream ss(s);
        std::string cell;
        GRow r;
        std::getline(ss, cell, ','); r.i = std::atoi(cell.c_str());
        std::getline(ss, cell, ','); r.j = std::atoi(cell.c_str());
        std::getline(ss, cell, ','); r.block = std::atoi(cell.c_str());
        std::getline(ss, cell, ','); r.f2 = std::strtod(cell.c_str(), nullptr);
        out.push_back(r);
        if (r.block > max_block) max_block = r.block;
        if (r.i > max_pop) max_pop = r.i;
        if (r.j > max_pop) max_pop = r.j;
    }
    return true;
}

// STPF2BK1 header (must match src/device/f2_disk_format.hpp F2DiskHeader, 64 bytes).
#pragma pack(push, 1)
struct DiskHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t dtype;
    std::int32_t P;
    std::int32_t n_block;
    std::uint64_t f2_offset;
    std::uint64_t vpair_offset;
    std::uint64_t block_sizes_offset;
    std::uint8_t reserved[16];
};
#pragma pack(pop)

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: %s <steppe-binary> <aadr-root> <golden-dir>\n", argv[0]);
        return 2;
    }
    const std::string steppe_bin = argv[1];
    const std::string aadr_root = argv[2];
    const std::string golden_dir = argv[3];
    const std::string prefix = aadr_root + "/raw/v66.p1_HO.aadr.patch.PUB";

    std::printf("=== qpfstats CLI parity (genotype-path smoothed f2 TENSOR; ON THE GPU) ===\n");
    std::printf("steppe: %s\nprefix: %s\ngolden: %s\n", steppe_bin.c_str(), prefix.c_str(),
                golden_dir.c_str());

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found)\n"); return 1;
    }
    if (!std::filesystem::exists(prefix + ".geno") || !std::filesystem::exists(prefix + ".snp") ||
        !std::filesystem::exists(prefix + ".ind")) {
        std::printf("\nRESULT: SKIP (genotype prefix absent: %s.{geno,snp,ind})\n", prefix.c_str());
        return 0;
    }

    std::vector<GRow> golden;
    int max_block = -1, max_pop = -1;
    if (!read_golden(golden_dir + "/csv/golden_qpfstats_geno.csv", golden, max_block, max_pop)) {
        std::printf("\nRESULT: SKIP (qpfstats golden absent)\n"); return 0;
    }
    const int Pg = max_pop + 1, NBg = max_block + 1;
    check_true("golden P == 9", Pg == 9);
    check_true("golden n_block == 711", NBg == 711);
    check_true("golden row count == 45*711", static_cast<int>(golden.size()) == 45 * 711);
    if (golden.empty()) { std::printf("\nRESULT: FAIL (empty golden)\n"); return 1; }

    // The 9 fit0 pops (sorted ASC == the golden dimnames order == steppe's internal sort).
    const std::vector<std::string> pops = {
        "Czechia_EBA_CordedWare", "England_BellBeaker", "Han", "Iran_GanjDareh_N",
        "Israel_Natufian", "Karitiana", "Mbuti", "Papuan", "Turkey_N"};

    std::error_code ec;
    const std::filesystem::path tmp_root =
        std::filesystem::temp_directory_path(ec) / ("steppe_cli_qpfstats_" +
            std::to_string(static_cast<long long>(
                std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp_root, ec);
    const std::filesystem::path out_dir = tmp_root / "smoothed_f2";

    std::string pops_arg;
    for (std::size_t i = 0; i < pops.size(); ++i) { if (i) pops_arg += ","; pops_arg += pops[i]; }

    const RunResult rr = run_steppe(steppe_bin,
        {"qpfstats", "--prefix", prefix, "--pops", pops_arg,
         "--out-dir", out_dir.string(), "--device", "0"},
        tmp_root);
    if (looks_like_no_gpu(rr.text)) {
        std::printf("  [SKIP] no CUDA device visible — qpfstats cannot run\n%s\n", rr.text.c_str());
        std::filesystem::remove_all(tmp_root, ec);
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;
    }
    check_true("qpfstats exit code == 0", rr.exit_code == 0);
    if (rr.exit_code != 0) { std::printf("steppe output:\n%s\n", rr.text.c_str()); }

    // Read the smoothed f2.bin.
    const std::filesystem::path f2bin = out_dir / "f2.bin";
    if (!std::filesystem::exists(f2bin)) {
        std::printf("  [FAIL] smoothed f2.bin not written: %s\nsteppe output:\n%s\n",
                    f2bin.string().c_str(), rr.text.c_str());
        std::filesystem::remove_all(tmp_root, ec);
        std::printf("\nRESULT: FAIL\n");
        return 1;
    }
    std::ifstream bf(f2bin, std::ios::binary);
    DiskHeader h;
    bf.read(reinterpret_cast<char*>(&h), sizeof(h));
    check_true("f2.bin magic STPF2BK1", std::memcmp(h.magic, "STPF2BK1", 8) == 0);
    check_true("f2.bin P == 9", h.P == 9);
    check_true("f2.bin n_block == 711", h.n_block == 711);
    const std::size_t nf2 = static_cast<std::size_t>(h.P) * h.P * h.n_block;
    std::vector<double> f2(nf2, 0.0);
    bf.seekg(static_cast<std::streamoff>(h.f2_offset), std::ios::beg);
    bf.read(reinterpret_cast<char*>(f2.data()), static_cast<std::streamsize>(nf2 * sizeof(double)));
    if (!bf) {
        std::printf("  [FAIL] could not read f2.bin payload\n");
        std::filesystem::remove_all(tmp_root, ec);
        std::printf("\nRESULT: FAIL\n");
        return 1;
    }

    // Diff every golden (i,j,block) vs f2[i + P*j + P*P*b]. The diagonal (i==j) golden is 0;
    // steppe's diagonal stays 0 by construction (the off-diagonal pair-basis scatter).
    const int P = h.P;
    int checked = 0;
    for (const GRow& g : golden) {
        const std::size_t idx = static_cast<std::size_t>(g.i) +
                                static_cast<std::size_t>(P) * static_cast<std::size_t>(g.j) +
                                static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
                                    static_cast<std::size_t>(g.block);
        char nm[64];
        std::snprintf(nm, sizeof(nm), "f2[%d,%d,b%d]", g.i, g.j, g.block);
        check_close(nm, f2[idx], g.f2, 1e-6, 1e-9);
        // Symmetry: f2[j,i,b] == f2[i,j,b].
        const std::size_t sym = static_cast<std::size_t>(g.j) +
                                static_cast<std::size_t>(P) * static_cast<std::size_t>(g.i) +
                                static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
                                    static_cast<std::size_t>(g.block);
        if (std::fabs(f2[idx] - f2[sym]) > 1e-12) { ++g_failures; }
        ++checked;
    }
    std::printf("  checked %d golden entries (9x9x711 upper-tri incl diag)\n", checked);

    std::filesystem::remove_all(tmp_root, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (smoothed f2 tensor reproduced element-wise through the GPU)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
