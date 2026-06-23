// tests/cli/test_cli_f3.cpp
//
// STANDALONE f3 acceptance gate (fit-engine §6): the `steppe f3` CLI reproduces the
// fixture-matched AT2 f3 golden (golden_fit0_f3_readf2.csv, 36 triple rows) THROUGH the CLI,
// ON THE GPU. The THREE-slab clone of test_cli_f4.cpp. The golden was made via
// admixtools::f3(read_f2(the 9-pop maxmiss=0 f2 fixture)) so a standalone f3 over the SAME
// committed fixture matches it at tolerance.
//
// NO SYNTHETIC DATA (memory real-data-only): it writes the COMMITTED real 9-pop golden f2
// fixture (fixtures/f2_fit0_9pop.bin — the SAME tensor golden_fit0 / test_qpadm / qpwave / f4
// use) into a temp f2-dir (STPF2BK1 f2.bin + pops.txt + meta.json), then runs the BUILT
// `steppe f3` binary for ALL 36 golden triples (the row-aligned --pop1/--pop2/--pop3 columns)
// and diffs the est/se/z/p table row-for-row.
//
// TIERS: est/se/z/p ALL TIGHT at rtol 1e-6 (atol 1e-9), p with a generous atol (huge |z|
// underflows to ~0). status == "ok".
//
// PLAIN C++ host TU (NO CUDA header): it spawns the steppe binary and parses stdout; the GPU
// work happens inside that child. SKIPs cleanly (exit 0) when no CUDA device is visible. The
// harness helpers are reused verbatim from the test_cli_f4 pattern. Self-checking main();
// CTest gates on the exit code.

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

#include "device/f2_disk_format.hpp"  // F2DiskHeader, kF2DiskMagic/Version/DtypeFp64 (CUDA-FREE)

namespace {

int g_failures = 0;

void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const double diff = std::fabs(got - want);
    const bool ok = diff <= tol;
    if (!ok)
        std::printf("  [FAIL] %-28s got=% .12e want=% .12e |d|=% .3e tol=% .3e\n",
                    what, got, want, diff, tol);
    if (!ok) ++g_failures;
}

void check_eq_int(const char* what, long long got, long long want) {
    const bool ok = (got == want);
    if (!ok) std::printf("  [FAIL] %-28s got=%lld want=%lld\n", what, got, want);
    if (!ok) ++g_failures;
}

void check_true(const char* what, bool ok) {
    if (!ok) std::printf("  [FAIL] %s\n", what);
    if (!ok) ++g_failures;
}

// ============================================================================
// Harness — reused verbatim from test_cli_f4.cpp / test_cli_qpwave.cpp.
// ============================================================================
struct RawF2 {
    int P = 0, nb = 0;
    std::vector<std::int32_t> block_sizes;
    std::vector<double> f2;  // P*P*nb
};

bool read_raw_fixture(const std::string& path, RawF2& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("  [FAIL] cannot open fixture: %s\n", path.c_str()); return false; }
    std::int32_t P = 0, nb = 0;
    f.read(reinterpret_cast<char*>(&P), sizeof(P));
    f.read(reinterpret_cast<char*>(&nb), sizeof(nb));
    if (!f || P <= 0 || nb <= 0) { std::printf("  [FAIL] bad fixture header\n"); return false; }
    out.P = P; out.nb = nb;
    out.block_sizes.resize(static_cast<std::size_t>(nb));
    f.read(reinterpret_cast<char*>(out.block_sizes.data()),
           static_cast<std::streamsize>(sizeof(std::int32_t) * static_cast<std::size_t>(nb)));
    const std::size_t n = static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
                          static_cast<std::size_t>(nb);
    out.f2.resize(n);
    f.read(reinterpret_cast<char*>(out.f2.data()),
           static_cast<std::streamsize>(sizeof(double) * n));
    if (!f) { std::printf("  [FAIL] fixture truncated\n"); return false; }
    return true;
}

bool write_f2_dir(const std::filesystem::path& dir, const RawF2& raw,
                  const std::vector<std::string>& pops) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) { std::printf("  [FAIL] mkdir %s: %s\n", dir.string().c_str(), ec.message().c_str()); return false; }

    const std::uint64_t P = static_cast<std::uint64_t>(raw.P);
    const std::uint64_t nb = static_cast<std::uint64_t>(raw.nb);
    const std::uint64_t slab_bytes = P * P * nb * sizeof(double);

    steppe::device::F2DiskHeader hdr{};
    for (int i = 0; i < 8; ++i) hdr.magic[i] = steppe::device::kF2DiskMagic[i];
    hdr.version = steppe::device::kF2DiskVersion;
    hdr.dtype = steppe::device::kF2DiskDtypeFp64;
    hdr.P = raw.P;
    hdr.n_block = raw.nb;
    hdr.f2_offset = steppe::device::kF2DiskHeaderSize;            // == 64
    hdr.vpair_offset = hdr.f2_offset + slab_bytes;
    hdr.block_sizes_offset = hdr.vpair_offset + slab_bytes;

    const std::filesystem::path bin = dir / "f2.bin";
    std::ofstream o(bin, std::ios::binary | std::ios::trunc);
    if (!o) { std::printf("  [FAIL] cannot write f2.bin\n"); return false; }
    o.write(reinterpret_cast<const char*>(&hdr), static_cast<std::streamsize>(sizeof(hdr)));
    o.write(reinterpret_cast<const char*>(raw.f2.data()),
            static_cast<std::streamsize>(slab_bytes));
    const std::vector<double> vpair_zeros(static_cast<std::size_t>(P * P * nb), 0.0);
    o.write(reinterpret_cast<const char*>(vpair_zeros.data()),
            static_cast<std::streamsize>(slab_bytes));
    o.write(reinterpret_cast<const char*>(raw.block_sizes.data()),
            static_cast<std::streamsize>(sizeof(std::int32_t) * nb));
    if (!o) { std::printf("  [FAIL] f2.bin write failed\n"); return false; }
    o.close();

    std::ofstream pf(dir / "pops.txt", std::ios::trunc);
    if (!pf) { std::printf("  [FAIL] cannot write pops.txt\n"); return false; }
    for (const std::string& s : pops) pf << s << "\n";
    pf.close();

    std::ofstream mf(dir / "meta.json", std::ios::trunc);
    if (!mf) { std::printf("  [FAIL] cannot write meta.json\n"); return false; }
    mf << "{\n  \"format\": \"STPF2BK1\",\n  \"P\": " << raw.P
       << ",\n  \"n_block\": " << raw.nb
       << ",\n  \"provenance\": \"test fixture (real-AADR 9-pop f3 golden f2)\"\n}\n";
    mf.close();
    return true;
}

struct RunResult {
    int exit_code = -1;
    std::string stdout_text;
};

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "cli_f3_stdout.txt";
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

// One golden triple row: the three pop names + est/se/z/p.
struct F3Row {
    std::string p1, p2, p3;
    double est = 0.0, se = 0.0, z = 0.0, p = 0.0;
};

// Parse golden_fit0_f3_readf2.csv (header pop1,pop2,pop3,est,se,z,p + 36 data rows).
bool read_golden_f3(const std::string& path, std::vector<F3Row>& out) {
    std::ifstream f(path);
    if (!f) { std::printf("  [SKIP] golden f3 csv absent: %s\n", path.c_str()); return false; }
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
        F3Row r;
        r.p1 = cells[col["pop1"]];
        r.p2 = cells[col["pop2"]];
        r.p3 = cells[col["pop3"]];
        r.est = cell_d(cells[col["est"]]);
        r.se  = cell_d(cells[col["se"]]);
        r.z   = cell_d(cells[col["z"]]);
        r.p   = cell_d(cells[col["p"]]);
        out.push_back(r);
    }
    return true;
}

// Map the steppe f3 CSV output (header pop1..pop3,est,se,z,p + N data rows) keyed on the
// triple name 3-tuple, so the diff is order-independent.
std::map<std::string, F3Row> parse_steppe_f3(const std::string& text) {
    std::map<std::string, F3Row> out;
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
        F3Row r;
        r.p1 = cells[col["pop1"]];
        r.p2 = cells[col["pop2"]];
        r.p3 = cells[col["pop3"]];
        r.est = cell_d(cells[col["est"]]);
        r.se  = cell_d(cells[col["se"]]);
        r.z   = cell_d(cells[col["z"]]);
        r.p   = cell_d(cells[col["p"]]);
        out[r.p1 + "|" + r.p2 + "|" + r.p3] = r;
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
    // argv[1] = the built steppe binary; argv[2] = the golden dir.
    if (argc < 3) {
        std::printf("usage: %s <steppe-binary> <golden-dir>\n", argv[0]);
        return 2;
    }
    const std::string steppe_bin = argv[1];
    const std::string golden_dir = argv[2];
    std::printf("=== standalone f3 CLI parity (THROUGH the CLI, ON THE GPU) ===\n");
    std::printf("steppe binary: %s\ngolden dir:    %s\n", steppe_bin.c_str(), golden_dir.c_str());

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", steppe_bin.c_str());
        return 1;
    }

    // ---- the committed 9-pop f2 fixture (the SAME tensor golden_fit0 / qpwave / f4 use) ---
    RawF2 raw;
    if (!read_raw_fixture(golden_dir + "/fixtures/f2_fit0_9pop.bin", raw)) {
        std::printf("\nRESULT: SKIP (f3 fixture absent)\n");
        return 0;  // absent fixture is a clean skip (like the parity test)
    }

    // ---- the fixture-matched golden (36 genuine triples) ------------------------------
    std::vector<F3Row> golden;
    if (!read_golden_f3(golden_dir + "/csv/golden_fit0_f3_readf2.csv", golden)) {
        std::printf("\nRESULT: SKIP (f3 golden absent)\n");
        return 0;
    }
    check_true("golden has 36 triple rows", golden.size() == 36);
    if (golden.empty()) { std::printf("\nRESULT: FAIL (empty golden)\n"); return 1; }

    const std::vector<std::string> pops = {
        "England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N", "Mbuti",
        "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"};
    check_eq_int("fixture P == pops size", raw.P, static_cast<int>(pops.size()));

    std::error_code ec;
    const std::filesystem::path tmp_root =
        std::filesystem::temp_directory_path(ec) / ("steppe_cli_f3_test_" + std::to_string(
            static_cast<long long>(std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp_root, ec);
    const std::filesystem::path dir = tmp_root / "f3";
    if (!write_f2_dir(dir, raw, pops)) {
        ++g_failures;
        std::filesystem::remove_all(tmp_root, ec);
        std::printf("\nRESULT: FAIL (could not write f2-dir)\n");
        return 1;
    }

    // Build the row-aligned --pop1/--pop2/--pop3 columns for ALL 36 golden triples.
    std::vector<std::string> c1, c2, c3;
    for (const F3Row& g : golden) {
        c1.push_back(g.p1); c2.push_back(g.p2); c3.push_back(g.p3);
    }

    // ---------------- CSV run (the full 36-triple batch) ----------------
    const RunResult csv = run_steppe(steppe_bin,
        {"f3", "--f2-dir", dir.string(),
         "--pop1", join(c1), "--pop2", join(c2), "--pop3", join(c3),
         "--format", "csv", "--device", "0"},
        tmp_root);
    if (looks_like_no_gpu(csv.stdout_text)) {
        std::printf("  [SKIP] no CUDA device visible — CLI GPU f3 cannot run\n");
        std::filesystem::remove_all(tmp_root, ec);
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;
    }
    check_eq_int("csv exit code == 0", csv.exit_code, 0);

    const std::map<std::string, F3Row> got = parse_steppe_f3(csv.stdout_text);
    check_eq_int("steppe emitted 36 rows", static_cast<long long>(got.size()), 36);

    for (const F3Row& g : golden) {
        const std::string key = g.p1 + "|" + g.p2 + "|" + g.p3;
        const auto it = got.find(key);
        if (it == got.end()) {
            std::printf("  [FAIL] triple missing from steppe output: %s\n", key.c_str());
            ++g_failures;
            continue;
        }
        const F3Row& s = it->second;
        char nm[160];
        std::snprintf(nm, sizeof(nm), "est %s", key.c_str());
        check_close(nm, s.est, g.est, 1e-6, 1e-9);
        std::snprintf(nm, sizeof(nm), "se %s", key.c_str());
        check_close(nm, s.se, g.se, 1e-6, 1e-9);
        std::snprintf(nm, sizeof(nm), "z %s", key.c_str());
        check_close(nm, s.z, g.z, 1e-6, 1e-9);
        std::snprintf(nm, sizeof(nm), "p %s", key.c_str());
        // p can underflow to ~0 for huge |z|; gate with a generous atol.
        check_close(nm, s.p, g.p, 1e-6, 1e-12);
    }

    // ---------------- single-triple --pops convenience (the SANITY) ----------------
    {
        const F3Row& g0 = golden.front();
        const std::string pops_arg = g0.p1 + "," + g0.p2 + "," + g0.p3;
        const RunResult one = run_steppe(steppe_bin,
            {"f3", "--f2-dir", dir.string(), "--pops", pops_arg,
             "--format", "csv", "--device", "0"},
            tmp_root);
        check_eq_int("single-triple exit code == 0", one.exit_code, 0);
        const std::map<std::string, F3Row> g1 = parse_steppe_f3(one.stdout_text);
        check_eq_int("single-triple 1 row", static_cast<long long>(g1.size()), 1);
        const std::string key = g0.p1 + "|" + g0.p2 + "|" + g0.p3;
        const auto it = g1.find(key);
        check_true("single-triple matches golden row 1", it != g1.end());
        if (it != g1.end()) {
            check_close("--pops est", it->second.est, g0.est, 1e-6, 1e-9);
            check_close("--pops se", it->second.se, g0.se, 1e-6, 1e-9);
            check_close("--pops z", it->second.z, g0.z, 1e-6, 1e-9);
            check_close("--pops p", it->second.p, g0.p, 1e-6, 1e-12);
        }
    }

    // ---------------- JSON run (prove JSON serializes the same first row) ----------------
    const RunResult js = run_steppe(steppe_bin,
        {"f3", "--f2-dir", dir.string(),
         "--pop1", c1[0], "--pop2", c2[0], "--pop3", c3[0],
         "--format", "json", "--device", "0"},
        tmp_root);
    check_eq_int("json exit code == 0", js.exit_code, 0);
    {
        const std::string& t = js.stdout_text;
        const std::size_t ek = t.find("\"est\"");
        check_true("json has est field", ek != std::string::npos);
        if (ek != std::string::npos) {
            const std::size_t colon = t.find(':', ek);
            const double est = std::strtod(t.c_str() + colon + 1, nullptr);
            check_close("json est[0]", est, golden.front().est, 1e-6, 1e-9);
        }
        check_true("json status ok", t.find("\"status\": \"ok\"") != std::string::npos);
    }

    std::filesystem::remove_all(tmp_root, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (36 f3 triples reproduced through the GPU; CSV + JSON + --pops)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
