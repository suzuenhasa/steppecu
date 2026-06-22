// tests/cli/test_cli_qpwave.cpp
//
// M(cli-2) ACCEPTANCE GATE (cli-bindings.md §7 row `tests/cli/test_cli_qpwave`): the
// `steppe qpwave` CLI reproduces the REAL-AADR AT2 qpWave golden (golden_qpwave.json,
// two models) THROUGH the CLI, ON THE GPU — the CLI counterpart of the engine-level
// test_qpwave_parity.cu (#22). NO AT2 re-run: the gated values are the pinned binary-
// fixture (f2-object-path) constants from test_qpwave_parity.cu:goldens(), which the
// committed fixture reproduces (the documented read-arg caveat, golden_qpwave.json:8).
//
// NO SYNTHETIC DATA (memory real-data-only): it writes the COMMITTED real 9-pop golden f2
// fixture (fixtures/f2_fit0_9pop.bin — the SAME tensor golden_fit0/test_qpadm use) into a
// temp f2-dir (STPF2BK1 f2.bin + pops.txt + meta.json) — exactly what a real <dir>
// contains — then runs the BUILT `steppe qpwave` binary with the golden's EXACT inputs.
// The distinguishing qpWave invocation: NO --target flag; --left[0] is the reference row.
//
//   M1 (3-left, est_rank=1): --left England_BellBeaker,Czechia_EBA_CordedWare,Turkey_N
//       (left[0]=reference). f4rank/est_rank=1; rankdrop (f4rank DESC) rows {1,0}, the
//       rank-1 model NOT rejected at alpha=0.05.
//   M2 (2-left CLADE, est_rank=0): --left Czechia_EBA_CordedWare,Turkey_N (left[0]=ref).
//       The clade is REJECTED (p<<0.05) ⇒ f4rank/est_rank=0; a SINGLE-row rankdrop (the
//       last-row NA shape: dofdiff=="NA").
//
// TIERS (test_qpwave_parity.cu tiers, fit-engine.md §4.1): f4rank/est_rank/rankdrop dof/
// rankdrop f4rank/dofdiff EXACT (==); rankdrop chisq/chisqdiff TIGHT (rtol 1e-6, atol
// 1e-9); rankdrop p/p_nested LOOSE (rtol 1e-3, atol 1e-9, the rank-decision tier). status
// == "ok"; the per_rank sweep len == rankdrop rows and per_rank.chisq[r=0] == the last
// rankdrop row chisq (the reverse-order cross-check, test_qpwave_parity.cu:222-225).
//
// PLAIN C++ host TU (NO CUDA header): it spawns the steppe binary and parses stdout; the
// GPU work happens inside that child. SKIPs cleanly (exit 0) when no CUDA device is
// visible — identical to test_cli_qpadm / test_cli_rotate (cli-bindings.md §7) — by
// detecting the child's "no CUDA device" fault. The harness helpers (read_raw_fixture/
// write_f2_dir/run_steppe/looks_like_no_gpu/parse_csv_sections/split_csv_line/cell_d) are
// reused verbatim from the test_cli_rotate pattern. Self-checking main(); CTest gates on
// the exit code.

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
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
// Harness — reused verbatim from test_cli_rotate.cpp / test_cli_qpadm.cpp
// (raw-fixture read, f2-dir write, child-process run, no-GPU detection, CSV
// section parse).
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

// Write the f2-dir: f2.bin (STPF2BK1) + pops.txt + meta.json. vpair written as zeros (the
// fit reads block_sizes, NOT vpair) — the same way the parity/CLI tests do it.
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
       << ",\n  \"provenance\": \"test fixture (real-AADR 9-pop qpwave golden f2)\"\n}\n";
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
    const std::filesystem::path outf = tmp / "cli_qpwave_stdout.txt";
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

struct CsvSections {
    std::map<std::string, std::vector<std::vector<std::string>>> sec;
};

CsvSections parse_csv_sections(const std::string& text) {
    CsvSections out;
    std::istringstream ss(text);
    std::string line;
    std::string cur_sec;
    const std::string prefix = "# section: ";
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind(prefix, 0) == 0) { cur_sec = line.substr(prefix.size()); continue; }
        if (cur_sec.empty() || line.empty()) continue;
        out.sec[cur_sec].push_back(split_csv_line(line));
    }
    return out;
}

double cell_d(const std::string& s) {
    if (s == "NA" || s == "null" || s.empty()) return std::nan("");
    return std::strtod(s.c_str(), nullptr);
}

// ============================================================================
// The pinned qpWave golden (the f2-object-path values from test_qpwave_parity.cu:
// goldens(); golden_qpwave.json's directory-path numbers differ by the read-arg
// caveat — the COMMITTED binary fixture reproduces THESE).
// ============================================================================
struct QpWaveGolden {
    std::string label;
    std::vector<std::string> left;   // left[0] is the reference (NO target)
    std::vector<std::string> right;  // right[0] is R0
    int est_rank = 0;
    int f4rank = 0;
    // rankdrop, f4rank-DESCENDING:
    std::vector<int> rd_f4rank, rd_dof;
    std::vector<double> rd_chisq, rd_p;
    int rd0_dofdiff = INT_MIN;   // row 0 only (last row = NA)
    double rd0_chisqdiff = 0.0;
    double rd0_p_nested = 0.0;
    bool has_nested = false;     // M1 has a nested-diff row 0; M2 (single row) does not
};

// ============================================================================
// Run + gate one qpWave model THROUGH the CLI (CSV) and (JSON), comparing every
// field at its tier. Returns false on a clean SKIP (no GPU).
// ============================================================================
bool run_case(const std::string& bin, const std::filesystem::path& tmp_root,
              const std::filesystem::path& dir, const QpWaveGolden& g) {
    std::printf("\n-- %s --\n", g.label.c_str());
    const std::string left = [&] {
        std::string s; for (std::size_t i = 0; i < g.left.size(); ++i) { if (i) s += ","; s += g.left[i]; } return s;
    }();
    const std::string right = [&] {
        std::string s; for (std::size_t i = 0; i < g.right.size(); ++i) { if (i) s += ","; s += g.right[i]; } return s;
    }();
    const int n = static_cast<int>(g.rd_f4rank.size());

    // ---------------- CSV run (NO --target: the qpWave invocation) ----------------
    const RunResult csv = run_steppe(bin,
        {"qpwave", "--f2-dir", dir.string(), "--left", left, "--right", right,
         "--format", "csv", "--device", "0"},
        tmp_root);
    if (looks_like_no_gpu(csv.stdout_text)) {
        std::printf("  [SKIP] no CUDA device visible — CLI GPU qpWave cannot run\n");
        return false;
    }
    check_eq_int("csv exit code == 0", csv.exit_code, 0);

    const CsvSections cs = parse_csv_sections(csv.stdout_text);
    const auto rd_it = cs.sec.find("rankdrop");
    const auto pr_it = cs.sec.find("per_rank");
    const auto su_it = cs.sec.find("summary");
    check_true("csv has rankdrop section", rd_it != cs.sec.end());
    check_true("csv has per_rank section", pr_it != cs.sec.end());
    check_true("csv has summary section", su_it != cs.sec.end());

    // --- rankdrop table (f4rank DESCENDING) ---
    if (rd_it != cs.sec.end()) {
        const auto& rows = rd_it->second;  // rows[0] = header; one data row per rank drop.
        std::map<std::string, std::size_t> col;
        if (!rows.empty()) for (std::size_t c = 0; c < rows[0].size(); ++c) col[rows[0][c]] = c;
        check_eq_int("rankdrop rows", static_cast<long long>(rows.size()) - 1, n);
        for (int k = 0; k < n && static_cast<std::size_t>(k + 1) < rows.size(); ++k) {
            const auto& r = rows[static_cast<std::size_t>(k + 1)];
            const std::size_t sk = static_cast<std::size_t>(k);
            char nm[56];
            std::snprintf(nm, sizeof(nm), "rd[%d].f4rank", k);
            check_eq_int(nm, static_cast<long long>(cell_d(r[col["f4rank"]])), g.rd_f4rank[sk]);
            std::snprintf(nm, sizeof(nm), "rd[%d].dof", k);
            check_eq_int(nm, static_cast<long long>(cell_d(r[col["dof"]])), g.rd_dof[sk]);
            std::snprintf(nm, sizeof(nm), "rd[%d].chisq", k);
            check_close(nm, cell_d(r[col["chisq"]]), g.rd_chisq[sk], 1e-6, 1e-9);
            std::snprintf(nm, sizeof(nm), "rd[%d].p", k);
            check_close(nm, cell_d(r[col["p"]]), g.rd_p[sk], 1e-3, 1e-9);
        }
        // nested diff: M1 row-0 EXACT(dofdiff)/TIGHT(chisqdiff)/LOOSE(p_nested).
        if (g.has_nested && rows.size() >= 2) {
            const auto& r0 = rows[1];
            check_eq_int("rd[0].dofdiff", static_cast<long long>(cell_d(r0[col["dofdiff"]])), g.rd0_dofdiff);
            check_close("rd[0].chisqdiff", cell_d(r0[col["chisqdiff"]]), g.rd0_chisqdiff, 1e-6, 1e-9);
            check_close("rd[0].p_nested", cell_d(r0[col["p_nested"]]), g.rd0_p_nested, 1e-3, 1e-9);
        }
        // last row NA: dofdiff column literal "NA" (the lowest rank has no nested compare).
        if (rows.size() >= 2) {
            const auto& rl = rows[rows.size() - 1];
            check_true("rd[last].dofdiff == NA", rl[col["dofdiff"]] == "NA");
        }
    }

    // --- per_rank sweep (ASCENDING r) + the reverse-order cross-check ---
    double pr0_chisq = std::nan("");
    long long pr0_dof = -1;
    if (pr_it != cs.sec.end()) {
        const auto& rows = pr_it->second;
        std::map<std::string, std::size_t> col;
        if (!rows.empty()) for (std::size_t c = 0; c < rows[0].size(); ++c) col[rows[0][c]] = c;
        check_eq_int("per_rank sweep len", static_cast<long long>(rows.size()) - 1, n);
        if (rows.size() >= 2) {
            // r=0 row is the FIRST data row (ascending); the `rank` column == 0.
            const auto& r0 = rows[1];
            check_eq_int("per_rank[0].rank == 0", static_cast<long long>(cell_d(r0[col["rank"]])), 0);
            pr0_chisq = cell_d(r0[col["chisq"]]);
            pr0_dof = static_cast<long long>(cell_d(r0[col["dof"]]));
        }
    }
    // The reverse-order cross-check: per_rank.chisq[r=0] == the LAST rankdrop row chisq
    // (the rd row with f4rank==0); per_rank.dof[r=0] == that row's dof.
    if (n >= 1) {
        check_close("per_rank.chisq[r=0] == rd[last].chisq", pr0_chisq, g.rd_chisq[static_cast<std::size_t>(n - 1)], 1e-6, 1e-9);
        check_eq_int("per_rank.dof[r=0] == rd[last].dof", pr0_dof, g.rd_dof[static_cast<std::size_t>(n - 1)]);
    }

    // --- summary (f4rank / est_rank / status) ---
    if (su_it != cs.sec.end()) {
        const auto& rows = su_it->second;
        std::map<std::string, std::size_t> col;
        if (!rows.empty()) for (std::size_t c = 0; c < rows[0].size(); ++c) col[rows[0][c]] = c;
        if (rows.size() >= 2) {
            const auto& r = rows[1];
            check_eq_int("summary f4rank", static_cast<long long>(cell_d(r[col["f4rank"]])), g.f4rank);
            check_eq_int("summary est_rank", static_cast<long long>(cell_d(r[col["est_rank"]])), g.est_rank);
            check_true("summary status == ok", r[col["status"]] == "ok");
            // reference column echoes left[0].
            check_true("summary reference == left[0]", r[col["reference"]] == g.left.front());
        }
    }

    // ---------------- JSON run (re-scrape to prove JSON serializes the same result) ----
    const RunResult js = run_steppe(bin,
        {"qpwave", "--f2-dir", dir.string(), "--left", left, "--right", right,
         "--format", "json", "--device", "0"},
        tmp_root);
    check_eq_int("json exit code == 0", js.exit_code, 0);
    {
        const std::string& t = js.stdout_text;
        // scrape summary f4rank / est_rank.
        auto scrape_int = [&](const char* key) -> long long {
            const std::size_t k = t.find(key);
            if (k == std::string::npos) return -999999;
            const std::size_t colon = t.find(':', k);
            return std::strtoll(t.c_str() + colon + 1, nullptr, 10);
        };
        // The summary block is the only place "f4rank":<scalar> appears as an object value
        // (the rankdrop block has "f4rank": [..]); scan the summary substring.
        const std::size_t su = t.find("\"summary\"");
        check_true("json has summary block", su != std::string::npos);
        if (su != std::string::npos) {
            const std::string sub = t.substr(su);
            auto scrape_in = [&](const char* key) -> long long {
                const std::size_t k = sub.find(key);
                if (k == std::string::npos) return -999999;
                const std::size_t colon = sub.find(':', k);
                return std::strtoll(sub.c_str() + colon + 1, nullptr, 10);
            };
            check_eq_int("json summary f4rank", scrape_in("\"f4rank\""), g.f4rank);
            check_eq_int("json summary est_rank", scrape_in("\"est_rank\""), g.est_rank);
            check_true("json summary status ok", sub.find("\"status\": \"ok\"") != std::string::npos);
        }
        (void)scrape_int;
        // scrape the FIRST rankdrop chisq array element (== rd[0].chisq) to prove the JSON
        // rankdrop block carries the same first row the CSV did.
        const std::size_t rdk = t.find("\"rankdrop\"");
        check_true("json has rankdrop block", rdk != std::string::npos);
        if (rdk != std::string::npos) {
            const std::size_t ck = t.find("\"chisq\"", rdk);
            const std::size_t lb = t.find('[', ck);
            const double first = std::strtod(t.c_str() + lb + 1, nullptr);
            check_close("json rd[0].chisq", first, g.rd_chisq.front(), 1e-6, 1e-9);
        }
    }
    return true;
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
    std::printf("=== M(cli-2) steppe qpwave CLI parity (THROUGH the CLI, ON THE GPU) ===\n");
    std::printf("steppe binary: %s\ngolden dir:    %s\n", steppe_bin.c_str(), golden_dir.c_str());

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", steppe_bin.c_str());
        return 1;
    }

    // ---- the committed 9-pop f2 fixture (the SAME tensor golden_fit0/test_qpadm use) ---
    RawF2 raw;
    if (!read_raw_fixture(golden_dir + "/fixtures/f2_fit0_9pop.bin", raw)) {
        std::printf("  [SKIP] f2_fit0_9pop fixture absent\n");
        std::printf("\nRESULT: SKIP (qpwave fixture absent)\n");
        return 0;  // absent fixture is a clean skip (like the parity test)
    }

    // pops.txt == the golden 9-pop index order (golden_qpwave.json:9). The CLI resolves
    // --left/--right against THESE labels.
    const std::vector<std::string> pops = {
        "England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N", "Mbuti",
        "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"};
    check_eq_int("fixture P == pops size", raw.P, static_cast<int>(pops.size()));

    std::error_code ec;
    const std::filesystem::path tmp_root =
        std::filesystem::temp_directory_path(ec) / ("steppe_cli_qpwave_test_" + std::to_string(
            static_cast<long long>(std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp_root, ec);
    std::printf("temp f2-dir root: %s\n", tmp_root.string().c_str());

    const std::filesystem::path dir = tmp_root / "qpwave";
    if (!write_f2_dir(dir, raw, pops)) {
        ++g_failures;
        std::filesystem::remove_all(tmp_root, ec);
        std::printf("\nRESULT: FAIL (could not write f2-dir)\n");
        return 1;
    }

    const std::vector<std::string> right6 = {
        "Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"};

    // ---- M1 (3-left, est_rank=1): left[0]=England_BellBeaker reference -------------
    QpWaveGolden m1;
    m1.label = "M1 3-left (reference=England_BellBeaker), est_rank=1";
    m1.left = {"England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N"};
    m1.right = right6;
    m1.est_rank = 1; m1.f4rank = 1;
    m1.rd_f4rank = {1, 0}; m1.rd_dof = {4, 10};
    m1.rd_chisq = {3.95682062790988, 1474.03320584515};
    m1.rd_p = {0.411881081897742, 1.02285567525252e-310};
    m1.rd0_dofdiff = 6; m1.rd0_chisqdiff = 1470.07638521724; m1.rd0_p_nested = 1.62084120329381e-314;
    m1.has_nested = true;

    // ---- M2 (2-left CLADE, est_rank=0): left[0]=CordedWare reference, single row ----
    QpWaveGolden m2;
    m2.label = "M2 2-left clade (reference=Czechia_EBA_CordedWare), est_rank=0";
    m2.left = {"Czechia_EBA_CordedWare", "Turkey_N"};
    m2.right = right6;
    m2.est_rank = 0; m2.f4rank = 0;
    m2.rd_f4rank = {0}; m2.rd_dof = {5};
    m2.rd_chisq = {1401.571655111};
    m2.rd_p = {6.284245136077e-301};
    m2.has_nested = false;  // single row ⇒ the one row is the NA last row

    const bool ran1 = run_case(steppe_bin, tmp_root, dir, m1);
    bool ran2 = true;
    if (ran1) ran2 = run_case(steppe_bin, tmp_root, dir, m2);

    std::filesystem::remove_all(tmp_root, ec);

    if (!ran1 || !ran2) {
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;  // CI-without-GPU degrades cleanly
    }
    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (both qpWave models reproduced through the GPU; CSV + JSON)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
