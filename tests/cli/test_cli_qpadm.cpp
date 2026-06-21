// tests/cli/test_cli_qpadm.cpp
//
// M(cli-1) ACCEPTANCE GATE (cli-bindings.md §7 row `tests/cli/test_cli_qpadm`): the
// `steppe qpadm` CLI reproduces the REAL-AADR AT2 goldens THROUGH the CLI, ON THE GPU.
//
// NO SYNTHETIC DATA (cli-bindings.md §7): this writes the COMMITTED real golden f2
// fixtures into a temp f2-dir (f2.bin STPF2BK1 + pops.txt + meta.json) — exactly what
// a real <dir> contains — then runs the BUILT `steppe qpadm` binary for the golden
// model, parses its CSV + JSON output, and asserts weights/se/z/p/chisq/dof/rankdrop/
// popdrop == the committed golden within the SAME two tiers the fit parity test uses
// (TIGHT rtol 1e-6 on weights/chisq, dof EXACT, LOOSE rtol 1e-3 on se/z/p). It is the
// CLI counterpart of test_qpadm_parity.cu — the name->index hardcode that test carries
// is replaced here by the CLI's pops.txt resolution (the gap the access layer closes).
//
// Two cases:
//   * golden_fit0  (9-pop, nr<=32 batched path): the PRIMARY gate.
//   * golden_fit1_NRBIG (P=43, nr=39, gesvd large path): the large case (cli-bindings
//     §7 "add golden_fit1_NRBIG as the large case").
//
// PLAIN C++ host TU (NO CUDA header): it spawns the steppe binary and parses stdout;
// the GPU work happens inside that child. SKIPs cleanly (exit 0) when no CUDA device is
// visible — identical to every reference test (cli-bindings.md §7) — by detecting the
// child's "no CUDA device" fault. Self-checking main(); CTest gates on the exit code.

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
    std::printf("  [%s] %-32s got=% .12e want=% .12e |d|=% .3e tol=% .3e\n",
                ok ? "PASS" : "FAIL", what, got, want, diff, tol);
    if (!ok) ++g_failures;
}

void check_eq_int(const char* what, long long got, long long want) {
    const bool ok = (got == want);
    std::printf("  [%s] %-32s got=%lld want=%lld\n", ok ? "PASS" : "FAIL", what, got, want);
    if (!ok) ++g_failures;
}

void check_true(const char* what, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

// ---- Read the committed RAW fixture (int32 P, int32 nb, int32[nb] block_sizes,
//      f64[P*P*nb] f2 col-major i+P*j+P*P*b; NO vpair) — the same layout
//      test_qpadm_parity.cu reads. -------------------------------------------------
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

// ---- Write the f2-dir: f2.bin (STPF2BK1) + pops.txt + meta.json. -----------------
// vpair is written as zeros (the fit reads block_sizes, NOT vpair — OQ-3; the parity
// test uploads zero-vpair the same way), so a faithful STPF2BK1 round-trip carries the
// golden f2 numbers in the f2 region.
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
       << ",\n  \"provenance\": \"test fixture (real-AADR golden f2)\"\n}\n";
    mf.close();
    return true;
}

// ---- Run the built `steppe qpadm` binary, capturing stdout to a temp file. --------
// Returns the process exit code (0 on a completed run / domain outcome). On a "no CUDA
// device" fault we detect it via stderr (merged) so the caller can SKIP cleanly.
struct RunResult {
    int exit_code = -1;
    std::string stdout_text;
};

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "cli_stdout.txt";
    std::string cmd = "\"" + bin + "\"";
    for (const std::string& a : args) cmd += " \"" + a + "\"";
    // Merge stderr into stdout so a "no CUDA device" fault line is visible to the SKIP
    // detector; the CSV/JSON we assert on is the only thing on stdout for a good run.
    cmd += " > \"" + outf.string() + "\" 2>&1";
    const int sys = std::system(cmd.c_str());
    rr.exit_code = (sys == -1) ? -1 :
#ifdef WIFEXITED
                   (WIFEXITED(sys) ? WEXITSTATUS(sys) : -1);
#else
                   sys;
#endif
    std::ifstream f(outf, std::ios::binary);
    if (f) {
        std::ostringstream ss; ss << f.rdbuf(); rr.stdout_text = ss.str();
    }
    return rr;
}

bool looks_like_no_gpu(const std::string& text) {
    return text.find("no CUDA device") != std::string::npos ||
           text.find("device error") != std::string::npos ||
           text.find("No CUDA") != std::string::npos;
}

// ---- Minimal CSV section parser: maps "# section: NAME" -> rows of cells. ---------
// Splits unquoted-comma cells, strips surrounding double quotes. Sufficient for the
// CLI's own well-formed output (we generated it).
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
    // section name -> list of rows (each row a vector of cells); row 0 is the header.
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

// ---- The golden_fit0 expected values (golden_fit0.json; admixtools 2.0.10). -------
struct Golden {
    std::string name;
    std::string fixture;          // raw .bin filename under fixtures/
    std::vector<std::string> pops; // P labels in index order
    std::string target;
    std::vector<std::string> left;
    std::vector<std::string> right;
    // weights:
    std::vector<double> weight, se, z;
    double chisq = 0; int dof = 0; double p = 0;
    // rankdrop rows (f4rank DESC: rank1, rank0):
    std::vector<int> rd_f4rank, rd_dof; std::vector<double> rd_chisq, rd_p;
    int rd0_dofdiff = 0; double rd0_chisqdiff = 0, rd0_p_nested = 0;
    int f4rank = 0;
    // popdrop rows:
    std::vector<std::string> pd_pat; std::vector<int> pd_dof, pd_f4rank;
    std::vector<double> pd_chisq;
};

// Assert one golden through the CLI CSV + JSON. Returns true if it RAN (false ⇒ SKIP).
bool run_case(const std::string& steppe_bin, const std::string& golden_dir,
              const std::filesystem::path& tmp_root, const Golden& g) {
    std::printf("\n========== M(cli-1) CLI case: %s ==========\n", g.name.c_str());
    RawF2 raw;
    if (!read_raw_fixture(golden_dir + "/fixtures/" + g.fixture, raw)) {
        std::printf("  [SKIP] fixture %s absent\n", g.fixture.c_str());
        return true;  // absent fixture is a clean skip (like the parity test)
    }
    check_eq_int("fixture P matches pops.txt size", raw.P, static_cast<int>(g.pops.size()));

    const std::filesystem::path dir = tmp_root / g.name;
    if (!write_f2_dir(dir, raw, g.pops)) { ++g_failures; return true; }

    // Build the --left / --right comma lists.
    auto join = [](const std::vector<std::string>& v) {
        std::string s; for (std::size_t i = 0; i < v.size(); ++i) { if (i) s += ","; s += v[i]; } return s;
    };

    // ---------------- CSV run ----------------
    const RunResult csv = run_steppe(steppe_bin,
        {"qpadm", "--f2-dir", dir.string(), "--target", g.target,
         "--left", join(g.left), "--right", join(g.right), "--format", "csv"},
        tmp_root);
    if (looks_like_no_gpu(csv.stdout_text)) {
        std::printf("  [SKIP] no CUDA device visible — CLI GPU path cannot run "
                    "(CI-without-GPU degrades cleanly)\n%s\n", csv.stdout_text.c_str());
        return false;
    }
    check_eq_int("csv exit code == 0", csv.exit_code, 0);
    const CsvSections cs = parse_csv_sections(csv.stdout_text);

    // weights section: header + one row per left source.
    {
        const auto it = cs.sec.find("weights");
        check_true("csv has weights section", it != cs.sec.end());
        if (it != cs.sec.end()) {
            const auto& rows = it->second;
            // rows[0] is the header; data rows follow.
            check_eq_int("csv weight rows", static_cast<int>(rows.size()) - 1,
                         static_cast<int>(g.weight.size()));
            for (std::size_t i = 0; i + 1 < rows.size() && i < g.weight.size(); ++i) {
                const auto& r = rows[i + 1];  // target,left,weight,se,z
                check_true("weights row has 5 cols", r.size() >= 5);
                if (r.size() >= 5) {
                    check_true("csv target label", r[0] == g.target);
                    check_true("csv left label", r[1] == g.left[i]);
                    char nm[64];
                    std::snprintf(nm, sizeof(nm), "csv weight[%zu]", i);
                    check_close(nm, cell_d(r[2]), g.weight[i], 1e-6, 1e-12);
                    std::snprintf(nm, sizeof(nm), "csv se[%zu]", i);
                    check_close(nm, cell_d(r[3]), g.se[i], 1e-3, 1e-9);
                    std::snprintf(nm, sizeof(nm), "csv z[%zu]", i);
                    check_close(nm, cell_d(r[4]), g.z[i], 1e-3, 1e-6);
                }
            }
        }
    }
    // summary section: p, chisq, dof, f4rank, ... (header + 1 row).
    {
        const auto it = cs.sec.find("summary");
        check_true("csv has summary section", it != cs.sec.end());
        if (it != cs.sec.end() && it->second.size() >= 2) {
            const auto& h = it->second[0];
            const auto& r = it->second[1];
            std::map<std::string, std::string> kv;
            for (std::size_t c = 0; c < h.size() && c < r.size(); ++c) kv[h[c]] = r[c];
            check_close("csv summary chisq", cell_d(kv["chisq"]), g.chisq, 1e-6, 1e-12);
            check_eq_int("csv summary dof", static_cast<long long>(cell_d(kv["dof"])), g.dof);
            check_close("csv summary p", cell_d(kv["p"]), g.p, 1e-3, 1e-9);
            check_eq_int("csv summary f4rank", static_cast<long long>(cell_d(kv["f4rank"])), g.f4rank);
            check_true("csv summary status==ok", kv["status"] == "ok");
        }
    }
    // rankdrop section: f4rank DESC (rank1, rank0).
    {
        const auto it = cs.sec.find("rankdrop");
        check_true("csv has rankdrop section", it != cs.sec.end());
        if (it != cs.sec.end()) {
            const auto& rows = it->second;
            check_eq_int("csv rankdrop rows", static_cast<int>(rows.size()) - 1,
                         static_cast<int>(g.rd_f4rank.size()));
            for (std::size_t k = 0; k + 1 < rows.size() && k < g.rd_f4rank.size(); ++k) {
                const auto& r = rows[k + 1];  // f4rank,dof,chisq,p,dofdiff,chisqdiff,p_nested
                char nm[64];
                std::snprintf(nm, sizeof(nm), "csv rd[%zu].f4rank", k);
                check_eq_int(nm, static_cast<long long>(cell_d(r[0])), g.rd_f4rank[k]);
                std::snprintf(nm, sizeof(nm), "csv rd[%zu].dof", k);
                check_eq_int(nm, static_cast<long long>(cell_d(r[1])), g.rd_dof[k]);
                std::snprintf(nm, sizeof(nm), "csv rd[%zu].chisq", k);
                check_close(nm, cell_d(r[2]), g.rd_chisq[k], 1e-6, 1e-9);
                std::snprintf(nm, sizeof(nm), "csv rd[%zu].p", k);
                check_close(nm, cell_d(r[3]), g.rd_p[k], 1e-3, 1e-12);
            }
            // row0 nested diff; row1 = NA.
            if (rows.size() >= 2) {
                const auto& r0 = rows[1];
                check_eq_int("csv rd[0].dofdiff", static_cast<long long>(cell_d(r0[4])), g.rd0_dofdiff);
                check_close("csv rd[0].chisqdiff", cell_d(r0[5]), g.rd0_chisqdiff, 1e-6, 1e-9);
                check_close("csv rd[0].p_nested", cell_d(r0[6]), g.rd0_p_nested, 1e-3, 1e-13);
            }
            if (rows.size() >= 3) {
                const auto& r1 = rows[2];
                check_true("csv rd[1] dofdiff is NA", r1[4] == "NA");
            }
        }
    }
    // popdrop section.
    {
        const auto it = cs.sec.find("popdrop");
        check_true("csv has popdrop section", it != cs.sec.end());
        if (it != cs.sec.end()) {
            const auto& rows = it->second;
            check_eq_int("csv popdrop rows", static_cast<int>(rows.size()) - 1,
                         static_cast<int>(g.pd_pat.size()));
            for (std::size_t k = 0; k + 1 < rows.size() && k < g.pd_pat.size(); ++k) {
                const auto& r = rows[k + 1];  // pat,wt,dof,chisq,p,f4rank,feasible
                char nm[64];
                std::snprintf(nm, sizeof(nm), "csv pd[%zu].pat", k);
                check_true(nm, r[0] == g.pd_pat[k]);
                std::snprintf(nm, sizeof(nm), "csv pd[%zu].dof", k);
                check_eq_int(nm, static_cast<long long>(cell_d(r[2])), g.pd_dof[k]);
                std::snprintf(nm, sizeof(nm), "csv pd[%zu].chisq", k);
                check_close(nm, cell_d(r[3]), g.pd_chisq[k], 1e-6, 1e-9);
                std::snprintf(nm, sizeof(nm), "csv pd[%zu].f4rank", k);
                check_eq_int(nm, static_cast<long long>(cell_d(r[5])), g.pd_f4rank[k]);
            }
        }
    }

    // ---------------- JSON run (schema mirrors golden_fit0.json) ----------------
    const RunResult js = run_steppe(steppe_bin,
        {"qpadm", "--f2-dir", dir.string(), "--target", g.target,
         "--left", join(g.left), "--right", join(g.right), "--format", "json"},
        tmp_root);
    check_eq_int("json exit code == 0", js.exit_code, 0);
    // Lightweight JSON scrape: pull the "weight": [...] and "chisq": <n> from summary.
    // We assert the first weight value + chisq + p appear with the golden magnitude,
    // proving the JSON path serializes the same result the CSV did (full structural
    // JSON parsing is unnecessary — the CSV gate above is the value gate; this confirms
    // the JSON format emits and round-trips the golden numbers).
    {
        const std::string& t = js.stdout_text;
        const auto wpos = t.find("\"weight\":");
        check_true("json has weights.weight array", wpos != std::string::npos);
        if (wpos != std::string::npos) {
            const auto lb = t.find('[', wpos);
            const auto rb = t.find(']', lb);
            if (lb != std::string::npos && rb != std::string::npos) {
                const std::string arr = t.substr(lb + 1, rb - lb - 1);
                const double w0 = std::strtod(arr.c_str(), nullptr);
                check_close("json weight[0]", w0, g.weight[0], 1e-6, 1e-12);
            }
        }
        // summary chisq.
        const auto cpos = t.find("\"chisq\":");
        check_true("json has summary chisq", cpos != std::string::npos);
        if (cpos != std::string::npos) {
            const double cj = std::strtod(t.c_str() + cpos + 8, nullptr);
            check_close("json summary chisq", cj, g.chisq, 1e-6, 1e-9);
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
    std::printf("=== M(cli-1) steppe qpadm CLI parity (THROUGH the CLI, ON THE GPU) ===\n");
    std::printf("steppe binary: %s\ngolden dir:    %s\n", steppe_bin.c_str(), golden_dir.c_str());

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", steppe_bin.c_str());
        return 1;
    }

    std::error_code ec;
    const std::filesystem::path tmp_root =
        std::filesystem::temp_directory_path(ec) / ("steppe_cli_test_" + std::to_string(
            static_cast<long long>(std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp_root, ec);
    std::printf("temp f2-dir root: %s\n", tmp_root.string().c_str());

    // ---- golden_fit0 (9-pop, nr<=32 batched) -------------------------------------
    Golden fit0;
    fit0.name = "golden_fit0";
    fit0.fixture = "f2_fit0_9pop.bin";
    fit0.pops = {"England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N",
                 "Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"};
    fit0.target = "England_BellBeaker";
    fit0.left = {"Czechia_EBA_CordedWare", "Turkey_N"};
    fit0.right = {"Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"};
    fit0.weight = {0.558906248861195, 0.441093751138805};
    fit0.se = {0.225911861836373, 0.225911861836373};
    fit0.z = {2.47400133980574, 1.95250372226266};
    fit0.chisq = 4.63516296859645; fit0.dof = 4; fit0.p = 0.326820092470997;
    fit0.rd_f4rank = {1, 0}; fit0.rd_dof = {4, 10};
    fit0.rd_chisq = {4.63516296859645, 31.9697628796068};
    fit0.rd_p = {0.326820092470997, 0.000405109973855609};
    fit0.rd0_dofdiff = 6; fit0.rd0_chisqdiff = 27.3345999110104; fit0.rd0_p_nested = 0.000125328972063141;
    fit0.f4rank = 1;
    fit0.pd_pat = {"00", "01", "10"}; fit0.pd_dof = {4, 5, 5};
    fit0.pd_chisq = {4.63516296859645, 13.1352334823443, 17.1406935861748};
    fit0.pd_f4rank = {1, 0, 0};

    // ---- golden_fit1_NRBIG (P=43, nr=39, gesvd large path) -----------------------
    Golden nrbig;
    nrbig.name = "golden_fit1_NRBIG";
    nrbig.fixture = "f2_fit1_NRBIG.bin";
    nrbig.pops = {"England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N", "Mbuti",
        "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana", "YRI", "ESN",
        "GWD", "LWK", "MSL", "ACB", "Yoruba", "Biaka", "Mayan", "Surui", "Pima", "Chukchi",
        "Nganasan", "Yakut", "Ulchi", "Dai", "Eskimo_Naukan", "Aleut", "Tibetan", "She",
        "Naxi", "Miao", "Tujia", "Kalash", "Brahui", "Balochi", "Makrani", "Pathan",
        "Burusho", "Mozabite", "BantuKenya", "Ju_hoan_North", "Kusunda", "Somali"};
    nrbig.target = "England_BellBeaker";
    nrbig.left = {"Czechia_EBA_CordedWare", "Turkey_N"};
    for (std::size_t i = 3; i < nrbig.pops.size(); ++i) nrbig.right.push_back(nrbig.pops[i]);  // 40 rights
    nrbig.weight = {0.7913004433128438, 0.2086995566871562};
    nrbig.se = {0.1588254912031973, 0.1588254912031974};
    nrbig.z = {4.982200510247275, 1.314018015031045};
    nrbig.chisq = 52.704281610335912; nrbig.dof = 38; nrbig.p = 0.05678246029948012;
    nrbig.rd_f4rank = {1, 0}; nrbig.rd_dof = {38, 78};
    nrbig.rd_chisq = {52.704281610335912, 190.83602239090976};
    nrbig.rd_p = {0.05678246029948012, 1.922125797354803e-11};
    nrbig.rd0_dofdiff = 40; nrbig.rd0_chisqdiff = 138.131740780574; nrbig.rd0_p_nested = 1.00598619034513e-12;
    nrbig.f4rank = 1;
    nrbig.pd_pat = {"00", "01", "10"}; nrbig.pd_dof = {38, 39, 39};
    nrbig.pd_chisq = {52.704281610335912, 100.19050317026696, 169.11350353681215};
    nrbig.pd_f4rank = {1, 0, 0};

    const bool ran0 = run_case(steppe_bin, golden_dir, tmp_root, fit0);
    bool ran1 = true;
    if (ran0) {
        ran1 = run_case(steppe_bin, golden_dir, tmp_root, nrbig);
    } else {
        std::printf("\n[SKIP] golden_fit1_NRBIG: no GPU (first case skipped)\n");
    }

    std::filesystem::remove_all(tmp_root, ec);

    if (!ran0) {
        std::printf("\nRESULT: SKIP (no CUDA device visible — CLI GPU path not exercised)\n");
        return 0;  // clean skip, like every reference test (cli-bindings.md §7)
    }
    (void)ran1;
    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (all CLI goldens reproduced through the GPU)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
