// tests/cli/test_cli_rotate.cpp
//
// M(cli-3) ACCEPTANCE GATE (cli-bindings.md §7 row `tests/cli/test_cli_rotate`): the
// `steppe qpadm-rotate` CLI reproduces the REAL-AADR AT2 rotation golden (golden_rot.json,
// 84 models) THROUGH the CLI, ON THE GPU — the CLI counterpart of the engine-level
// test_qpadm_rotation.cu.
//
// NO SYNTHETIC DATA (memory real-data-only): it writes the COMMITTED real golden f2
// fixture (fixtures/f2_rot.bin) into a temp f2-dir (STPF2BK1 f2.bin + pops.txt = the
// golden pop_order + meta.json) — exactly what a real <dir> contains — then runs the
// BUILT `steppe qpadm-rotate` binary with the golden's EXACT inputs (target/pool/right,
// --min-sources 2 --max-sources 3, defaults reproducing fudge 1e-4 / als 20 / alpha 0.05
// / jackknife All), parses the per-model CSV (and a second JSON run), and diffs every one
// of the 84 rows against golden_rot.json at the SAME tiers the engine gate uses
// (test_qpadm_rotation.cu:421-480): weights TIGHT (rtol 1e-5 + atol 1e-5), f4rank EXACT,
// feasible DECISION-match, p LOOSE (rtol 1e-3 + atol 1e-6), left labels exact, status ok.
//
// PLAIN C++ host TU (NO CUDA header): it spawns the steppe binary and parses stdout; the
// GPU work happens inside that child. SKIPs cleanly (exit 0) when no CUDA device is
// visible — identical to every reference/CLI test (cli-bindings.md §7) — by detecting the
// child's "no CUDA device" fault. The harness helpers (write_f2_dir/run_steppe/
// looks_like_no_gpu/parse_csv_sections/split_csv_line/cell_d) are reused from the
// test_cli_qpadm pattern; the golden JSON parser is reused from test_qpadm_rotation.cu.
// Self-checking main(); CTest gates on the exit code.

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
// Harness — reused verbatim from test_cli_qpadm.cpp (raw-fixture read, f2-dir
// write, child-process run, no-GPU detection, CSV section parse).
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

// Write the f2-dir: f2.bin (STPF2BK1) + pops.txt + meta.json. vpair written as zeros
// (the fit reads block_sizes, NOT vpair) — the same way the parity/CLI tests do it.
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
       << ",\n  \"provenance\": \"test fixture (real-AADR rotation golden f2)\"\n}\n";
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
    const std::filesystem::path outf = tmp / "cli_rot_stdout.txt";
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
// Golden JSON parser — reused from test_qpadm_rotation.cu:168-303 (dependency-free,
// relies on the machine-generated regular schema steppe.golden.at2.qpadm.rotation/1).
// ============================================================================
struct GoldenModel {
    std::vector<std::string> left;
    std::vector<double> weight, se, z;
    double p = 0.0;
    int f4rank = 0;
    bool feasible = false;
};
struct GoldenRot {
    std::vector<std::string> pop_order;
    std::string target;
    std::vector<std::string> right;
    std::vector<GoldenModel> models;
};

bool slurp(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return true;
}

std::vector<std::string> parse_str_array(const std::string& s, std::size_t& pos) {
    std::vector<std::string> out;
    const std::size_t lb = s.find('[', pos);
    const std::size_t rb = s.find(']', lb);
    std::size_t i = lb + 1;
    while (i < rb) {
        const std::size_t q0 = s.find('"', i);
        if (q0 == std::string::npos || q0 >= rb) break;
        const std::size_t q1 = s.find('"', q0 + 1);
        out.push_back(s.substr(q0 + 1, q1 - q0 - 1));
        i = q1 + 1;
    }
    pos = rb + 1;
    return out;
}

std::vector<double> parse_num_array(const std::string& s, std::size_t& pos) {
    std::vector<double> out;
    const std::size_t lb = s.find('[', pos);
    const std::size_t rb = s.find(']', lb);
    std::size_t i = lb + 1;
    while (i < rb) {
        while (i < rb && (s[i] == ' ' || s[i] == ',' || s[i] == '\n' || s[i] == '\t')) ++i;
        if (i >= rb) break;
        char* end = nullptr;
        const double v = std::strtod(s.c_str() + i, &end);
        out.push_back(v);
        i = static_cast<std::size_t>(end - s.c_str());
    }
    pos = rb + 1;
    return out;
}

std::string find_str_value(const std::string& s, const char* key, std::size_t from) {
    const std::size_t k = s.find(key, from);
    if (k == std::string::npos) return {};
    const std::size_t colon = s.find(':', k);
    const std::size_t q0 = s.find('"', colon);
    const std::size_t q1 = s.find('"', q0 + 1);
    return s.substr(q0 + 1, q1 - q0 - 1);
}

bool parse_golden(const std::string& path, GoldenRot& g) {
    std::string s;
    if (!slurp(path, s)) { std::printf("  [FAIL] cannot read %s\n", path.c_str()); return false; }
    g.target = find_str_value(s, "\"target\"", 0);
    {
        std::size_t p = s.find("\"right\"");
        if (p != std::string::npos) { p = s.find(':', p); g.right = parse_str_array(s, p); }
    }
    {
        std::size_t p = s.find("\"pop_order\"");
        if (p != std::string::npos) { p = s.find(':', p); g.pop_order = parse_str_array(s, p); }
    }
    std::size_t cur = s.find("\"models\"");
    if (cur == std::string::npos) { std::printf("  [FAIL] no models[]\n"); return false; }
    while (true) {
        const std::size_t mi = s.find("\"model_index\"", cur);
        if (mi == std::string::npos) break;
        const std::size_t next = s.find("\"model_index\"", mi + 1);
        const std::size_t blk_end = (next == std::string::npos) ? s.size() : next;
        GoldenModel m;
        std::size_t p = s.find("\"left\"", mi);
        if (p != std::string::npos && p < blk_end) { p = s.find(':', p); m.left = parse_str_array(s, p); }
        p = s.find("\"weight\"", mi);
        if (p != std::string::npos && p < blk_end) { p = s.find(':', p); m.weight = parse_num_array(s, p); }
        p = s.find("\"se\"", mi);
        if (p != std::string::npos && p < blk_end) { p = s.find(':', p); m.se = parse_num_array(s, p); }
        p = s.find("\"z\"", mi);
        if (p != std::string::npos && p < blk_end) { p = s.find(':', p); m.z = parse_num_array(s, p); }
        {
            std::size_t pp = s.find("\"p\"", mi);
            if (pp != std::string::npos && pp < blk_end) {
                const std::size_t colon = s.find(':', pp);
                m.p = std::strtod(s.c_str() + colon + 1, nullptr);
            }
        }
        {
            std::size_t fp = s.find("\"f4rank\"", mi);
            if (fp != std::string::npos && fp < blk_end) {
                const std::size_t colon = s.find(':', fp);
                m.f4rank = static_cast<int>(std::strtol(s.c_str() + colon + 1, nullptr, 10));
            }
        }
        {
            std::size_t fb = s.find("\"feasible\"", mi);
            if (fb != std::string::npos && fb < blk_end) {
                const std::size_t colon = s.find(':', fb);
                m.feasible = (s.find("true", colon) < s.find("false", colon));
            }
        }
        g.models.push_back(std::move(m));
        cur = blk_end;
        if (next == std::string::npos) break;
    }
    return !g.models.empty();
}

// Join a vector of labels with sep (build the CLI --pool/--right comma lists + compare
// the row's semicolon-joined `left` field against the golden's left[]).
std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) { if (i) s += sep; s += v[i]; }
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
    std::printf("=== M(cli-3) steppe qpadm-rotate CLI parity (THROUGH the CLI, ON THE GPU) ===\n");
    std::printf("steppe binary: %s\ngolden dir:    %s\n", steppe_bin.c_str(), golden_dir.c_str());

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", steppe_bin.c_str());
        return 1;
    }

    // ---- read the rotation golden + the co-matching f2 fixture -------------------
    GoldenRot G;
    if (!parse_golden(golden_dir + "/golden_rot.json", G)) {
        std::printf("RESULT: FAIL (golden parse)\n"); return 1;
    }
    std::printf("  golden: target=%s, %zu models, pop_order P=%zu\n",
                G.target.c_str(), G.models.size(), G.pop_order.size());

    RawF2 raw;
    if (!read_raw_fixture(golden_dir + "/fixtures/f2_rot.bin", raw)) {
        std::printf("  [SKIP] f2_rot fixture absent\n");
        std::printf("\nRESULT: SKIP (rotation fixture absent)\n");
        return 0;  // absent fixture is a clean skip (like the parity test)
    }
    check_eq_int("fixture P == pop_order size", raw.P, static_cast<int>(G.pop_order.size()));

    std::error_code ec;
    const std::filesystem::path tmp_root =
        std::filesystem::temp_directory_path(ec) / ("steppe_cli_rot_test_" + std::to_string(
            static_cast<long long>(std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp_root, ec);
    std::printf("temp f2-dir root: %s\n", tmp_root.string().c_str());

    // pops.txt == the golden pop_order (== target ++ pool ++ right; the rotation fixture's
    // P axis order). The CLI resolves --target/--pool/--right against THESE labels.
    const std::filesystem::path dir = tmp_root / "rot";
    if (!write_f2_dir(dir, raw, G.pop_order)) {
        ++g_failures;
        std::filesystem::remove_all(tmp_root, ec);
        std::printf("\nRESULT: FAIL (could not write f2-dir)\n");
        return 1;
    }

    // ---- the golden's EXACT CLI inputs (golden_rot.json metadata) ----------------
    const std::string target = "England_BellBeaker";
    const std::string pool =
        "Czechia_EBA_CordedWare,Turkey_N,Russia_Samara_EBA_Yamnaya,"
        "Luxembourg_Loschbour_Mesolithic,Russia_Karelia_Mesolithic,Spain_EN,"
        "England_N,Russia_Khakassia_Afanasievo";
    const std::string right = join(G.right, ",");  // Mbuti,...,Karitiana (nr=5)

    // ---------------- CSV run ----------------
    const RunResult csv = run_steppe(steppe_bin,
        {"qpadm-rotate", "--f2-dir", dir.string(), "--target", target,
         "--pool", pool, "--right", right,
         "--min-sources", "2", "--max-sources", "3",
         "--format", "csv", "--device", "0"},
        tmp_root);
    if (looks_like_no_gpu(csv.stdout_text)) {
        std::printf("  [SKIP] no CUDA device visible — CLI GPU rotation cannot run "
                    "(CI-without-GPU degrades cleanly)\n");
        std::filesystem::remove_all(tmp_root, ec);
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;
    }
    check_eq_int("csv exit code == 0", csv.exit_code, 0);

    const CsvSections cs = parse_csv_sections(csv.stdout_text);
    const auto it = cs.sec.find("rotation");
    check_true("csv has rotation section", it != cs.sec.end());
    if (it != cs.sec.end()) {
        const auto& rows = it->second;  // rows[0] = header; one data row per model.
        check_eq_int("csv rotation rows == 84 models",
                     static_cast<int>(rows.size()) - 1, static_cast<int>(G.models.size()));
        // Map the header to column indices (order-independent).
        std::map<std::string, std::size_t> col;
        if (!rows.empty()) for (std::size_t c = 0; c < rows[0].size(); ++c) col[rows[0][c]] = c;
        auto has = [&](const char* k) { return col.find(k) != col.end(); };
        check_true("csv header has model_index", has("model_index"));
        check_true("csv header has left", has("left"));
        check_true("csv header has p", has("p"));
        check_true("csv header has f4rank", has("f4rank"));
        check_true("csv header has feasible", has("feasible"));
        check_true("csv header has status", has("status"));
        check_true("csv header has weights", has("weights"));

        for (std::size_t i = 0; i + 1 < rows.size() && i < G.models.size(); ++i) {
            const auto& r = rows[i + 1];
            const GoldenModel& gm = G.models[i];
            // model_index dense 0..n-1 (enumeration order aligns with the golden rows).
            check_eq_int("model_index", static_cast<long long>(cell_d(r[col["model_index"]])),
                         static_cast<long long>(i));
            // left labels — EXACT string match (semicolon-joined == golden left[]).
            check_true(("m" + std::to_string(i) + " left").c_str(),
                       r[col["left"]] == join(gm.left, ";"));
            // status — "ok" throughout.
            check_true(("m" + std::to_string(i) + " status ok").c_str(),
                       r[col["status"]] == "ok");
            // f4rank — EXACT (== est_rank, the fitted rank nl-1).
            check_eq_int(("m" + std::to_string(i) + " f4rank").c_str(),
                         static_cast<long long>(cell_d(r[col["f4rank"]])), gm.f4rank);
            // feasible — DECISION match.
            const bool feas = (r[col["feasible"]] == "TRUE");
            check_true(("m" + std::to_string(i) + " feasible").c_str(), feas == gm.feasible);
            // p — LOOSE (rtol 1e-3 + atol 1e-6).
            check_close(("m" + std::to_string(i) + " p").c_str(),
                        cell_d(r[col["p"]]), gm.p, 1e-3, 1e-6);
            // weights — TIGHT (rtol 1e-5 + atol 1e-5), per weight. The weights column is
            // semicolon-joined; split and compare element-wise.
            const std::vector<std::string> wcells = [&] {
                std::vector<std::string> v; std::string cur;
                const std::string& w = r[col["weights"]];
                for (char ch : w) { if (ch == ';') { v.push_back(cur); cur.clear(); } else cur += ch; }
                v.push_back(cur);
                return v;
            }();
            check_eq_int(("m" + std::to_string(i) + " weights len").c_str(),
                         static_cast<long long>(wcells.size()),
                         static_cast<long long>(gm.weight.size()));
            for (std::size_t k = 0; k < gm.weight.size() && k < wcells.size(); ++k) {
                char nm[48]; std::snprintf(nm, sizeof(nm), "m%zu w[%zu]", i, k);
                check_close(nm, std::strtod(wcells[k].c_str(), nullptr), gm.weight[k], 1e-5, 1e-5);
            }
        }
    }

    // ---------------- JSON run (schema mirrors golden_rot.json models[]) ----------
    const RunResult js = run_steppe(steppe_bin,
        {"qpadm-rotate", "--f2-dir", dir.string(), "--target", target,
         "--pool", pool, "--right", right,
         "--min-sources", "2", "--max-sources", "3",
         "--format", "json", "--device", "0"},
        tmp_root);
    check_eq_int("json exit code == 0", js.exit_code, 0);
    {
        // Re-parse the CLI's own JSON with the golden parser (same schema), then diff the
        // per-model weights/p/f4rank/feasible/left against the golden — proving the JSON
        // path serializes the SAME 84-model result the CSV did.
        const std::filesystem::path jf = tmp_root / "cli_rot.json";
        { std::ofstream o(jf, std::ios::trunc); o << js.stdout_text; }
        GoldenRot J;
        const bool jok = parse_golden(jf.string(), J);
        check_true("json parses as rotation schema", jok);
        if (jok) {
            check_eq_int("json model count == 84",
                         static_cast<int>(J.models.size()), static_cast<int>(G.models.size()));
            for (std::size_t i = 0; i < J.models.size() && i < G.models.size(); ++i) {
                const GoldenModel& jm = J.models[i];
                const GoldenModel& gm = G.models[i];
                check_true(("json m" + std::to_string(i) + " left").c_str(),
                           jm.left == gm.left);
                check_eq_int(("json m" + std::to_string(i) + " f4rank").c_str(),
                             jm.f4rank, gm.f4rank);
                check_true(("json m" + std::to_string(i) + " feasible").c_str(),
                           jm.feasible == gm.feasible);
                check_close(("json m" + std::to_string(i) + " p").c_str(),
                            jm.p, gm.p, 1e-3, 1e-6);
                check_eq_int(("json m" + std::to_string(i) + " weights len").c_str(),
                             static_cast<long long>(jm.weight.size()),
                             static_cast<long long>(gm.weight.size()));
                for (std::size_t k = 0; k < gm.weight.size() && k < jm.weight.size(); ++k) {
                    char nm[56]; std::snprintf(nm, sizeof(nm), "json m%zu w[%zu]", i, k);
                    check_close(nm, jm.weight[k], gm.weight[k], 1e-5, 1e-5);
                }
            }
        }
    }

    std::filesystem::remove_all(tmp_root, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (all 84 rotation models reproduced through the GPU; "
                    "CSV + JSON)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
