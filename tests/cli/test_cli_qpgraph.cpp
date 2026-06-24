// tests/cli/test_cli_qpgraph.cpp
//
// Single-graph qpGraph CLI acceptance gate: the BUILT `steppe qpgraph` reproduces the AT2
// golden (golden_qpgraph_score.csv / golden_qpgraph_weights.csv; admixtools 2.0.10) for the
// committed WELL-IDENTIFIED topology, RUNNING ON THE GPU. Mirrors test_cli_f3.cpp's harness.
//
// NO SYNTHETIC DATA (memory real-data-only): it stages the COMMITTED real-AADR afprod=FALSE
// f2 fixture (fixtures/f2_qpgraph_9pop.bin — the SAME f2 admixtools::qpgraph(read_f2(...))
// reads) into a temp STPF2BK1 f2-dir + writes the golden edge list to a temp --graph file,
// then runs `steppe qpgraph --f2-dir <dir> --graph <file>` and parses the CSV output.
//
// GATE: the `summary` section's score (1e-4 — the converged optimum, not step-identical
// L-BFGS-B) + the `edges` section's admix weight keyed by parent NAME (pSteppe->aCW, 1e-5).
// PLAIN C++ host TU (NO CUDA header): it spawns the binary; the GPU work is in the child.
// SKIPs cleanly (exit 0) when no CUDA device is visible. Self-checking main(); CTest gates.

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
#  include <sys/wait.h>
#endif

#include "device/f2_disk_format.hpp"  // F2DiskHeader (CUDA-FREE)

namespace {

int g_failures = 0;

void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const double diff = std::fabs(got - want);
    const bool ok = diff <= tol;
    std::printf("  [%s] %-24s got=% .12e want=% .12e |d|=% .3e tol=% .3e\n",
                ok ? "PASS" : "FAIL", what, got, want, diff, tol);
    if (!ok) ++g_failures;
}

struct RawF2 {
    int P = 0, nb = 0;
    std::vector<std::int32_t> block_sizes;
    std::vector<double> f2;
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
    const std::size_t n = static_cast<std::size_t>(P) * static_cast<std::size_t>(P) * static_cast<std::size_t>(nb);
    out.f2.resize(n);
    f.read(reinterpret_cast<char*>(out.f2.data()), static_cast<std::streamsize>(sizeof(double) * n));
    if (!f) { std::printf("  [FAIL] fixture truncated\n"); return false; }
    return true;
}

bool write_f2_dir(const std::filesystem::path& dir, const RawF2& raw,
                  const std::vector<std::string>& pops) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) { std::printf("  [FAIL] mkdir: %s\n", ec.message().c_str()); return false; }
    const std::uint64_t P = static_cast<std::uint64_t>(raw.P);
    const std::uint64_t nb = static_cast<std::uint64_t>(raw.nb);
    const std::uint64_t slab_bytes = P * P * nb * sizeof(double);
    steppe::device::F2DiskHeader hdr{};
    for (int i = 0; i < 8; ++i) hdr.magic[i] = steppe::device::kF2DiskMagic[i];
    hdr.version = steppe::device::kF2DiskVersion;
    hdr.dtype = steppe::device::kF2DiskDtypeFp64;
    hdr.P = raw.P; hdr.n_block = raw.nb;
    hdr.f2_offset = steppe::device::kF2DiskHeaderSize;
    hdr.vpair_offset = hdr.f2_offset + slab_bytes;
    hdr.block_sizes_offset = hdr.vpair_offset + slab_bytes;
    std::ofstream o(dir / "f2.bin", std::ios::binary | std::ios::trunc);
    if (!o) { std::printf("  [FAIL] cannot write f2.bin\n"); return false; }
    o.write(reinterpret_cast<const char*>(&hdr), static_cast<std::streamsize>(sizeof(hdr)));
    o.write(reinterpret_cast<const char*>(raw.f2.data()), static_cast<std::streamsize>(slab_bytes));
    const std::vector<double> vpair_zeros(static_cast<std::size_t>(P * P * nb), 0.0);
    o.write(reinterpret_cast<const char*>(vpair_zeros.data()), static_cast<std::streamsize>(slab_bytes));
    o.write(reinterpret_cast<const char*>(raw.block_sizes.data()),
            static_cast<std::streamsize>(sizeof(std::int32_t) * nb));
    if (!o) { std::printf("  [FAIL] f2.bin write failed\n"); return false; }
    o.close();
    std::ofstream pf(dir / "pops.txt", std::ios::trunc);
    for (const std::string& s : pops) pf << s << "\n";
    pf.close();
    std::ofstream mf(dir / "meta.json", std::ios::trunc);
    mf << "{\n  \"format\": \"STPF2BK1\",\n  \"P\": " << raw.P << ",\n  \"n_block\": " << raw.nb
       << ",\n  \"provenance\": \"test fixture (real-AADR afprod=FALSE qpgraph f2)\"\n}\n";
    mf.close();
    return true;
}

struct RunResult { int exit_code = -1; std::string stdout_text; };

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "cli_qpgraph_stdout.txt";
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
    std::vector<std::string> cells; std::string cur;
    for (char c : line) { if (c == ',') { cells.push_back(cur); cur.clear(); } else cur += c; }
    cells.push_back(cur);
    return cells;
}

const std::vector<std::string> kPops9 = {
    "England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N", "Mbuti", "Israel_Natufian",
    "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"};

const char* kEdgeListCsv =
    "from,to\n"
    "R,Mbuti\nR,nOOA\nnOOA,Papuan\nnOOA,nEAS\nnEAS,Han\nnEAS,Karitiana\nnOOA,nWE\n"
    "nWE,Israel_Natufian\nnWE,nAnat\nnAnat,Turkey_N\nnAnat,England_BellBeaker\nnWE,pIran\n"
    "pIran,Iran_GanjDareh_N\nnEAS,pSteppe\npSteppe,aCW\npIran,aCW\naCW,Czechia_EBA_CordedWare\n";

constexpr double kGoldenScore = 80.0674246076313;
constexpr double kGoldenWeightSteppeToCW = 0.153483829987482;

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: %s <steppe-binary> <golden-dir>\n", argv[0]);
        return 2;
    }
    const std::string bin = argv[1];
    const std::string golden_dir = argv[2];
    std::printf("=== cli_qpgraph: `steppe qpgraph` reproduces the AT2 golden ===\n");

    RawF2 raw;
    if (!read_raw_fixture(golden_dir + "/fixtures/f2_qpgraph_9pop.bin", raw)) return 1;

    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / "steppe_cli_qpgraph_test";
    std::error_code ec; std::filesystem::create_directories(tmp, ec);
    const std::filesystem::path f2dir = tmp / "f2dir";
    if (!write_f2_dir(f2dir, raw, kPops9)) return 1;
    const std::filesystem::path graphf = tmp / "graph.csv";
    { std::ofstream g(graphf, std::ios::trunc); g << kEdgeListCsv; }

    const RunResult rr = run_steppe(
        bin, {"qpgraph", "--f2-dir", f2dir.string(), "--graph", graphf.string()}, tmp);

    if (looks_like_no_gpu(rr.stdout_text)) {
        std::printf("  [SKIP] no CUDA device visible — qpgraph CLI not exercised.\n%s\n",
                    rr.stdout_text.c_str());
        return 0;
    }
    if (rr.exit_code != 0) {
        std::printf("  [FAIL] `steppe qpgraph` exit=%d\n%s\n", rr.exit_code, rr.stdout_text.c_str());
        return 1;
    }

    // Parse the two CSV sections (# section: edges / # section: summary).
    double score = std::nan(""), w_steppe = std::nan("");
    {
        std::istringstream ss(rr.stdout_text);
        std::string line, section;
        std::map<std::string, std::size_t> col;
        bool need_header = false;
        while (std::getline(ss, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (line.rfind("# section:", 0) == 0) {
                section = line.substr(line.find(':') + 1);
                while (!section.empty() && section.front() == ' ') section.erase(section.begin());
                need_header = true;
                col.clear();
                continue;
            }
            const std::vector<std::string> cells = split_csv_line(line);
            if (need_header) {
                for (std::size_t c = 0; c < cells.size(); ++c) col[cells[c]] = c;
                need_header = false;
                continue;
            }
            if (section == "summary") {
                if (col.count("score")) score = std::strtod(cells[col["score"]].c_str(), nullptr);
            } else if (section == "edges") {
                // a row: from,to,type,weight — the admix row pSteppe->aCW.
                if (col.count("from") && col.count("to") && col.count("weight") && col.count("type")) {
                    if (cells[col["from"]] == "pSteppe" && cells[col["to"]] == "aCW" &&
                        cells[col["type"]] == "admix")
                        w_steppe = std::strtod(cells[col["weight"]].c_str(), nullptr);
                }
            }
        }
    }

    check_close("score", score, kGoldenScore, 0.0, 1e-4);
    check_close("weight pSteppe->aCW", w_steppe, kGoldenWeightSteppeToCW, 0.0, 1e-5);

    std::filesystem::remove_all(tmp, ec);
    std::printf("\nRESULT: %s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
