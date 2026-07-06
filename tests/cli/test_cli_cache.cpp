// tests/cli/test_cli_cache.cpp
//
// Native `steppe cache` host-only inspector gate. Drives the BUILT steppe binary's
// `cache ls/show/verify` over the committed real 9-pop example cache
// (docs/examples/f2_9pop) and asserts the same integrity behaviors the steppe-cache
// Python tool has — so the two front ends stay in parity:
//   * verify the pristine cache            -> exit 0, all [OK] rows
//   * a truncated f2.bin                   -> verify FAIL (nonzero)
//   * meta.json P that contradicts header  -> verify [FAIL] header P (nonzero)
//   * a tampered f2_cache_id (no sha256:)  -> verify [FAIL] f2_cache_id (nonzero) — NOT "no stored id"
//   * a minimal meta with no stored id     -> verify report-only (exit 0)
//   * a non-FP64 dtype header              -> show rejects (not a cache)
//   * ls of a missing root                 -> exit 0 (os.walk-yields-nothing parity)
//   * ls of a tree                         -> lists the cache header-only
//
// PLAIN C++ HOST TU: no CUDA header, NO GPU — the cache tool is host-only, so this
// runs everywhere and SKIPs (exit 0) only if the example cache is absent. argv[1] =
// the built steppe binary; argv[2] = the example cache dir. NO SYNTHETIC DATA: the
// fixture is the committed real-AADR example cache. Self-checking main(); CTest gates
// on the exit code.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#  include <sys/wait.h>  // WIFEXITED / WEXITSTATUS for std::system's status word
#endif

namespace {

namespace fs = std::filesystem;
int g_failures = 0;

void check_true(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

struct RunResult {
    int exit_code = -1;
    std::string out;
};

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const fs::path& tmp) {
    RunResult rr;
    const fs::path outf = tmp / "cli_cache_stdout.txt";
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
    if (f) { std::ostringstream ss; ss << f.rdbuf(); rr.out = ss.str(); }
    return rr;
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

void write_file(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << s;
}

// Copy the example cache into dst (recursive).
bool copy_cache(const fs::path& src, const fs::path& dst) {
    std::error_code ec;
    fs::create_directories(dst, ec);
    fs::copy(src, dst, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    if (ec) {
        std::printf("  [FAIL] copy %s -> %s: %s\n", src.string().c_str(),
                    dst.string().c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: %s <steppe-binary> <example-cache-dir>\n", argv[0]);
        return 2;
    }
    const std::string bin = argv[1];
    const fs::path example = argv[2];
    std::printf("=== native `steppe cache` host-only parity gate ===\n");
    std::printf("steppe binary: %s\nexample cache: %s\n", bin.c_str(), example.string().c_str());

    if (!fs::exists(bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", bin.c_str());
        return 1;
    }
    if (!fs::exists(example / "f2.bin")) {
        std::printf("\nRESULT: SKIP (example cache absent: %s)\n", example.string().c_str());
        return 0;  // absent fixture is a clean skip
    }

    std::error_code ec;
    const fs::path T = fs::temp_directory_path(ec) / ("steppe_cli_cache_" + std::to_string(
        static_cast<long long>(fs::file_time_type::clock::now().time_since_epoch().count())));
    fs::create_directories(T, ec);

    // 1) verify the pristine committed cache -> exit 0, every check OK.
    {
        const RunResult r = run_steppe(bin, {"cache", "verify", example.string()}, T);
        check_true("verify golden exit 0", r.exit_code == 0);
        check_true("verify golden: f2.bin size OK", contains(r.out, "[OK  ] f2.bin size"));
        check_true("verify golden: header P OK", contains(r.out, "[OK  ] header P"));
        check_true("verify golden: f2_cache_id OK", contains(r.out, "[OK  ] f2_cache_id"));
        check_true("verify golden: pops_sha256 OK", contains(r.out, "[OK  ] pops_sha256"));
    }

    // 2) show -> exit 0, header facts + size OK.
    {
        const RunResult r = run_steppe(bin, {"cache", "show", example.string()}, T);
        check_true("show golden exit 0", r.exit_code == 0);
        check_true("show golden: prints P", contains(r.out, "P (populations):"));
        check_true("show golden: size OK mark", contains(r.out, "[OK]"));
    }

    // 3) ls a tree holding one copy -> lists it header-only.
    {
        const fs::path root = T / "tree";
        check_true("copy for ls", copy_cache(example, root / "mycache"));
        const RunResult r = run_steppe(bin, {"cache", "ls", root.string()}, T);
        check_true("ls tree exit 0", r.exit_code == 0);
        check_true("ls tree: N_BLOCK header", contains(r.out, "N_BLOCK"));
        check_true("ls tree: lists mycache", contains(r.out, "mycache"));
    }

    // 4) truncated f2.bin -> verify FAIL (size + hash both diverge).
    {
        const fs::path d = T / "trunc";
        copy_cache(example, d);
        const auto sz = fs::file_size(d / "f2.bin", ec);
        fs::resize_file(d / "f2.bin", sz - 1, ec);
        const RunResult r = run_steppe(bin, {"cache", "verify", d.string()}, T);
        check_true("verify truncated -> nonzero", r.exit_code != 0);
    }

    // 5) meta.json P contradicts the header -> [FAIL] header P (the cross-check).
    {
        const fs::path d = T / "badP";
        copy_cache(example, d);
        std::string m = read_file(d / "meta.json");
        m = std::regex_replace(m, std::regex("\"P\"\\s*:\\s*[0-9]+"), "\"P\": 8");
        write_file(d / "meta.json", m);
        const RunResult r = run_steppe(bin, {"cache", "verify", d.string()}, T);
        check_true("verify bad meta P -> nonzero", r.exit_code != 0);
        check_true("verify bad meta P -> [FAIL] header P", contains(r.out, "[FAIL] header P"));
    }

    // 6) tampered f2_cache_id lacking the sha256: prefix -> [FAIL], NOT read as absent.
    {
        const fs::path d = T / "badid";
        copy_cache(example, d);
        std::string m = read_file(d / "meta.json");
        m = std::regex_replace(m, std::regex("(\"f2_cache_id\"\\s*:\\s*\")[^\"]*(\")"), "$1TAMPERED$2");
        write_file(d / "meta.json", m);
        const RunResult r = run_steppe(bin, {"cache", "verify", d.string()}, T);
        check_true("verify tampered id -> nonzero", r.exit_code != 0);
        check_true("verify tampered id -> [FAIL] f2_cache_id", contains(r.out, "[FAIL] f2_cache_id"));
        check_true("verify tampered id NOT read as 'no stored id'",
                   !contains(r.out, "[--  ] f2_cache_id"));
    }

    // 7) minimal meta with no stored id -> report-only, exit 0.
    {
        const fs::path d = T / "min";
        copy_cache(example, d);
        write_file(d / "meta.json", "{\"format\":\"STPF2BK1\",\"P\":9,\"n_block\":710}");
        const RunResult r = run_steppe(bin, {"cache", "verify", d.string()}, T);
        check_true("verify minimal meta -> exit 0", r.exit_code == 0);
        check_true("verify minimal meta -> no stored id", contains(r.out, "no stored id"));
    }

    // 8) non-FP64 dtype header -> show rejects it (the dtype gate).
    {
        const fs::path d = T / "baddt";
        copy_cache(example, d);
        std::fstream f(d / "f2.bin", std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(12);  // the dtype word: after the 8-byte magic + 4-byte version
        const char two = 2; f.write(&two, 1); f.close();
        const RunResult r = run_steppe(bin, {"cache", "show", d.string()}, T);
        check_true("show non-FP64 dtype -> nonzero", r.exit_code != 0);
        check_true("show non-FP64 dtype -> not a cache", contains(r.out, "not an STPF2BK1 cache"));
    }

    // 9) ls a missing root -> exit 0 (os.walk-yields-nothing parity, not an error).
    {
        const RunResult r = run_steppe(bin, {"cache", "ls", (T / "does_not_exist").string()}, T);
        check_true("ls missing root -> exit 0", r.exit_code == 0);
    }

    fs::remove_all(T, ec);
    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (native cache ls/show/verify parity; 9 behaviors)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
