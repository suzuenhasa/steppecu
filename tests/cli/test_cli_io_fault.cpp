// tests/cli/test_cli_io_fault.cpp
//
// B1 (kimiactions §B1 — I/O fault taxonomy) acceptance gate: a TORN / SHORT WRITE on the
// --out destination must return kExitIoError (4), NOT silently exit 0 with a truncated file.
// The shared emit_to_destination helper (src/app/cmd_emit.hpp) does open->write->FLUSH->
// good() after every emit; the flush is load-bearing (a small buffered write has not hit the
// fd until flush, so the badbit is only observable after it).
//
// THE FAULT INJECTION: --out /dev/full. open() succeeds, but every write()/flush() to
// /dev/full returns ENOSPC, so the ofstream sets badbit -> finish_emit() returns
// kExitIoError. This is exactly the full-disk / quota case the goldens never exercise.
//
// CONTROL: a normal --out <tmp>/ok.csv exits 0 and produces a non-empty, well-formed table
// (the success path is byte-unchanged: flush is a content no-op, good() is true).
//
// NO SYNTHETIC DATA (memory real-data-only): it writes the COMMITTED real 9-pop golden f2
// fixture (fixtures/f2_fit0_9pop.bin — the SAME tensor cli_f4 / qpadm use) into a temp
// f2-dir, then runs the BUILT `steppe f4` binary. The f4 command is the representative emit
// site; every cmd_*.cpp now routes through the SAME helper, so one site proves the contract.
//
// PLAIN C++ host TU (NO CUDA header): it spawns the steppe binary and inspects the exit code
// + stderr. SKIPs cleanly (exit 0) when no CUDA device is visible (the f4 compute runs on the
// GPU before the emit) — identical to test_cli_f4. The harness helpers (read_raw_fixture /
// write_f2_dir / run_steppe / looks_like_no_gpu) are reused verbatim from the cli_f4 pattern.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#  include <sys/wait.h>  // WIFEXITED / WEXITSTATUS for std::system's status word
#endif

#include "device/f2_disk_format.hpp"  // F2DiskHeader, kF2DiskMagic/Version/DtypeFp64 (CUDA-FREE)

namespace {

int g_failures = 0;

void check_eq_int(const char* what, long long got, long long want) {
    const bool ok = (got == want);
    if (!ok) std::printf("  [FAIL] %-32s got=%lld want=%lld\n", what, got, want);
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
    if (!f) { std::printf("  [SKIP] cannot open fixture: %s\n", path.c_str()); return false; }
    std::int32_t P = 0, nb = 0;
    f.read(reinterpret_cast<char*>(&P), sizeof(P));
    f.read(reinterpret_cast<char*>(&nb), sizeof(nb));
    if (!f || P <= 0 || nb <= 0) { std::printf("  [SKIP] bad fixture header\n"); return false; }
    out.P = P; out.nb = nb;
    out.block_sizes.resize(static_cast<std::size_t>(nb));
    f.read(reinterpret_cast<char*>(out.block_sizes.data()),
           static_cast<std::streamsize>(sizeof(std::int32_t) * static_cast<std::size_t>(nb)));
    const std::size_t n = static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
                          static_cast<std::size_t>(nb);
    out.f2.resize(n);
    f.read(reinterpret_cast<char*>(out.f2.data()),
           static_cast<std::streamsize>(sizeof(double) * n));
    if (!f) { std::printf("  [SKIP] fixture truncated\n"); return false; }
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
       << ",\n  \"provenance\": \"test fixture (real-AADR 9-pop f2; B1 io-fault gate)\"\n}\n";
    mf.close();
    return true;
}

struct RunResult {
    int exit_code = -1;
    std::string text;  // merged stdout+stderr
};

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "cli_io_fault_capture.txt";
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

bool looks_like_no_gpu(const std::string& text) {
    return text.find("no CUDA device") != std::string::npos ||
           text.find("device error") != std::string::npos ||
           text.find("No CUDA") != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    // argv[1] = the built steppe binary; argv[2] = the golden dir (for the fixture).
    if (argc < 3) {
        std::printf("usage: %s <steppe-binary> <golden-dir>\n", argv[0]);
        return 2;
    }
    const std::string steppe_bin = argv[1];
    const std::string golden_dir = argv[2];
    std::printf("=== B1 I/O fault taxonomy: torn --out write -> kExitIoError(4) ===\n");
    std::printf("steppe binary: %s\n", steppe_bin.c_str());

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", steppe_bin.c_str());
        return 1;
    }

    RawF2 raw;
    if (!read_raw_fixture(golden_dir + "/fixtures/f2_fit0_9pop.bin", raw)) {
        std::printf("\nRESULT: SKIP (f2 fixture absent)\n");
        return 0;
    }

    const std::vector<std::string> pops = {
        "England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N", "Mbuti",
        "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"};
    check_eq_int("fixture P == pops size", raw.P, static_cast<int>(pops.size()));

    std::error_code ec;
    const std::filesystem::path tmp_root =
        std::filesystem::temp_directory_path(ec) / ("steppe_cli_iofault_" + std::to_string(
            static_cast<long long>(std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp_root, ec);
    const std::filesystem::path dir = tmp_root / "f2";
    if (!write_f2_dir(dir, raw, pops)) {
        ++g_failures;
        std::filesystem::remove_all(tmp_root, ec);
        std::printf("\nRESULT: FAIL (could not write f2-dir)\n");
        return 1;
    }

    // One valid quartet (the first four fixture pops) — the compute succeeds; only the WRITE
    // is forced to fail, so this isolates the emit-path fault from any domain outcome.
    const std::string quartet =
        pops[0] + "," + pops[1] + "," + pops[2] + "," + pops[3];

    // ---------------- 1. TORN WRITE: --out /dev/full -> exit 4 ----------------
    // /dev/full accepts open() but every write returns ENOSPC: finish_emit's flush()+good()
    // fires and returns kExitIoError. Without the B1 fix this exited 0 with a truncated file.
    const RunResult full = run_steppe(steppe_bin,
        {"f4", "--f2-dir", dir.string(), "--pops", quartet,
         "--format", "csv", "--device", "0", "--out", "/dev/full"},
        tmp_root);
    if (looks_like_no_gpu(full.text)) {
        std::printf("  [SKIP] no CUDA device visible — CLI GPU f4 cannot run\n");
        std::filesystem::remove_all(tmp_root, ec);
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;
    }
    check_eq_int("torn write (/dev/full) exit == kExitIoError(4)", full.exit_code, 4);
    check_true("torn write logs a 'write failed' diagnostic",
               full.text.find("write failed") != std::string::npos);

    // ---------------- 2. CONTROL: a normal --out file -> exit 0, non-empty ----------------
    const std::filesystem::path okf = tmp_root / "ok.csv";
    const RunResult ok = run_steppe(steppe_bin,
        {"f4", "--f2-dir", dir.string(), "--pops", quartet,
         "--format", "csv", "--device", "0", "--out", okf.string()},
        tmp_root);
    check_eq_int("normal --out exit == 0", ok.exit_code, 0);
    {
        std::ifstream f(okf, std::ios::binary);
        std::string body;
        if (f) { std::ostringstream ss; ss << f.rdbuf(); body = ss.str(); }
        check_true("normal --out wrote a non-empty file", !body.empty());
        check_true("normal --out file has the f4 header", body.find("pop1") != std::string::npos);
    }

    std::filesystem::remove_all(tmp_root, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (torn write -> exit 4; clean write -> exit 0, valid file)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
