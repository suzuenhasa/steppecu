// tests/cli/test_cli_dates.cpp
//
// DATES acceptance gate: the `steppe dates --prefix <geno> --target PUR --left CEU,YRI` CLI
// reproduces the DATES reference golden (admixture date in generations) THROUGH the CLI, ON
// THE GPU (the cuFFT autocorrelation LD engine). The golden was produced by building the
// reference github.com/priyamoorjani/DATES (Version 750) on box5090 and running it on the
// REAL AADR v66 TGENO decoded to packedancestrymap (target PUR, sources CEU+YRI):
//   date = 9.742 generations, SE = 0.317  (tests/reference/goldens/dates/aadr_PUR_CEU_YRI).
// Literature-consistent (PUR European-African admixture ~9-12 gen / colonial era;
// Moreno-Estrada 2013, Browning 2018).
//
// steppe reads the SAME raw v66 TGENO directly (steppe's TGENO decode is correct — the
// reference decoded TGENO itself to sidestep the AT2-misreads-TGENO bug; steppe never routes
// genotypes through AT2). The raw TGENO carries 584131 SNPs with the genetic map; the
// reference's packedancestrymap dropped 4411 (convertf filtering) to 579720, so the curves are
// the same up to that small SNP-set delta — the DATE is reproduced within a tight tier and the
// SE within an absolute band. The covariance decay is computed with NO host O(M²) SNP-pair
// loop (the ALDER FFT reformulation: cov(lag) = IFFT(|FFT(grid)|²)).
//
// PLAIN C++ host TU (NO CUDA header): it spawns the steppe binary and parses stdout; the GPU
// work happens inside that child. SKIPs cleanly (exit 0) when no CUDA device is visible OR the
// raw AADR prefix is absent (the STEPPE_AADR_ROOT raw/ convention), like test_cli_dstat_geno.
// NO SYNTHETIC DATA (memory real-data-only). Self-checking main(); CTest gates on the exit.

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

void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const double diff = std::fabs(got - want);
    const bool ok = diff <= tol;
    if (!ok)
        std::printf("  [FAIL] %-22s got=% .6f want=% .6f |d|=% .4e tol=% .4e\n", what, got, want,
                    diff, tol);
    else
        std::printf("  [ ok ] %-22s got=% .6f want=% .6f |d|=% .4e tol=% .4e\n", what, got, want,
                    diff, tol);
    if (!ok) ++g_failures;
}

void check_eq_int(const char* what, long long got, long long want) {
    const bool ok = (got == want);
    if (!ok) std::printf("  [FAIL] %-22s got=%lld want=%lld\n", what, got, want);
    if (!ok) ++g_failures;
}

struct RunResult {
    int exit_code = -1;
    std::string stdout_text;
};

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "cli_dates_stdout.txt";
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

bool looks_like_no_gpu(const std::string& t) {
    return t.find("no CUDA device") != std::string::npos ||
           t.find("device error") != std::string::npos ||
           t.find("No CUDA") != std::string::npos;
}

std::vector<std::string> split(const std::string& line, char sep) {
    std::vector<std::string> cells;
    std::string cur;
    for (char c : line) {
        if (c == sep) { cells.push_back(cur); cur.clear(); }
        else cur += c;
    }
    cells.push_back(cur);
    return cells;
}

double cell_d(const std::string& s) {
    if (s == "NA" || s == "null" || s.empty()) return std::nan("");
    return std::strtod(s.c_str(), nullptr);
}

// Parse the steppe dates CSV: header target,source1,source2,date_gen,se,fit_error_sd,status.
bool parse_dates(const std::string& text, double& date, double& se, std::string& status) {
    std::istringstream ss(text);
    std::string line;
    std::map<std::string, std::size_t> col;
    bool header = true;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> cells = split(line, ',');
        if (header) {
            for (std::size_t c = 0; c < cells.size(); ++c) col[cells[c]] = c;
            header = false;
            continue;
        }
        if (!col.count("date_gen")) return false;
        date = cell_d(cells[col["date_gen"]]);
        se = cell_d(cells[col["se"]]);
        status = col.count("status") ? cells[col["status"]] : "";
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    // argv[1] = the built steppe binary; argv[2] = the AADR root.
    if (argc < 3) {
        std::printf("usage: %s <steppe-binary> <aadr-root>\n", argv[0]);
        return 2;
    }
    const std::string steppe_bin = argv[1];
    const std::string aadr_root = argv[2];
    std::printf("=== DATES CLI golden gate (admixture date THROUGH the CLI, ON THE GPU) ===\n");
    std::printf("steppe binary: %s\naadr root:     %s\n", steppe_bin.c_str(), aadr_root.c_str());

    if (!std::filesystem::exists(steppe_bin)) {
        std::printf("RESULT: FAIL (steppe binary not found)\n");
        return 1;
    }

    const std::string prefix = aadr_root + "/raw/v66.p1_HO.aadr.patch.PUB";
    if (!std::filesystem::exists(prefix + ".geno") || !std::filesystem::exists(prefix + ".snp") ||
        !std::filesystem::exists(prefix + ".ind")) {
        std::printf("\nRESULT: SKIP (raw AADR TGENO prefix absent: %s)\n", prefix.c_str());
        return 0;
    }

    // The DATES reference golden (aadr_PUR_CEU_YRI): date = 9.742 gen, SE = 0.317.
    const double golden_date = 9.742;
    const double golden_se = 0.317;

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_cli_dates_" +
         std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const RunResult r = run_steppe(steppe_bin,
        {"dates", "--prefix", prefix, "--target", "PUR", "--left", "CEU,YRI",
         "--format", "csv", "--device", "0"},
        tmp);

    if (looks_like_no_gpu(r.stdout_text)) {
        std::printf("  [SKIP] no CUDA device visible — CLI GPU dates cannot run\n");
        std::filesystem::remove_all(tmp, ec);
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;
    }

    check_eq_int("exit code == 0", r.exit_code, 0);

    double date = std::nan(""), se = std::nan("");
    std::string status;
    const bool parsed = parse_dates(r.stdout_text, date, se, status);
    if (!parsed) {
        std::printf("  [FAIL] could not parse dates output:\n%s\n", r.stdout_text.c_str());
        ++g_failures;
    } else {
        // The DATE reproduces the reference golden tightly (the small SNP-set delta between the
        // raw TGENO 584131 SNPs and the reference's convertf-filtered 579720 perturbs it only
        // marginally). The SE is more SNP-set-sensitive, so it gets an absolute band.
        check_close("date_gen", date, golden_date, /*rtol=*/0.02, /*atol=*/0.0);
        check_close("se",       se,   golden_se,   /*rtol=*/0.0,  /*atol=*/0.10);
        if (status != "ok") { std::printf("  [FAIL] status != ok: %s\n", status.c_str()); ++g_failures; }
    }

    std::filesystem::remove_all(tmp, ec);
    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (DATES date %.3f gen reproduced through the GPU cuFFT LD "
                    "engine; golden %.3f)\n", date, golden_date);
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
