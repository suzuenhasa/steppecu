// tests/cli/test_cli_readv2.cpp
//
// Real-data end-to-end gate for `steppe readv2`, measured against the Phase-0 ruler
// (`steppe readv2-concord`). Plain C++ host TU (spawns the built binary; needs a live
// CUDA device for the readv2 run). It runs
//     steppe readv2 --prefix <FIXTURE>/relatives --samples <FIXTURE>/keep.txt
//                   --window-snps <W> --norm median --format tsv --out steppe.tsv
// then
//     steppe readv2-concord --a steppe.tsv --b <FIXTURE>/oracle.tsv
// and asserts `RESULT: PASS` (the ruler's degree-agreement / P0_norm / coverage floors).
//
// This is the SKELETON for the supervised real-data step: the AADR relative-pair
// oracle fixture (docs/planning/readv2-oracle-fixture-recipe.md) is produced there, not
// here. The fixture dir holds the autosome-only PLINK subset built by make_subset_bed.py
// (prefix `readv2_subset.{bed,bim,fam}`), the `keep_genetic_ids.txt` sample list, and the
// `readv2_oracle.tsv` reference table. Until the subset .bed exists on the box the test
// SKIPs cleanly (exit 0) when the binary, the genotype prefix, the keep list, or the
// oracle table is absent — NO synthetic science is ever presented. The .bed is
// autosome-only (chr 1-22), which is what the oracle was scored on.
//
// TWO cases run against the SAME autosome-only oracle:
//   (1) the pre-filtered autosome-only subset (readv2_subset) — validates the estimator;
//   (2) the FULL sex-chromosome-bearing 1240K prefix (argv[3], if present) — proves
//       `steppe readv2` performs the autosome restriction ITSELF (default --auto-only),
//       i.e. the exact code path a real user hits on a raw 1240K prefix. Without case (2)
//       the missing-autosome-filter defect is structurally invisible to the suite.
// argv[1] = the steppe binary; argv[2] = the fixture dir; argv[3] = OPTIONAL full 1240K
// prefix (skipped when absent).
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

struct RunResult {
    int exit_code = -1;
    std::string text;
};

RunResult run(const std::string& bin, const std::vector<std::string>& args,
              const std::filesystem::path& logf) {
    RunResult rr;
    std::string cmd = "\"" + bin + "\"";
    for (const std::string& a : args) cmd += " \"" + a + "\"";
    cmd += " > \"" + logf.string() + "\" 2>&1";
    const int sys = std::system(cmd.c_str());
    rr.exit_code = (sys == -1) ? -1 :
#ifdef WIFEXITED
                   (WIFEXITED(sys) ? WEXITSTATUS(sys) : -1);
#else
                   sys;
#endif
    std::ifstream f(logf, std::ios::binary);
    if (f) { std::ostringstream ss; ss << f.rdbuf(); rr.text = ss.str(); }
    return rr;
}

bool path_exists(const std::filesystem::path& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

// Accept either a PLINK .bed or a TGENO/PACKEDANCESTRYMAP .geno at a genotype prefix.
bool has_genotype(const std::filesystem::path& prefix) {
    return path_exists(std::filesystem::path(prefix.string() + ".bed")) ||
           path_exists(std::filesystem::path(prefix.string() + ".geno"));
}

// Outcome of one prefix case. Skipped cases do not fail the gate.
enum class CaseResult { Pass, Fail, Skip };

// Run `steppe readv2 --prefix <prefix>` then `readv2-concord` against `oracle`, in a
// per-case temp dir. `label` names the case in the log.
CaseResult run_case(const std::string& bin, const std::string& label,
                    const std::filesystem::path& prefix, const std::filesystem::path& keep,
                    const std::filesystem::path& oracle) {
    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) / ("steppe_readv2_gate_" + label);
    std::filesystem::create_directories(tmp, ec);
    const std::filesystem::path out = tmp / "steppe.tsv";
    const std::filesystem::path log1 = tmp / "readv2.log";
    const std::filesystem::path log2 = tmp / "concord.log";

    // --auto-only is the readv2 default, so the full-prefix case restricts to autosomes
    // in-tool and concords with the autosome-only oracle without any external pre-filter.
    const RunResult r1 = run(bin, {"readv2", "--prefix", prefix.string(), "--samples",
                                   keep.string(), "--window-snps", "1000", "--norm", "median",
                                   "--format", "tsv", "--out", out.string(), "--device", "0"},
                             log1);
    if (r1.exit_code != 0) {
        if (r1.text.find("no CUDA device available") != std::string::npos) {
            std::printf("[%s] SKIP: no CUDA device available\n", label.c_str());
            std::filesystem::remove_all(tmp, ec);
            return CaseResult::Skip;
        }
        std::printf("[%s] RESULT: FAIL (steppe readv2 exited %d)\n%s\n", label.c_str(),
                    r1.exit_code, r1.text.c_str());
        std::filesystem::remove_all(tmp, ec);
        return CaseResult::Fail;
    }

    const RunResult r2 = run(bin, {"readv2-concord", "--a", out.string(), "--b", oracle.string()},
                             log2);
    const bool pass = r2.text.find("RESULT: PASS") != std::string::npos && r2.exit_code == 0;
    std::filesystem::remove_all(tmp, ec);
    if (pass) {
        std::printf("[%s] RESULT: PASS (concords with the oracle within the ruler floors)\n",
                    label.c_str());
        return CaseResult::Pass;
    }
    std::printf("[%s] RESULT: FAIL (concord exited %d)\n%s\n", label.c_str(), r2.exit_code,
                r2.text.c_str());
    return CaseResult::Fail;
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("=== steppe readv2 real-data gate (vs readv2-concord ruler) ===\n");
    if (argc < 3) {
        std::printf("SKIP: usage %s <steppe-binary> <fixture-dir> [full-1240K-prefix]\n", argv[0]);
        return 0;
    }
    const std::string bin = argv[1];
    const std::filesystem::path fixture = argv[2];

    const std::filesystem::path subset = fixture / "readv2_subset";  // autosome-only .bed/.geno
    const std::filesystem::path keep = fixture / "keep_genetic_ids.txt";
    const std::filesystem::path oracle = fixture / "readv2_oracle.tsv";

    if (!path_exists(bin)) { std::printf("SKIP: steppe binary not found: %s\n", bin.c_str()); return 0; }
    if (!has_genotype(subset) || !path_exists(keep) || !path_exists(oracle)) {
        std::printf("SKIP: AADR relative-pair oracle fixture absent under %s "
                    "(the readv2_subset.{bed,bim,fam} genotype matrix is produced in the "
                    "supervised real-data step)\n", fixture.string().c_str());
        return 0;
    }

    bool any_ran = false;
    bool any_failed = false;

    // Case 1: pre-filtered autosome-only subset — validates the estimator.
    const CaseResult c1 = run_case(bin, "subset", subset, keep, oracle);
    if (c1 == CaseResult::Skip) { std::printf("SKIP: no CUDA device available\n"); return 0; }
    any_ran = true;
    any_failed = any_failed || (c1 == CaseResult::Fail);

    // Case 2 (optional): the FULL sex-chromosome-bearing 1240K prefix — proves steppe's
    // OWN --auto-only restriction on the raw code path a real user hits. Skipped when the
    // full prefix is not provided or its genotype matrix is absent (e.g. local dev).
    if (argc >= 4 && std::string(argv[3]).size() > 0) {
        const std::filesystem::path full = argv[3];
        if (has_genotype(full)) {
            const CaseResult c2 = run_case(bin, "full-1240K", full, keep, oracle);
            if (c2 != CaseResult::Skip) any_failed = any_failed || (c2 == CaseResult::Fail);
        } else {
            std::printf("[full-1240K] SKIP: full prefix genotype absent: %s\n", full.c_str());
        }
    }

    if (!any_ran) { std::printf("SKIP: no case ran\n"); return 0; }
    if (any_failed) { std::printf("RESULT: FAIL (a case did not concord)\n"); return 1; }
    std::printf("RESULT: PASS (all readv2 real-data cases concord within the ruler floors)\n");
    return 0;
}
