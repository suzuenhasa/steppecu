// tests/cli/test_cli_paint.cpp
//
// `steppe paint` CLI + validator smoke gate. Plain C++ host TU (no CUDA): it writes
// tiny EIGENSTRAT triples to a temp dir and drives the built `steppe paint` binary,
// asserting the validator contracts AND the Phase-2 compute path THROUGH the CLI:
//   1. a well-formed PHASED HAPLOID panel (self-painted, leave-one-out) RUNS and emits
//      a coancestry table (exit 0, `expected_chunks` header) — the CPU backend serves
//      the compute when no GPU is visible;
//   2. a DIPLOID recipient triple (a heterozygous call) is REJECTED with a
//      "phase first" error (kExitInvalidConfig);
//   3. a missing cM map is REJECTED unless --bp-fallback is opted in;
//   4. a missing --donors is REJECTED;
//   5. OPTIONAL real-data spot check (SKIPPED unless the env fixture is set): paint a
//      real multi-population phased panel and assert the per-label coancestry is
//      diagonal-dominant (each recipient copies most length from its own population) —
//      an ancestry-agnostic sanity check, REAL data only, no hard-coded answer.
// The hand-written fixtures are a smoke test (not a scientific result). Self-checking
// main(); CTest gates on the exit code. Needs NO CUDA device (falls back to the CPU
// reference backend). argv[1] = the built steppe binary.

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

int g_fail = 0;

struct RunResult {
    int exit_code = -1;
    std::string text;
};

RunResult run_steppe(const std::string& bin, const std::vector<std::string>& args,
                     const std::filesystem::path& tmp) {
    RunResult rr;
    const std::filesystem::path outf = tmp / "paint_stdout.txt";
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

void write_file(const std::filesystem::path& p, const std::string& s) {
    std::ofstream f(p);
    f << s;
}

// Write an EIGENSTRAT triple. geno_rows = SNP-major ASCII rows (one char/indiv);
// snp lines carry the cM map in genpos_morgans (col 3). n_ind individuals, pop PANEL.
void write_triple(const std::filesystem::path& dir, const std::string& stem, int n_ind,
                  const std::vector<std::string>& geno_rows,
                  const std::vector<double>& genpos_morgans) {
    std::string ind;
    for (int i = 0; i < n_ind; ++i)
        ind += "h" + std::to_string(i) + " U PANEL\n";
    write_file(dir / (stem + ".ind"), ind);

    std::string snp;
    for (std::size_t s = 0; s < geno_rows.size(); ++s) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "rs%zu 1 %.6f %zu A C\n", s, genpos_morgans[s],
                      1000 * (s + 1));
        snp += buf;
    }
    write_file(dir / (stem + ".snp"), snp);

    std::string geno;
    for (const std::string& r : geno_rows) geno += r + "\n";
    write_file(dir / (stem + ".geno"), geno);
}

void check(bool cond, const char* what, const RunResult& r) {
    if (!cond) {
        std::printf("  [FAIL] %s (exit=%d)\n%s\n", what, r.exit_code, r.text.c_str());
        ++g_fail;
    } else {
        std::printf("  [ ok ] %s (exit=%d)\n", what, r.exit_code);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <steppe-binary>\n", argv[0]);
        return 2;
    }
    const std::string bin = argv[1];
    std::printf("=== steppe paint CLI + validator smoke gate (Phase 0, host-only) ===\n");
    if (!std::filesystem::exists(bin)) {
        std::printf("RESULT: FAIL (steppe binary not found: %s)\n", bin.c_str());
        return 1;
    }

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_cli_paint_" +
         std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const int N = 6;
    const std::vector<double> gp = {0.01, 0.02, 0.03, 0.04, 0.05};
    // A phased HAPLOID panel: pseudo-haploid codes {0,2}, no heterozygotes.
    write_triple(tmp, "hap", N,
                 {"020020", "200200", "022002", "002200", "220022"}, gp);
    // A DIPLOID triple: individual 0 carries a heterozygous call (code 1) at SNP 0.
    write_triple(tmp, "dip", N,
                 {"120020", "200200", "022002", "002200", "220022"}, gp);
    // The haploid geno/ind with an ABSENT map (all genpos 0).
    write_triple(tmp, "nomap", N,
                 {"020020", "200200", "022002", "002200", "220022"},
                 {0.0, 0.0, 0.0, 0.0, 0.0});

    const std::string hap = (tmp / "hap").string();
    const std::string dip = (tmp / "dip").string();
    const std::string nomap = (tmp / "nomap").string();

    // 1. Well-formed self-painted haploid panel runs and emits a coancestry table
    //    (leave-one-out on; CPU backend serves the compute with no GPU visible).
    {
        RunResult r = run_steppe(bin, {"paint", "--prefix", hap, "--donors", hap}, tmp);
        check(r.exit_code == 0 && r.text.find("expected_chunks") != std::string::npos,
              "phased haploid panel paints (coancestry table emitted)", r);
    }

    // 2. Diploid recipient triple is rejected with a phase-first error.
    {
        RunResult r = run_steppe(bin, {"paint", "--prefix", dip, "--donors", hap}, tmp);
        check(r.exit_code != 0 && r.text.find("PRE-PHASED") != std::string::npos,
              "diploid input rejected (phase first)", r);
    }

    // 3. Absent cM map is rejected by default, accepted with --bp-fallback.
    {
        RunResult r = run_steppe(bin, {"paint", "--prefix", nomap, "--donors", hap}, tmp);
        check(r.exit_code != 0 && r.text.find("cM map") != std::string::npos,
              "absent cM map rejected by default", r);
        // Self-paint nomap (recipients and donors share the same marker set) so the
        // check exercises the --bp-fallback opt-in, not the common-marker guard.
        RunResult r2 =
            run_steppe(bin, {"paint", "--prefix", nomap, "--donors", nomap, "--bp-fallback"}, tmp);
        check(r2.exit_code == 0, "absent map + --bp-fallback accepted", r2);
    }

    // 4. Missing --donors is rejected.
    {
        RunResult r = run_steppe(bin, {"paint", "--prefix", hap}, tmp);
        check(r.exit_code != 0 && r.text.find("--donors") != std::string::npos,
              "missing --donors rejected", r);
    }

    // 4b. --face localanc smoke: the same self-painted haploid panel runs and emits a
    //     per-SNP ancestry posterior table (long format). The synthetic panel has one
    //     population label (PANEL) so P=1 and every posterior is 1 — a trivial but real
    //     end-to-end exercise of the localanc face + emitter (header + sum-to-1).
    {
        const std::filesystem::path outf = tmp / "localanc_smoke.csv";
        RunResult r = run_steppe(
            bin, {"paint", "--face", "localanc", "--prefix", hap, "--donors", hap, "--out",
                  outf.string(), "--format", "csv"}, tmp);
        bool ok = (r.exit_code == 0);
        // Parse: recipient,snp_id,chrom,pos_bp,genpos_cM,ancestry_label,posterior. Group the
        // posterior by (recipient, snp_id) and assert each group sums to ~1.
        std::map<std::string, double> sum_by_key;
        bool header_ok = false;
        std::ifstream cf(outf);
        std::string line;
        bool header = true;
        while (std::getline(cf, line)) {
            if (header) {
                header = false;
                header_ok = line.find("ancestry_label") != std::string::npos &&
                            line.find("posterior") != std::string::npos;
                continue;
            }
            if (line.empty()) continue;
            std::vector<std::string> f;
            std::string tok; std::istringstream ss(line);
            while (std::getline(ss, tok, ',')) f.push_back(tok);
            if (f.size() < 7) continue;
            sum_by_key[f[0] + "|" + f[1]] += std::strtod(f[6].c_str(), nullptr);
        }
        ok = ok && header_ok && !sum_by_key.empty();
        for (const auto& [key, s] : sum_by_key)
            if (std::fabs(s - 1.0) > 1e-9) {
                std::printf("  [FAIL] localanc column %s sums to %.6f (expected 1)\n",
                            key.c_str(), s);
                ok = false;
            }
        check(ok, "--face localanc emits a well-formed sum-to-1 per-SNP posterior table", r);
    }

    // 4c. OPTIONAL real-data localanc spot check (SKIP-still-PASS unless the env fixture is
    //     set). Runs `--face localanc` on a REAL admixed phased sample against a labelled
    //     donor panel and asserts the emitted per-SNP posterior is well-formed (header +
    //     every column sums to ~1). The FLARE/RFMix concordance itself is scored by
    //     tests/concordance/localanc_concordance.py (needs the external tool + real panels),
    //     NOT asserted in CI. REAL data only, no hard-coded ancestry.
    //       STEPPE_LOCALANC_REAL_RECIP  = admixed recipient triple PREFIX (phased .ind)
    //       STEPPE_LOCALANC_REAL_DONORS = donor panel triple PREFIX (phased, pop-labelled)
    //       STEPPE_LOCALANC_REAL_LABELS = --labels file (one label per donor haplotype col)
    {
        const char* er = std::getenv("STEPPE_LOCALANC_REAL_RECIP");
        const char* ed = std::getenv("STEPPE_LOCALANC_REAL_DONORS");
        const char* el = std::getenv("STEPPE_LOCALANC_REAL_LABELS");
        if (er == nullptr || ed == nullptr || std::string(er).empty() || std::string(ed).empty()) {
            std::printf("  [SKIP] real-data localanc spot check "
                        "(set STEPPE_LOCALANC_REAL_RECIP / _DONORS [/ _LABELS])\n");
        } else {
            const std::filesystem::path outf = tmp / "localanc_real.csv";
            std::vector<std::string> args = {"paint", "--face", "localanc", "--prefix", er,
                                             "--donors", ed};
            if (el != nullptr && !std::string(el).empty()) { args.push_back("--labels"); args.push_back(el); }
            args.push_back("--out"); args.push_back(outf.string());
            args.push_back("--format"); args.push_back("csv");
            RunResult r = run_steppe(bin, args, tmp);
            bool ok = (r.exit_code == 0);
            std::map<std::string, double> sum_by_key;
            bool header_ok = false;
            std::ifstream cf(outf);
            std::string line;
            bool header = true;
            while (std::getline(cf, line)) {
                if (header) {
                    header = false;
                    header_ok = line.find("ancestry_label") != std::string::npos;
                    continue;
                }
                if (line.empty()) continue;
                std::vector<std::string> f;
                std::string tok; std::istringstream ss(line);
                while (std::getline(ss, tok, ',')) f.push_back(tok);
                if (f.size() < 7) continue;
                sum_by_key[f[0] + "|" + f[1]] += std::strtod(f[6].c_str(), nullptr);
            }
            ok = ok && header_ok && !sum_by_key.empty();
            int bad = 0;
            for (const auto& [key, s] : sum_by_key)
                if (std::fabs(s - 1.0) > 1e-6) { (void)key; ++bad; }
            if (bad > 0) {
                std::printf("  [FAIL] %d real localanc columns do not sum to 1\n", bad);
                ok = false;
            }
            check(ok, "real localanc: well-formed per-SNP posterior (every column sums to 1)", r);
        }
    }

    // 5. OPTIONAL real-data diagonal-dominance spot check. Provide a REAL multi-pop
    //    phased panel via env vars (recipients OUT of the donor set — clean out-of-panel
    //    painting so the diagonal is driven by genuine cross-population copying, not
    //    self-copy). SKIP (still PASS) when the fixture is absent so CI stays green.
    //      STEPPE_PAINT_REAL_RECIP  = recipient triple PREFIX (phased, pop-labelled .ind)
    //      STEPPE_PAINT_REAL_DONORS = donor panel triple PREFIX (phased, pop-labelled .ind)
    {
        const char* er = std::getenv("STEPPE_PAINT_REAL_RECIP");
        const char* ed = std::getenv("STEPPE_PAINT_REAL_DONORS");
        if (er == nullptr || ed == nullptr || std::string(er).empty() || std::string(ed).empty()) {
            std::printf("  [SKIP] real-data diagonal-dominance spot check "
                        "(set STEPPE_PAINT_REAL_RECIP / STEPPE_PAINT_REAL_DONORS)\n");
        } else {
            const std::filesystem::path outf = tmp / "paint_real.csv";
            RunResult r = run_steppe(
                bin, {"paint", "--prefix", er, "--donors", ed, "--out", outf.string(),
                      "--format", "csv"}, tmp);
            bool ok = (r.exit_code == 0);
            // Parse: recipient,donor_label,expected_chunks,expected_length_cM. Aggregate the
            // total copied LENGTH per (recipient own-population, donor label) — the recipient
            // key is "<poplabel>:<hap-idx>", so grouping by the own pop pools a sample's two
            // haplotypes (the ancestry-relevant granularity; a single phased haplotype on one
            // chromosome arm is noisy). Assert each recipient population copies the most total
            // length from its OWN population's donors (ancestry-agnostic, no hard-coded answer).
            std::map<std::string, std::map<std::string, double>> by_pop;  // own-pop -> label -> len
            auto own_pop = [](const std::string& recip) {
                const auto c = recip.rfind(':');
                return c == std::string::npos ? recip : recip.substr(0, c);
            };
            std::ifstream cf(outf);
            std::string line;
            bool header = true;
            while (std::getline(cf, line)) {
                if (header) { header = false; continue; }  // skip the column header row
                if (line.empty()) continue;
                std::vector<std::string> f;
                std::string tok; std::istringstream ss(line);
                while (std::getline(ss, tok, ',')) f.push_back(tok);
                if (f.size() < 4) continue;
                by_pop[own_pop(f[0])][f[1]] += std::strtod(f[3].c_str(), nullptr);
            }
            int checked = 0;
            for (const auto& [pop, hist] : by_pop) {
                ++checked;
                std::string best_label; double best_len = -1.0;
                for (const auto& [label, len] : hist)
                    if (len > best_len) { best_len = len; best_label = label; }
                if (best_label != pop) {
                    std::printf("  [FAIL] recipient pop %s copies most length from %s\n",
                                pop.c_str(), best_label.c_str());
                    ok = false;
                }
            }
            ok = ok && (checked > 0);
            check(ok, "real multi-pop panel: per-label coancestry is diagonal-dominant", r);
        }
    }

    std::filesystem::remove_all(tmp, ec);
    if (g_fail == 0) {
        std::printf("\nRESULT: PASS\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s))\n", g_fail);
    return 1;
}
