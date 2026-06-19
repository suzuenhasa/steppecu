// tests/reference/test_qpadm_parity.cpp
//
// M(fit-1) ACCEPTANCE: the first steppe qpAdm GLS fit (CpuBackend reference,
// native FP64, ONE model) validated to tolerance parity against the af6a8c2
// ADMIXTOOLS 2 golden (tests/reference/goldens/at2/, admixtools 2.0.10, R 4.3.3).
//
// f2-SOURCE (the chosen isolation strategy, design §5, BINDING): feed steppe the
// SAME f2 AT2 used — the committed binary fixture
// tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin is the 9-pop f2 tensor
// AT2's qpadm(f2_dir, ...) reads INTERNALLY (f2_from_precomp(dir, pops=c(left,
// right), afprod=TRUE) — the canonical directory-path read; README §"Canonical-
// invocation caveat"), dumped in steppe's [P × P × n_block] layout (i + P*j +
// P*P*b) with the per-block AT2 block_lengths. This ISOLATES the fit math from any
// steppe-f2-vs-AT2-f2 difference; the steppe weights then match AT2 to ~1e-12.
//
// CROSS-CHECKS (localize S3/S4 vs S6/S7, design §5):
//   * X slice — f4(Czechia, Turkey_N; Mbuti, Rj) for the 5 non-Mbuti rights, the
//     committed golden_fit0_X.csv (a leftref=Czechia/single-source slice, NOT the
//     internal m=10 X; FROZEN CONTRACT §0). Localizes S3.
//   * Q slice — the 5×5 jackknife covariance of that slice, the committed
//     golden_fit0_Q.csv + the fudged_diag. Localizes S4 (the OQ-3 block-weight).
//   NOTE: the committed X/Q CSV were generated from the NON-afprod f2 cache, so
//   they differ from the afprod fixture at ~1e-4; this slice is a LOCALIZER (loose
//   tier here), the primary gate is weights/chisq/dof/p (afprod, tight).
//
// Self-checking main(): returns non-zero on any failure (CTest gates on the exit
// code). No GoogleTest dependency. CUDA-FREE (CpuBackend only) — this is the §4
// layering proof: the fit reference compiles + runs with no GPU.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "device/backend.hpp"          // steppe::ComputeBackend, F4Blocks, JackknifeCov, Precision
#include "device/backend_factory.hpp"  // steppe::device::make_cpu_backend
#include "device/resources.hpp"        // steppe::device::Resources / PerGpuResources
#include "steppe/error.hpp"            // steppe::Status
#include "steppe/fstats.hpp"           // steppe::F2BlockTensor
#include "steppe/qpadm.hpp"            // steppe::run_qpadm + model/result/options

namespace {

int g_failures = 0;

void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const double diff = std::fabs(got - want);
    const bool ok = diff <= tol;
    std::printf("  [%s] %-28s got=% .12e want=% .12e |d|=% .3e tol=% .3e\n",
                ok ? "PASS" : "FAIL", what, got, want, diff, tol);
    if (!ok) ++g_failures;
}

void check_eq_int(const char* what, int got, int want) {
    const bool ok = (got == want);
    std::printf("  [%s] %-28s got=%d want=%d\n", ok ? "PASS" : "FAIL", what, got, want);
    if (!ok) ++g_failures;
}

// Minimal §12 metadata gate (design §5): refuse to run if any of the six required
// keys is missing from the committed golden_fit0.json. A substring presence check
// (the JSON is the committed authoritative source).
bool metadata_gate(const std::string& golden_dir) {
    const std::string path = golden_dir + "/golden_fit0.json";
    std::ifstream f(path);
    if (!f) {
        std::printf("  [FAIL] cannot open golden json: %s\n", path.c_str());
        return false;
    }
    const std::string js((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* keys[] = {"\"R\"", "\"admixtools\"", "\"blgsize\"",
                          "\"maxmiss\"", "\"boot\"", "\"fudge\""};
    bool ok = true;
    for (const char* k : keys) {
        if (js.find(k) == std::string::npos) {
            std::printf("  [FAIL] §12 metadata key missing: %s\n", k);
            ok = false;
        }
    }
    if (ok) std::printf("  [PASS] §12 metadata gate (R, admixtools, blgsize, maxmiss, boot, fudge present)\n");
    return ok;
}

// Read the committed binary f2 fixture: int32 P, int32 n_block, n_block × int32
// block_sizes, P*P*n_block × float64 f2 (column-major i + P*j + P*P*b).
bool read_fixture(const std::string& path, steppe::F2BlockTensor& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("  [FAIL] cannot open fixture: %s\n", path.c_str()); return false; }
    std::int32_t P = 0, nb = 0;
    f.read(reinterpret_cast<char*>(&P), sizeof(P));
    f.read(reinterpret_cast<char*>(&nb), sizeof(nb));
    if (!f || P <= 0 || nb <= 0) { std::printf("  [FAIL] bad fixture header\n"); return false; }
    out.P = P;
    out.n_block = nb;
    out.block_sizes.resize(static_cast<std::size_t>(nb));
    f.read(reinterpret_cast<char*>(out.block_sizes.data()),
           static_cast<std::streamsize>(sizeof(std::int32_t) * static_cast<std::size_t>(nb)));
    const std::size_t n = static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
                          static_cast<std::size_t>(nb);
    out.f2.resize(n);
    f.read(reinterpret_cast<char*>(out.f2.data()),
           static_cast<std::streamsize>(sizeof(double) * n));
    // vpair is not needed by the fit (OQ-3 uses block_sizes); leave it empty.
    if (!f) { std::printf("  [FAIL] fixture truncated\n"); return false; }
    std::printf("  fixture: P=%d n_block=%d (sum block_sizes=%lld)\n", P, nb, [&] {
        long long s = 0; for (int v : out.block_sizes) s += v; return s; }());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // Golden dir: argv[1] override, else the in-tree committed path.
    std::string golden_dir = (argc > 1) ? argv[1]
                                        : "tests/reference/goldens/at2";
    std::printf("=== M(fit-1) qpAdm parity test (CpuBackend, FP64) ===\n");
    std::printf("golden dir: %s\n", golden_dir.c_str());

    if (!metadata_gate(golden_dir)) {
        std::printf("RESULT: FAIL (metadata gate)\n");
        return 1;
    }

    steppe::F2BlockTensor f2;
    if (!read_fixture(golden_dir + "/fixtures/f2_fit0_9pop.bin", f2)) {
        std::printf("RESULT: FAIL (fixture read)\n");
        return 1;
    }

    // Resources with a single CpuBackend (no GPU; the M(fit-1) oracle path).
    steppe::device::Resources resources;
    steppe::device::PerGpuResources cpu;
    cpu.device_id = -1;  // CPU
    cpu.backend = steppe::device::make_cpu_backend();
    resources.gpus.push_back(std::move(cpu));

    // The af6a8c2 model. Fixture pop order (index):
    //   0=England_BellBeaker 1=Czechia_EBA_CordedWare 2=Turkey_N
    //   3=Mbuti 4=Israel_Natufian 5=Iran_GanjDareh_N 6=Han 7=Papuan 8=Karitiana
    steppe::QpAdmModel model;
    model.target = 0;                 // England_BellBeaker
    model.left = {1, 2};              // CordedWare, Turkey_N
    model.right = {3, 4, 5, 6, 7, 8}; // R0=Mbuti, then 5 outgroups
    model.model_index = 0;

    const steppe::QpAdmOptions opts;  // fudge=1e-4, als_iterations=20, rank=-1 (nl-1)
    const steppe::QpAdmResult res = steppe::run_qpadm(f2, model, opts, resources);

    if (res.status != steppe::Status::Ok) {
        std::printf("  [FAIL] run_qpadm status != Ok (%d)\n", static_cast<int>(res.status));
        std::printf("RESULT: FAIL\n");
        return 1;
    }

    // ---- Golden values (golden_fit0.json; af6a8c2 / admixtools 2.0.10) --------
    const double g_w0 = 0.558906248861195;   // CordedWare
    const double g_w1 = 0.441093751138805;   // Turkey_N
    const double g_se = 0.225911861836373;   // both (R-path se ~1.6e-4 off ⇒ loose)
    const double g_z0 = 2.47400133980574;
    const double g_z1 = 1.95250372226266;
    const double g_chisq = 4.63516296859645;
    const int    g_dof = 4;
    const double g_p = 0.326820092470997;

    std::printf("\n-- weights (TIGHT rtol 1e-6) --\n");
    check_close("weight[CordedWare]", res.weight.at(0), g_w0, 1e-6, 1e-12);
    check_close("weight[Turkey_N]",   res.weight.at(1), g_w1, 1e-6, 1e-12);

    std::printf("-- chisq (TIGHT) / dof (EXACT) --\n");
    check_close("chisq", res.chisq, g_chisq, 1e-6, 1e-12);
    check_eq_int("dof", res.dof, g_dof);

    std::printf("-- se / z / p (LOOSE rtol 1e-3) --\n");
    check_close("se[CordedWare]", res.se.at(0), g_se, 1e-3, 1e-9);
    check_close("se[Turkey_N]",   res.se.at(1), g_se, 1e-3, 1e-9);
    check_close("z[CordedWare]",  res.z.at(0),  g_z0, 1e-3, 1e-6);
    check_close("z[Turkey_N]",    res.z.at(1),  g_z1, 1e-3, 1e-6);
    check_close("p", res.p, g_p, 1e-3, 1e-9);

    // ---- S3/S4 localizer: the committed X/Q slice = f4(Czechia, Turkey_N; Mbuti,
    //      Rj), i.e. a model with target=Czechia, single source=Turkey_N. This is
    //      the FROZEN CONTRACT §0 single-leftref slice (NOT the internal m=10 X).
    //      Generated from the non-afprod cache ⇒ ~1e-4 off the afprod fixture ⇒
    //      LOOSE tier (a localizer, not the gate). r=0 here ⇒ no weights, but the
    //      X (x_total) + Q come out of S3/S4 unchanged. We re-run a slice model and
    //      read its intermediate X/Q via a dedicated rank-0 fit. -----------------
    {
        steppe::QpAdmModel slice;
        slice.target = 1;          // Czechia (the slice's L0 reference)
        slice.left = {2};          // Turkey_N (single source)
        slice.right = {3, 4, 5, 6, 7, 8};
        slice.model_index = 1;
        steppe::QpAdmOptions sopts;  // rank -1 ⇒ nl-1 = 0 for a single source
        const steppe::QpAdmResult sres = steppe::run_qpadm(f2, slice, sopts, resources);
        // The slice's status must be Ok (rank-0 trivial weight=1).
        std::printf("\n-- S3/S4 slice cross-check (LOCALIZER, loose rtol 1e-3) --\n");
        std::printf("  slice status=%d est_rank=%d (rank-0 single-source f4 slice)\n",
                    static_cast<int>(sres.status), sres.est_rank);
        // Golden committed X (golden_fit0_X.csv) and Q diag (golden_fit0_Q.csv).
        const double gX[5] = {0.000204208644854152, 0.000158461166756911,
                              -2.44579443823133e-05, -2.42885897838109e-05,
                              -3.27534454121373e-05};
        const double gQdiag[5] = {4.83261400481559e-09, 4.38937359295631e-09,
                                  2.43374449452477e-09, 3.18630101668274e-09,
                                  4.5704478312779e-09};
        // We need the intermediate X/Q for the slice. Expose them via a direct
        // backend call (the same the orchestrator uses). Build the slice f4 and Q.
        const double fudged_tr_factor = 1e-4;
        (void)fudged_tr_factor;
        // Recompute via the backend seam directly to read X/Q.
        steppe::ComputeBackend& be = *resources.gpus.at(0).backend;
        const std::vector<int> lidx = {1, 2};        // [target=Czechia, src=Turkey_N]
        const std::vector<int> ridx = {3, 4, 5, 6, 7, 8};
        const steppe::Precision prec{steppe::Precision::Kind::Fp64};
        const steppe::F4Blocks X = be.assemble_f4(f2, std::span<const int>(lidx),
                                                  std::span<const int>(ridx), prec);
        const steppe::JackknifeCov cov =
            be.jackknife_cov(X, std::span<const int>(f2.block_sizes), 1e-4, prec);
        // X slice: x_total has m = nl*nr = 1*5 = 5, k = j + nr*i = j (i=0).
        for (int j = 0; j < 5; ++j) {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "X_slice[%d]", j);
            check_close(nm, X.x_total.at(static_cast<std::size_t>(j)), gX[j], 1e-3, 1e-9);
        }
        // Q diag: Q is m×m = 5×5 column-major, diag at k + 5*k.
        for (int k = 0; k < 5; ++k) {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "Q_diag[%d]", k);
            check_close(nm, cov.Q.at(static_cast<std::size_t>(k) + 5u * static_cast<std::size_t>(k)),
                        gQdiag[k], 1e-3, 1e-9);
        }
    }

    std::printf("\n%s\n", g_failures == 0 ? "RESULT: PASS" : "RESULT: FAIL");
    return g_failures == 0 ? 0 : 1;
}
