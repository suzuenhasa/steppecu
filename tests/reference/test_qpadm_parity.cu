// tests/reference/test_qpadm_parity.cpp
//
// qpAdm fit ACCEPTANCE: the steppe qpAdm GLS fit (ONE model, native FP64) validated
// to tolerance parity against the af6a8c2 ADMIXTOOLS 2 golden (tests/reference/
// goldens/at2/, admixtools 2.0.10, R 4.3.3) on BOTH backends — the CpuBackend
// reference (M(fit-1): the bit-exact diff oracle + the no-GPU fallback) AND the CUDA
// backend (M(fit-4): the PRODUCTION GPU path, f2 RESIDENT in VRAM).
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
// code). No GoogleTest dependency.
//
// M(fit-4) EXTENSION: this now exercises BOTH backends — the CpuBackend (the
// bit-exact diff ORACLE + the no-GPU fallback) AND the CUDA backend (the PRODUCTION
// GPU PATH). The GPU run uploads the golden f2 fixture to VRAM as a DeviceF2Blocks,
// asserts f2 is RESIDENT + a real GPU is bound (caps.compute_major), runs the fit on
// the GPU (run_qpadm(DeviceF2Blocks) → the f4-gather kernel reading resident f2 →
// jackknife SYRK/cuSOLVER → on-device SVD/ALS/weight → batched LOO), and asserts the
// GPU weights/chisq/dof/p match the af6a8c2 golden AND match the CpuBackend oracle to
// 1e-9 (localization). If no CUDA device is visible the GPU block prints SKIP and the
// CpuBackend path alone still gates (CI-without-GPU degrades cleanly; box5090 always
// has a GPU). The CpuBackend block ALWAYS runs.

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "device/backend.hpp"          // steppe::ComputeBackend, F4Blocks, JackknifeCov, Precision
#include "device/backend_factory.hpp"  // steppe::device::make_cpu_backend, make_cuda_backend, visible_device_count
#include "device/device_f2_blocks.hpp" // steppe::device::DeviceF2Blocks, upload_f2_blocks_to_device (the GPU-resident input)
#include "device/resources.hpp"        // steppe::device::Resources / PerGpuResources
#include "steppe/error.hpp"            // steppe::Status
#include "steppe/fstats.hpp"           // steppe::F2BlockTensor
#include "steppe/qpadm.hpp"            // steppe::run_qpadm + model/result/options

namespace {

int g_failures = 0;

// NRBIG (nl=2, nr=39, nb=701) PRE-FIX (serial large-LOO) GPU jackknife se/z — the
// BIT-IDENTITY anchor for the parallel large-LOO. Captured from the BEFORE run on
// box5090 (serial host nb-loop). The parallel kernel reuses the SAME cuSOLVER SVD
// seed + the SAME als_large/weight math (only the loop parallelizes) ⇒ the post-fix
// GPU se/z MUST equal these EXACTLY (rtol=atol=0). See the gate at the NRBIG block.
// Captured from the BEFORE (serial large-LOO) GPU run on box5090 (cuSOLVER gesvd
// seed path, qpadm_parity = 369.79 s). The GPU differs from the CpuBackend ORACLE
// (Jacobi seed) at the ~2e-11 SVD-seed localizer level — so the anchor is the GPU
// SERIAL value, the one the parallel kernel (same cuSOLVER seed) must reproduce EXACTLY.
constexpr double kNrbigPreSe0 = 0.15871216584857264;
constexpr double kNrbigPreSe1 = 0.15871216584857259;
constexpr double kNrbigPreZ0  = 4.9857579543926072;
constexpr double kNrbigPreZ1  = 1.3149562640378851;

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

// NON-GATING delta report for the exploratory promoted-emulated measurement
// (ROADMAP §6 seam): print got/want and abs+rel delta WITHOUT touching g_failures.
// This is forward evidence for S8, REPORTED not gated (per the fix-pass contract).
void report_delta(const char* what, double got, double want) {
    const double absd = std::fabs(got - want);
    const double reld = std::fabs(want) > 0.0 ? absd / std::fabs(want) : absd;
    std::printf("  [INFO] %-34s got=% .12e want=% .12e |d|=% .3e rel=% .3e\n",
                what, got, want, absd, reld);
}

// True iff |got-want| <= atol + rtol*|want| — the same tier predicate check_close
// uses, but returns a bool for the informational tier summary (no side effects).
bool rel_within(double got, double want, double rtol, double atol) {
    return std::fabs(got - want) <= atol + rtol * std::fabs(want);
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

    // FAST/THOROUGH split (frozen design): plain `ctest` is the FAST dev loop that
    // validates the GPU path vs the AT2 goldens ONLY. The CpuBackend oracle re-
    // derivation + the NRBIG GPU full LOO-SE (~347 s, never asserted) are opt-in via
    // STEPPE_THOROUGH=1. The CpuBackend ALSO runs automatically when no GPU is visible
    // (the CI-without-GPU acceptance gate). Read the env once at start.
    const bool g_thorough = [] {
        const char* e = std::getenv("STEPPE_THOROUGH");
        return e && e[0] == '1';
    }();
    // Hoist the GPU probe to the top so Block A (the CpuBackend oracle) can gate on
    // (g_thorough || gpu_count <= 0): run the CpuBackend whenever the user opted into
    // thorough OR there is no GPU to validate against.
    int gpu_count = 0;
    try {
        gpu_count = steppe::device::visible_device_count();
    } catch (const std::exception& e) {
        std::printf("  [INFO] visible_device_count threw at probe: %s\n", e.what());
        gpu_count = 0;
    }
    std::printf("MODE: %s (set STEPPE_THOROUGH=1 for the CpuBackend oracle + NRBIG full SE)\n",
                g_thorough ? "THOROUGH" : "FAST (GPU-vs-golden only)");

    if (!metadata_gate(golden_dir)) {
        std::printf("RESULT: FAIL (metadata gate)\n");
        return 1;
    }

    steppe::F2BlockTensor f2;
    if (!read_fixture(golden_dir + "/fixtures/f2_fit0_9pop.bin", f2)) {
        std::printf("RESULT: FAIL (fixture read)\n");
        return 1;
    }

    // The af6a8c2 model. Fixture pop order (index):
    //   0=England_BellBeaker 1=Czechia_EBA_CordedWare 2=Turkey_N
    //   3=Mbuti 4=Israel_Natufian 5=Iran_GanjDareh_N 6=Han 7=Papuan 8=Karitiana
    steppe::QpAdmModel model;
    model.target = 0;                 // England_BellBeaker
    model.left = {1, 2};              // CordedWare, Turkey_N
    model.right = {3, 4, 5, 6, 7, 8}; // R0=Mbuti, then 5 outgroups
    model.model_index = 0;

    const steppe::QpAdmOptions opts;  // fudge=1e-4, als_iterations=20, rank=-1 (nl-1)

    // ---- Golden values (golden_fit0.json; af6a8c2 / admixtools 2.0.10) --------
    // HOISTED above the THOROUGH gate: the default GPU block re-asserts these SAME
    // golden constants, so they must be in scope in FAST mode (the CpuBackend re-
    // derivation below is the THOROUGH-only oracle).
    // CORRECTED golden_fit0 (convertf-PA source; the prior values were AT2 2.0.10's
    // silent misread of the raw v66 TGENO — 500848 SNPs / weights [0.559,0.441]). The
    // fixture is read_f2(dir)'s f2-OBJECT tensor, so this test reproduces the golden's
    // fixture_f2_object_path block (golden_fit0.json), NOT the directory-path headline
    // (they differ by the documented ~1e-5 read-arg caveat). 391333 SNPs / 710 blocks.
    const double g_w0 = 0.868755109981416;   // CordedWare (fixture_f2_object_path)
    const double g_w1 = 0.131244890018584;   // Turkey_N
    const double g_se = 0.0248167157892669;  // both (R-path se ~1.7e-5 off ⇒ loose 1e-3)
    const double g_z0 = 35.006852532725;
    const double g_z1 = 5.2885680415193;
    const double g_chisq = 3.95682062790988;
    const int    g_dof = 4;
    const double g_p = 0.411881081897742;
    // The CpuBackend oracle rank_Q (full m=10 model); the GPU localizer reads it.
    // Stays 0 in FAST mode (the gpu-vs-cpu rank_Q localizer is THOROUGH-only).
    int rsw_rank_Q_ref = 0;
    // The CpuBackend host-oracle 9-pop fit result, hoisted to outer scope so the
    // THOROUGH-only GPU-vs-CPU localizers (Block C) can diff against it. Populated
    // inside the Block-A gate; left default-constructed (and unread) in FAST mode.
    steppe::QpAdmResult res;

    // =====================================================================
    // THOROUGH / no-GPU ONLY: the CpuBackend host-oracle re-derivation (the M(fit-1)
    // bit-exact diff ORACLE + the no-GPU fallback). The GPU path below re-asserts the
    // SAME golden constants, so this whole span is redundant for a routine GPU run —
    // it is the on-demand diff localizer + the CI-without-GPU acceptance gate. It also
    // carries the ~23 s NRBIG CpuBackend run_qpadm (rbig) + the ~347 s NRBIG GPU full
    // LOO-SE (run_qpadm(devbig)), neither of which is asserted-for-SE. Gate on
    // (g_thorough || gpu_count <= 0): run whenever thorough OR no GPU is visible.
    // =====================================================================
    if (g_thorough || gpu_count <= 0) {
    // Resources with a single CpuBackend (no GPU; the M(fit-1) oracle path).
    steppe::device::Resources resources;
    steppe::device::PerGpuResources cpu;
    cpu.device_id = -1;  // CPU
    cpu.backend = steppe::device::make_cpu_backend();
    resources.gpus.push_back(std::move(cpu));

    res = steppe::run_qpadm(f2, model, opts, resources);

    if (res.status != steppe::Status::Ok) {
        std::printf("  [FAIL] run_qpadm status != Ok (%d)\n", static_cast<int>(res.status));
        ++g_failures;
    }

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

    // =====================================================================
    // M(fit-2) RANK TEST / qpWave on the CpuBackend (the native ORACLE; the
    // parity step, NOT the deliverable). Validates the rank sweep (per-rank
    // chisq/dof/p), the AT2 res$rankdrop nested table, and the AT2 res$popdrop
    // leave-one-LEFT-SOURCE-out table against golden_fit0.json (the nr<=32 batched
    // gate). f4rank/dof/dofdiff EXACT; chisq/chisqdiff TIGHT (rtol 1e-6); p/p_nested
    // /rank-decision LOOSE (rtol 1e-3). The GPU deliverable (CudaBackend::rank_sweep)
    // is the NEXT phase; this asserts the oracle the GPU will be diffed against.
    // =====================================================================
    std::printf("\n========== M(fit-2) RANK TEST / qpWave (CpuBackend ORACLE) ==========\n");
    {
        // ---- golden_fit0.json fixture_f2_object_path res$rankdrop (rows: f4rank
        //      DESCENDING = rank1, rank0). CORRECTED (convertf-PA; the fixture is the
        //      read_f2 f2-object tensor). row0 = rank1: dof=4, chisq=3.9568..., p=0.4119;
        //      row1 = rank0: dof=10, chisq=1474.033..., p~1.0e-310 (model strongly
        //      rejected at rank0 — the corrected data is far more decisive). ----------
        const int    grd_f4rank[2]   = {1, 0};
        const int    grd_dof[2]      = {4, 10};
        const double grd_chisq[2]    = {3.95682062790988, 1474.03320584515};
        const double grd_p[2]        = {0.411881081897742, 1.02285567525252e-310};
        const int    grd_dofdiff     = 6;                    // row0 only (row1 = NA)
        const double grd_chisqdiff   = 1470.07638521724;     // row0 only
        const double grd_p_nested    = 1.62084120329381e-314; // row0 only
        const int    g_f4rank        = 1;                    // AT2 res$f4rank

        // The full QpAdmResult already carries the M(fit-2) surface (run_impl filled it).
        std::printf("-- rankdrop table (from run_qpadm result) --\n");
        check_eq_int("rankdrop rows", static_cast<int>(res.rankdrop_f4rank.size()), 2);
        check_eq_int("f4rank (res$f4rank)", res.f4rank, g_f4rank);
        for (int k = 0; k < 2; ++k) {
            char nm[48];
            std::snprintf(nm, sizeof(nm), "rd[%d].f4rank", k);
            check_eq_int(nm, res.rankdrop_f4rank.at(static_cast<std::size_t>(k)), grd_f4rank[k]);
            std::snprintf(nm, sizeof(nm), "rd[%d].dof", k);
            check_eq_int(nm, res.rankdrop_dof.at(static_cast<std::size_t>(k)), grd_dof[k]);
            std::snprintf(nm, sizeof(nm), "rd[%d].chisq", k);
            check_close(nm, res.rankdrop_chisq.at(static_cast<std::size_t>(k)), grd_chisq[k], 1e-6, 1e-12);
            std::snprintf(nm, sizeof(nm), "rd[%d].p", k);
            check_close(nm, res.rankdrop_p.at(static_cast<std::size_t>(k)), grd_p[k], 1e-3, 1e-9);
        }
        // nested diff: row0 (rank1 vs rank0) populated; row1 (rank0) = NA.
        check_eq_int("rd[0].dofdiff", res.rankdrop_dofdiff.at(0), grd_dofdiff);
        check_close("rd[0].chisqdiff", res.rankdrop_chisqdiff.at(0), grd_chisqdiff, 1e-6, 1e-12);
        check_close("rd[0].p_nested",  res.rankdrop_p_nested.at(0),  grd_p_nested, 1e-3, 1e-9);
        const bool row1_na = (res.rankdrop_dofdiff.at(1) == INT_MIN) &&
                             std::isnan(res.rankdrop_chisqdiff.at(1)) &&
                             std::isnan(res.rankdrop_p_nested.at(1));
        std::printf("  [%s] %-28s (dofdiff=%d chisqdiff/p_nested=NaN)\n",
                    row1_na ? "PASS" : "FAIL", "rd[1] is NA (last row)",
                    res.rankdrop_dofdiff.at(1));
        if (!row1_na) ++g_failures;

        // ---- per-rank sweep: chisq(0)=rank0, chisq(1)=rank1 (ASCENDING index) ----
        std::printf("-- per-rank sweep (chisq[r], dof[r]) --\n");
        check_eq_int("sweep ranks", static_cast<int>(res.rank_chisq.size()), 2);
        check_close("chisq[r=0]", res.rank_chisq.at(0), grd_chisq[1], 1e-6, 1e-12); // rank0
        check_close("chisq[r=1]", res.rank_chisq.at(1), grd_chisq[0], 1e-6, 1e-12); // rank1
        check_eq_int("dof[r=0]", res.rank_dof.at(0), grd_dof[1]);
        check_eq_int("dof[r=1]", res.rank_dof.at(1), grd_dof[0]);

        // ---- fixture_f2_object_path res$popdrop (rows "00","01","10"): leave-one-
        //      LEFT-SOURCE-out. CORRECTED (convertf-PA). "00" full: dof=4, chisq=3.9568,
        //      p=0.4119, f4rank=1; "01" drop Turkey_N: dof=5, chisq=43.591, p=2.80e-08;
        //      "10" drop CordedWare: dof=5, chisq=1215.218, p~1.5e-260; all feasible. ---
        const char*  gpd_pat[3]    = {"00", "01", "10"};
        const int    gpd_wt[3]     = {0, 1, 1};
        const int    gpd_dof[3]    = {4, 5, 5};
        const double gpd_chisq[3]  = {3.95682062790988, 43.5911692259, 1215.21839405374};
        const double gpd_p[3]      = {0.411881081897742, 2.80379850883969e-08, 1.48439876266533e-260};
        const int    gpd_f4rank[3] = {1, 0, 0};
        const bool   gpd_feas[3]   = {true, true, true};
        std::printf("-- popdrop table (from run_qpadm result) --\n");
        check_eq_int("popdrop rows", static_cast<int>(res.popdrop_pat.size()), 3);
        for (int k = 0; k < 3 && static_cast<std::size_t>(k) < res.popdrop_pat.size(); ++k) {
            char nm[48];
            const bool pat_ok = (res.popdrop_pat.at(static_cast<std::size_t>(k)) == gpd_pat[k]);
            std::printf("  [%s] pd[%d].pat got=%s want=%s\n", pat_ok ? "PASS" : "FAIL",
                        k, res.popdrop_pat.at(static_cast<std::size_t>(k)).c_str(), gpd_pat[k]);
            if (!pat_ok) ++g_failures;
            std::snprintf(nm, sizeof(nm), "pd[%d].wt", k);
            check_eq_int(nm, res.popdrop_wt.at(static_cast<std::size_t>(k)), gpd_wt[k]);
            std::snprintf(nm, sizeof(nm), "pd[%d].dof", k);
            check_eq_int(nm, res.popdrop_dof.at(static_cast<std::size_t>(k)), gpd_dof[k]);
            std::snprintf(nm, sizeof(nm), "pd[%d].f4rank", k);
            check_eq_int(nm, res.popdrop_f4rank.at(static_cast<std::size_t>(k)), gpd_f4rank[k]);
            std::snprintf(nm, sizeof(nm), "pd[%d].chisq", k);
            check_close(nm, res.popdrop_chisq.at(static_cast<std::size_t>(k)), gpd_chisq[k], 1e-6, 1e-12);
            std::snprintf(nm, sizeof(nm), "pd[%d].p", k);
            check_close(nm, res.popdrop_p.at(static_cast<std::size_t>(k)), gpd_p[k], 1e-3, 1e-9);
            const bool feas_ok = (res.popdrop_feasible.at(static_cast<std::size_t>(k)) != 0) == gpd_feas[k];
            std::snprintf(nm, sizeof(nm), "pd[%d].feasible", k);
            std::printf("  [%s] %-24s got=%d want=%d\n", feas_ok ? "PASS" : "FAIL", nm,
                        res.popdrop_feasible.at(static_cast<std::size_t>(k)) != 0, gpd_feas[k]);
            if (!feas_ok) ++g_failures;
        }

        // ---- DIRECT backend rank_sweep call (the dispatch report + localizer) ----
        std::printf("-- direct CpuBackend::rank_sweep (dispatch report) --\n");
        steppe::ComputeBackend& be2 = *resources.gpus.at(0).backend;
        const std::vector<int> lidx2 = {0, 1, 2};  // [target, CordedWare, Turkey_N]
        const std::vector<int> ridx2 = {3, 4, 5, 6, 7, 8};
        const steppe::Precision prec2{steppe::Precision::Kind::Fp64};
        const steppe::F4Blocks X2 = be2.assemble_f4(f2, std::span<const int>(lidx2),
                                                    std::span<const int>(ridx2), prec2);
        const steppe::JackknifeCov cov2 =
            be2.jackknife_cov(X2, std::span<const int>(f2.block_sizes), 1e-4, prec2);
        const steppe::RankSweep rsw = be2.rank_sweep(X2, cov2, opts.rank_alpha, opts, prec2);
        rsw_rank_Q_ref = rsw.rank_Q;  // captured for the GPU localizer (GPU rank_Q == this)
        check_eq_int("direct f4rank", rsw.f4rank, g_f4rank);
        // rank_Q is observability (non-gating). The full 2-source model's Q is m=10
        // (full numerical rank 10). The golden model_well_determined.rank_Q=5 is the
        // SLICE (single-leftref nr×nr=5×5) Q rank, a DIFFERENT quantity — so we
        // REPORT rsw.rank_Q here, not gate it against the slice's 5.
        std::printf("  [INFO] direct rank_Q (full m=10 model) = %d "
                    "(golden model_well_determined.rank_Q=5 is the 5x5 SLICE Q)\n",
                    rsw.rank_Q);
        const char* svd_name = (rsw.svd_path == 1) ? "gesvdjBatched would-select"
                             : (rsw.svd_path == 2) ? "gesvd would-select"
                                                   : "jacobi(executed)";
        std::printf("  [INFO] svd_path=%d (%s); executed=on-device Jacobi (nl=%d,nr=%d both<=32)\n",
                    rsw.svd_path, svd_name, X2.nl, X2.nr);
        check_eq_int("svd_path (nl,nr<=32 ⇒ gesvdjBatched)", rsw.svd_path, 1);
    }

    // ---- nr>32 fixture (cuSOLVER gesvd LARGE path): the rank test on a
    //      REAL-AADR >32-right model (golden_fit1_NRBIG.json: nl=2, nr=39, P=43).
    //      Validates the SWEEP MATH (chisq/dof/p/rankdrop/popdrop) at a large model on
    //      the CpuBackend ORACLE, then RUNS THE SAME MODEL ON THE GPU (the LARGE path:
    //      cuSOLVER gesvd SVD seed + VRAM-scratch ALS/weight/chisq, f2 RESIDENT) and
    //      GATES the GPU result against the golden — svd_path 2 = gesvd EXECUTED.
    //      The fixture is optional (skip cleanly if absent).
    //
    //      The golden rankdrop/popdrop (golden_fit1_NRBIG.json) was computed by AT2
    //      from the SAME read_f2(outdir,pops=...) f2 object that this fixture was
    //      dumped from (golden_fit1_generate.R), so it is BIT-CONSISTENT with the
    //      fixture — chisq is TIGHT (rtol 1e-6), dof/dofdiff/f4rank EXACT, like the
    //      9-pop gate. This is the NON-GATING fallback fixture (the 9-pop nr≤32
    //      batched golden above is the PRIMARY gate); it validates the SWEEP MATH at
    //      a >32-right model + the gesvd dispatch report. ---------------------------
    {
        std::printf("\n-- nr>32 gesvd-FALLBACK fixture (CpuBackend ORACLE; golden_fit1_NRBIG) --\n");
        steppe::F2BlockTensor f2big;
        if (!read_fixture(golden_dir + "/fixtures/f2_fit1_NRBIG.bin", f2big)) {
            std::printf("  [SKIP] nr>32 fixture absent — 9-pop gate stands; dispatch PENDING\n");
        } else {
            // pop_order: 0=England_BellBeaker 1=CordedWare 2=Turkey_N 3..41=right (39).
            steppe::QpAdmModel big;
            big.target = 0; big.left = {1, 2};
            big.right.clear();
            for (int j = 3; j < 43; ++j) big.right.push_back(j);  // 40 right pops (R0..R39) ⇒ nr=39
            big.model_index = 2;
            const steppe::QpAdmResult rbig = steppe::run_qpadm(f2big, big, opts, resources);
            if (rbig.status != steppe::Status::Ok) {
                std::printf("  [FAIL] nr>32 run_qpadm status != Ok (%d)\n",
                            static_cast<int>(rbig.status));
                ++g_failures;
            } else {
                // NRBIG CpuBackend-ORACLE jackknife SE/z (the ground-truth the GPU large
                // LOO must match; the GPU uses cuSOLVER SVD vs the oracle Jacobi seed ⇒
                // bit-close, the SVD-seed localizer tier). Dumped to full precision.
                if (rbig.se.size() >= 2 && rbig.z.size() >= 2) {
                    std::printf("  [INFO] NRBIG ORACLE se=[%.17g, %.17g] z=[%.17g, %.17g]\n",
                                rbig.se.at(0), rbig.se.at(1), rbig.z.at(0), rbig.z.at(1));
                }
                // golden_fit1_NRBIG.json res$rankdrop (rows rank1, rank0; read_f2 form):
                const int    b_f4rank = 1;
                const int    brd_dof[2]    = {38, 78};
                const double brd_chisq[2]  = {52.704281610335912, 190.83602239090976};
                const double brd_p[2]      = {0.05678246029948012, 1.922125797354803e-11};
                const int    brd_dofdiff   = 40;
                const double brd_chisqdiff = 138.131740780574;
                const double brd_p_nested  = 1.00598619034513e-12;
                check_eq_int("nr>32 f4rank", rbig.f4rank, b_f4rank);
                check_eq_int("nr>32 rankdrop rows", static_cast<int>(rbig.rankdrop_f4rank.size()), 2);
                for (int k = 0; k < 2; ++k) {
                    char nm[48];
                    std::snprintf(nm, sizeof(nm), "nr>32 rd[%d].dof", k);
                    check_eq_int(nm, rbig.rankdrop_dof.at(static_cast<std::size_t>(k)), brd_dof[k]);
                    std::snprintf(nm, sizeof(nm), "nr>32 rd[%d].chisq", k);
                    check_close(nm, rbig.rankdrop_chisq.at(static_cast<std::size_t>(k)), brd_chisq[k], 1e-6, 1e-9);
                    std::snprintf(nm, sizeof(nm), "nr>32 rd[%d].p", k);
                    check_close(nm, rbig.rankdrop_p.at(static_cast<std::size_t>(k)), brd_p[k], 1e-3, 1e-12);
                }
                check_eq_int("nr>32 rd[0].dofdiff", rbig.rankdrop_dofdiff.at(0), brd_dofdiff);
                check_close("nr>32 rd[0].chisqdiff", rbig.rankdrop_chisqdiff.at(0), brd_chisqdiff, 1e-6, 1e-9);
                check_close("nr>32 rd[0].p_nested",  rbig.rankdrop_p_nested.at(0),  brd_p_nested, 1e-3, 1e-13);

                // popdrop ("00","01","10"):
                const char*  bpd_pat[3]   = {"00", "01", "10"};
                const int    bpd_dof[3]   = {38, 39, 39};
                const double bpd_chisq[3] = {52.704281610335912, 100.19050317026696, 169.11350353681215};
                const int    bpd_f4rank[3]= {1, 0, 0};
                check_eq_int("nr>32 popdrop rows", static_cast<int>(rbig.popdrop_pat.size()), 3);
                for (int k = 0; k < 3 && static_cast<std::size_t>(k) < rbig.popdrop_pat.size(); ++k) {
                    char nm[48];
                    const bool pat_ok = (rbig.popdrop_pat.at(static_cast<std::size_t>(k)) == bpd_pat[k]);
                    if (!pat_ok) ++g_failures;
                    std::printf("  [%s] nr>32 pd[%d].pat got=%s want=%s\n", pat_ok ? "PASS" : "FAIL",
                                k, rbig.popdrop_pat.at(static_cast<std::size_t>(k)).c_str(), bpd_pat[k]);
                    std::snprintf(nm, sizeof(nm), "nr>32 pd[%d].dof", k);
                    check_eq_int(nm, rbig.popdrop_dof.at(static_cast<std::size_t>(k)), bpd_dof[k]);
                    std::snprintf(nm, sizeof(nm), "nr>32 pd[%d].f4rank", k);
                    check_eq_int(nm, rbig.popdrop_f4rank.at(static_cast<std::size_t>(k)), bpd_f4rank[k]);
                    std::snprintf(nm, sizeof(nm), "nr>32 pd[%d].chisq", k);
                    check_close(nm, rbig.popdrop_chisq.at(static_cast<std::size_t>(k)), bpd_chisq[k], 1e-6, 1e-9);
                }

                // dispatch report: nr=39>32 ⇒ gesvd would-select (svd_path=2).
                steppe::ComputeBackend& beb = *resources.gpus.at(0).backend;
                const std::vector<int> lidxb = {0, 1, 2};
                std::vector<int> ridxb;
                for (int j = 3; j < 43; ++j) ridxb.push_back(j);  // 40 right pops ⇒ nr=39
                const steppe::Precision precb{steppe::Precision::Kind::Fp64};
                const steppe::F4Blocks Xb = beb.assemble_f4(f2big, std::span<const int>(lidxb),
                                                            std::span<const int>(ridxb), precb);
                const steppe::JackknifeCov covb =
                    beb.jackknife_cov(Xb, std::span<const int>(f2big.block_sizes), 1e-4, precb);
                const steppe::RankSweep rswb = beb.rank_sweep(Xb, covb, opts.rank_alpha, opts, precb);
                std::printf("  [INFO] nr>32 svd_path=%d (gesvd; nl=%d nr=%d); CpuBackend "
                            "oracle (on-device Jacobi); GPU large path checked below\n",
                            rswb.svd_path, Xb.nl, Xb.nr);
                check_eq_int("nr>32 svd_path (nr>32 ⇒ gesvd)", rswb.svd_path, 2);

                // ---- nr>32 RUNNING ON THE GPU (the FROZEN CONTRACT §5.1 GATE) ----
                // The LARGE path (cuSOLVER gesvd SVD seed + VRAM-scratch ALS/weight/
                // chisq, the FROZEN CONTRACT §1/§2) now SERVES nr=39 RESIDENT on the GPU.
                // The GPU rank_sweep no longer throws — it EXECUTES and GATES against the
                // AT2 golden (dof 38/78, chisq 52.7043/190.836 rtol 1e-6, dofdiff 40,
                // chisqdiff) + a GPU-vs-CPU-oracle localizer. f2 is RESIDENT (devbig in
                // VRAM); the gather/jackknife/SVD/ALS run on the device — no host round-
                // trip of the big tensor. -------------------------------------------
                int gpu_count_big = 0;
                try { gpu_count_big = steppe::device::visible_device_count(); }
                catch (...) { gpu_count_big = 0; }
                if (gpu_count_big <= 0) {
                    std::printf("  [SKIP] nr>32 GPU sweep: no CUDA device visible\n");
                } else {
                    steppe::device::Resources gpu_big;
                    steppe::device::PerGpuResources gb;
                    gb.device_id = 0;
                    gb.backend = steppe::device::make_cuda_backend(0);
                    gb.caps = gb.backend->capabilities();
                    gpu_big.gpus.push_back(std::move(gb));
                    steppe::F2BlockTensor f2big_up = f2big;
                    f2big_up.vpair.assign(f2big.f2.size(), 0.0);
                    steppe::device::DeviceF2Blocks devbig =
                        steppe::device::upload_f2_blocks_to_device(f2big_up, 0);
                    // Assert f2 RESIDENT in VRAM (the binding requirement: no host round-trip).
                    check_eq_int("nr>32 f2 resident (f2_device != null)",
                                 devbig.f2_device() != nullptr ? 1 : 0, 1);
                    check_eq_int("nr>32 f2 resident !empty", devbig.empty() ? 0 : 1, 1);
                    steppe::ComputeBackend& gbeb = *gpu_big.gpus.at(0).backend;
                    const steppe::F4Blocks gXb =
                        gbeb.assemble_f4(devbig, std::span<const int>(lidxb),
                                         std::span<const int>(ridxb), precb);
                    const steppe::JackknifeCov gcovb =
                        gbeb.jackknife_cov(gXb, std::span<const int>(f2big.block_sizes), 1e-4, precb);
                    const steppe::RankSweep grswb =
                        gbeb.rank_sweep(gXb, gcovb, opts.rank_alpha, opts, precb);
                    std::printf("  [INFO] nr>32 GPU rank_sweep EXECUTED (cuSOLVER gesvd "
                                "large path, f2 RESIDENT); svd_path=%d (2=gesvd)\n",
                                grswb.svd_path);
                    check_eq_int("nr>32 GPU f4rank", grswb.f4rank, b_f4rank);              // 1
                    check_eq_int("nr>32 GPU svd_path (gesvd executed)", grswb.svd_path, 2);
                    check_eq_int("nr>32 GPU rd rows", static_cast<int>(grswb.rd_chisq.size()), 2);
                    check_close("nr>32 GPU rd[0].chisq", grswb.rd_chisq.at(0), brd_chisq[0], 1e-6, 1e-9);  // 52.7043
                    check_close("nr>32 GPU rd[1].chisq", grswb.rd_chisq.at(1), brd_chisq[1], 1e-6, 1e-9);  // 190.836
                    check_eq_int("nr>32 GPU rd[0].dof", grswb.rd_dof.at(0), brd_dof[0]);   // 38
                    check_eq_int("nr>32 GPU rd[1].dof", grswb.rd_dof.at(1), brd_dof[1]);   // 78
                    check_eq_int("nr>32 GPU rd[0].dofdiff", grswb.rd_dofdiff.at(0), brd_dofdiff);  // 40
                    check_close("nr>32 GPU rd[0].chisqdiff", grswb.rd_chisqdiff.at(0), brd_chisqdiff, 1e-6, 1e-9);
                    // GPU-vs-CPU-oracle localizer (cuSOLVER SVD vs core::jacobi_svd seed,
                    // then identical ALS): bit-close in practice. The golden tier (1e-6)
                    // above is non-negotiable; this localizer is 1e-7 (the §5.1 relaxation
                    // allowed ONLY for the SVD-seed delta, NOT the golden tier).
                    check_close("nr>32 GPU-vs-CPU rd[0].chisq", grswb.rd_chisq.at(0), rswb.rd_chisq.at(0), 0.0, 1e-7);
                    check_close("nr>32 GPU-vs-CPU rd[1].chisq", grswb.rd_chisq.at(1), rswb.rd_chisq.at(1), 0.0, 1e-7);

                    // ---- nr>32 popdrop ON THE GPU (the FROZEN CONTRACT §5.2) ----
                    // Run the full run_qpadm on the RESIDENT devbig (mirrors the 9-pop GPU
                    // block); this drives gls_weights' large path for the reduced popdrop
                    // models. Assert the popdrop 00/01/10 rows match the golden ON THE GPU.
                    // The NRBIG jackknife SE (the ~701-block LOO) is the parallel large-LOO
                    // gate: time this run (was ~371 s SERIAL; now the parallel kernel) and
                    // dump the se/z to prove BIT-IDENTITY vs the pre-fix value.
                    const auto t_big0 = std::chrono::steady_clock::now();
                    const steppe::QpAdmResult gpu_big_res =
                        steppe::run_qpadm(devbig, big, opts, gpu_big);
                    const auto t_big1 = std::chrono::steady_clock::now();
                    const double big_secs =
                        std::chrono::duration<double>(t_big1 - t_big0).count();
                    std::printf("  [TIME] NRBIG GPU run_qpadm (incl. ~701-block LOO SE) = "
                                "%.3f s\n", big_secs);
                    if (gpu_big_res.status != steppe::Status::Ok) {
                        std::printf("  [FAIL] nr>32 GPU run_qpadm status != Ok (%d)\n",
                                    static_cast<int>(gpu_big_res.status));
                        ++g_failures;
                    } else {
                        std::printf("  [INFO] nr>32 GPU run_qpadm EXECUTED (f2 RESIDENT); "
                                    "f4rank=%d\n", gpu_big_res.f4rank);
                        // ---- NRBIG GPU jackknife SE/z (the parallel large-LOO output) ----
                        // BIT-IDENTITY gate: these MUST equal the PRE-FIX (serial large LOO)
                        // GPU se/z exactly (the parallel kernel reuses the SAME cuSOLVER SVD
                        // seed + the SAME als_large/weight math, only the loop parallelizes).
                        // The pre-fix values were captured from the BEFORE run; assert ==.
                        if (gpu_big_res.se.size() >= 2 && gpu_big_res.z.size() >= 2) {
                            std::printf("  [INFO] NRBIG GPU se=[%.17g, %.17g] z=[%.17g, %.17g]\n",
                                        gpu_big_res.se.at(0), gpu_big_res.se.at(1),
                                        gpu_big_res.z.at(0), gpu_big_res.z.at(1));
                            // Pre-fix (serial large-LOO) GPU se/z — the BIT-IDENTITY anchor.
                            const double pre_se[2] = {kNrbigPreSe0, kNrbigPreSe1};
                            const double pre_z[2]  = {kNrbigPreZ0,  kNrbigPreZ1};
                            const char* qn[2] = {"CordedWare", "Turkey_N"};
                            for (int i = 0; i < 2; ++i) {
                                char nm[72];
                                const double dse = std::fabs(gpu_big_res.se.at(static_cast<std::size_t>(i)) - pre_se[i]);
                                const double dz  = std::fabs(gpu_big_res.z.at(static_cast<std::size_t>(i)) - pre_z[i]);
                                std::snprintf(nm, sizeof(nm), "NRBIG GPU se[%s] bit-identical", qn[i]);
                                check_close(nm, gpu_big_res.se.at(static_cast<std::size_t>(i)), pre_se[i], 0.0, 0.0);
                                std::snprintf(nm, sizeof(nm), "NRBIG GPU z[%s] bit-identical", qn[i]);
                                check_close(nm, gpu_big_res.z.at(static_cast<std::size_t>(i)), pre_z[i], 0.0, 0.0);
                                std::printf("  [INFO] NRBIG GPU |dse[%s]|=%.3g |dz[%s]|=%.3g (vs pre-fix)\n",
                                            qn[i], dse, qn[i], dz);
                            }
                        }
                        check_eq_int("nr>32 GPU rankdrop f4rank", gpu_big_res.f4rank, b_f4rank);
                        check_eq_int("nr>32 GPU rankdrop rows",
                                     static_cast<int>(gpu_big_res.rankdrop_f4rank.size()), 2);
                        for (int k = 0; k < 2; ++k) {
                            char nm[56];
                            std::snprintf(nm, sizeof(nm), "nr>32 GPU rd[%d].dof", k);
                            check_eq_int(nm, gpu_big_res.rankdrop_dof.at(static_cast<std::size_t>(k)), brd_dof[k]);
                            std::snprintf(nm, sizeof(nm), "nr>32 GPU rd[%d].chisq", k);
                            check_close(nm, gpu_big_res.rankdrop_chisq.at(static_cast<std::size_t>(k)), brd_chisq[k], 1e-6, 1e-9);
                        }
                        std::printf("-- nr>32 GPU popdrop vs GOLDEN (rows 00/01/10) [ran ON THE GPU] --\n");
                        check_eq_int("nr>32 GPU popdrop rows",
                                     static_cast<int>(gpu_big_res.popdrop_pat.size()), 3);
                        for (int k = 0; k < 3 && static_cast<std::size_t>(k) < gpu_big_res.popdrop_pat.size(); ++k) {
                            char nm[56];
                            const bool pat_ok = (gpu_big_res.popdrop_pat.at(static_cast<std::size_t>(k)) == bpd_pat[k]);
                            std::printf("  [%s] nr>32 GPU pd[%d].pat got=%s want=%s\n", pat_ok ? "PASS" : "FAIL",
                                        k, gpu_big_res.popdrop_pat.at(static_cast<std::size_t>(k)).c_str(), bpd_pat[k]);
                            if (!pat_ok) ++g_failures;
                            std::snprintf(nm, sizeof(nm), "nr>32 GPU pd[%d].dof", k);
                            check_eq_int(nm, gpu_big_res.popdrop_dof.at(static_cast<std::size_t>(k)), bpd_dof[k]);
                            std::snprintf(nm, sizeof(nm), "nr>32 GPU pd[%d].f4rank", k);
                            check_eq_int(nm, gpu_big_res.popdrop_f4rank.at(static_cast<std::size_t>(k)), bpd_f4rank[k]);
                            std::snprintf(nm, sizeof(nm), "nr>32 GPU pd[%d].chisq", k);
                            check_close(nm, gpu_big_res.popdrop_chisq.at(static_cast<std::size_t>(k)), bpd_chisq[k], 1e-6, 1e-9);
                        }
                    }
                }
            }
        }
    }

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
        // CORRECTED golden X (golden_fit0.json X_f4_estimate_vector) and Q diag
        // (Q_jackknife_covariance matrix diagonal) — the directory-path slice values
        // (convertf-PA source). The steppe fit reproduces these to ~1e-6 (the read-arg
        // caveat is on the m=10 internal X, not this single-leftref slice).
        const double gX[5] = {0.00362619866462659, -0.00124206979395638,
                              -0.00191622501527577, -0.00158336156336976,
                              -0.00403296857500433};
        const double gQdiag[5] = {3.21777968355993e-08, 2.18613885872559e-08,
                                  2.02104009961407e-08, 2.69523306005569e-08,
                                  3.04583757986008e-08};
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
    }  // end THOROUGH/no-GPU CpuBackend oracle gate (Block A)

    // =====================================================================
    // CUDA BACKEND — the PRODUCTION GPU PATH (M(fit-4)). Upload the golden f2
    // fixture to VRAM as a DeviceF2Blocks, assert it is RESIDENT + a real GPU is
    // bound, run the fit ON THE GPU (run_qpadm(DeviceF2Blocks) → the f4-gather
    // kernel reading resident f2 → jackknife SYRK/cuSOLVER → on-device SVD/ALS/
    // weight → batched LOO), and assert the GPU result matches the golden. SKIP
    // cleanly if no CUDA device is visible (gpu_count was probed at the top of main).
    // The DEFAULT (FAST) path is exactly this GPU-vs-golden block: 9-pop fit + SE +
    // rank/popdrop + X/Q, all vs the AT2 goldens.
    // =====================================================================
    std::printf("\n=== CUDA backend (GPU path, f2 RESIDENT) ===\n");
    if (gpu_count <= 0) {
        std::printf("  [SKIP] no CUDA device visible — CpuBackend path alone gates "
                    "(CI-without-GPU degrades cleanly)\n");
    } else {
        // Build a GPU Resources (one CudaBackend bound to device 0).
        steppe::device::Resources gpu_res;
        steppe::device::PerGpuResources g0;
        g0.device_id = 0;
        g0.backend = steppe::device::make_cuda_backend(0);
        g0.caps = g0.backend->capabilities();
        gpu_res.gpus.push_back(std::move(g0));

        // Prove a real CudaBackend (not CpuBackend whose caps are all-zero).
        const steppe::BackendCapabilities& caps = gpu_res.gpus.at(0).caps;
        check_eq_int("gpu caps.device_count>=1", caps.device_count >= 1 ? 1 : 0, 1);
        check_eq_int("gpu caps.compute_major>=1", caps.compute_major >= 1 ? 1 : 0, 1);
        std::printf("  GPU compute capability sm_%d%d, device_count=%d\n",
                    caps.compute_major, caps.compute_minor, caps.device_count);

        // Upload the golden f2 fixture to VRAM as a DeviceF2Blocks (the contract's
        // "the golden f2 fixture is loaded into a DeviceF2Blocks for the GPU fit").
        // The fit reads block_sizes, NOT vpair (OQ-3), and read_fixture leaves vpair
        // empty; upload_f2_blocks_to_device copies BOTH slabs, so size vpair to match
        // (zeros — the fit never reads it; the f2 slab carries the golden numbers).
        steppe::F2BlockTensor f2_up = f2;
        f2_up.vpair.assign(f2.f2.size(), 0.0);
        steppe::device::DeviceF2Blocks dev_f2 =
            steppe::device::upload_f2_blocks_to_device(f2_up, 0);
        // Assert f2 RESIDENT in VRAM (the binding requirement: no host round-trip).
        check_eq_int("f2 resident (f2_device != null)", dev_f2.f2_device() != nullptr ? 1 : 0, 1);
        check_eq_int("f2 resident device_id==0", dev_f2.device_id, 0);
        check_eq_int("f2 resident !empty", dev_f2.empty() ? 0 : 1, 1);
        check_eq_int("f2 resident n_block", dev_f2.n_block, f2.n_block);

        // Run the fit ON THE GPU (the DeviceF2Blocks overload = the production entry).
        const steppe::QpAdmResult gpu = steppe::run_qpadm(dev_f2, model, opts, gpu_res);
        if (gpu.status != steppe::Status::Ok) {
            std::printf("  [FAIL] GPU run_qpadm status != Ok (%d)\n",
                        static_cast<int>(gpu.status));
            ++g_failures;
        } else {
            // The DEFAULT GPU path now runs the covariance SYRK EMULATED{40} by
            // default (the unified fit precision policy; fit-engine.md §1.4). These
            // GPU-vs-golden checks are therefore the FIRST informative emulated-SYRK
            // measurement in the fit (the d6d3cbb cuSOLVER probe degraded to native and
            // was uninformative). The precision_tag the run reports is logged below.
            std::printf("\n-- GPU weights vs GOLDEN (TIGHT rtol 1e-6) [SYRK = emulated{40} by default] --\n");
            std::printf("  [INFO] gpu precision_tag = %s (1=EmulatedFp64, 0=Fp64)\n",
                        gpu.precision_tag == steppe::Precision::Kind::EmulatedFp64
                            ? "EmulatedFp64" : "Fp64");
            check_close("gpu weight[CordedWare]", gpu.weight.at(0), g_w0, 1e-6, 1e-12);
            check_close("gpu weight[Turkey_N]",   gpu.weight.at(1), g_w1, 1e-6, 1e-12);
            std::printf("-- GPU chisq (TIGHT) / dof (EXACT) --\n");
            check_close("gpu chisq", gpu.chisq, g_chisq, 1e-6, 1e-12);
            check_eq_int("gpu dof", gpu.dof, g_dof);
            std::printf("-- GPU se / z / p (LOOSE rtol 1e-3) --\n");
            check_close("gpu se[CordedWare]", gpu.se.at(0), g_se, 1e-3, 1e-9);
            check_close("gpu se[Turkey_N]",   gpu.se.at(1), g_se, 1e-3, 1e-9);
            check_close("gpu z[CordedWare]",  gpu.z.at(0),  g_z0, 1e-3, 1e-6);
            check_close("gpu z[Turkey_N]",    gpu.z.at(1),  g_z1, 1e-3, 1e-6);
            check_close("gpu p", gpu.p, g_p, 1e-3, 1e-9);

            // Diff ORACLE: GPU == CpuBackend to 1e-9 (localizes a GPU regression
            // against the bit-exact reference, not only the golden constants). The
            // CpuBackend ignores `precision` ⇒ always native ⇒ it is the native oracle
            // the emulated-SYRK GPU path is diffed against. THOROUGH-only: reads the
            // CpuBackend oracle `res`, which is only populated under Block A.
            if (g_thorough) {
                std::printf("-- GPU(emulated-SYRK) vs CpuBackend oracle (NATIVE) (diff localizer, 1e-9) --\n");
                check_close("gpu-vs-cpu weight[0]", gpu.weight.at(0), res.weight.at(0), 0.0, 1e-9);
                check_close("gpu-vs-cpu weight[1]", gpu.weight.at(1), res.weight.at(1), 0.0, 1e-9);
                check_close("gpu-vs-cpu chisq",     gpu.chisq,        res.chisq,        0.0, 1e-9);
            }

            // REAL per-quantity deltas of the (now emulated-by-default) SYRK path
            // vs the af6a8c2 golden. Reported (the gating is the check_close tiers
            // above); the vs-native-oracle deltas are THOROUGH-only (they read `res`).
            std::printf("-- [INFO] emulated-SYRK per-quantity deltas (vs golden) --\n");
            report_delta("emuSYRK weight[CordedWare] vs golden", gpu.weight.at(0), g_w0);
            report_delta("emuSYRK weight[Turkey_N]   vs golden", gpu.weight.at(1), g_w1);
            report_delta("emuSYRK chisq              vs golden", gpu.chisq,        g_chisq);
            report_delta("emuSYRK se[CordedWare]     vs golden", gpu.se.at(0),     g_se);
            report_delta("emuSYRK p                  vs golden", gpu.p,            g_p);
            if (g_thorough) {
                report_delta("emuSYRK weight[0] vs native-oracle",   gpu.weight.at(0), res.weight.at(0));
                report_delta("emuSYRK weight[1] vs native-oracle",   gpu.weight.at(1), res.weight.at(1));
                report_delta("emuSYRK chisq     vs native-oracle",   gpu.chisq,        res.chisq);
            }

            // =================================================================
            // EXPLORATORY (NON-GATING): the PROMOTED-emulated solve measurement
            // (ROADMAP §6 the fit-solve promotion seam). Re-run the SAME GPU fit
            // with the cuSOLVER SOLVE stage PROMOTED to EmulatedFp64{40} via
            // ComputeBackend::set_solve_precision — forward evidence on whether
            // the emulated tensor-core solve holds parity at THIS Q/X conditioning
            // (the S8 question). This does NOT touch g_failures: it is REPORTED,
            // not a pass condition. The default path above remains the gate. After
            // measuring we RESTORE the solve precision to native so any later use
            // of this backend is the default (the seam is local, but be explicit).
            //
            // NOTE: on CUDA 13.0 / cuSOLVER 12.0 there is no FP64-emulated cuSOLVER
            // math mode (cusolverMathMode_t lacks it; verified on box5090), so the
            // promotion DEGRADES to native with a one-shot capability tag — the
            // engage+restore round-trip still runs (the seam is live), and the
            // numbers are therefore expected to equal the native path. When a newer
            // cuSOLVER adds the mode the same path will actually promote and this
            // measurement becomes the real emulated-vs-golden delta.
            // THOROUGH-only: this re-runs the FULL GPU fit a SECOND time (pure cost,
            // zero assertions — forward S8 evidence, not an acceptance gate).
            if (g_thorough) {
                std::printf("\n-- [EXPLORATORY, NON-GATING] solves PROMOTED to "
                            "EmulatedFp64{40} vs GOLDEN (ROADMAP §6 seam) --\n");
                steppe::ComputeBackend& seam_be = *gpu_res.gpus.at(0).backend;
                const steppe::Precision emu40{steppe::Precision::Kind::EmulatedFp64, 40};
                seam_be.set_solve_precision(emu40);
                const steppe::QpAdmResult gpu_emu =
                    steppe::run_qpadm(dev_f2, model, opts, gpu_res);
                seam_be.set_solve_precision(steppe::Precision{steppe::Precision::Kind::Fp64});
                if (gpu_emu.status != steppe::Status::Ok) {
                    std::printf("  [INFO] promoted-emulated run status != Ok (%d) — "
                                "reported only, not a gate\n",
                                static_cast<int>(gpu_emu.status));
                } else {
                    report_delta("emu40 weight[CordedWare] vs golden", gpu_emu.weight.at(0), g_w0);
                    report_delta("emu40 weight[Turkey_N]   vs golden", gpu_emu.weight.at(1), g_w1);
                    report_delta("emu40 chisq               vs golden", gpu_emu.chisq,        g_chisq);
                    report_delta("emu40 se[CordedWare]      vs golden", gpu_emu.se.at(0),     g_se);
                    report_delta("emu40 se[Turkey_N]        vs golden", gpu_emu.se.at(1),     g_se);
                    report_delta("emu40 z[CordedWare]       vs golden", gpu_emu.z.at(0),      g_z0);
                    report_delta("emu40 z[Turkey_N]         vs golden", gpu_emu.z.at(1),      g_z1);
                    report_delta("emu40 p                   vs golden", gpu_emu.p,            g_p);
                    // and against the native GPU path (does promotion perturb the solve?)
                    report_delta("emu40 weight[0] vs native-GPU", gpu_emu.weight.at(0), gpu.weight.at(0));
                    report_delta("emu40 chisq     vs native-GPU", gpu_emu.chisq,        gpu.chisq);
                    std::printf("  [INFO] tier check (informational): weights/chisq within "
                                "1e-6 of golden? %s ; se/z/p within 1e-3? %s\n",
                                (rel_within(gpu_emu.weight.at(0), g_w0, 1e-6, 1e-12) &&
                                 rel_within(gpu_emu.weight.at(1), g_w1, 1e-6, 1e-12) &&
                                 rel_within(gpu_emu.chisq, g_chisq, 1e-6, 1e-12)) ? "YES" : "NO",
                                (rel_within(gpu_emu.se.at(0), g_se, 1e-3, 1e-9) &&
                                 rel_within(gpu_emu.p, g_p, 1e-3, 1e-9)) ? "YES" : "NO");
                }
            }

            // =================================================================
            // M(fit-2) GPU RANK TEST / qpWave — THE DELIVERABLE, THE GATE. The
            // `gpu` QpAdmResult above ALREADY carries the rankdrop/popdrop tables
            // filled by run_impl through CudaBackend::rank_sweep + run_popdrop —
            // i.e. THE RANK SWEEP RAN ON THE GPU (f2 resident, the f4-gather →
            // jackknife SYRK → on-device seed/ALS/weight/chisq per r → host
            // dof/p/rankdrop/f4rank). Assert it reproduces the AT2 goldens
            // (golden_fit0.json rankdrop/popdrop) AND matches the CpuBackend
            // oracle to 1e-9 (the localizer). The CpuBackend constants are the
            // SAME row-for-row goldens checked in the oracle block above.
            // -----------------------------------------------------------------
            std::printf("\n-- GPU RANK TEST vs GOLDEN (rankdrop, TIGHT chisq rtol 1e-6) [ran ON THE GPU] --\n");
            const int    grd_f4rank[2] = {1, 0};
            const int    grd_dof[2]    = {4, 10};
            const double grd_chisq[2]  = {3.95682062790988, 1474.03320584515};
            const double grd_p[2]      = {0.411881081897742, 1.02285567525252e-310};
            const int    grd_dofdiff   = 6;
            const double grd_chisqdiff = 1470.07638521724;
            const double grd_p_nested  = 1.62084120329381e-314;
            const int    g_f4rank      = 1;
            check_eq_int("gpu rankdrop rows", static_cast<int>(gpu.rankdrop_f4rank.size()), 2);
            check_eq_int("gpu f4rank (res$f4rank)", gpu.f4rank, g_f4rank);
            for (int k = 0; k < 2 && static_cast<std::size_t>(k) < gpu.rankdrop_f4rank.size(); ++k) {
                char nm[56];
                std::snprintf(nm, sizeof(nm), "gpu rd[%d].f4rank", k);
                check_eq_int(nm, gpu.rankdrop_f4rank.at(static_cast<std::size_t>(k)), grd_f4rank[k]);
                std::snprintf(nm, sizeof(nm), "gpu rd[%d].dof", k);
                check_eq_int(nm, gpu.rankdrop_dof.at(static_cast<std::size_t>(k)), grd_dof[k]);
                std::snprintf(nm, sizeof(nm), "gpu rd[%d].chisq", k);
                check_close(nm, gpu.rankdrop_chisq.at(static_cast<std::size_t>(k)), grd_chisq[k], 1e-6, 1e-12);
                std::snprintf(nm, sizeof(nm), "gpu rd[%d].p", k);
                check_close(nm, gpu.rankdrop_p.at(static_cast<std::size_t>(k)), grd_p[k], 1e-3, 1e-9);
            }
            check_eq_int("gpu rd[0].dofdiff", gpu.rankdrop_dofdiff.at(0), grd_dofdiff);
            check_close("gpu rd[0].chisqdiff", gpu.rankdrop_chisqdiff.at(0), grd_chisqdiff, 1e-6, 1e-12);
            check_close("gpu rd[0].p_nested",  gpu.rankdrop_p_nested.at(0),  grd_p_nested, 1e-3, 1e-9);
            const bool gpu_row1_na = (gpu.rankdrop_dofdiff.at(1) == INT_MIN) &&
                                     std::isnan(gpu.rankdrop_chisqdiff.at(1)) &&
                                     std::isnan(gpu.rankdrop_p_nested.at(1));
            std::printf("  [%s] gpu rd[1] is NA (last row, dofdiff=%d)\n",
                        gpu_row1_na ? "PASS" : "FAIL", gpu.rankdrop_dofdiff.at(1));
            if (!gpu_row1_na) ++g_failures;

            std::printf("-- GPU per-rank sweep (chisq[r], dof[r]) --\n");
            check_eq_int("gpu sweep ranks", static_cast<int>(gpu.rank_chisq.size()), 2);
            check_close("gpu chisq[r=0]", gpu.rank_chisq.at(0), grd_chisq[1], 1e-6, 1e-12);
            check_close("gpu chisq[r=1]", gpu.rank_chisq.at(1), grd_chisq[0], 1e-6, 1e-12);
            check_eq_int("gpu dof[r=0]", gpu.rank_dof.at(0), grd_dof[1]);
            check_eq_int("gpu dof[r=1]", gpu.rank_dof.at(1), grd_dof[0]);

            std::printf("-- GPU popdrop vs GOLDEN (rows 00/01/10) [ran ON THE GPU] --\n");
            const char*  gpd_pat[3]    = {"00", "01", "10"};
            const int    gpd_wt[3]     = {0, 1, 1};
            const int    gpd_dof[3]    = {4, 5, 5};
            const double gpd_chisq[3]  = {3.95682062790988, 43.5911692259, 1215.21839405374};
            const double gpd_p[3]      = {0.411881081897742, 2.80379850883969e-08, 1.48439876266533e-260};
            const int    gpd_f4rank[3] = {1, 0, 0};
            const bool   gpd_feas[3]   = {true, true, true};
            check_eq_int("gpu popdrop rows", static_cast<int>(gpu.popdrop_pat.size()), 3);
            for (int k = 0; k < 3 && static_cast<std::size_t>(k) < gpu.popdrop_pat.size(); ++k) {
                char nm[56];
                const bool pat_ok = (gpu.popdrop_pat.at(static_cast<std::size_t>(k)) == gpd_pat[k]);
                std::printf("  [%s] gpu pd[%d].pat got=%s want=%s\n", pat_ok ? "PASS" : "FAIL",
                            k, gpu.popdrop_pat.at(static_cast<std::size_t>(k)).c_str(), gpd_pat[k]);
                if (!pat_ok) ++g_failures;
                std::snprintf(nm, sizeof(nm), "gpu pd[%d].wt", k);
                check_eq_int(nm, gpu.popdrop_wt.at(static_cast<std::size_t>(k)), gpd_wt[k]);
                std::snprintf(nm, sizeof(nm), "gpu pd[%d].dof", k);
                check_eq_int(nm, gpu.popdrop_dof.at(static_cast<std::size_t>(k)), gpd_dof[k]);
                std::snprintf(nm, sizeof(nm), "gpu pd[%d].f4rank", k);
                check_eq_int(nm, gpu.popdrop_f4rank.at(static_cast<std::size_t>(k)), gpd_f4rank[k]);
                std::snprintf(nm, sizeof(nm), "gpu pd[%d].chisq", k);
                check_close(nm, gpu.popdrop_chisq.at(static_cast<std::size_t>(k)), gpd_chisq[k], 1e-6, 1e-12);
                std::snprintf(nm, sizeof(nm), "gpu pd[%d].p", k);
                check_close(nm, gpu.popdrop_p.at(static_cast<std::size_t>(k)), gpd_p[k], 1e-3, 1e-9);
                const bool feas_ok = (gpu.popdrop_feasible.at(static_cast<std::size_t>(k)) != 0) == gpd_feas[k];
                std::printf("  [%s] gpu pd[%d].feasible got=%d want=%d\n", feas_ok ? "PASS" : "FAIL",
                            k, gpu.popdrop_feasible.at(static_cast<std::size_t>(k)) != 0, gpd_feas[k]);
                if (!feas_ok) ++g_failures;
            }

            // GPU == CpuBackend ORACLE to 1e-9 (the localizer: any GPU rank-test
            // regression localizes against the native reference, not only the golden).
            // THOROUGH-only: reads the CpuBackend oracle `res` (Block A).
            if (g_thorough) {
                std::printf("-- GPU rank test vs CpuBackend ORACLE (diff localizer, 1e-9) --\n");
                check_eq_int("gpu-vs-cpu f4rank", gpu.f4rank, res.f4rank);
                check_close("gpu-vs-cpu rd[0].chisq", gpu.rankdrop_chisq.at(0), res.rankdrop_chisq.at(0), 0.0, 1e-9);
                check_close("gpu-vs-cpu rd[1].chisq", gpu.rankdrop_chisq.at(1), res.rankdrop_chisq.at(1), 0.0, 1e-9);
                check_close("gpu-vs-cpu rd[0].chisqdiff", gpu.rankdrop_chisqdiff.at(0), res.rankdrop_chisqdiff.at(0), 0.0, 1e-9);
                check_close("gpu-vs-cpu pd[1].chisq", gpu.popdrop_chisq.at(1), res.popdrop_chisq.at(1), 0.0, 1e-9);
                check_close("gpu-vs-cpu pd[2].chisq", gpu.popdrop_chisq.at(2), res.popdrop_chisq.at(2), 0.0, 1e-9);
            }

            // DIRECT GPU rank_sweep (the dispatch report + assert the GPU backend ran it).
            std::printf("-- direct CudaBackend::rank_sweep (dispatch report, ON THE GPU) --\n");
            steppe::ComputeBackend& gbe2 = *gpu_res.gpus.at(0).backend;
            const std::vector<int> glidx2 = {0, 1, 2};   // [target, CordedWare, Turkey_N]
            const std::vector<int> gridx2 = {3, 4, 5, 6, 7, 8};
            const steppe::Precision gprec2{steppe::Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
            const steppe::F4Blocks gX2 =
                gbe2.assemble_f4(dev_f2, std::span<const int>(glidx2),
                                 std::span<const int>(gridx2), gprec2);
            const steppe::JackknifeCov gcov2 =
                gbe2.jackknife_cov(gX2, std::span<const int>(f2.block_sizes), opts.fudge, gprec2);
            const steppe::RankSweep grsw =
                gbe2.rank_sweep(gX2, gcov2, opts.rank_alpha, opts, gprec2);
            check_eq_int("direct gpu f4rank", grsw.f4rank, g_f4rank);
            const char* gsvd_name = (grsw.svd_path == 1) ? "gesvdjBatched would-select"
                                  : (grsw.svd_path == 2) ? "gesvd would-select"
                                                         : "on-device Jacobi (executed)";
            std::printf("  [INFO] gpu svd_path=%d (%s); executed=on-device Jacobi "
                        "(nl=%d,nr=%d both<=32); rank_Q=%d\n",
                        grsw.svd_path, gsvd_name, gX2.nl, gX2.nr, grsw.rank_Q);
            check_eq_int("gpu svd_path (nl,nr<=32 ⇒ gesvdjBatched)", grsw.svd_path, 1);
            // THOROUGH-only: rsw_rank_Q_ref is the CpuBackend oracle rank_Q (Block A),
            // 0 in FAST mode (gpu rank_Q is reported above either way).
            if (g_thorough)
                check_eq_int("gpu-vs-cpu rank_Q", grsw.rank_Q, rsw_rank_Q_ref);
        }

        // GPU X/Q localizers: call the CUDA backend's assemble_f4(DeviceF2Blocks) +
        // jackknife_cov directly (proving S3/S4 ran on the GPU over resident f2).
        std::printf("\n-- GPU S3/S4 slice cross-check (LOCALIZER, loose rtol 1e-3) --\n");
        const double gX[5] = {0.00362619866462659, -0.00124206979395638,
                              -0.00191622501527577, -0.00158336156336976,
                              -0.00403296857500433};
        const double gQdiag[5] = {3.21777968355993e-08, 2.18613885872559e-08,
                                  2.02104009961407e-08, 2.69523306005569e-08,
                                  3.04583757986008e-08};
        steppe::ComputeBackend& gbe = *gpu_res.gpus.at(0).backend;
        const std::vector<int> lidx = {1, 2};
        const std::vector<int> ridx = {3, 4, 5, 6, 7, 8};
        const steppe::Precision gprec{steppe::Precision::Kind::Fp64};
        const steppe::F4Blocks gX_blocks =
            gbe.assemble_f4(dev_f2, std::span<const int>(lidx),
                            std::span<const int>(ridx), gprec);
        const steppe::JackknifeCov gcov =
            gbe.jackknife_cov(gX_blocks, std::span<const int>(f2.block_sizes), 1e-4, gprec);
        for (int j = 0; j < 5; ++j) {
            char nm[40];
            std::snprintf(nm, sizeof(nm), "gpu X_slice[%d]", j);
            check_close(nm, gX_blocks.x_total.at(static_cast<std::size_t>(j)), gX[j], 1e-3, 1e-9);
        }
        for (int k = 0; k < 5; ++k) {
            char nm[40];
            std::snprintf(nm, sizeof(nm), "gpu Q_diag[%d]", k);
            check_close(nm, gcov.Q.at(static_cast<std::size_t>(k) + 5u * static_cast<std::size_t>(k)),
                        gQdiag[k], 1e-3, 1e-9);
        }
    }

    std::printf("\n%s\n", g_failures == 0 ? "RESULT: PASS" : "RESULT: FAIL");
    return g_failures == 0 ? 0 : 1;
}
