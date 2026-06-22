// tests/reference/test_qpwave_parity.cu
//
// F4 (fit-engine-finish-punchlist): the standalone qpWave acceptance gate. run_qpwave
// is a first-class PUBLIC entry (include/steppe/qpadm.hpp: the DeviceF2Blocks +
// F2BlockTensor overloads, impl src/core/qpadm/qpadm_fit.cpp run_qpwave_impl) that, until
// now, had ZERO direct test coverage — its distinguishing semantic (NO target prepend;
// `left` IS the rows, left[0] is the qpWave reference) was unvalidated. This test calls
// run_qpwave DIRECTLY on BOTH backends and gates est_rank / f4rank / the rankdrop table /
// chisq against a pinned REAL AT2 qpwave() golden (golden_qpwave.json, admixtools 2.0.10 /
// R 4.3.3, boot=FALSE) — the same golden-test pattern as test_qpadm_parity.cu /
// test_qpadm_domain.cu.
//
// NO SYNTHETIC DATA (memory real-data-only). Both qpWave models REUSE the committed
// REAL-AADR 9-pop f2 fixture (goldens/at2/fixtures/f2_fit0_9pop.bin — the tensor AT2 read
// internally for golden_fit0), arranged as qpWave left/right splits over REAL pops; pop
// index 0=England_BellBeaker 1=Czechia_EBA_CordedWare 2=Turkey_N 3=Mbuti 4=Israel_Natufian
// 5=Iran_GanjDareh_N 6=Han 7=Papuan 8=Karitiana:
//   M1 (3-left, est_rank=1): left={England_BellBeaker, CordedWare, Turkey_N} (left[0]=
//       reference), right={the 6 outgroups}. The qpadm target + 2 sources arranged as ALL
//       left rows — validates that run_qpwave's no-target path builds the SAME f4 matrix as
//       qpadm's internal left=c(target,sources) (its rankdrop equals golden_fit0's). The
//       rank-1 model is NOT rejected at alpha=0.05 ⇒ f4rank/est_rank = 1.
//   M2 (2-left CLADE, est_rank=0): left={CordedWare, Turkey_N} (left[0]=reference),
//       right={the 6 outgroups}. The canonical qpWave clade question — REJECTED
//       (p=7.24e-5 < 0.05) ⇒ all candidate ranks rejected ⇒ f4rank/est_rank = rmax = 0.
//       The single candidate rank ⇒ a SINGLE-row rankdrop (qpWave's distinguishing
//       single-difference-row sweep, a shape the qpadm goldens never exercise).
//
// TIERS (fit-engine.md §4.1 / SPEC §12): f4rank / est_rank / dof / dofdiff EXACT (==);
// chisq / chisqdiff TIGHT (rtol 1e-6); p / p_nested LOOSE (rtol 1e-3, the rank-decision
// tier). Both backends MUST agree (CpuBackend oracle == CudaBackend deliverable); a
// GPU-vs-oracle localizer (1e-9) runs under THOROUGH.
//
// Self-checking main() (returns non-zero on any failure; CTest gates on the exit code; no
// GoogleTest). The CpuBackend ORACLE block runs under STEPPE_THOROUGH=1 OR when no GPU is
// visible (the CI-without-GPU acceptance gate); the CudaBackend DELIVERABLE block runs
// whenever a CUDA device is visible (the default FAST path) — mirroring test_qpadm_*.cu.

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "device/backend_factory.hpp"  // make_cpu_backend / make_cuda_backend / visible_device_count
#include "device/device_f2_blocks.hpp" // DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"        // Resources / PerGpuResources
#include "steppe/error.hpp"            // Status
#include "steppe/fstats.hpp"           // F2BlockTensor
#include "steppe/qpadm.hpp"            // run_qpwave + QpWaveResult / QpAdmOptions

namespace {

int g_failures = 0;

void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const double diff = std::fabs(got - want);
    const bool ok = diff <= tol;
    std::printf("  [%s] %-30s got=% .12e want=% .12e |d|=% .3e tol=% .3e\n",
                ok ? "PASS" : "FAIL", what, got, want, diff, tol);
    if (!ok) ++g_failures;
}

void check_eq_int(const char* what, int got, int want) {
    const bool ok = (got == want);
    std::printf("  [%s] %-30s got=%d want=%d\n", ok ? "PASS" : "FAIL", what, got, want);
    if (!ok) ++g_failures;
}

void check_true(const char* what, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

// Read the committed binary f2 fixture (same format as test_qpadm_parity.cu):
// int32 P, int32 n_block, n_block × int32 block_sizes, P*P*n_block × float64 f2.
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
    if (!f) { std::printf("  [FAIL] fixture truncated\n"); return false; }
    std::printf("  fixture: P=%d n_block=%d\n", P, nb);
    return true;
}

// A pinned REAL qpwave() golden model (golden_qpwave.json). The rankdrop rows are f4rank
// DESCENDING (AT2 res$rankdrop order; the LAST row is the lowest rank with NA nested
// columns). rank_chisq/rank_dof in QpWaveResult are per-rank ASCENDING (the sweep order).
struct QpWaveGolden {
    const char*         label;
    std::vector<int>    left;      // left[0] is the reference (NO target)
    std::vector<int>    right;     // right[0] is R0
    int                 est_rank;  // == f4rank (AT2)
    int                 f4rank;
    // rankdrop, f4rank-DESCENDING:
    std::vector<int>    rd_f4rank, rd_dof;
    std::vector<double> rd_chisq, rd_p;
    int                 rd0_dofdiff;     // row 0 only (last row = NA)
    double              rd0_chisqdiff;
    double              rd0_p_nested;
    bool                has_nested;      // M1 has a nested-diff row 0; M2 (single row) does not
};

std::vector<QpWaveGolden> goldens() {
    std::vector<QpWaveGolden> g;

    // M1 (3-left, est_rank=1): left[0]=England_BellBeaker reference, then CordedWare,
    // Turkey_N; right = the 6 outgroups. rankdrop == golden_fit0's (same internal f4).
    {
        QpWaveGolden m;
        m.label = "M1 3-left (reference=England_BellBeaker), est_rank=1";
        m.left  = {0, 1, 2};
        m.right = {3, 4, 5, 6, 7, 8};
        m.est_rank = 1;
        m.f4rank   = 1;
        // CORRECTED (convertf-PA; rankdrop == golden_fit0 fixture_f2_object_path's, same
        // internal f4). The prior 4.635/31.97 was AT2 2.0.10's TGENO misread.
        m.rd_f4rank = {1, 0};
        m.rd_dof    = {4, 10};
        m.rd_chisq  = {3.95682062790988, 1474.03320584515};
        m.rd_p      = {0.411881081897742, 1.02285567525252e-310};
        m.rd0_dofdiff   = 6;
        m.rd0_chisqdiff = 1470.07638521724;
        m.rd0_p_nested  = 1.62084120329381e-314;
        m.has_nested    = true;
        g.push_back(m);
    }

    // M2 (2-left CLADE, est_rank=0): left[0]=CordedWare reference, then Turkey_N; right =
    // the 6 outgroups. The clade is REJECTED (p < 0.05) ⇒ est_rank=0. Single rankdrop row.
    {
        QpWaveGolden m;
        m.label = "M2 2-left clade (reference=CordedWare), est_rank=0";
        m.left  = {1, 2};
        m.right = {3, 4, 5, 6, 7, 8};
        m.est_rank = 0;
        m.f4rank   = 0;
        m.rd_f4rank = {0};
        m.rd_dof    = {5};
        // CORRECTED (convertf-PA): the clade is REJECTED far more decisively than the
        // corrupt TGENO golden (was chisq 26.47). chisq is the committed binary-fixture
        // f2-object value (read-arg caveat vs AT2's afprod directory object chisq
        // 1395.716; the clade is rejected at p<<0.05 either way ⇒ est_rank=0 is robust).
        m.rd_chisq  = {1401.571655111};
        m.rd_p      = {6.284245136077e-301};
        m.rd0_dofdiff   = INT_MIN;   // unused (single row ⇒ no nested diff)
        m.rd0_chisqdiff = 0.0;
        m.rd0_p_nested  = 0.0;
        m.has_nested    = false;
        g.push_back(m);
    }

    return g;
}

// Gate one QpWaveResult against its pinned golden. `tag` labels the backend.
void gate_qpwave(const char* tag, const QpWaveGolden& g, const steppe::QpWaveResult& qw) {
    std::printf("-- %s [%s] --\n", g.label, tag);
    // status: a well-formed qpWave model returns Ok (a domain outcome would be a VALUE,
    // never a crash; both these REAL models are non-degenerate ⇒ Ok).
    check_eq_int("status==Ok", static_cast<int>(qw.status),
                 static_cast<int>(steppe::Status::Ok));
    // est_rank / f4rank EXACT (the qpWave rank decision — the headline contract).
    check_eq_int("est_rank", qw.est_rank, g.est_rank);
    check_eq_int("f4rank",   qw.f4rank,   g.f4rank);

    // rankdrop table (f4rank DESCENDING). Rows EXACT; dof EXACT; chisq TIGHT; p LOOSE.
    const int n = static_cast<int>(g.rd_f4rank.size());
    check_eq_int("rankdrop rows", static_cast<int>(qw.rankdrop_f4rank.size()), n);
    for (int k = 0; k < n && static_cast<std::size_t>(k) < qw.rankdrop_f4rank.size(); ++k) {
        const std::size_t sk = static_cast<std::size_t>(k);
        char nm[56];
        std::snprintf(nm, sizeof(nm), "rd[%d].f4rank", k);
        check_eq_int(nm, qw.rankdrop_f4rank.at(sk), g.rd_f4rank[sk]);
        std::snprintf(nm, sizeof(nm), "rd[%d].dof", k);
        check_eq_int(nm, qw.rankdrop_dof.at(sk), g.rd_dof[sk]);
        std::snprintf(nm, sizeof(nm), "rd[%d].chisq", k);
        check_close(nm, qw.rankdrop_chisq.at(sk), g.rd_chisq[sk], 1e-6, 1e-9);
        std::snprintf(nm, sizeof(nm), "rd[%d].p", k);
        check_close(nm, qw.rankdrop_p.at(sk), g.rd_p[sk], 1e-3, 1e-9);
    }

    // nested diff: M1 has a populated row-0 (rank1 vs rank0); the LAST row is always NA
    // (dofdiff == INT_MIN sentinel; chisqdiff/p_nested == NaN). M2 is a single row ⇒ that
    // one row is the NA last row.
    if (g.has_nested) {
        check_eq_int("rd[0].dofdiff", qw.rankdrop_dofdiff.at(0), g.rd0_dofdiff);
        check_close("rd[0].chisqdiff", qw.rankdrop_chisqdiff.at(0), g.rd0_chisqdiff, 1e-6, 1e-9);
        check_close("rd[0].p_nested",  qw.rankdrop_p_nested.at(0),  g.rd0_p_nested,  1e-3, 1e-9);
    }
    // last row NA (no nested-rank comparison for the lowest rank).
    const std::size_t last = static_cast<std::size_t>(n - 1);
    const bool last_na = (qw.rankdrop_dofdiff.at(last) == INT_MIN) &&
                         std::isnan(qw.rankdrop_chisqdiff.at(last)) &&
                         std::isnan(qw.rankdrop_p_nested.at(last));
    std::printf("  [%s] %-30s (dofdiff=%d chisqdiff/p_nested=NaN)\n",
                last_na ? "PASS" : "FAIL", "rd[last] is NA",
                qw.rankdrop_dofdiff.at(last));
    if (!last_na) ++g_failures;

    // per-rank sweep (ASCENDING r): the rank_chisq/rank_dof vectors carry chisq(r)/dof(r)
    // for r = 0..rmax. They are the rankdrop rows in REVERSE order. Cross-check r=0.
    check_eq_int("per-rank sweep len", static_cast<int>(qw.rank_chisq.size()), n);
    if (static_cast<int>(qw.rank_chisq.size()) == n && n >= 1) {
        // rank_chisq[0] == the rd row with f4rank==0 (the last rankdrop row).
        check_close("rank_chisq[r=0]", qw.rank_chisq.at(0), g.rd_chisq.at(last), 1e-6, 1e-9);
        check_eq_int("rank_dof[r=0]", qw.rank_dof.at(0), g.rd_dof.at(last));
    }
    // No NaN leak in the chisq sweep (a real domain result, not a sentinel).
    bool clean = true;
    for (double c : qw.rank_chisq) if (!std::isfinite(c)) clean = false;
    check_true("rank_chisq all finite (no NaN leak)", clean);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string golden_dir = (argc > 1) ? argv[1]
                                              : "tests/reference/goldens/at2";
    std::printf("=== F4 qpWave parity test (run_qpwave, REAL AADR, both backends) ===\n");
    std::printf("golden dir: %s\n", golden_dir.c_str());

    const bool g_thorough = [] {
        const char* e = std::getenv("STEPPE_THOROUGH");
        return e && e[0] == '1';
    }();
    int gpu_count = 0;
    try {
        gpu_count = steppe::device::visible_device_count();
    } catch (const std::exception& e) {
        std::printf("  [INFO] visible_device_count threw at probe: %s\n", e.what());
        gpu_count = 0;
    }
    std::printf("MODE: %s ; GPUs visible: %d\n",
                g_thorough ? "THOROUGH" : "FAST (GPU deliverable)", gpu_count);

    // §12 metadata gate (the golden refuses to run without the required keys).
    {
        const std::string path = golden_dir + "/golden_qpwave.json";
        std::ifstream f(path);
        if (!f) { std::printf("RESULT: FAIL (cannot open %s)\n", path.c_str()); return 1; }
        const std::string js((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const char* keys[] = {"\"R\"", "\"admixtools\"", "\"blgsize\"",
                              "\"maxmiss\"", "\"boot\"", "\"fudge\"", "\"alpha\""};
        bool ok = true;
        for (const char* k : keys)
            if (js.find(k) == std::string::npos) {
                std::printf("  [FAIL] §12 metadata key missing: %s\n", k); ok = false;
            }
        if (!ok) { std::printf("RESULT: FAIL (metadata gate)\n"); return 1; }
        std::printf("  [PASS] §12 metadata gate (R, admixtools, blgsize, maxmiss, boot, fudge, alpha)\n");
    }

    steppe::F2BlockTensor f2;
    if (!read_fixture(golden_dir + "/fixtures/f2_fit0_9pop.bin", f2)) {
        std::printf("RESULT: FAIL (fixture read)\n");
        return 1;
    }

    const steppe::QpAdmOptions opts;  // fudge=1e-4, rank_alpha=0.05 (the golden's alpha)
    const std::vector<QpWaveGolden> gs = goldens();

    // ---- CpuBackend ORACLE: under STEPPE_THOROUGH OR when no GPU is visible (the
    //      CI-without-GPU acceptance gate; mirrors test_qpadm_parity.cu Block A). ----
    if (g_thorough || gpu_count <= 0) {
        std::printf("\n=== CpuBackend ORACLE (run_qpwave on host f2) ===\n");
        steppe::device::Resources resources;
        steppe::device::PerGpuResources cpu;
        cpu.device_id = -1;
        cpu.backend = steppe::device::make_cpu_backend();
        resources.gpus.push_back(std::move(cpu));
        for (const QpWaveGolden& g : gs) {
            const steppe::QpWaveResult qw =
                steppe::run_qpwave(f2, std::span<const int>(g.left),
                                   std::span<const int>(g.right), opts, resources);
            gate_qpwave("CpuBackend ORACLE", g, qw);
        }
    } else {
        std::printf("\n[INFO] CpuBackend oracle block skipped (FAST mode, GPU present); "
                    "set STEPPE_THOROUGH=1 to run the oracle.\n");
    }

    // ---- CudaBackend DELIVERABLE: the production GPU path, f2 RESIDENT in VRAM (the
    //      default FAST gate). SKIP cleanly if no CUDA device is visible. ----
    if (gpu_count <= 0) {
        std::printf("\n[SKIP] no CUDA device visible — CpuBackend oracle alone gates "
                    "(CI-without-GPU degrades cleanly)\n");
    } else {
        std::printf("\n=== CudaBackend DELIVERABLE (run_qpwave, f2 RESIDENT in VRAM) ===\n");
        steppe::device::Resources gpu_res;
        steppe::device::PerGpuResources g0;
        g0.device_id = 0;
        g0.backend = steppe::device::make_cuda_backend(0);
        g0.caps = g0.backend->capabilities();
        gpu_res.gpus.push_back(std::move(g0));

        const steppe::BackendCapabilities& caps = gpu_res.gpus.at(0).caps;
        check_true("gpu caps.compute_major>=1", caps.compute_major >= 1);
        std::printf("  GPU compute capability sm_%d%d, device_count=%d\n",
                    caps.compute_major, caps.compute_minor, caps.device_count);

        // Upload the REAL f2 fixture to VRAM (f2 RESIDENT; the fit reads block_sizes, not
        // vpair, so size vpair to match with zeros exactly as test_qpadm_parity.cu does).
        steppe::F2BlockTensor f2_up = f2;
        f2_up.vpair.assign(f2.f2.size(), 0.0);
        steppe::device::DeviceF2Blocks dev_f2 =
            steppe::device::upload_f2_blocks_to_device(f2_up, 0);
        check_true("f2 resident (f2_device != null)", dev_f2.f2_device() != nullptr);
        check_true("f2 resident !empty", !dev_f2.empty());

        for (const QpWaveGolden& g : gs) {
            // run_qpwave on the RESIDENT DeviceF2Blocks = the production GPU entry.
            const steppe::QpWaveResult qw =
                steppe::run_qpwave(dev_f2, std::span<const int>(g.left),
                                   std::span<const int>(g.right), opts, gpu_res);
            gate_qpwave("CudaBackend (GPU, f2 RESIDENT)", g, qw);

            // GPU == CpuBackend ORACLE localizer (RELATIVE rtol 1e-9): any GPU qpWave
            // regression localizes against the native reference, not only the golden.
            // THOROUGH-only (re-runs the oracle here so the oracle `res` need not be
            // hoisted). The delta is the cuSOLVER/emulated-SYRK vs native/Jacobi SVD-seed
            // difference, which is RELATIVE (a few ulps of the singular values through the
            // rank test); on the corrected convertf-PA golden_fit0/qpwave fixture the
            // rank-0 chisq is ~1474/~1402 (vs the old corrupt small value), so an ABSOLUTE
            // 1e-9 floor was magnitude-specific — it fires at ~9e-9/~3e-8 absolute even
            // though the relative delta is ~2e-11. The relative tier scales with the chisq
            // and stays ~50000x tighter than the golden gate (rtol 1e-6); the GOLDEN
            // comparison (gate_qpwave above, rtol 1e-6) is unchanged and still PASSES.
            if (g_thorough) {
                steppe::device::Resources cpu_res;
                steppe::device::PerGpuResources cpu;
                cpu.device_id = -1;
                cpu.backend = steppe::device::make_cpu_backend();
                cpu_res.gpus.push_back(std::move(cpu));
                const steppe::QpWaveResult ow =
                    steppe::run_qpwave(f2, std::span<const int>(g.left),
                                       std::span<const int>(g.right), opts, cpu_res);
                std::printf("  -- GPU vs CpuBackend ORACLE localizer (rtol 1e-9) [%s] --\n", g.label);
                check_eq_int("gpu-vs-cpu f4rank", qw.f4rank, ow.f4rank);
                for (std::size_t k = 0; k < qw.rankdrop_chisq.size() &&
                                         k < ow.rankdrop_chisq.size(); ++k) {
                    char nm[48];
                    std::snprintf(nm, sizeof(nm), "gpu-vs-cpu rd[%zu].chisq", k);
                    check_close(nm, qw.rankdrop_chisq.at(k), ow.rankdrop_chisq.at(k), 1e-9, 1e-9);
                }
            }
        }
    }

    std::printf("\n%s\n", g_failures == 0 ? "RESULT: PASS" : "RESULT: FAIL");
    return g_failures == 0 ? 0 : 1;
}
