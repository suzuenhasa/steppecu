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

#include <cmath>
#include <cstdio>
#include <cstdint>
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

    // =====================================================================
    // CUDA BACKEND — the PRODUCTION GPU PATH (M(fit-4)). Upload the golden f2
    // fixture to VRAM as a DeviceF2Blocks, assert it is RESIDENT + a real GPU is
    // bound, run the fit ON THE GPU (run_qpadm(DeviceF2Blocks) → the f4-gather
    // kernel reading resident f2 → jackknife SYRK/cuSOLVER → on-device SVD/ALS/
    // weight → batched LOO), and assert the GPU result matches the golden AND the
    // CpuBackend oracle (localization). SKIP cleanly if no CUDA device is visible.
    // =====================================================================
    std::printf("\n=== CUDA backend (GPU path, f2 RESIDENT) ===\n");
    int gpu_count = 0;
    try {
        gpu_count = steppe::device::visible_device_count();
    } catch (const std::exception& e) {
        std::printf("  [SKIP] visible_device_count threw: %s\n", e.what());
        gpu_count = 0;
    }
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
            // the emulated-SYRK GPU path is diffed against.
            std::printf("-- GPU(emulated-SYRK) vs CpuBackend oracle (NATIVE) (diff localizer, 1e-9) --\n");
            check_close("gpu-vs-cpu weight[0]", gpu.weight.at(0), res.weight.at(0), 0.0, 1e-9);
            check_close("gpu-vs-cpu weight[1]", gpu.weight.at(1), res.weight.at(1), 0.0, 1e-9);
            check_close("gpu-vs-cpu chisq",     gpu.chisq,        res.chisq,        0.0, 1e-9);

            // REAL per-quantity deltas of the (now emulated-by-default) SYRK path —
            // (a) vs the af6a8c2 golden, (b) vs the CpuBackend native oracle. Reported
            // (the gating is the check_close tiers above); this is the first
            // informative emulated measurement in the fit.
            std::printf("-- [INFO] emulated-SYRK per-quantity deltas (vs golden / vs native oracle) --\n");
            report_delta("emuSYRK weight[CordedWare] vs golden", gpu.weight.at(0), g_w0);
            report_delta("emuSYRK weight[Turkey_N]   vs golden", gpu.weight.at(1), g_w1);
            report_delta("emuSYRK chisq              vs golden", gpu.chisq,        g_chisq);
            report_delta("emuSYRK se[CordedWare]     vs golden", gpu.se.at(0),     g_se);
            report_delta("emuSYRK p                  vs golden", gpu.p,            g_p);
            report_delta("emuSYRK weight[0] vs native-oracle",   gpu.weight.at(0), res.weight.at(0));
            report_delta("emuSYRK weight[1] vs native-oracle",   gpu.weight.at(1), res.weight.at(1));
            report_delta("emuSYRK chisq     vs native-oracle",   gpu.chisq,        res.chisq);

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

        // GPU X/Q localizers: call the CUDA backend's assemble_f4(DeviceF2Blocks) +
        // jackknife_cov directly (proving S3/S4 ran on the GPU over resident f2).
        std::printf("\n-- GPU S3/S4 slice cross-check (LOCALIZER, loose rtol 1e-3) --\n");
        const double gX[5] = {0.000204208644854152, 0.000158461166756911,
                              -2.44579443823133e-05, -2.42885897838109e-05,
                              -3.27534454121373e-05};
        const double gQdiag[5] = {4.83261400481559e-09, 4.38937359295631e-09,
                                  2.43374449452477e-09, 3.18630101668274e-09,
                                  4.5704478312779e-09};
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
