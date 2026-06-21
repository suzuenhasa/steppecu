// tests/reference/test_qpadm_domain.cu
//
// M(fit-5) DOMAIN-OUTCOME acceptance gate (fit-engine.md §2 M(fit-5):433;
// architecture.md §13:803; §18 DoD:939). The spec mandates that a deliberately
// degenerate model returns a per-model DOMAIN-OUTCOME STATUS VALUE — never a crash,
// never an exception, never a NaN silently reported as Ok — on BOTH the CpuBackend
// ORACLE and the production CudaBackend. Every prior status assertion in
// test_qpadm_parity.cu / test_qpadm_rotation.cu checks `== Status::Ok`; the contract
// requires the NON-Ok outcomes be covered too. This is that gate.
//
// THREE domain outcomes (architecture.md §10 STEPPE_ERR_* taxonomy; error.hpp):
//   (a) RankDeficient     — the GLS weight solve hit a rank-deficient design X (the
//                           model is unidentifiable).
//   (b) NonSpdCovariance  — the covariance Q is not SPD; Cholesky/inverse failed.
//   (c) ChisqUndefined    — dof <= 0; the chi-squared tail-p is undefined (an over-
//                           parameterized model). p is the NaN sentinel BY CONTRACT
//                           (filter on status, not on p) — that is NOT a leak.
//
// NO SYNTHETIC DATA (memory real-data-only). Every degenerate model is built from
// the SAME committed REAL-AADR f2 fixture the parity gate uses
// (goldens/at2/fixtures/f2_fit0_9pop.bin — the 9-pop tensor AT2 read internally;
// pop order index 0=England_BellBeaker 1=Czechia_EBA_CordedWare 2=Turkey_N 3=Mbuti
// 4=Israel_Natufian 5=Iran_GanjDareh_N 6=Han 7=Papuan 8=Karitiana), arranged
// DEGENERATE by selecting real pops in a degenerate configuration — NOT a made-up
// matrix:
//   (a) a left set with a DUPLICATED real source (left = {CordedWare, CordedWare}):
//       the two left rows of X are bit-identical ⇒ the constrained weight solve is
//       singular ⇒ RankDeficient.
//   (b) the SAME duplicated-source model with the fudge ridge REMOVED (opts.fudge=0):
//       the duplicated source makes Q exactly singular (a zero eigenvalue) and with no
//       ridge to regularize it back to SPD, BOTH the CPU LU inverse (an exact-zero
//       pivot) and the GPU Cholesky potrf (not SPD) reject it ⇒ NonSpdCovariance.
//       This is the punch-list (b) case chosen so the two backends AGREE: the
//       EXACTLY-singular (duplicate-row) Q is rejected by LU AND Cholesky alike, so it
//       sidesteps the indefinite-but-nonsingular divergence the punch-list caveats
//       (CPU LU vs GPU Cholesky). With the DEFAULT fudge the same model is
//       RankDeficient (case a) because the ridge restores SPD — both are valid domain
//       outcomes; the fudge=0 variant isolates the non-SPD path.
//   (c) an over-parameterized model (target=England_BellBeaker, left={CordedWare,
//       Turkey_N, Natufian} ⇒ nl=4, right={Iran_GanjDareh_N, Han, Papuan} ⇒ nr=2):
//       at the default rank r=nl-1=3, dof=(nl-r)*(nr-r) <= 0 ⇒ ChisqUndefined. The
//       weights still fit (finite); only the tail-p is the NaN sentinel.
//
// Both backends MUST return the SAME status VALUE for each (the CpuBackend oracle ==
// the CudaBackend deliverable), and no reported numeric field other than the
// contractual ChisqUndefined `p` sentinel may be NaN.
//
// Self-checking main() (returns non-zero on any failure; CTest gates on the exit
// code). No GoogleTest dependency. The CpuBackend (ORACLE) block runs under
// STEPPE_THOROUGH=1 OR when no GPU is visible (the CI-without-GPU acceptance gate),
// mirroring test_qpadm_parity.cu's gate; the CudaBackend (DELIVERABLE) block runs
// whenever a CUDA device is visible (the default FAST path). REAL AADR throughout.

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
#include "steppe/qpadm.hpp"            // run_qpadm + model/result/options

namespace {

int g_failures = 0;

const char* status_name(steppe::Status s) {
    switch (s) {
        case steppe::Status::Ok:               return "Ok";
        case steppe::Status::DeviceOom:        return "DeviceOom";
        case steppe::Status::RankDeficient:    return "RankDeficient";
        case steppe::Status::NonSpdCovariance: return "NonSpdCovariance";
        case steppe::Status::ChisqUndefined:   return "ChisqUndefined";
        case steppe::Status::InvalidConfig:    return "InvalidConfig";
    }
    return "?";
}

void check_status(const char* what, steppe::Status got, steppe::Status want) {
    const bool ok = (got == want);
    std::printf("  [%s] %-40s got=%s want=%s\n", ok ? "PASS" : "FAIL", what,
                status_name(got), status_name(want));
    if (!ok) ++g_failures;
}

void check_true(const char* what, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

// No NaN/Inf may leak into a reported numeric field. `p` is EXEMPT only when the
// status is ChisqUndefined (the documented NaN sentinel; error.hpp / architecture.md
// §10) — every other reported field, and `p` for every other status, must be finite.
void check_no_nan_leak(const char* tag, const steppe::QpAdmResult& r) {
    bool clean = true;
    for (double w : r.weight) if (!std::isfinite(w)) clean = false;
    for (double s : r.se)     if (!std::isfinite(s)) clean = false;
    for (double z : r.z)      if (!std::isfinite(z)) clean = false;
    if (!std::isfinite(r.chisq)) clean = false;
    // p is the NaN sentinel BY CONTRACT for ChisqUndefined (filter on status, not p);
    // for any other status it must be finite.
    if (r.status != steppe::Status::ChisqUndefined && !std::isfinite(r.p)) clean = false;
    char nm[96];
    std::snprintf(nm, sizeof(nm), "%s: no NaN/Inf leak in reported fields", tag);
    check_true(nm, clean);
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

// The three degenerate models + their expected domain outcome, built from REAL pops.
struct DomainCase {
    const char*           label;
    steppe::QpAdmModel    model;
    steppe::QpAdmOptions  opts;
    steppe::Status        want;
};

std::vector<DomainCase> build_cases() {
    std::vector<DomainCase> cases;

    // (a) RankDeficient — DUPLICATED real left source (CordedWare twice). The two
    //     left rows of X are identical ⇒ the constrained weight normal-equations are
    //     singular ⇒ the GLS solve degenerates. Default fudge (the ridge keeps Q SPD,
    //     so the degeneracy surfaces at the weight solve, not the covariance).
    {
        DomainCase c;
        c.label = "(a) collinear left (CordedWare duplicated)";
        c.model.target = 0;            // England_BellBeaker
        c.model.left = {1, 1};         // CordedWare, CordedWare (DUPLICATE real source)
        c.model.right = {3, 4, 5, 6, 7, 8};
        c.model.model_index = 100;
        c.want = steppe::Status::RankDeficient;
        cases.push_back(c);
    }

    // (b) NonSpdCovariance — the SAME duplicated-source model with the fudge ridge
    //     REMOVED (fudge=0). The duplicate makes Q exactly singular (a zero
    //     eigenvalue); with no ridge BOTH the CPU LU inverse (exact-zero pivot) and
    //     the GPU Cholesky potrf (not SPD) reject it ⇒ NonSpdCovariance on both.
    {
        DomainCase c;
        c.label = "(b) non-SPD Q (duplicated source, fudge=0)";
        c.model.target = 0;
        c.model.left = {1, 1};         // duplicate ⇒ singular Q
        c.model.right = {3, 4, 5, 6, 7, 8};
        c.model.model_index = 101;
        c.opts.fudge = 0.0;            // remove the ridge ⇒ Q stays singular ⇒ non-SPD
        c.want = steppe::Status::NonSpdCovariance;
        cases.push_back(c);
    }

    // (c) ChisqUndefined — over-parameterized: nl=4 (target + 3 real sources),
    //     nr=2 (3 real right pops, R0 + 2). At r=nl-1=3, dof=(nl-r)*(nr-r) <= 0 ⇒ the
    //     tail-p is undefined. The weights still fit (finite); only p is the sentinel.
    {
        DomainCase c;
        c.label = "(c) dof<=0 over-parameterized (nl=4, nr=2)";
        c.model.target = 0;            // England_BellBeaker
        c.model.left = {1, 2, 4};      // CordedWare, Turkey_N, Natufian (3 sources ⇒ nl=4)
        c.model.right = {5, 6, 7};     // R0=Iran_GanjDareh_N, Han, Papuan ⇒ nr=2
        c.model.model_index = 102;
        c.want = steppe::Status::ChisqUndefined;
        cases.push_back(c);
    }

    return cases;
}

// Run all cases on one Resources (one backend) and assert the status VALUE + no
// crash + no NaN leak. `tag` labels the backend ("CpuBackend ORACLE" / "CudaBackend").
void run_cases_on(const char* tag, const steppe::F2BlockTensor& f2_host,
                  const steppe::device::DeviceF2Blocks* dev_f2,
                  steppe::device::Resources& resources) {
    std::printf("\n=== %s domain outcomes (REAL AADR, degenerate) ===\n", tag);
    for (const DomainCase& c : build_cases()) {
        std::printf("-- %s --\n", c.label);
        // The fit MUST return a value, never throw/abort: any exception escaping here
        // fails the contract (the harness would crash). We do not catch — a domain
        // outcome is a STATUS, so an exception is itself the failure mode under test.
        const steppe::QpAdmResult res =
            dev_f2 ? steppe::run_qpadm(*dev_f2, c.model, c.opts, resources)
                   : steppe::run_qpadm(f2_host, c.model, c.opts, resources);
        char nm[96];
        std::snprintf(nm, sizeof(nm), "%s status", c.label);
        check_status(nm, res.status, c.want);
        check_no_nan_leak(c.label, res);
        // ChisqUndefined: assert the documented behavior — weights ARE fit (finite,
        // non-empty) and p IS the NaN sentinel (the fit succeeded; only the tail-p is
        // undefined). This pins the F3 contract (dof<=0 ⇒ ChisqUndefined, p=NaN).
        if (c.want == steppe::Status::ChisqUndefined) {
            std::snprintf(nm, sizeof(nm), "%s: dof<=0", c.label);
            check_true(nm, res.dof <= 0);
            std::snprintf(nm, sizeof(nm), "%s: p is NaN sentinel", c.label);
            check_true(nm, std::isnan(res.p));
            bool w_ok = !res.weight.empty();
            for (double w : res.weight) if (!std::isfinite(w)) w_ok = false;
            std::snprintf(nm, sizeof(nm), "%s: weights finite (fit succeeded)", c.label);
            check_true(nm, w_ok);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string golden_dir = (argc > 1) ? argv[1]
                                              : "tests/reference/goldens/at2";
    std::printf("=== M(fit-5) qpAdm DOMAIN-OUTCOME test (REAL AADR, both backends) ===\n");
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

    steppe::F2BlockTensor f2;
    if (!read_fixture(golden_dir + "/fixtures/f2_fit0_9pop.bin", f2)) {
        std::printf("RESULT: FAIL (fixture read)\n");
        return 1;
    }

    // ---- CpuBackend ORACLE: under STEPPE_THOROUGH OR when no GPU is visible (the
    //      CI-without-GPU acceptance gate; mirrors test_qpadm_parity.cu Block A). ----
    if (g_thorough || gpu_count <= 0) {
        steppe::device::Resources resources;
        steppe::device::PerGpuResources cpu;
        cpu.device_id = -1;  // CPU
        cpu.backend = steppe::device::make_cpu_backend();
        resources.gpus.push_back(std::move(cpu));
        run_cases_on("CpuBackend ORACLE", f2, /*dev_f2=*/nullptr, resources);
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
        steppe::device::Resources gpu_res;
        steppe::device::PerGpuResources g0;
        g0.device_id = 0;
        g0.backend = steppe::device::make_cuda_backend(0);
        g0.caps = g0.backend->capabilities();
        gpu_res.gpus.push_back(std::move(g0));

        // Upload the REAL f2 fixture to VRAM as a DeviceF2Blocks (f2 RESIDENT — the
        // production seam; the fit reads block_sizes, not vpair, so size vpair to
        // match with zeros exactly as test_qpadm_parity.cu does).
        steppe::F2BlockTensor f2_up = f2;
        f2_up.vpair.assign(f2.f2.size(), 0.0);
        steppe::device::DeviceF2Blocks dev_f2 =
            steppe::device::upload_f2_blocks_to_device(f2_up, 0);
        check_true("f2 resident (f2_device != null)", dev_f2.f2_device() != nullptr);
        check_true("f2 resident !empty", !dev_f2.empty());
        run_cases_on("CudaBackend (GPU, f2 RESIDENT)", f2, &dev_f2, gpu_res);
    }

    std::printf("\n%s\n", g_failures == 0 ? "RESULT: PASS" : "RESULT: FAIL");
    return g_failures == 0 ? 0 : 1;
}
