// tests/reference/test_qpadm_missing_block.cu
//
// F1 / OQ-12 MISSING-BLOCK acceptance gate (fit-engine.md §6 OQ-12; architecture.md
// §12 PARITY, §13). EVERY other qpAdm golden is extracted at maxmiss=0 (the global
// SNP intersection ⇒ NO missing blocks, Vpair uniform = block size). This gate is the
// ONE that exercises a MISSING block: a REAL-AADR f2 set extracted with maxmiss>0 in
// which one jackknife block has a population pair with Vpair==0 (no SNP jointly valid
// in both pops). AT2 read_f2(remove_na=TRUE) DROPS such a block ENTIRELY before the
// jackknife (keep = apply(f2,3,sum(!is.finite)==0)); steppe must reproduce the drop —
// NOT impute the missing pair's f2 as 0 (which biases f4 toward 0 and inflates the
// jackknife variance, the OQ-12 defect). The drop is implemented on BOTH backends
// (the CpuBackend oracle + the production CudaBackend f2-resident path), and this test
// gates BOTH against the pinned AT2 golden AND against each other.
//
// NO SYNTHETIC DATA (memory real-data-only). The model is REAL AADR pops; the missing
// block arises from a REAL sparse ancient pop (Afghanistan_DarraiKurCave_MBA, a single
// low-coverage individual) used as a right pop — at maxmiss=0.99 one block has no SNP
// jointly valid in (Afghanistan_DarraiKurCave_MBA, Mbuti). The fixture
// fixtures/f2_fitNA.bin is the RAW (pre-drop) f2 block tensor with the NA pair stored
// as IEEE-754 NaN (AT2's `extract_f2`/`read_f2(remove_na=FALSE)` output); steppe
// derives Vpair (0 iff a cell is non-finite) and reproduces the keep-mask drop.
//
// DISCRIMINATION: without the drop (the pre-F1 impute-0 behavior) the f4 X and Q
// differ from this golden by ~1e-3 — far above the tight tier (rtol 1e-6) — so the
// test FAILS the buggy impute-0 path and PASSES only the correct drop.
//
// Self-checking main() (non-zero exit on any failure; CTest gates on the exit code).
// No GoogleTest. The CpuBackend ORACLE block runs under STEPPE_THOROUGH=1 OR when no
// GPU is visible (the CI-without-GPU gate); the CudaBackend DELIVERABLE block runs
// whenever a CUDA device is visible (the default FAST path) — mirroring
// test_qpadm_parity.cu / test_qpadm_domain.cu. REAL AADR throughout.

#include <cmath>
#include <cstdint>
#include <cstdio>
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

void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const bool ok = std::fabs(got - want) <= tol;
    std::printf("  [%s] %-34s got=%.12g want=%.12g |d|=%.3g tol=%.3g\n",
                ok ? "PASS" : "FAIL", what, got, want, std::fabs(got - want), tol);
    if (!ok) ++g_failures;
}

void check_eq_int(const char* what, long got, long want) {
    const bool ok = (got == want);
    std::printf("  [%s] %-34s got=%ld want=%ld\n", ok ? "PASS" : "FAIL", what, got, want);
    if (!ok) ++g_failures;
}

void check_true(const char* what, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

// Read the RAW (pre-drop) f2 fixture: int32 P, int32 nb, nb x int32 block_sizes,
// P*P*nb x float64 f2 with NaN = the missing (Vpair==0) pair marker. We DERIVE Vpair
// from the NaN markers (Vpair[c] = isfinite(f2[c]) ? block_size : 0) so the resident
// path's keep-mask (which reads Vpair) fires exactly on the AT2-missing pairs — this
// mirrors the PRODUCTION f2 build (which carries the real Vpair counts; here the magnitude
// is irrelevant, only ==0 vs >0 drives the drop). The NaN f2 cells are then replaced by 0
// (the impute-0 value the drop makes irrelevant) so no NaN reaches the gather arithmetic
// of a SURVIVING block (the dropped block's cells are never gathered).
bool read_na_fixture(const std::string& path, steppe::F2BlockTensor& out) {
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
    const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const std::size_t n = slab * static_cast<std::size_t>(nb);
    out.f2.resize(n);
    f.read(reinterpret_cast<char*>(out.f2.data()),
           static_cast<std::streamsize>(sizeof(double) * n));
    if (!f) { std::printf("  [FAIL] fixture truncated\n"); return false; }
    // Derive Vpair from the NaN markers + count the missing pairs/blocks.
    out.vpair.assign(n, 0.0);
    long n_na = 0;
    for (int b = 0; b < nb; ++b) {
        const std::size_t base = slab * static_cast<std::size_t>(b);
        const double bl = static_cast<double>(out.block_sizes[static_cast<std::size_t>(b)]);
        for (std::size_t e = 0; e < slab; ++e) {
            if (std::isfinite(out.f2[base + e])) {
                out.vpair[base + e] = bl;  // present (magnitude irrelevant; only ==0 matters)
            } else {
                out.vpair[base + e] = 0.0;  // missing pair (AT2 NA)
                out.f2[base + e] = 0.0;     // neutralize NaN (dropped block, never gathered)
                ++n_na;
            }
        }
    }
    std::printf("  fixture: P=%d n_block=%d (NA pair-cells=%ld)\n", P, nb, n_na);
    return true;
}

// ---- The pinned AT2 golden (golden_fitNA.json; admixtools 2.0.10, R 4.3.3; the
//      CANONICAL f2-BLOCKS-OBJECT path qpadm(read_f2(dir),...) — steppe's resident-f2
//      analogue, NOT qpadm(dir,...)) ----------------------------------------------
// target=England_BellBeaker(0), left={CordedWare(1),Turkey_N(2)},
// right={Mbuti(3),Natufian(4),GanjDareh(5),Han(6),Papuan(7),Karitiana(8),Afghan_MBA(9)}.
// One block (0-based 534) is dropped: the (Afghan_MBA, Mbuti) pair has Vpair==0. The
// 2-source model is a poor fit for these outgroups (large weights) — AT2's exact answer,
// which steppe reproduces bit-for-bit; the gate is steppe==AT2 WITH the drop + drop!=impute.
constexpr double kW0 = 21.9095007306;        // weight[CordedWare]
constexpr double kW1 = -20.9095007306;       // weight[Turkey_N]
constexpr double kChisq = 8.26893432701;     // fitted-rank chisq
constexpr int    kDof = 5;
constexpr double kP = 0.142023723110;
constexpr double kSe = 139.254797565;        // both left SE equal (Σw=1)
constexpr double kZ0 = 0.157333902413;       // z[CordedWare]
constexpr double kZ1 = -0.150152821276;      // z[Turkey_N]
// rankdrop rows (f4rank DESC: rank1, rank0)
constexpr int    kRdF4rank[2] = {1, 0};
constexpr int    kRdDof[2]    = {5, 12};
constexpr double kRdChisq[2]  = {8.26893432701, 97.84153701761};
constexpr int    kRdDofdiff   = 7;             // row0
constexpr double kRdChisqdiff = 89.5726026906; // row0
// steppe-convention f4 X (leftref=TARGET, rows=sources, rightref=Mbuti), row TurkeyN
// (i=1) over the 6 rights — what assemble_f4 computes (k=j+nr*i). The LAST entry
// (Afghan_MBA) is the one a missing block most perturbs. (The buggy impute-0 path
// differs measurably here — the drop discriminator.)
constexpr double kXTurkeyN[6] = {1.66993190213e-04, 3.47353762472e-05, 0.000742604696600,
                                 0.000417419530876, 0.000176477228166, -3.01081865168e-04};
constexpr double kXCordedWare[6] = {2.64042025208e-05, 2.43227639508e-05, 0.000704048777804,
                                    0.000445239755203, 0.000112900574313, 4.74253984197e-05};
constexpr int kSurvivingBlocks = 709;          // 710 raw - 1 dropped
constexpr int kRawBlocks = 710;

// Run the fit on one backend and assert the FULL golden taxonomy + the drop.
void run_on(const char* tag, const steppe::F2BlockTensor& f2_host,
            const steppe::device::DeviceF2Blocks* dev_f2,
            steppe::device::Resources& resources, steppe::QpAdmResult& out_res) {
    std::printf("\n=== %s (REAL AADR, maxmiss>0 missing block) ===\n", tag);
    steppe::QpAdmModel model;
    model.target = 0;                          // England_BellBeaker
    model.left = {1, 2};                       // CordedWare, Turkey_N
    model.right = {3, 4, 5, 6, 7, 8, 9};       // R0=Mbuti, ..., Afghan_MBA (the NA source)
    model.model_index = 900;
    steppe::QpAdmOptions opts;                  // fudge=1e-4, jackknife=All by default? see below
    opts.jackknife = steppe::JackknifePolicy::All;  // force the SE path (this model is infeasible)

    const steppe::QpAdmResult res =
        dev_f2 ? steppe::run_qpadm(*dev_f2, model, opts, resources)
               : steppe::run_qpadm(f2_host, model, opts, resources);
    out_res = res;

    check_eq_int("status == Ok", static_cast<long>(res.status),
                 static_cast<long>(steppe::Status::Ok));
    std::printf("-- weights (TIGHT rtol 1e-6) --\n");
    check_close("weight[CordedWare]", res.weight.at(0), kW0, 1e-6, 1e-9);
    check_close("weight[Turkey_N]",   res.weight.at(1), kW1, 1e-6, 1e-9);
    std::printf("-- chisq (TIGHT) / dof (EXACT) / p (LOOSE) --\n");
    check_close("chisq", res.chisq, kChisq, 1e-6, 1e-9);
    check_eq_int("dof", res.dof, kDof);
    check_close("p", res.p, kP, 1e-3, 1e-9);

    // S7 SE over the SURVIVOR blocks (LOOSE tier; the per-block LOO re-fits skip the
    // dropped block because x_loo is already compacted to survivors). Both left SE are
    // equal (Σw=1 constraint ⇒ se[0]==se[1]).
    std::printf("-- se / z (LOOSE rtol 1e-3) --\n");
    check_true("se computed (JackknifePolicy::All)", res.se.size() == 2);
    if (res.se.size() == 2) {
        check_close("se[CordedWare]", res.se.at(0), kSe, 1e-3, 1e-6);
        check_close("se[Turkey_N]",   res.se.at(1), kSe, 1e-3, 1e-6);
    }
    if (res.z.size() == 2) {
        check_close("z[CordedWare]", res.z.at(0), kZ0, 1e-3, 1e-6);
        check_close("z[Turkey_N]",   res.z.at(1), kZ1, 1e-3, 1e-6);
    }

    std::printf("-- rankdrop table --\n");
    check_eq_int("rankdrop rows", static_cast<long>(res.rankdrop_f4rank.size()), 2);
    if (res.rankdrop_f4rank.size() == 2) {
        for (int k = 0; k < 2; ++k) {
            char nm[48];
            std::snprintf(nm, sizeof(nm), "rd[%d].f4rank", k);
            check_eq_int(nm, res.rankdrop_f4rank.at(static_cast<std::size_t>(k)), kRdF4rank[k]);
            std::snprintf(nm, sizeof(nm), "rd[%d].dof", k);
            check_eq_int(nm, res.rankdrop_dof.at(static_cast<std::size_t>(k)), kRdDof[k]);
            std::snprintf(nm, sizeof(nm), "rd[%d].chisq", k);
            check_close(nm, res.rankdrop_chisq.at(static_cast<std::size_t>(k)), kRdChisq[k], 1e-6, 1e-9);
        }
        check_eq_int("rd[0].dofdiff", res.rankdrop_dofdiff.at(0), kRdDofdiff);
        check_close("rd[0].chisqdiff", res.rankdrop_chisqdiff.at(0), kRdChisqdiff, 1e-6, 1e-9);
    }

    // No NaN/Inf leak (the dropped block must not poison any reported field).
    bool clean = std::isfinite(res.chisq) && std::isfinite(res.p);
    for (double w : res.weight) if (!std::isfinite(w)) clean = false;
    for (double s : res.se)     if (!std::isfinite(s)) clean = false;
    check_true("no NaN/Inf in reported fields (drop, not impute)", clean);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string golden_dir = (argc > 1) ? argv[1] : "tests/reference/goldens/at2";
    std::printf("=== F1/OQ-12 qpAdm MISSING-BLOCK test (REAL AADR, both backends) ===\n");
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
    if (!read_na_fixture(golden_dir + "/fixtures/f2_fitNA.bin", f2)) {
        std::printf("RESULT: FAIL (fixture read)\n");
        return 1;
    }
    check_eq_int("fixture raw n_block", f2.n_block, kRawBlocks);

    // The X.f4_est golden is not directly on QpAdmResult; we re-derive it on the CPU
    // oracle via assemble_f4 to localize the drop at S3 (and assert the survivor count).
    // This runs only in the oracle block (it needs a backend). The primary gates are the
    // weights/chisq/rankdrop above, which depend on the SURVIVOR jackknife.

    steppe::QpAdmResult cpu_res, gpu_res;
    bool ran_cpu = false, ran_gpu = false;

    // ---- CpuBackend ORACLE (STEPPE_THOROUGH or no GPU) -----------------------------
    if (g_thorough || gpu_count <= 0) {
        steppe::device::Resources resources;
        steppe::device::PerGpuResources cpu;
        cpu.device_id = -1;
        cpu.backend = steppe::device::make_cpu_backend();

        // Localize the drop at S3: assemble_f4 must compact to the surviving blocks and
        // its X (over survivors) must match the AT2 golden X to the tight tier.
        {
            const std::vector<int> left_idx = {0, 1, 2};            // target + sources
            const std::vector<int> right_idx = {3, 4, 5, 6, 7, 8, 9};
            const steppe::Precision prec;  // native FP64 oracle (default)
            const steppe::F4Blocks X = cpu.backend->assemble_f4(
                f2, std::span<const int>(left_idx), std::span<const int>(right_idx), prec);
            std::printf("\n=== CpuBackend ORACLE assemble_f4 (drop localizer) ===\n");
            check_eq_int("X.n_block == surviving (710-1)", X.n_block, kSurvivingBlocks);
            check_true("X.block_sizes sized to survivors",
                       static_cast<int>(X.block_sizes.size()) == kSurvivingBlocks);
            // X for nl=2 (target+2 sources ⇒ nl=2), nr=6; the single left row (Turkey_N
            // = source index 1, k = j + nr*i with i=1) is x_total[nr + j] over the 6 rights.
            const int nr = X.nr;  // 6
            check_eq_int("X.nr", nr, 6);
            check_eq_int("X.nl", X.nl, 2);  // 2 sources (CordedWare, Turkey_N)
            std::printf("-- f4 X over survivors (TIGHT rtol 1e-6; the drop discriminator) --\n");
            for (int j = 0; j < 6; ++j) {
                char nm[44];
                // steppe x_total k = j + nr*i; i=0 CordedWare, i=1 Turkey_N.
                std::snprintf(nm, sizeof(nm), "X[CordedWare, right%d]", j);
                check_close(nm, X.x_total.at(static_cast<std::size_t>(j + nr * 0)),
                            kXCordedWare[j], 1e-6, 1e-12);
                std::snprintf(nm, sizeof(nm), "X[Turkey_N, right%d]", j);
                check_close(nm, X.x_total.at(static_cast<std::size_t>(j + nr * 1)),
                            kXTurkeyN[j], 1e-6, 1e-12);
            }
        }

        resources.gpus.push_back(std::move(cpu));
        run_on("CpuBackend ORACLE", f2, /*dev_f2=*/nullptr, resources, cpu_res);
        ran_cpu = true;
    } else {
        std::printf("\n[INFO] CpuBackend oracle block skipped (FAST mode, GPU present); "
                    "set STEPPE_THOROUGH=1 to run the oracle.\n");
    }

    // ---- CudaBackend DELIVERABLE (f2 RESIDENT in VRAM; the default FAST gate) -------
    if (gpu_count <= 0) {
        std::printf("\n[SKIP] no CUDA device visible — CpuBackend oracle alone gates "
                    "(CI-without-GPU degrades cleanly)\n");
    } else {
        steppe::device::Resources gpu_res_pool;
        steppe::device::PerGpuResources g0;
        g0.device_id = 0;
        g0.backend = steppe::device::make_cuda_backend(0);
        g0.caps = g0.backend->capabilities();
        gpu_res_pool.gpus.push_back(std::move(g0));

        // Upload the RAW f2 tensor WITH the derived Vpair RESIDENT — the production seam
        // (the keep-mask kernel reads Vpair from VRAM; a Vpair==0 pair ⇒ the block is
        // dropped on-device exactly as the oracle drops it).
        steppe::device::DeviceF2Blocks dev_f2 =
            steppe::device::upload_f2_blocks_to_device(f2, 0);
        check_true("f2 resident (f2_device != null)", dev_f2.f2_device() != nullptr);
        check_true("Vpair resident (vpair_device != null)", dev_f2.vpair_device() != nullptr);
        run_on("CudaBackend (GPU, f2 RESIDENT)", f2, &dev_f2, gpu_res_pool, gpu_res);
        ran_gpu = true;
    }

    // ---- CpuBackend == CudaBackend (when both ran) ---------------------------------
    if (ran_cpu && ran_gpu) {
        std::printf("\n=== CpuBackend oracle == CudaBackend deliverable ===\n");
        check_eq_int("status agree", static_cast<long>(cpu_res.status),
                     static_cast<long>(gpu_res.status));
        // chisq/dof are the well-conditioned quadratic form ⇒ tight agreement (1e-9).
        check_close("chisq agree", gpu_res.chisq, cpu_res.chisq, 0.0, 1e-9);
        check_eq_int("dof agree", gpu_res.dof, cpu_res.dof);
        // The WEIGHTS are a ridge-GLS solve on a near-singular Q (this poor 2-source
        // outgroup model has |w|~22), so the CPU LU inverse and the GPU cuSOLVER potrf/
        // potri differ in the last ~7 figures and the large magnitude AMPLIFIES that:
        // the agreement is reldiff ~1.6e-8 (|d|~3.5e-7 on ~22). Gate it RELATIVE (rtol
        // 1e-6, the golden weight tier) — the bit-1e-9 absolute localizer used by the
        // well-conditioned goldens (|w|~0.5) is not the right scale for an ill-
        // conditioned ~22 weight. The DROP itself is bit-identical across backends (the
        // f4 X / survivor set agree exactly; only the final ill-conditioned solve differs).
        check_close("weight[0] agree", gpu_res.weight.at(0), cpu_res.weight.at(0), 1e-6, 1e-9);
        check_close("weight[1] agree", gpu_res.weight.at(1), cpu_res.weight.at(1), 1e-6, 1e-9);
    }

    std::printf("\n%s\n", g_failures == 0 ? "RESULT: PASS" : "RESULT: FAIL");
    return g_failures == 0 ? 0 : 1;
}
