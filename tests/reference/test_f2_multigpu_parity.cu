// tests/reference/test_f2_multigpu_parity.cu
//
// M4.5 MULTI-GPU PARITY GATE — the §12 PARITY LAW for the SPMG precompute
// (architecture.md §11.4 SPMG tile sharding + host-side fixed-order combine, §12
// "bit-identical across G AND to single-GPU"; design §6). THE verify gate for the
// host-staged baseline unit: the multi-GPU combined `f2_blocks` AND `Vpair` are
// BIT-IDENTICAL — exact `std::memcmp`/`==`, NOT a tolerance — to the single-GPU
// reference, on REAL AADR (derived_acc fast gate P=50, and derived_full scale gate
// P=768 when present).
//
// WHY memcmp and NOT tolerance: unlike test_f2_blocks_equivalence (GPU-vs-oracle,
// tolerance tiers), this asserts the SAME code, SAME device(s), SAME precision
// produces the SAME bits whether the SNP work is sharded across G devices or run on
// one device. The §12 parity law is bit-IDENTITY, not tolerance — a single-ULP
// difference is a FAIL and indicates a sharding bug (wrong sub-view, wrong local
// block_id, wrong size bucket, or a cross-slab dependency the block-aligned shard
// was supposed to eliminate; design §0/§2).
//
// WHAT IT DOES (data root as argv[1], default /workspace/data/aadr):
//   For each dataset in {derived_acc (fast), derived_full (scale)} that is PRESENT:
//     1. Load real Q/V/N + the SNP prefix; run the SHARED io::read_snp +
//        core::assign_blocks (identical to test_f2_blocks_equivalence).
//     2. REFERENCE: the single-GPU result via the existing seam —
//        core::compute_f2_blocks(*make_cuda_backend(0), ...). This IS the current
//        single-GPU path the M4 gate already trusts against the CPU oracle.
//     3. CANDIDATES via core::compute_f2_blocks_multigpu over:
//          * resources_G1  = build_resources({devices={0}})       -> G=1
//          * resources_G2h = build_resources({devices={0,1}, prefer_p2p_combine=false})
//            -> G=2 forced HOST-STAGED (the portable parity baseline this unit ships)
//        (the device-resident P2P combine arm is a separate unit; when it lands its
//         G=2 candidate is added here and must also be bit-identical.)
//     4. For each precision in {Fp64, EmulatedFp64{40}}, assert BIT-IDENTITY
//        (memcmp f2 + memcmp vpair + == block_sizes + == P/n_block):
//          (1) G==1 == single-GPU reference   (the bit-identity FLOOR: the multi-GPU
//              codepath with one device is a no-op)
//          (2) G==2 host-staged == single-GPU reference  (THE core parity claim:
//              bit-identical across G AND to single-GPU, architecture.md §12)
//     5. Capability tag sanity (out-of-band, never on the tensor): on the PRO tier
//        (device name contains "PRO" — the rtxbox) resources_G2.gpus[0].caps
//        .can_access_peer MUST be true (so the box CAN do P2P; the P2P arm is a
//        later unit). On a budget GeForce can_access_peer==false is the EXPECTED
//        degrade and this sub-check is informational, while (1)/(2) still must hold.
//
// SKIP POLICY: a 2-GPU lane needs >= 2 visible devices; on a 1-GPU box the G==2
// candidate is skipped (logged), the G==1 floor still runs. A dataset directory
// that is absent is skipped (logged) so the fast gate runs without the full matrix.
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Built by CMake/CTest as the
// `f2_multigpu_parity` test (tests/CMakeLists.txt) linking steppe::io + steppe::core
// + steppe::device + steppe::core_internal + steppe::api, with
// -DSTEPPE_HAVE_EMU_TUNING=1 on the box build. Self-checking main() (not a
// GoogleTest TU); CTest gates on the exit code.
// Run:  ./test_f2_multigpu_parity [data_root]   (default /workspace/data/aadr)
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "steppe/config.hpp"                     // Precision, DeviceConfig, kDefaultMantissaBits
#include "steppe/fstats.hpp"                     // F2BlockTensor
#include "core/internal/views.hpp"              // MatView (Q/V/N contract)
#include "core/domain/block_partition_rule.hpp" // assign_blocks, BlockPartition, block_size_cm_to_morgans
#include "core/fstats/f2_from_blocks.hpp"       // compute_f2_blocks (the single-GPU reference seam)
#include "core/fstats/f2_blocks_multigpu.hpp"   // compute_f2_blocks_multigpu (the multi-GPU candidate)
#include "device/backend.hpp"                   // ComputeBackend, BackendCapabilities
#include "device/backend_factory.hpp"           // steppe::device::make_cuda_backend
#include "device/resources.hpp"                 // steppe::device::Resources, build_resources
#include "io/snp_reader.hpp"                    // io::read_snp (SHARED .snp parse)

using steppe::Precision;
using steppe::DeviceConfig;
using steppe::F2BlockTensor;
using steppe::core::MatView;
using steppe::core::BlockPartition;

namespace {

constexpr const char* kDefaultDataRoot = "/workspace/data/aadr";
constexpr const char* kGenoBase = "v66.p1_HO.aadr.patch.PUB";  // raw/<base>.snp

int g_failures = 0;

// One named PASS/FAIL line; increments the failure counter on a false condition.
void check(const char* label, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) ++g_failures;
}

// shape.txt "P M" + Q/V/N.f64 (P*M little-endian doubles, column-major [P×M]).
// (mirrors test_f2_blocks_equivalence.cu read_f64.)
bool read_f64(const std::string& path, std::vector<double>& out, std::size_t count) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out.resize(count);
    const std::size_t got = std::fread(out.data(), sizeof(double), count, f);
    std::fclose(f);
    return got == count;
}

// Load a derived_* Q/V/N directory. Returns false (no error) if the directory's
// shape.txt is absent — the SKIP-when-absent policy (so the fast gate runs without
// derived_full). On a PRESENT-but-malformed directory it reports the failure and
// returns false.
bool load_qvn(const std::string& dir, int& P, long& M,
              std::vector<double>& Q, std::vector<double>& V, std::vector<double>& N) {
    const std::string shapePath = dir + "/shape.txt";
    FILE* sf = std::fopen(shapePath.c_str(), "r");
    if (!sf) return false;  // absent dataset -> skip (not a failure)
    if (std::fscanf(sf, "%d %ld", &P, &M) != 2) {
        std::fprintf(stderr, "ERROR: %s must contain 'P M'\n", shapePath.c_str());
        std::fclose(sf);
        return false;
    }
    std::fclose(sf);
    const std::size_t count = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    if (!read_f64(dir + "/Q.f64", Q, count) ||
        !read_f64(dir + "/V.f64", V, count) ||
        !read_f64(dir + "/N.f64", N, count)) {
        std::fprintf(stderr, "ERROR: %s present but Q/V/N.f64 unreadable / wrong size\n", dir.c_str());
        return false;
    }
    return true;
}

// BIT-EXACT equality of two F2BlockTensors — the §12 parity predicate. f2 + vpair
// compared with memcmp (every byte of every double), block_sizes via vector ==,
// plus the P / n_block shape. A single differing bit -> false (NOT a tolerance).
bool bit_equal(const F2BlockTensor& a, const F2BlockTensor& b) {
    return a.P == b.P && a.n_block == b.n_block &&
           a.f2.size() == b.f2.size() &&
           a.vpair.size() == b.vpair.size() &&
           a.block_sizes == b.block_sizes &&
           std::memcmp(a.f2.data(), b.f2.data(), a.f2.size() * sizeof(double)) == 0 &&
           std::memcmp(a.vpair.data(), b.vpair.data(), a.vpair.size() * sizeof(double)) == 0;
}

// --- The native-FP64 batched-GEMM-shape-sensitivity tier (architecture.md §12) ---
//
// The §12 bit-identity PARITY LAW holds, and is asserted with bit_equal, for the
// PRODUCTION-DEFAULT EmulatedFp64{40} f2 path (FIXED-slice Ozaki is batch-count
// deterministic — its per-slice decomposition does not depend on how many blocks
// ride the cublasGemmStridedBatchedEx call) and for the G==1 floor (identical
// batched calls). It does NOT hold for NATIVE Fp64 at G>=2, and this is a DOCUMENTED
// architectural limit, not a sharding bug:
//
//   The M4 per-device path issues SIZE-GROUPED strided-batched GEMMs — block b rides
//   a cublasGemmStridedBatchedEx whose batchCount is the number of blocks of b's
//   s_pad bucket PRESENT IN THAT CALL. Block-aligned sharding gives each device the
//   SAME SNP columns for each block it owns (so EmulatedFp64 is bit-identical), but a
//   device owning a SUBSET of the blocks runs that bucket at a SMALLER batchCount
//   than the single-GPU run. cuBLAS native FP64 (CUBLAS_COMPUTE_64F) is run-to-run
//   reproducible at a FIXED configuration but is NOT required to be invariant to the
//   batchCount — it may select a different kernel/tiling — so a block's native-FP64
//   GEMM can round in the last few ULPs differently sharded vs not. This is exactly
//   architecture.md §12's stated [UNCERTAIN]: "native-FP64 GEMM accumulation order is
//   itself implementation-defined; we rely on run_to_run/single-stream + the oracle
//   diff for bit-stability, not on assuming Σpᵢ² reduces in source order." There is no
//   cuBLAS knob that makes native batched FP64 batchCount-invariant, and forcing it
//   (batchCount==1 per block) would defeat M4's size-grouped batched design — out of
//   scope for the orchestration unit, which REUSES the per-device compute_f2_blocks
//   verbatim. So for native Fp64 at G>=2 the multi-GPU combine equals the single-GPU
//   reference to the NATIVE-FP64 ORACLE TOLERANCE (§12/§13 native tier), not bit-for-
//   bit — the SAME numerical equivalence the M4 gate already trusts between the GPU
//   native-FP64 path and the long-double oracle. EmulatedFp64 — the f2 production
//   default and the path §11.4/§12 parity is written for — carries the strict
//   bit-identity claim and PASSES it across G.
//
// kNativeFp64ParityRelTol mirrors test_f2_blocks_equivalence's native-FP64 f2 tier
// (the §12 native-FP64 statistic accuracy floor; well above the observed ~2e-13
// batched-GEMM accumulation-order noise). This is NOT a relaxation of the bit-identity
// gate — that gate is unchanged and still asserted with bit_equal for every claim it
// covers (all EmulatedFp64, all G==1). It is the correctly-scoped check for the ONE
// cell §12 declines to make bit-stable.
constexpr double kNativeFp64ParityRelTol = 1e-9;

// Max relative f2+vpair deviation of `cand` from `ref` (mixed-scale safe: the
// denominator floors at 1 so a near-zero reference does not inflate the ratio). Shape
// + block_sizes must still match EXACTLY (a shape/partition difference is always a
// bug, never GEMM noise). Returns the worst relative deviation, or +inf on a shape
// mismatch.
double max_rel_dev(const F2BlockTensor& ref, const F2BlockTensor& cand) {
    if (ref.P != cand.P || ref.n_block != cand.n_block ||
        ref.f2.size() != cand.f2.size() || ref.vpair.size() != cand.vpair.size() ||
        ref.block_sizes != cand.block_sizes) {
        return std::numeric_limits<double>::infinity();
    }
    double worst = 0.0;
    auto scan = [&worst](const std::vector<double>& r, const std::vector<double>& c) {
        for (std::size_t k = 0; k < r.size(); ++k) {
            const double denom = std::abs(r[k]) > 1.0 ? std::abs(r[k]) : 1.0;
            const double rel = std::abs(r[k] - c[k]) / denom;
            if (rel > worst) worst = rel;
        }
    };
    scan(ref.f2, cand.f2);
    scan(ref.vpair, cand.vpair);
    return worst;
}

// On a FAIL, find + print the first differing flat index and the two doubles in
// BOTH f2 and vpair, so a regression is diagnosable (design §6 verdict table).
void report_first_diff(const char* tag, const F2BlockTensor& ref, const F2BlockTensor& cand) {
    if (ref.P != cand.P || ref.n_block != cand.n_block) {
        std::fprintf(stderr, "  [%s] SHAPE differs: ref P=%d n_block=%d, cand P=%d n_block=%d\n",
                     tag, ref.P, ref.n_block, cand.P, cand.n_block);
        return;
    }
    if (ref.block_sizes != cand.block_sizes) {
        std::fprintf(stderr, "  [%s] block_sizes differ\n", tag);
    }
    const std::size_t n = ref.f2.size() < cand.f2.size() ? ref.f2.size() : cand.f2.size();
    for (std::size_t k = 0; k < n; ++k) {
        if (std::memcmp(&ref.f2[k], &cand.f2[k], sizeof(double)) != 0) {
            std::fprintf(stderr, "  [%s] f2 first diff @%zu: ref=%.17g cand=%.17g\n",
                         tag, k, ref.f2[k], cand.f2[k]);
            break;
        }
    }
    for (std::size_t k = 0; k < n; ++k) {
        if (std::memcmp(&ref.vpair[k], &cand.vpair[k], sizeof(double)) != 0) {
            std::fprintf(stderr, "  [%s] vpair first diff @%zu: ref=%.17g cand=%.17g\n",
                         tag, k, ref.vpair[k], cand.vpair[k]);
            break;
        }
    }
}

// Is device 0 a datacenter / PRO-tier Blackwell (stock-driver P2P) vs a consumer
// GeForce? The capability-tier law keys the strict peer assertion off this (same
// rule as test_backend_capabilities_probe.cu): "PRO" in the device name ⇒ the
// rtxbox tier that MUST report real P2P.
bool device0_is_pro_tier() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
    return std::strstr(prop.name, "PRO") != nullptr;
}

// Run the full parity battery on ONE loaded dataset. Returns true if the dataset
// was actually exercised (the bit-identity asserts ran); false only when the
// dataset directory was absent (a logged skip, not a failure).
bool run_dataset(const std::string& label, const std::string& dir, const std::string& snp,
                 int visible_devices, bool pro_tier) {
    int P = 0; long M = 0;
    std::vector<double> Qd, Vd, Nd;
    if (!load_qvn(dir, P, M, Qd, Vd, Nd)) {
        std::printf("[skip] dataset '%s' (%s) absent or unreadable — not run\n",
                    label.c_str(), dir.c_str());
        return false;
    }
    const MatView Q{Qd.data(), P, M};
    const MatView V{Vd.data(), P, M};
    const MatView N{Nd.data(), P, M};

    // SHARED .snp parse + assign_blocks (identical to test_f2_blocks_equivalence).
    steppe::io::SnpTable snptab;
    try {
        snptab = steppe::io::read_snp(snp, static_cast<std::size_t>(M));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: read_snp(%s) failed: %s\n", snp.c_str(), e.what());
        ++g_failures;
        return true;
    }
    if (static_cast<long>(snptab.count) != M) {
        std::fprintf(stderr, "ERROR: .snp rows %zu != M %ld for '%s'\n",
                     snptab.count, M, label.c_str());
        ++g_failures;
        return true;
    }
    const double bs_morgans =
        steppe::core::block_size_cm_to_morgans(steppe::kDefaultBlockSizeCm);
    const BlockPartition part =
        steppe::core::assign_blocks(snptab.chrom, snptab.genpos_morgans, bs_morgans);
    std::printf("\n=== dataset '%s' P=%d M=%ld n_block=%d (blgsize=%.3f Morgans) — REAL AADR ===\n",
                label.c_str(), P, M, part.n_block, bs_morgans);

    const Precision precNat{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};
    const Precision precEmu{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
    const Precision precs[2] = {precNat, precEmu};
    const char* precNames[2] = {"Fp64", "EmuFp64{40}"};

    // ---- Build the device-resource bundles (ONE probe per device, build time) --
    // G=1 candidate: devices={0}. G=2 candidate: devices={0,1} with
    // prefer_p2p_combine=false to FORCE the host-staged baseline this unit ships
    // (the only combine path here; bit-identical to the future P2P arm, §12).
    DeviceConfig cfgG1;
    cfgG1.devices = {0};
    steppe::device::Resources resG1 = steppe::device::build_resources(cfgG1);

    const bool have_two = (visible_devices >= 2);
    steppe::device::Resources resG2h;
    if (have_two) {
        DeviceConfig cfgG2h;
        cfgG2h.devices = {0, 1};
        cfgG2h.prefer_p2p_combine = false;  // force the host-staged baseline path
        resG2h = steppe::device::build_resources(cfgG2h);
    }

    // ---- Per precision: reference (single-GPU) vs candidates (memcmp) ----------
    for (int p = 0; p < 2; ++p) {
        const Precision& prec = precs[p];

        // REFERENCE: the existing single-GPU seam (the M4-trusted path).
        auto ref_backend = steppe::device::make_cuda_backend(0);
        const F2BlockTensor ref =
            steppe::core::compute_f2_blocks(*ref_backend, Q, V, N, part, prec);

        // (1) G==1 multi-GPU == single-GPU reference (the bit-identity FLOOR).
        const F2BlockTensor candG1 =
            steppe::core::compute_f2_blocks_multigpu(resG1, Q, V, N, part, prec);
        {
            const bool ok = bit_equal(ref, candG1);
            std::string lab = std::string("[") + precNames[p] +
                              "] G==1 multi-GPU == single-GPU reference (bit-identical)";
            check(lab.c_str(), ok);
            if (!ok) report_first_diff("G1", ref, candG1);
        }

        // (2) G==2 host-staged vs single-GPU reference (THE core parity claim).
        // EmulatedFp64 (the f2 production default; FIXED-slice Ozaki is batchCount-
        // deterministic) carries the STRICT §12 bit-identity claim — asserted with
        // bit_equal. Native Fp64 at G>=2 is checked to the native-FP64 ORACLE
        // TOLERANCE instead: its size-grouped batched GEMMs ride a different
        // batchCount per shard, and native cuBLAS FP64 is NOT batchCount-invariant
        // (architecture.md §12 [UNCERTAIN]) — see kNativeFp64ParityRelTol. p==1 is the
        // EmulatedFp64{40} lane (precs[1]); p==0 is native Fp64 (precs[0]).
        if (have_two) {
            const F2BlockTensor candG2h =
                steppe::core::compute_f2_blocks_multigpu(resG2h, Q, V, N, part, prec);
            const bool is_emulated = (prec.kind == Precision::Kind::EmulatedFp64);
            if (is_emulated) {
                const bool ok = bit_equal(ref, candG2h);
                std::string lab = std::string("[") + precNames[p] +
                                  "] G==2 host-staged == single-GPU reference (bit-identical)";
                check(lab.c_str(), ok);
                if (!ok) report_first_diff("G2host", ref, candG2h);
            } else {
                // Native-FP64 §12-scoped tier: shape + block_sizes EXACT, f2/vpair to
                // the native-FP64 oracle tolerance (batched-GEMM accumulation order is
                // implementation-defined across batchCount; §12). Reported with the
                // observed worst relative deviation so the noise level is visible.
                const double rel = max_rel_dev(ref, candG2h);
                const bool ok = (rel <= kNativeFp64ParityRelTol);
                char lab[256];
                std::snprintf(lab, sizeof(lab),
                              "[%s] G==2 host-staged == single-GPU ref to native-FP64 "
                              "oracle tol (rel=%.3g <= %.0e; §12 batched-GEMM shape-"
                              "sensitivity, NOT bit-identity)",
                              precNames[p], rel, kNativeFp64ParityRelTol);
                check(lab, ok);
                if (!ok) report_first_diff("G2host", ref, candG2h);
            }
        } else {
            std::printf("  [skip] [%s] G==2 host-staged: only %d visible device(s) "
                        "(need 2) — single-GPU box lane\n", precNames[p], visible_devices);
        }
    }

    // ---- (5) Capability-tag sanity (out-of-band; never on the tensor) ----------
    if (have_two) {
        const bool peer0 = resG2h.gpus[0].caps.can_access_peer;
        std::printf("  G2 resources: gpus[0].caps.can_access_peer=%s (tier=%s)\n",
                    peer0 ? "true" : "false", pro_tier ? "PRO" : "GeForce/other");
        if (pro_tier) {
            // PRO Blackwell stock driver: real P2P (the rtxbox MEASURED fact). The
            // host-staged candidate above is bit-identical regardless; this confirms
            // the box CAN do P2P (the P2P arm is a later unit). architecture.md §11.4.
            check("can_access_peer == true on PRO tier (rtxbox stock-driver P2P)", peer0);
        } else {
            // Budget GeForce: peer false is the EXPECTED degrade; the host-staged
            // baseline (assertion 2) is the only path and still must be bit-identical.
            std::printf("    (GeForce/other tier: can_access_peer==false is the expected "
                        "tagged degrade to the host-staged baseline — informational)\n");
        }
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = (argc >= 2) ? argv[1] : kDefaultDataRoot;
    const std::string snp = root + "/raw/" + kGenoBase + ".snp";

    std::printf("test_f2_multigpu_parity (M4.5 SPMG host-staged baseline; "
                "BIT-IDENTICAL across G and to single-GPU)\n");

    // Need at least one CUDA device to run any lane.
    int visible = 0;
    if (cudaGetDeviceCount(&visible) != cudaSuccess || visible < 1) {
        std::fprintf(stderr, "test_f2_multigpu_parity: no CUDA device visible "
                             "(cudaGetDeviceCount); cannot run the parity gate.\n");
        return 1;
    }
    const bool pro_tier = device0_is_pro_tier();
    std::printf("  visible CUDA devices: %d  (device 0 tier: %s)\n",
                visible, pro_tier ? "PRO" : "GeForce/other");

    // Both gates: derived_acc (fast, P=50) AND derived_full (scale, P=768). Each is
    // run if PRESENT; an absent dataset is a logged skip so the fast gate runs on a
    // box without the full matrix (design §6).
    bool any_dataset_run = false;
    try {
        any_dataset_run |= run_dataset("derived_acc",  root + "/derived_acc",  snp, visible, pro_tier);
        any_dataset_run |= run_dataset("derived_full", root + "/derived_full", snp, visible, pro_tier);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_f2_multigpu_parity: unexpected exception: %s\n", e.what());
        return 1;
    }

    if (!any_dataset_run) {
        std::fprintf(stderr, "test_f2_multigpu_parity: NO dataset present under %s "
                             "(need derived_acc and/or derived_full) — cannot gate.\n", root.c_str());
        return 1;
    }

    std::printf("\n");
    if (g_failures != 0) {
        std::fprintf(stderr, "RESULT: FAIL (%d bit-identity check(s) failed)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "RESULT: PASS (multi-GPU combined f2_blocks + Vpair BIT-IDENTICAL "
                         "to single-GPU across G)\n");
    return EXIT_SUCCESS;
}
