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
//            -> G=2 forced HOST-STAGED (the portable parity baseline)
//          * resources_G2p = build_resources({devices={0,1}, prefer_p2p_combine=true})
//            -> G=2 device-resident cudaMemcpyPeer combine (the OPT-IN P2P fast-path,
//               exercised when can_access_peer==true — the rtxbox PRO tier). On a box
//               WITHOUT peer access this candidate degrades (tagged) to host-staged
//               and is therefore still bit-identical; the last_combine_path tag
//               records which transport actually ran.
//     4. For each precision in {Fp64, EmulatedFp64{40}}, assert BIT-IDENTITY
//        (memcmp f2 + memcmp vpair + == block_sizes + == P/n_block):
//          (1) G==1 == single-GPU reference   (the bit-identity FLOOR: the multi-GPU
//              codepath with one device is a no-op)
//          (2) G==2 host-staged == single-GPU reference  (THE core parity claim:
//              bit-identical across G AND to single-GPU, architecture.md §12)
//          (3) G==2 P2P == single-GPU reference  (the device-resident cudaMemcpyPeer
//              combine moves the same bytes / sums the same fixed order → identical;
//              §11.4, §12)  [P2P arm — runs when can_access_peer]
//          (4) G==2 P2P == G==2 host-staged  (the two-tier NEUTRALITY: both combine
//              tiers bit-identical to each other, the explicit workflow ask)
//        (claims (2)-(4) carry the strict bit-identity gate for EmulatedFp64{40}, the
//         f2 production default; native Fp64 at G>=2 is the documented batched-GEMM
//         shape-sensitivity cell — checked to the native-FP64 oracle tolerance, see
//         kNativeFp64ParityRelTol. The P2P vs host-staged neutrality (4) IS strictly
//         bit-identical for BOTH precisions: both combine tiers consume the SAME
//         per-device partials and sum the SAME fixed order, so they agree bit-for-bit
//         regardless of the per-device GEMM precision.)
//     5. EmulatedFp64 ENGAGED + capability/which-path tag sanity (out-of-band, never
//        on the tensor):
//          * (F-COV-1) under STEPPE_HAVE_EMU_TUNING the EmulatedFp64 lane MUST be
//            honorable (resG1.gpus[0].caps.emulated_fp64_honorable == true) AND its
//            single-GPU reference MUST DIFFER bit-for-bit from native Fp64 — so the
//            §12 fixed-slice Ozaki path actually engaged and the EmuFp64 bit-identity
//            claims did not silently certify a native fallback.
//          * (L8 capability BICONDITIONAL — NOT a device-name match) for resG2p
//            (prefer_p2p_combine=true): can_access_peer ⇔ last_combine_path ==
//            P2pDeviceResident, and !can_access_peer ⇔ last_combine_path == HostStaged.
//            This is the SAME field the production §4 fork gates on, so it holds on
//            EVERY tier — the PRO 6000 (stock-driver P2P) AND the consumer 5090 (P2P
//            driver-disabled), with no marketing-string coupling.
//          * resG2h.last_combine_path MUST be HostStaged (prefer_p2p_combine=false
//            forces the baseline — the explicit workflow ask, on every tier).
//
// SKIP POLICY: a 2-GPU lane needs >= 2 visible devices; on a 1-GPU box the G==2
// candidate is skipped (logged), the G==1 floor still runs. A dataset directory that
// is ABSENT is skipped (logged); a dataset that is PRESENT-but-MALFORMED is a FAIL
// (F-BUG-2 — never a silent skip on a parity gate). A dataset whose estimated device-0
// peak does NOT fit free VRAM (cudaMemGetInfo) is skipped with an EXPLICIT reason
// ("[skip] <name> P=.. : insufficient VRAM (need ~X MiB, free ~Y MiB)") rather than
// OOMing — so derived_full (P=768, ~36 GB) is cleanly skipped on a 32 GB consumer 5090
// while derived_acc (P=50) runs, and a 96 GB PRO box clears both.
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
#include "device/device_f2_blocks.hpp"          // steppe::device::DeviceF2Blocks (the device-resident primary + opt-in to_host)
#include "device/f2_blocks_out.hpp"             // steppe::device::F2BlocksOut, OutputTier (M5 adaptive tiered result)
#include "device/tier_select.hpp"               // steppe::device::select_output_tier, free_host_ram_bytes (M5 tier policy)
#include "device/vram_budget.hpp"               // resident_tensor_bytes (the §11.2 footprint home — VRAM gate)
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

// Human-readable name of the out-of-band which-path tag (Resources::last_combine_path)
// for the verdict log — confirms WHICH combine transport actually ran (architecture
// .md §11.4 tagged combine; cleanup §(2).2).
const char* combine_path_name(steppe::device::CombinePath p) {
    switch (p) {
        case steppe::device::CombinePath::None:              return "None";
        case steppe::device::CombinePath::HostStaged:        return "HostStaged";
        case steppe::device::CombinePath::P2pDeviceResident: return "P2pDeviceResident";
    }
    return "?";
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

// Outcome of loading a derived_* Q/V/N directory — the F-BUG-2 absent-vs-malformed
// distinction. ABSENT (no shape.txt) is the intended SKIP (so the fast gate runs
// without derived_full); MALFORMED (present but unparseable shape, non-positive P/M,
// or unreadable/wrong-size Q/V/N) is a real ERROR the caller MUST count as a failure —
// NOT a silent skip. A parity gate that advertises a scale dataset must FAIL, never
// quietly pass, if that dataset is present-but-truncated (§2 fail-fast).
enum class LoadResult { Absent, Malformed, Ok };

// Load a derived_* Q/V/N directory. Distinguishes ABSENT (skip) from MALFORMED
// (error) per LoadResult — closing the F-BUG-2 silent-skip-of-the-scale-dataset hole.
// A non-positive P or M is rejected up front (F-BUG-3) so a negative M can never wrap
// `count` into a huge size_t and bad_alloc instead of reporting a clean malformed-shape
// diagnostic.
LoadResult load_qvn(const std::string& dir, int& P, long& M,
                    std::vector<double>& Q, std::vector<double>& V, std::vector<double>& N) {
    const std::string shapePath = dir + "/shape.txt";
    FILE* sf = std::fopen(shapePath.c_str(), "r");
    if (!sf) return LoadResult::Absent;  // absent dataset -> intended skip (not a failure)
    if (std::fscanf(sf, "%d %ld", &P, &M) != 2) {
        std::fprintf(stderr, "ERROR: %s must contain 'P M'\n", shapePath.c_str());
        std::fclose(sf);
        return LoadResult::Malformed;
    }
    std::fclose(sf);
    if (P <= 0 || M <= 0) {  // F-BUG-3: a non-positive shape is malformed, not absent
        std::fprintf(stderr, "ERROR: %s has non-positive shape P=%d M=%ld\n",
                     shapePath.c_str(), P, M);
        return LoadResult::Malformed;
    }
    const std::size_t count = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    if (!read_f64(dir + "/Q.f64", Q, count) ||
        !read_f64(dir + "/V.f64", V, count) ||
        !read_f64(dir + "/N.f64", N, count)) {
        std::fprintf(stderr, "ERROR: %s present but Q/V/N.f64 unreadable / wrong size\n", dir.c_str());
        return LoadResult::Malformed;
    }
    return LoadResult::Ok;
}

// Safety headroom applied to the raw peak estimate before the skip decision. The raw
// term below is the SUM of the documented co-resident DeviceBuffers; the real high-
// water is HIGHER because of (a) the CUDA caching allocator's fragmentation/rounding,
// (b) the up-to-three concurrent idle backend workspaces device 0 holds (resG1,
// resG2h[0], resG2p[0]) while a full-M compute runs, and (c) the per-chunk slabs that,
// although they reuse the freed-raw VRAM, can transiently exceed it. A skip decision
// must err HIGH (skipping a dataset that might just barely fit is correct; an OOM that
// crashes the whole gate is not), so the estimate is inflated by this factor. Tuned so
// derived_full (P=768, raw ~32.0 GiB) is gated OFF a 32 GB 5090 (~31.6 GiB free) while
// derived_acc (P=50, raw ~0.33 GiB) clears trivially and a 96 GB PRO box clears both.
constexpr double kVramHeadroomFactor = 1.15;

// Peak device-0 VRAM (bytes) one dataset's parity battery needs, used to GATE the
// scale dataset against a 32 GB consumer 5090 (cudaMemGetInfo) so derived_full is
// SKIPPED-WITH-REASON rather than OOMing (F-COV-1/L8 VRAM gate). The per-precision
// reference + G==1 candidate each run the SINGLE-GPU compute_f2_blocks over the FULL
// P×M on device 0 — that full-M call is the binding peak (the G==2 shards each see
// ~M/2). One such call's co-resident device footprint, mirroring cuda_backend.cu's own
// budget comment (the FEEDER phase holds the raw inputs dQ_raw+dV_raw+dN_raw = 3·P·M
// AND the persisted feeder outputs dQ+dV+dS = 4·P·M, on TOP of the two resident
// f2/Vpair tensors that were allocated before the feeder scope), is:
//     feeder phase (raw + feeder outputs)  : 7 · P · M · 8 bytes
//   + resident f2 + Vpair tensors          : resident_tensor_bytes(P, n_block)  (§11.2 home, DRY)
//   + cuBLAS determinism workspace         : kCublasWorkspaceBytes
//   ──────────────────────────────────────────────────────────────
//   × kVramHeadroomFactor (allocator overhead + concurrent idle workspaces; see above)
// The chunk slabs reuse the freed-raw VRAM (cuda_backend.cu) so they fit under the same
// envelope. This is a conservative PEAK ESTIMATE for a skip decision, NOT a hard
// allocator: the headroom factor makes it err toward skipping rather than OOMing. (The
// pre-headroom 7·P·M+resident sum is ~32.3 GiB at P=768/M=584k, just under the 31.6 GiB
// free on a 32 GB 5090 — which OOMed in practice; the headroom closes that gap.)
[[nodiscard]] std::size_t estimate_peak_vram_bytes(int P, long M, int n_block) {
    if (P <= 0 || M <= 0 || n_block <= 0) return 0;
    const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    const std::size_t feeder_peak = 7u * pm * sizeof(double);
    const std::size_t raw = feeder_peak + steppe::device::resident_tensor_bytes(P, n_block) +
                            steppe::kCublasWorkspaceBytes;
    return static_cast<std::size_t>(kVramHeadroomFactor * static_cast<double>(raw));
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

// Run the full parity battery on ONE loaded dataset. Returns true if the dataset
// was actually exercised OR a real error was counted (the caller treats both as
// "covered"); false ONLY when the dataset directory was absent or VRAM-skipped (a
// logged skip, not a failure). The capability tier is NOT a parameter: the tier
// dispatch is read in-band from each Resources' probed caps.can_access_peer (L8
// biconditional below), the SAME field the production §4 fork gates on — no device-
// name match. `device_name` is observability only (printed, never a dispatch key).
bool run_dataset(const std::string& label, const std::string& dir, const std::string& snp,
                 int visible_devices, const char* device_name) {
    int P = 0; long M = 0;
    std::vector<double> Qd, Vd, Nd;
    const LoadResult lr = load_qvn(dir, P, M, Qd, Vd, Nd);
    if (lr == LoadResult::Absent) {
        std::printf("[skip] dataset '%s' (%s) absent — not run\n",
                    label.c_str(), dir.c_str());
        return false;
    }
    if (lr == LoadResult::Malformed) {
        // F-BUG-2: present-but-malformed is a real ERROR on a parity gate, NOT a
        // silent skip. Count it so a truncated derived_full cannot leave the gate
        // green while the scale dataset it advertises never ran.
        std::fprintf(stderr, "ERROR: dataset '%s' (%s) PRESENT but malformed — counting as FAIL\n",
                     label.c_str(), dir.c_str());
        ++g_failures;
        return true;
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

    // ---- VRAM GATE (F-COV-1/L8): never OOM, never silent-skip the scale dataset ----
    // The battery builds several concurrent Resources/backends and drives the
    // single-GPU compute_f2_blocks over the FULL P×M on device 0 — at P=768/M=584k the
    // resident envelope (~36 GB) exceeds a 32 GB consumer 5090. Probe device-0 free
    // VRAM and the estimated peak; if it will not fit, SKIP this dataset with an
    // EXPLICIT reason (need ~X MiB, free ~Y MiB) — NOT an OOM (a crash would taint the
    // whole gate), NOT a silent skip (the absent-vs-malformed law is upheld; this is a
    // third, explicitly-logged outcome). On a 96 GB PRO box both datasets clear the
    // gate. The skip returns false (a logged skip, like Absent), so a derived_acc-only
    // run still gates green and `any_dataset_run` reflects what actually ran.
    {
        // The estimate is a device-0 peak; build the candidates on device 0, so probe
        // device 0. cudaMemGetInfo reports the CURRENT device's free/total VRAM
        // (CUDA Runtime API, group CUDART_MEMORY) — make device 0 current first.
        if (cudaSetDevice(0) != cudaSuccess) {
            std::fprintf(stderr, "ERROR: cudaSetDevice(0) failed before the VRAM probe for '%s'\n",
                         label.c_str());
            ++g_failures;
            return true;
        }
        std::size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            std::fprintf(stderr, "ERROR: cudaMemGetInfo failed before the VRAM probe for '%s'\n",
                         label.c_str());
            ++g_failures;
            return true;
        }
        const std::size_t need_b = estimate_peak_vram_bytes(P, M, part.n_block);
        const double need_mib = static_cast<double>(need_b) / (1024.0 * 1024.0);
        const double free_mib = static_cast<double>(free_b) / (1024.0 * 1024.0);
        if (need_b > free_b) {
            std::printf("[skip] %s P=%d : insufficient VRAM (need ~%.0f MiB, free ~%.0f MiB)\n",
                        label.c_str(), P, need_mib, free_mib);
            return false;  // logged VRAM skip — not a failure, not silent
        }
        std::printf("  VRAM gate OK: need ~%.0f MiB, free ~%.0f MiB on device 0\n",
                    need_mib, free_mib);
    }

    const Precision precNat{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};
    const Precision precEmu{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
    const Precision precs[2] = {precNat, precEmu};
    const char* precNames[2] = {"Fp64", "EmuFp64{40}"};

    // ---- Build the device-resource bundles (ONE probe per device, build time) --
    // G=1 candidate: devices={0}. G=2 candidates: devices={0,1} TWICE —
    //   resG2h: prefer_p2p_combine=false  → FORCES the host-staged baseline,
    //   resG2p: prefer_p2p_combine=true   → the device-resident cudaMemcpyPeer P2P
    //           fast-path (exercised when can_access_peer; the rtxbox PRO tier).
    // Both G=2 candidates must be bit-identical to single-GPU AND to each other (the
    // two-tier neutrality, architecture.md §11.4 §12).
    DeviceConfig cfgG1;
    cfgG1.devices = {0};
    steppe::device::Resources resG1 = steppe::device::build_resources(cfgG1);

    const bool have_two = (visible_devices >= 2);
    steppe::device::Resources resG2h;
    steppe::device::Resources resG2p;
    if (have_two) {
        DeviceConfig cfgG2h;
        cfgG2h.devices = {0, 1};
        cfgG2h.prefer_p2p_combine = false;  // force the host-staged baseline path
        resG2h = steppe::device::build_resources(cfgG2h);

        DeviceConfig cfgG2p;
        cfgG2p.devices = {0, 1};
        cfgG2p.prefer_p2p_combine = true;   // prefer the device-resident P2P combine
        resG2p = steppe::device::build_resources(cfgG2p);
    }

    // F-COV-1 capture: the native (p==0) and EmulatedFp64 (p==1) single-GPU references,
    // kept so that AFTER the precision loop we can assert the EmulatedFp64 lane actually
    // ENGAGED the fixed-slice Ozaki path (it must produce DIFFERENT bits than native) —
    // closing the silent-native-fallback blind spot: every EmuFp64 bit-identity claim
    // compares sharded-EmuFp64 to single-GPU-EmuFp64, so if both silently ran native
    // Fp64 they would still match and the gate would certify the WRONG path.
    F2BlockTensor natRef;
    F2BlockTensor emuRef;

    // ---- Per precision: reference (single-GPU) vs candidates (memcmp) ----------
    for (int p = 0; p < 2; ++p) {
        const Precision& prec = precs[p];
        const bool is_emu_prec = (prec.kind == Precision::Kind::EmulatedFp64);

        // REFERENCE: the existing single-GPU seam (the M4-trusted path).
        auto ref_backend = steppe::device::make_cuda_backend(0);
        const F2BlockTensor ref =
            steppe::core::compute_f2_blocks(*ref_backend, Q, V, N, part, prec);
        (is_emu_prec ? emuRef : natRef) = ref;  // F-COV-1: capture for the engaged-check

        // (1) G==1 multi-GPU == single-GPU reference (the bit-identity FLOOR).
        // Drive the DEVICE-RESIDENT primary and materialize ONCE (.to_host()) for the
        // memcmp — proves the resident result is bit-identical to single-GPU (§12).
        const F2BlockTensor candG1 =
            steppe::core::compute_f2_blocks_multigpu_device(resG1, Q, V, N, part, prec).to_host();
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
                steppe::core::compute_f2_blocks_multigpu_device(resG2h, Q, V, N, part, prec).to_host();
            // The P2P candidate. On the PRO tier (can_access_peer) this runs the
            // device-resident cudaMemcpyPeer combine (combine_f2_partials_resident_device,
            // the no-final-D2H assembly); on a no-peer box it degrades (tagged) to
            // host-staged + a re-upload — either way the .to_host() result is
            // bit-identical. last_combine_path (checked below) records which transport ran.
            const F2BlockTensor candG2p =
                steppe::core::compute_f2_blocks_multigpu_device(resG2p, Q, V, N, part, prec).to_host();

            const bool is_emulated = (prec.kind == Precision::Kind::EmulatedFp64);
            if (is_emulated) {
                // (2) G==2 host-staged == single-GPU reference — STRICT bit-identity.
                {
                    const bool ok = bit_equal(ref, candG2h);
                    std::string lab = std::string("[") + precNames[p] +
                                      "] G==2 host-staged == single-GPU reference (bit-identical)";
                    check(lab.c_str(), ok);
                    if (!ok) report_first_diff("G2host", ref, candG2h);
                }
                // (3) G==2 P2P == single-GPU reference — STRICT bit-identity. The
                // device-resident cudaMemcpyPeer combine moves the same bytes and sums
                // the same fixed g=0..G-1 order ⇒ identical (architecture.md §11.4,
                // §12). Asserted only when P2P actually ran (PRO tier); on a no-peer
                // box candG2p IS the host-staged result (already covered by (2)) and
                // this is informational.
                if (resG2p.gpus[0].caps.can_access_peer) {
                    const bool ok = bit_equal(ref, candG2p);
                    std::string lab = std::string("[") + precNames[p] +
                                      "] G==2 P2P device-resident == single-GPU reference (bit-identical)";
                    check(lab.c_str(), ok);
                    if (!ok) report_first_diff("G2p2p", ref, candG2p);
                } else {
                    std::printf("  [skip] [%s] G==2 P2P == single-GPU ref: no peer access "
                                "(degraded to host-staged — covered by (2))\n", precNames[p]);
                }
            } else {
                // Native-FP64 §12-scoped tier: shape + block_sizes EXACT, f2/vpair to
                // the native-FP64 oracle tolerance (batched-GEMM accumulation order is
                // implementation-defined across batchCount; §12). Reported with the
                // observed worst relative deviation so the noise level is visible.
                {
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
                if (resG2p.gpus[0].caps.can_access_peer) {
                    const double rel = max_rel_dev(ref, candG2p);
                    const bool ok = (rel <= kNativeFp64ParityRelTol);
                    char lab[256];
                    std::snprintf(lab, sizeof(lab),
                                  "[%s] G==2 P2P device-resident == single-GPU ref to "
                                  "native-FP64 oracle tol (rel=%.3g <= %.0e; §12 batched-GEMM "
                                  "shape-sensitivity, NOT bit-identity)",
                                  precNames[p], rel, kNativeFp64ParityRelTol);
                    check(lab, ok);
                    if (!ok) report_first_diff("G2p2p", ref, candG2p);
                }
            }

            // (4) THE TWO-TIER NEUTRALITY — STRICT bit-identity for BOTH precisions.
            // Both combine tiers consume the SAME per-device partials and sum the SAME
            // fixed g=0..G-1 order onto a zero-initialized full tensor; the transport
            // (host-stage vs cudaMemcpyPeer) only moves bytes (architecture.md §11.4).
            // So P2P and host-staged agree BIT-FOR-BIT regardless of per-device GEMM
            // precision — this is the explicit workflow ask, and it is bit-identity
            // even in the native-Fp64 lane (the §12 batched-GEMM shape-sensitivity
            // affects the per-device GEMM, which is IDENTICAL in both candidates here,
            // not the combine). When P2P degraded to host-staged (no peer) this is a
            // host==host tautology that still must hold.
            {
                const bool ok = bit_equal(candG2h, candG2p);
                std::string lab = std::string("[") + precNames[p] +
                                  "] G==2 P2P == G==2 host-staged (two-tier neutrality, bit-identical)";
                check(lab.c_str(), ok);
                if (!ok) report_first_diff("G2 p2p-vs-host", candG2h, candG2p);
            }
        } else {
            std::printf("  [skip] [%s] G==2 host-staged/P2P: only %d visible device(s) "
                        "(need 2) — single-GPU box lane\n", precNames[p], visible_devices);
        }
    }

    // ---- (F-COV-1) EmulatedFp64 ACTUALLY ENGAGED — the silent-native-fallback check --
    // Every EmuFp64 bit-identity claim above compares sharded-EmuFp64 to single-GPU-
    // EmuFp64; if the EmulatedFp64 request silently degraded to native Fp64 on BOTH
    // sides, the bits would still match and the gate would print PASS while never
    // exercising the §12 fixed-slice Ozaki path it exists to certify. So, on a build
    // where the fixed-mantissa tuning API IS compiled in (STEPPE_HAVE_EMU_TUNING), the
    // probe field caps.emulated_fp64_honorable MUST be true AND the EmuFp64 reference
    // MUST differ bit-for-bit from native — exactly the guard the sibling
    // test_f2_blocks_equivalence makes. Read via the CUDA-FREE caps probe field
    // (backend.hpp:184), NOT the device-private emulation_honorable() predicate, so this
    // TU keeps its clean layering (F-LAY-2). When STEPPE_HAVE_EMU_TUNING is NOT defined
    // (a build without the tuning API) EmulatedFp64 is EXPECTED to honorably degrade to
    // native, so this check does not apply.
#ifdef STEPPE_HAVE_EMU_TUNING
    {
        check("caps.emulated_fp64_honorable == true under STEPPE_HAVE_EMU_TUNING "
              "(EmulatedFp64 lane must be honorable, not a silent native fallback)",
              resG1.gpus[0].caps.emulated_fp64_honorable);
        const bool emu_differs =
            emuRef.f2.size() == natRef.f2.size() &&
            std::memcmp(emuRef.f2.data(), natRef.f2.data(),
                        emuRef.f2.size() * sizeof(double)) != 0;
        check("EmulatedFp64 single-GPU ref DIFFERS bit-for-bit from native Fp64 "
              "(the fixed-slice Ozaki path engaged; not a silent native fallback)",
              emu_differs);
        if (!emu_differs) {
            std::fprintf(stderr,
                "  [F-COV-1] EmulatedFp64 produced IDENTICAL bits to native Fp64 — the "
                "§12 emulated-FP64 path did NOT engage (silent native fallback); every "
                "EmuFp64 bit-identity PASS above certified the WRONG path.\n");
        }
    }
#endif  // STEPPE_HAVE_EMU_TUNING

    // ---- (5) Capability + which-path tag sanity (out-of-band; never on the tensor) ----
    if (have_two) {
        const bool peer0 = resG2p.gpus[0].caps.can_access_peer;
        std::printf("  G2 resources: gpus[0].caps.can_access_peer=%s (device 0: %s)\n"
                    "    last_combine_path: resG2h=%s  resG2p=%s\n",
                    peer0 ? "true" : "false", device_name,
                    combine_path_name(resG2h.last_combine_path),
                    combine_path_name(resG2p.last_combine_path));

        // prefer_p2p_combine=false ALWAYS forces the host-staged baseline (the
        // explicit workflow ask, every tier). The which-path tag is the out-of-band
        // record (Resources, never on F2BlockTensor; architecture.md §11.4 / cleanup
        // §(2).2) — confirms the gate honored the user's intent.
        check("resG2h.last_combine_path == HostStaged (prefer_p2p_combine=false forces baseline)",
              resG2h.last_combine_path == steppe::device::CombinePath::HostStaged);

        // L8 / F-IDIOM-2: select the tier by the CAPABILITY BICONDITIONAL, NOT a "PRO"
        // device-name match. The production §4 fork (f2_blocks_multigpu.cpp:171-172)
        // takes the device-resident P2P combine IFF prefer_p2p_combine && can_access_peer;
        // resG2p set prefer_p2p_combine=true, so for it the path is determined SOLELY by
        // can_access_peer. Assert exactly that invariant — it holds on EVERY tier (the
        // PRO box with stock-driver P2P AND the consumer 5090 with P2P driver-disabled),
        // with no marketing-string coupling and no false-FAIL on a future "PRO"-named SKU
        // that lacks stock P2P:
        //   can_access_peer  ⇔  last_combine_path == P2pDeviceResident
        //  !can_access_peer  ⇔  last_combine_path == HostStaged   (the tagged degrade)
        // This is the SAME field the strict claim-(3) guard reads (line ~485), so the
        // gate cannot assert P2P where production would not have taken it, nor miss a P2P
        // that did run silently as host-staged.
        if (peer0) {
            // Peer access available (the PRO 6000 / datacenter-Blackwell stock-driver
            // case): resG2p MUST have actually taken the device-resident cudaMemcpyPeer
            // combine — so claim (3) genuinely exercised P2P, not a silent fallback.
            check("can_access_peer ⇒ resG2p.last_combine_path == P2pDeviceResident "
                  "(P2P actually ran, not a silent host-staged fallback)",
                  resG2p.last_combine_path == steppe::device::CombinePath::P2pDeviceResident);
        } else {
            // No peer access (the consumer GeForce / stock-5090 P2P-driver-disabled
            // case): peer false is the EXPECTED degrade; resG2p degrades (tagged) to the
            // host-staged baseline, which is still bit-identical (claims (2)/(4)).
            std::printf("    (no peer access: can_access_peer==false is the expected "
                        "tagged degrade to the host-staged baseline)\n");
            check("!can_access_peer ⇒ resG2p.last_combine_path == HostStaged "
                  "(tagged degrade to the parity baseline)",
                  resG2p.last_combine_path == steppe::device::CombinePath::HostStaged);
        }
    }

    // ===================================================================================
    // (M5) ADAPTIVE TIERED OUTPUT — FORCE EACH TIER, READ BACK, MEMCMP vs the
    // device-resident reference (architecture.md §12; the FROZEN per-tier read-back
    // battery). The reference is the EmulatedFp64{40} single-GPU result materialized to
    // host (emuRef above — the existing trusted path). For each tier in
    // {Resident, HostRam, Disk} we build a DeviceConfig with force_tier set to that tier
    // (the override exercises Disk/HostRam at the SMALL P=50 derived_acc where Auto would
    // pick Resident), run compute_f2_blocks_multigpu_tiered, call out.to_host(), and
    // assert bit_equal(emuRef, out.to_host()). Block-axis streaming is EXACT by
    // construction — each block is computed identically/independently and the stream only
    // changes WHEN/WHERE a slab lands, never its bits (§12). EmulatedFp64{40} is the f2
    // production default and is batchCount-deterministic; at G==1 all three tiers share
    // ONE per-block run_f2_blocks_resident code, so even native Fp64 would be bit-identical
    // — we assert with the strict bit_equal predicate.
    {
        const Precision precEmuTier{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
        struct TierCase { steppe::DeviceConfig::ForceTier ft; steppe::device::OutputTier expect; const char* name; };
        const TierCase tiers[3] = {
            {steppe::DeviceConfig::ForceTier::Resident, steppe::device::OutputTier::Resident, "Resident"},
            {steppe::DeviceConfig::ForceTier::HostRam,  steppe::device::OutputTier::HostRam,  "HostRam"},
            {steppe::DeviceConfig::ForceTier::Disk,     steppe::device::OutputTier::Disk,     "Disk"},
        };
        for (const TierCase& tc : tiers) {
            steppe::DeviceConfig cfgT;
            cfgT.devices = {0};
            cfgT.force_tier = tc.ft;
            if (tc.ft == steppe::DeviceConfig::ForceTier::Disk)
                cfgT.disk_cache_path = std::string("/tmp/steppe_f2_") + label + ".cache";
            steppe::device::Resources resT = steppe::device::build_resources(cfgT);

            steppe::device::F2BlocksOut outT = steppe::core::compute_f2_blocks_multigpu_tiered(
                resT, Q, V, N, part, precEmuTier);

            // The override was honored (out.tier == the forced tier).
            {
                char lab[160];
                std::snprintf(lab, sizeof(lab),
                              "[EmuFp64{40}] force_tier=%s ⇒ out.tier == %s (override honored)",
                              tc.name, tc.name);
                check(lab, outT.tier == tc.expect);
            }

            // to_host() read-back is BIT-IDENTICAL to the single-GPU EmuFp64 reference.
            const F2BlockTensor backT = outT.to_host();
            {
                char lab[160];
                std::snprintf(lab, sizeof(lab),
                              "[EmuFp64{40}] tier=%s to_host() == single-GPU reference (bit-identical)",
                              tc.name);
                const bool ok = bit_equal(emuRef, backT);
                check(lab, ok);
                if (!ok) report_first_diff(tc.name, emuRef, backT);
            }

            // Disk-only: the FIT's per-block accessor read_block_to_host(b) is byte-exact
            // for EVERY block (proves the pread offsets, f2_disk_format.hpp §4, are right).
            if (tc.ft == steppe::DeviceConfig::ForceTier::Disk) {
                const std::size_t slab =
                    static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
                std::vector<double> f2_slab(slab), vp_slab(slab);
                bool all_ok = (outT.n_block == emuRef.n_block);
                for (int b = 0; b < outT.n_block && all_ok; ++b) {
                    outT.read_block_to_host(b, f2_slab.data(), vp_slab.data());
                    const std::size_t off = slab * static_cast<std::size_t>(b);
                    if (std::memcmp(f2_slab.data(), emuRef.f2.data() + off, slab * sizeof(double)) != 0 ||
                        std::memcmp(vp_slab.data(), emuRef.vpair.data() + off, slab * sizeof(double)) != 0) {
                        all_ok = false;
                        std::fprintf(stderr, "  [Disk] read_block_to_host(%d) differs from reference slab\n", b);
                    }
                }
                check("[EmuFp64{40}] tier=Disk read_block_to_host(b) == reference slab for every b "
                      "(the fit's per-block accessor is byte-exact; §4 pread offsets correct)",
                      all_ok);
            }
        }

        // AUTO at small P picks Resident (streaming NOT forced on small P — the dominant
        // correctness/perf gate: opt-in-by-need). select_output_tier reads the SAME runtime
        // probes the production orchestrator uses (free VRAM caps + sysinfo free host RAM).
        {
            steppe::DeviceConfig cfgA;
            cfgA.devices = {0};
            steppe::device::Resources resA = steppe::device::build_resources(cfgA);
            const std::size_t free_vram = resA.gpus[0].caps.free_vram_bytes;
            const std::size_t free_host = steppe::device::free_host_ram_bytes();
            const steppe::device::OutputTier auto_tier =
                steppe::device::select_output_tier(P, M, part.n_block, free_vram, free_host);
            char lab[200];
            std::snprintf(lab, sizeof(lab),
                          "[EmuFp64{40}] Auto select_output_tier(P=%d) == Resident "
                          "(streaming NOT forced on small P; opt-in-by-need)", P);
            check(lab, auto_tier == steppe::device::OutputTier::Resident);
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
    // Device-0 name for OBSERVABILITY ONLY — it is printed, never a dispatch key
    // (L8/F-IDIOM-2: the tier branch is driven by caps.can_access_peer, not the name).
    // cudaGetDeviceProperties queries by ordinal, no cudaSetDevice needed (CUDA Runtime
    // API, group CUDART_DEVICE); on failure the name is left "(unknown)" — purely
    // cosmetic, it cannot mis-classify anything anymore.
    char device_name[256] = "(unknown)";
    {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            std::snprintf(device_name, sizeof(device_name), "%s", prop.name);
        } else {
            std::fprintf(stderr, "WARN: cudaGetDeviceProperties(0) failed; device name unknown "
                                 "(observability only — tier is driven by can_access_peer)\n");
        }
    }
    std::printf("  visible CUDA devices: %d  (device 0: %s)\n", visible, device_name);

    // Both gates: derived_acc (fast, P=50) AND derived_full (scale, P=768). Each is
    // run if PRESENT and it FITS in device-0 VRAM; an absent dataset OR a dataset that
    // does not fit is a logged skip so the fast gate runs on a box without the full
    // matrix / on a 32 GB consumer 5090 (design §6; F-COV-1/L8 VRAM gate).
    bool any_dataset_run = false;
    try {
        any_dataset_run |= run_dataset("derived_acc",  root + "/derived_acc",  snp, visible, device_name);
        any_dataset_run |= run_dataset("derived_full", root + "/derived_full", snp, visible, device_name);
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
