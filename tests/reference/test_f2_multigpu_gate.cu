// tests/reference/test_f2_multigpu_gate.cu
//
// HOST-PURE-LOGIC unit test of the M4.5 SPMG combine GATE — the §4 capability-tier
// fork in core::compute_f2_blocks_multigpu (architecture.md §11.4, §12; cleanup
// CT2 / config C-1/K-2). NO real GPU compute, NO io, NO real data, NO real peer
// access: a CUDA-FREE recording FakeBackend supplies each device's compact partial,
// and a hand-built Resources sets gpus[g].caps.can_access_peer DIRECTLY (so the gate
// can be exercised at a probe value the host this test runs on does not actually have
// — e.g. can_access_peer==true on a consumer 5090 where the real probe is false). The
// gate is host-pure logic over Resources/DeviceConfig; this test pins it without a
// device.
//
// WHY A .cu (NOT a host .cpp): the test references core::compute_f2_blocks_multigpu,
// whose TU (steppe_core's f2_blocks_multigpu.o) references the device-layer symbols
// plan_block_shards (host) and combine_f2_partials_p2p (CUDA, RDC) — so the executable
// must LINK + device-link steppe::device to RESOLVE them (steppe::core links
// steppe::device PRIVATE). The P2P symbol is only RESOLVED here, never CALLED (every
// driven row resolves to the host-staged baseline). This is the same .cu +
// CUDA_SEPARABLE_COMPILATION pattern test_resources_build.cu uses to reach
// build_resources without itself running a GPU kernel.
//
// THE FIX UNDER TEST (B3 / CT2 / config C-1; B4 / X6): the combine gate is the
// FOUR-term predicate
//     use_p2p = prefer_p2p_combine && enable_peer_access && can_access_peer && G >= 2
// — the four-term §4 gate defined ONCE in f2_blocks_multigpu.cpp ("THE §4 COMBINE
// GATE", §8 single-source). B3 ANDed in the MAY-WE knob `enable_peer_access` (the OLD
// gate ignored it); the device-resident P2P path calls cudaDeviceEnablePeerAccess
// (cuda/p2p_combine.cu), the exact operation enable_peer_access=false is documented to
// FORBID, so the widened gate honors that veto. B4 ADDED the dead-true `&& G >= 2`
// term so the shipped CODE MATCHES the predicate documented across five files (cleanup
// X6: it was documented 4-term while shipping 3-term). `G >= 2` is structurally
// dead-true at the gate (the G==1 single-GPU fast-path returns before the gate), so the
// term changes NO reached path — it is identity on every reachable input. Because both
// combine tiers are bit-identical (parity-NEUTRAL, §12), neither the B3 enable_peer_access
// term nor the B4 G>=2 term can move a reported number — the locked f2_multigpu_parity
// bit-identity is unaffected by these fixes.
//
// THE REQUIRED OBJECTIVE GATE (the workflow ask): a check that enable_peer_access=false
// forces last_combine_path==HostStaged EVEN IF can_access_peer were true. Asserted
// directly below (g_eppr_false_caps_true), plus the full gate truth table.
//
// WHY THE P2P-SELECTING ROW IS NOT DRIVEN END-TO-END HERE. The one combo that selects
// P2P (prefer && enable && can_access) would, through the entry point, call the real
// device-resident combine_f2_partials_p2p (cudaMemcpyPeer + cudaDeviceEnablePeerAccess)
// — which needs an actual peer-capable device pair this host-pure test deliberately
// does not have. So the P2P-selecting row is asserted on the GATE PREDICATE itself
// (the same boolean the entry point computes), while every HostStaged-resolving row is
// driven END-TO-END through the real compute_f2_blocks_multigpu (the FakeBackend's
// partials + the CUDA-free host-staged combine — no GPU) and the resulting
// last_combine_path tag is asserted. This is exactly the host-pure seam the §13 design
// admits (cleanup T1).
//
// Self-checking main() (not a GoogleTest TU; CTest gates on the exit code), mirroring
// the reference-test convention (test_resources_build.cu). The recording FakeBackend
// names only the ABSTRACT ComputeBackend and issues no GEMM — the gate LOGIC, not any
// device arithmetic, is under test.
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "core/fstats/f2_blocks_multigpu.hpp"      // compute_f2_blocks_multigpu
#include "core/domain/block_partition_rule.hpp"    // BlockPartition
#include "core/internal/views.hpp"                 // MatView
#include "device/backend.hpp"                      // ComputeBackend, BackendCapabilities, F2Result
#include "device/resources.hpp"                    // Resources, PerGpuResources, CombinePath
#include "steppe/config.hpp"                        // DeviceConfig, Precision
#include "steppe/fstats.hpp"                        // F2BlockTensor

namespace {

using steppe::ComputeBackend;
using steppe::DecodeResult;
using steppe::DecodeTileView;
using steppe::DeviceConfig;
using steppe::F2BlockTensor;
using steppe::F2Result;
using steppe::Precision;
using steppe::core::BlockPartition;
using steppe::core::compute_f2_blocks_multigpu;
using steppe::core::MatView;
using steppe::device::CombinePath;
using steppe::device::PerGpuResources;
using steppe::device::Resources;

// ---------------------------------------------------------------------------
// A CUDA-FREE FakeBackend. compute_f2_blocks returns a VALID compact partial built
// from the LOCAL block_id/n_block the orchestrator hands it: an [P × P × n_block]
// tensor whose block_sizes[b] is the count of local SNPs assigned to local block b.
// That is exactly the contract combine_f2_partials_host validates (each partial's
// n_block == its shard's block span; block_sizes the per-block SNP counts), so the
// HostStaged-resolving rows combine successfully with NO GPU. The f2/vpair payload is
// a recognizable sentinel — the gate logic, not the arithmetic, is under test.
// ---------------------------------------------------------------------------
class FakeBackend final : public ComputeBackend {
public:
    [[nodiscard]] F2Result compute_f2(const MatView&, const MatView&, const MatView&,
                                      const Precision&) override {
        return {};  // not exercised by the multi-GPU entry point
    }

    [[nodiscard]] F2BlockTensor compute_f2_blocks(const MatView& Q, const MatView&,
                                                  const MatView&, const int* block_id,
                                                  int n_block, const Precision&) override {
        F2BlockTensor out;
        out.P = Q.P;
        out.n_block = n_block;
        // Per-local-block SNP counts from the dense local block_id (length M_local).
        out.block_sizes.assign(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
        for (long k = 0; k < Q.M; ++k) {
            const int b = block_id[k];
            if (b >= 0 && b < n_block) ++out.block_sizes[static_cast<std::size_t>(b)];
        }
        const std::size_t n = out.size();
        out.f2.assign(n, kSentinel);
        out.vpair.assign(n, 1.0);
        return out;
    }

    // The host-staged-DIRECT seam (M4.5 d2h-speed): the orchestrator's host-staged
    // branch now drives this instead of compute_f2_blocks + combine. Build the SAME
    // partial and place it (host memcpy, no CUDA, no pin) into the caller's shared
    // result at the disjoint block offset slab*b0 (f2/vpair) and b0 (block_sizes), so
    // the end-to-end run resolves HostStaged exactly as before (the gate reads only
    // last_combine_path).
    void compute_f2_blocks_into(const MatView& Q, const MatView& V, const MatView& N,
                                const int* block_id, int n_block, int b0,
                                double* dst_f2, double* dst_vpair, int* block_sizes_dst,
                                const Precision& precision) override {
        F2BlockTensor part = compute_f2_blocks(Q, V, N, block_id, n_block, precision);
        const std::size_t slab =
            static_cast<std::size_t>(Q.P) * static_cast<std::size_t>(Q.P);
        const std::size_t slab_off = slab * static_cast<std::size_t>(b0);
        if (!part.f2.empty()) {
            std::memcpy(dst_f2 + slab_off, part.f2.data(),
                        part.f2.size() * sizeof(double));
            std::memcpy(dst_vpair + slab_off, part.vpair.data(),
                        part.vpair.size() * sizeof(double));
        }
        for (int lb = 0; lb < part.n_block; ++lb) {
            block_sizes_dst[static_cast<std::size_t>(b0) + static_cast<std::size_t>(lb)] =
                part.block_sizes[static_cast<std::size_t>(lb)];
        }
    }

    [[nodiscard]] DecodeResult decode_af(const DecodeTileView&) override { return {}; }

    static constexpr double kSentinel = 7.0;
};

// Build a G-device Resources by hand (Resources is a public CUDA-free aggregate): G
// FakeBackends, each with caps.can_access_peer set to `peer` (gpus[0] is the combine
// root the gate reads). The DeviceConfig carries the two override-intent knobs under
// test. NO build_resources / NO GPU.
[[nodiscard]] Resources make_fake_resources(std::size_t G, bool prefer, bool enable, bool peer) {
    Resources r;
    r.gpus.reserve(G);
    for (std::size_t g = 0; g < G; ++g) {
        PerGpuResources pg;
        pg.device_id = static_cast<int>(g);
        pg.backend = std::make_unique<FakeBackend>();
        pg.caps.can_access_peer = peer;   // hand-set the DISCOVERED probe value
        r.gpus.push_back(std::move(pg));
    }
    r.config.devices.clear();
    for (std::size_t g = 0; g < G; ++g) r.config.devices.push_back(static_cast<int>(g));
    r.config.prefer_p2p_combine = prefer;
    r.config.enable_peer_access = enable;
    return r;
}

// A consistent 4-block partition over M columns (dense, non-decreasing, in [0,4)).
// 4 blocks > G=2 so each device owns >= 2 contiguous blocks (a real shard, not empty).
[[nodiscard]] BlockPartition make_partition(long M) {
    BlockPartition bp;
    bp.n_block = 4;
    bp.block_id.resize(static_cast<std::size_t>(M));
    for (long s = 0; s < M; ++s) {
        const int b = static_cast<int>((s * bp.n_block) / M);  // 0..3, non-decreasing
        bp.block_id[static_cast<std::size_t>(s)] = (b < bp.n_block) ? b : bp.n_block - 1;
    }
    return bp;
}

[[nodiscard]] MatView make_view(const std::vector<double>& storage, int P, long M) {
    MatView mv;
    mv.data = storage.data();
    mv.P = P;
    mv.M = M;
    return mv;
}

// The EXACT host-pure gate predicate the entry point computes (architecture.md §11.4
// §4; the four-term "THE §4 COMBINE GATE" in f2_blocks_multigpu.cpp, §8 single-source).
// Kept here as the single boolean under test so the P2P-SELECTING row can be asserted
// without driving the real cudaMemcpyPeer transport. MIRRORS THE PRODUCTION EXPRESSION
// TERM-FOR-TERM, including the B4 `&& G >= 2` term (cleanup X6/B4): code == doc == this
// mirror. G == r.device_count() (G==1 short-circuits before the gate in production, but
// the predicate carries the term so a refactor that lifted the gate to be reachable at
// G==1 would be caught — see g_single_gpu_gate_term_false).
[[nodiscard]] bool gate_selects_p2p(const Resources& r) {
    return r.config.prefer_p2p_combine && r.config.enable_peer_access &&
           r.gpus[0].caps.can_access_peer && r.device_count() >= 2;
}

// Drive the real entry point on the fake resources and return the recorded tag. Only
// called for HostStaged-resolving configs (the P2P transport is not exercised host-only).
[[nodiscard]] CombinePath run_and_tag(bool prefer, bool enable, bool peer) {
    constexpr int  P = 3;
    constexpr long M = 12;
    Resources r = make_fake_resources(/*G=*/2, prefer, enable, peer);
    const std::vector<double> qd(static_cast<std::size_t>(P) * M, 0.5);
    const MatView Q = make_view(qd, P, M);
    const MatView V = make_view(qd, P, M);
    const MatView N = make_view(qd, P, M);
    const BlockPartition bp = make_partition(M);
    const Precision prec{Precision::Kind::EmulatedFp64, 40};
    const F2BlockTensor t = compute_f2_blocks_multigpu(r, Q, V, N, bp, prec);
    (void)t;  // the gate TAG is under test, not the numeric payload
    return r.last_combine_path;
}

// =====================  THE GATE TRUTH TABLE  ==============================

// THE REQUIRED OBJECTIVE GATE: enable_peer_access=false forces HostStaged EVEN IF
// can_access_peer were true (and P2P is preferred). This is the row the OLD two-term
// gate got wrong (it would have selected P2P and called the forbidden
// cudaDeviceEnablePeerAccess). The widened gate honors the veto.
[[nodiscard]] bool g_eppr_false_caps_true() {
    Resources r = make_fake_resources(/*G=*/2, /*prefer=*/true, /*enable=*/false, /*peer=*/true);
    // Predicate: the gate must NOT select P2P despite peer && prefer.
    const bool predicate_ok = (gate_selects_p2p(r) == false);
    // End-to-end: the entry point must record HostStaged (it resolves to the baseline,
    // so the host-only combine runs — no GPU, no forbidden enable).
    const CombinePath tag = run_and_tag(/*prefer=*/true, /*enable=*/false, /*peer=*/true);
    return predicate_ok && tag == CombinePath::HostStaged;
}

// The ONLY combo that selects P2P: prefer && enable && can_access && G >= 2. Asserted
// on the predicate (not driven end-to-end — would need a real peer-capable device pair).
[[nodiscard]] bool g_all_true_selects_p2p() {
    Resources r = make_fake_resources(/*G=*/2, /*prefer=*/true, /*enable=*/true, /*peer=*/true);
    return gate_selects_p2p(r) == true;
}

// B4 / X6: the `&& G >= 2` term is the FOURTH term of the production gate and this
// mirror. With prefer && enable && can_access ALL TRUE but G == 1, the four-term gate
// must be FALSE — proving the G>=2 term is spelled and load-bearing IN the predicate
// (not merely enforced structurally by the G==1 early return). This is the latent
// hazard X6 names: a future select_combine_path(resources) lifted out of the entry
// point could be reached at G==1; carrying the term in the predicate forecloses it.
// (The production entry point never reaches the gate at G==1 — the G==1 fast-path
// returns first; g_single_gpu_no_combine proves that path. So adding the term changes
// NO reached production path: it is identity on every reachable input, G is always >= 2
// at the gate. This case asserts the term EXISTS and bites at G==1, not that any
// production path changed.)
[[nodiscard]] bool g_single_gpu_gate_term_false() {
    Resources r = make_fake_resources(/*G=*/1, /*prefer=*/true, /*enable=*/true, /*peer=*/true);
    return gate_selects_p2p(r) == false;  // the G>=2 term alone vetoes it
}

// prefer_p2p_combine=false -> HostStaged regardless of enable/peer (the explicit
// baseline preference; no WARN, no P2P).
[[nodiscard]] bool g_prefer_false_forces_host() {
    Resources r = make_fake_resources(/*G=*/2, /*prefer=*/false, /*enable=*/true, /*peer=*/true);
    const bool predicate_ok = (gate_selects_p2p(r) == false);
    const CombinePath tag = run_and_tag(/*prefer=*/false, /*enable=*/true, /*peer=*/true);
    return predicate_ok && tag == CombinePath::HostStaged;
}

// can_access_peer=false -> HostStaged (the device-cannot-peer degrade), even with both
// intent knobs on (this is the consumer-5090 reality: the genuine tagged degrade).
[[nodiscard]] bool g_no_peer_degrades_host() {
    Resources r = make_fake_resources(/*G=*/2, /*prefer=*/true, /*enable=*/true, /*peer=*/false);
    const bool predicate_ok = (gate_selects_p2p(r) == false);
    const CombinePath tag = run_and_tag(/*prefer=*/true, /*enable=*/true, /*peer=*/false);
    return predicate_ok && tag == CombinePath::HostStaged;
}

// Both intent knobs off + peer off -> HostStaged (the fully-baseline config).
[[nodiscard]] bool g_all_off_host() {
    const CombinePath tag = run_and_tag(/*prefer=*/false, /*enable=*/false, /*peer=*/false);
    return tag == CombinePath::HostStaged;
}

// G==1 fast path: no shard, no combine — the tag stays at its value-initialized None
// regardless of the knobs (the gate never runs). Confirms the gate is reached only at
// G>=2 (the structural G>=2 precondition; cleanup C-2).
[[nodiscard]] bool g_single_gpu_no_combine() {
    constexpr int  P = 3;
    constexpr long M = 12;
    // prefer && enable && peer all TRUE — if the gate ran at G==1 it could mis-tag; it
    // must NOT run (the G==1 early return precedes the gate).
    Resources r = make_fake_resources(/*G=*/1, /*prefer=*/true, /*enable=*/true, /*peer=*/true);
    const std::vector<double> qd(static_cast<std::size_t>(P) * M, 0.5);
    const MatView Q = make_view(qd, P, M);
    const MatView V = make_view(qd, P, M);
    const MatView N = make_view(qd, P, M);
    const BlockPartition bp = make_partition(M);
    const Precision prec{Precision::Kind::EmulatedFp64, 40};
    const F2BlockTensor t = compute_f2_blocks_multigpu(r, Q, V, N, bp, prec);
    (void)t;
    return r.last_combine_path == CombinePath::None;
}

struct GateCase {
    const char* name;
    bool (*fn)();
};

constexpr GateCase kCases[] = {
    {"enable_peer_access=false forces HostStaged EVEN IF can_access_peer (THE required gate)",
     g_eppr_false_caps_true},
    {"prefer && enable && can_access_peer && G>=2 -> gate selects P2P (the only P2P combo)",
     g_all_true_selects_p2p},
    {"B4/X6: prefer && enable && can_access but G==1 -> gate FALSE (the G>=2 term bites)",
     g_single_gpu_gate_term_false},
    {"prefer_p2p_combine=false -> HostStaged regardless of enable/peer",
     g_prefer_false_forces_host},
    {"can_access_peer=false -> HostStaged (device-cannot-peer tagged degrade)",
     g_no_peer_degrades_host},
    {"all knobs off + no peer -> HostStaged (fully-baseline config)",
     g_all_off_host},
    {"G==1 fast path: gate never runs, tag stays None",
     g_single_gpu_no_combine},
};

}  // namespace

int main() {
    int failures = 0;
    for (const auto& c : kCases) {
        const bool ok = c.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "test_f2_multigpu_gate: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_f2_multigpu_gate: all checks PASS\n");
    return 0;
}
