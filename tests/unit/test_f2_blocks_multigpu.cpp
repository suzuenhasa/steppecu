// tests/unit/test_f2_blocks_multigpu.cpp
//
// HOST-PURE, GPU-FREE, DATA-FREE unit test of the SPMG precompute ORCHESTRATOR core
// — the block-aligned shard PLAN (core::plan_multigpu_shards) + the per-device
// concurrent FAN-OUT (core::compute_multigpu_partials), factored out of the public
// entry point compute_f2_blocks_multigpu into f2_blocks_multigpu_core.{hpp,cpp}
// (cleanup D1/T1, B9; architecture.md §11.4 SPMG, §12 PARITY LAW, §13). NO GPU, NO
// CUDA call, NO io, NO real data: the core is host-pure logic over the CUDA-free
// ComputeBackend seam, driven here by a FAKE recording backend, so it links ONLY the
// host-pure steppe::core (no device-link of the CUDA P2P combine — that is the WHOLE
// point of the D1 extraction, the audit's T1).
//
// WHY A GPU-FREE .cpp (vs the existing test_f2_multigpu_gate.cu). The public
// compute_f2_blocks_multigpu references the device-RDC symbol combine_f2_partials_p2p
// (the §4 P2P fast-path), so a test that drives it END-TO-END must be a .cu that
// device-links steppe::device (that is test_f2_multigpu_gate.cu, which covers the §4
// gate truth table). The orchestrator's HOST-PURE heart — the sub-view / dense-local-
// block-id transform, the std::jthread fan-out, the empty-/n_block<G-shard handling —
// references NEITHER the P2P symbol NOR any CUDA type, so the D1 extraction lets THIS
// test exercise it on the cheap host path in microseconds: the fast inner-loop gate
// the slow GPU parity test (test_f2_multigpu_parity.cu) cannot be. A CUDA leak in the
// extracted core would fail this host compile/link (the §4 layering proof, same
// discipline as test_f2_from_blocks.cpp / test_shard_plan.cpp).
//
// It pins (the audit's T1 (a)/(b)/(c)/(e)):
//   1. SUB-VIEW + LOCAL-ID MATH — each device g is handed EXACTLY its shard's columns
//      (Q.data + P*s0, M_local == s1-s0; same offset for Q/V/N) and a DENSE ZERO-BASED
//      local block_id (block_id_local[k] == global block_id[s0+k] - b0), so the
//      backend's [P×P×n_block_local] tensor indexes 0..n_block_local-1. Asserted by a
//      recording fake that captures the views + the local ids it received.
//   2. FAN-OUT + COMBINE REASSEMBLY — the G partials, each keyed on its GLOBAL block
//      id by the fake, recombine via the host-staged fixed-order combine into the full
//      [P×P×n_block] tensor BIT-FOR-BIT equal to an independent per-block reference.
//      Driven over MANY blocks/devices so the concurrent fan-out writes G distinct
//      partials[g] slots with no aliasing (race-freedom is also TSan-checkable).
//   3. G==1 IDENTITY — the single-device plan is the one full shard [0,n_block) over
//      all columns, and the lone partial spans the whole partition with local==global
//      ids: the structural single-GPU invariance the public entry point's G==1 fast
//      path encodes (here proven on the planning/transform, GPU-free).
//   4. EMPTY / n_block<G — trailing empty shards (b0==b1) hand the backend
//      n_block_local==0 / M_local==0; the fake returns an empty partial and the
//      combine places nothing for that device; the non-empty prefix still reassembles
//      bit-exact.
//   5. EXCEPTION PROPAGATION — a worker whose backend THROWS surfaces as a normal
//      throw out of compute_multigpu_partials (the std::exception_ptr rethrow of the
//      first/lowest-g failure), never std::terminate from an escaped thread exception.
//
// These assert NO statistic — they exercise the PLACEMENT / sub-view-and-local-id math
// / control flow of the orchestrator core on hand-built layouts (ROADMAP §0: no
// synthetic data for PRECISION claims; this is index math + control flow + bit-level
// placement, not a precision claim). The combine's own bit-faithfulness (incl. the
// −0.0 case) is covered in test_f2_combine.cpp; the planner's skew/edge tiling in
// test_shard_plan.cpp; the §4 gate truth table in test_f2_multigpu_gate.cu. This
// closes B9's third leg: the orchestrator core, GPU-free.
//
// Self-checking main() (CTest gates on the exit code), mirroring the host-unit-test
// convention (test_shard_plan.cpp): each named case returns bool; main runs the table
// and returns non-zero if any failed.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/fstats/f2_blocks_multigpu_core.hpp"  // plan_multigpu_shards, compute_multigpu_partials
#include "core/fstats/f2_combine.hpp"               // combine_f2_partials_host
#include "core/domain/block_partition_rule.hpp"     // BlockPartition
#include "core/internal/views.hpp"                  // MatView
#include "device/backend.hpp"                       // ComputeBackend, F2Result
#include "device/resources.hpp"                     // Resources, PerGpuResources
#include "device/shard_plan.hpp"                    // DeviceShard
#include "steppe/config.hpp"                         // Precision
#include "steppe/fstats.hpp"                         // F2BlockTensor

namespace {

using steppe::ComputeBackend;
using steppe::DecodeResult;
using steppe::DecodeTileView;
using steppe::F2BlockTensor;
using steppe::F2Result;
using steppe::Precision;
using steppe::core::BlockPartition;
using steppe::core::combine_f2_partials_host;
using steppe::core::compute_multigpu_partials;
using steppe::core::MatView;
using steppe::core::plan_multigpu_shards;
using steppe::device::DeviceShard;
using steppe::device::PerGpuResources;
using steppe::device::Resources;

// The deterministic synthetic per-block payload keyed on the GLOBAL block id. The
// single source the fake backend EMITS and the reference RECONSTRUCTS, so a wrong
// placement / a swapped local-id transform is caught bit-for-bit.
[[nodiscard]] double f2_value(int global_block, std::size_t e) {
    return static_cast<double>(global_block) * 1000.0 + static_cast<double>(e) + 1.0;
}
[[nodiscard]] double vpair_value(int global_block, std::size_t e) {
    return static_cast<double>(global_block) * 1000.0 + static_cast<double>(e) + 500000.0;
}
[[nodiscard]] int block_size_value(int global_block) { return 100 + global_block; }

[[nodiscard]] bool bits_equal(double a, double b) {
    std::uint64_t ua, ub;
    std::memcpy(&ua, &a, sizeof ua);
    std::memcpy(&ub, &b, sizeof ub);
    return ua == ub;
}

// ---------------------------------------------------------------------------
// A CUDA-FREE recording fake backend. compute_f2_blocks:
//   * VERIFIES the orchestrator handed it a DENSE ZERO-BASED local block_id
//     (block_id[0..M_local) in [0, n_block_local), non-decreasing) — recording any
//     breach in `local_id_ok`;
//   * recovers each LOCAL block's GLOBAL id from the recorded shard base b0 and emits
//     the keyed payload f2_value(global, e) / vpair_value(global, e), so the combine
//     must place each device's run at the right GLOBAL offset to reconstruct it;
//   * records the Q sub-view base pointer + M it received per device (for the sub-view
//     math assertion) and a per-device call count (to prove every device was driven).
// Optionally THROWS on a chosen device id (for the exception-propagation case). The
// fake names only the ABSTRACT ComputeBackend + the CUDA-free views — no GPU.
// ---------------------------------------------------------------------------
class FakeBackend final : public ComputeBackend {
public:
    // The per-device shard base (global block offset b0) the orchestrator should hand
    // this backend, set by the harness before the run so the fake can recover global
    // ids from the local ones. -1 means "not expected to be called with work".
    int expected_b0 = 0;
    int throw_on_call = 0;  // if nonzero, throw std::runtime_error on compute_f2_blocks

    // Recorded observations (single-call per device on the SPMG path).
    const double* seen_q_data = nullptr;
    long  seen_M = -1;
    int   seen_n_block = -1;
    int   calls = 0;
    bool  local_id_ok = true;   // the local block_id was dense/zero-based/in-range

    [[nodiscard]] F2Result compute_f2(const MatView&, const MatView&, const MatView&,
                                      const Precision&) override {
        return {};  // not exercised by the multi-GPU core
    }

    [[nodiscard]] F2BlockTensor compute_f2_blocks(const MatView& Q, const MatView&,
                                                  const MatView&, const int* block_id,
                                                  int n_block, const Precision&) override {
        ++calls;
        seen_q_data = Q.data;
        seen_M = Q.M;
        seen_n_block = n_block;
        if (throw_on_call != 0) {
            throw std::runtime_error("FakeBackend: forced device fault");
        }

        F2BlockTensor out;
        out.P = Q.P;
        out.n_block = n_block;
        const std::size_t slab =
            static_cast<std::size_t>(Q.P) * static_cast<std::size_t>(Q.P);
        const std::size_t total =
            slab * static_cast<std::size_t>(n_block > 0 ? n_block : 0);
        out.f2.resize(total);
        out.vpair.resize(total);
        out.block_sizes.assign(static_cast<std::size_t>(n_block > 0 ? n_block : 0), 0);

        // VERIFY the local block_id is dense, zero-based, non-decreasing, in
        // [0, n_block_local) — the transform contract — and key the payload on the
        // recovered GLOBAL block id (local + expected_b0).
        int prev = 0;
        for (long k = 0; k < Q.M; ++k) {
            const int lb = block_id[k];
            if (lb < 0 || lb >= n_block) local_id_ok = false;
            if (lb < prev) local_id_ok = false;  // non-decreasing
            prev = lb;
        }
        for (int lb = 0; lb < n_block; ++lb) {
            const int gb = expected_b0 + lb;  // recovered global block id
            for (std::size_t e = 0; e < slab; ++e) {
                out.f2[slab * static_cast<std::size_t>(lb) + e] = f2_value(gb, e);
                out.vpair[slab * static_cast<std::size_t>(lb) + e] = vpair_value(gb, e);
            }
            out.block_sizes[static_cast<std::size_t>(lb)] = block_size_value(gb);
        }
        return out;
    }

    // The host-staged-DIRECT seam (M4.5 d2h-speed). Mirrors the real CUDA backend's
    // PLACEMENT — compute the SAME keyed payload as compute_f2_blocks, then write
    // (host memcpy, no CUDA, no pin — pinning is a parity-neutral perf lever) DIRECTLY
    // into the caller's shared result at the disjoint block offset slab*b0 (f2/vpair)
    // and b0 (block_sizes). The fake reuses compute_f2_blocks to build the per-device
    // partial, then copies its slabs into the destination slice — so the recordings
    // (calls/seen_q_data/seen_M/seen_n_block/local_id_ok) are populated identically.
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
};

// Build a G-device Resources by hand (Resources is a public CUDA-free aggregate): G
// FakeBackends, gpus[g] owns FakeBackend g. NO build_resources / NO GPU. Returns the
// raw FakeBackend pointers (still owned by Resources) so the harness can set per-device
// expectations and read recordings after the run.
struct FakeRig {
    Resources resources;
    std::vector<FakeBackend*> backends;
};

[[nodiscard]] FakeRig make_rig(std::size_t G) {
    FakeRig rig;
    rig.resources.gpus.reserve(G);
    rig.backends.reserve(G);
    for (std::size_t g = 0; g < G; ++g) {
        auto fb = std::make_unique<FakeBackend>();
        rig.backends.push_back(fb.get());
        PerGpuResources pg;
        pg.device_id = static_cast<int>(g);
        pg.backend = std::move(fb);
        rig.resources.gpus.push_back(std::move(pg));
    }
    return rig;
}

// A dense, non-decreasing partition: block b owns `sizes[b]` contiguous SNP columns.
[[nodiscard]] BlockPartition partition_from_sizes(const std::vector<int>& sizes) {
    BlockPartition bp;
    bp.n_block = static_cast<int>(sizes.size());
    for (int b = 0; b < bp.n_block; ++b) {
        for (int s = 0; s < sizes[static_cast<std::size_t>(b)]; ++s) {
            bp.block_id.push_back(b);
        }
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

// Reassemble + assert: drive plan + fan-out + host-staged combine, then check the full
// tensor against the independent per-global-block reference. Also assert the sub-view
// math (each device saw Q.data + P*s0 with M == s1-s0) and the dense-local-id contract.
[[nodiscard]] bool run_and_check(const char* tag, int P, const std::vector<int>& sizes,
                                 std::size_t G) {
    const BlockPartition bp = partition_from_sizes(sizes);
    const long M = static_cast<long>(bp.block_id.size());
    const int  n_block = bp.n_block;

    FakeRig rig = make_rig(G);

    // Plan first so we can set each device's expected global base b0 on its fake.
    const std::vector<DeviceShard> shards = plan_multigpu_shards(bp, M, n_block, G);
    if (shards.size() != G) {
        std::fprintf(stderr, "[%s] plan returned %zu shards, expected G=%zu\n",
                     tag, shards.size(), G);
        return false;
    }
    for (std::size_t g = 0; g < G; ++g) {
        rig.backends[g]->expected_b0 = shards[g].b0;
    }

    // Q/V/N storage: P×M, distinct values so a swapped/offset sub-view would be caught
    // by the seen_q_data pointer assertion (the fake records the base it received).
    std::vector<double> qd(static_cast<std::size_t>(P) * static_cast<std::size_t>(M), 0.0);
    for (std::size_t i = 0; i < qd.size(); ++i) qd[i] = static_cast<double>(i) + 0.5;
    const MatView Q = make_view(qd, P, M);
    const MatView Vv = make_view(qd, P, M);
    const MatView Nv = make_view(qd, P, M);
    const Precision prec{Precision::Kind::EmulatedFp64, 40};

    const std::vector<F2BlockTensor> partials = compute_multigpu_partials(
        rig.resources, Q, Vv, Nv, bp,
        std::span<const DeviceShard>(shards.data(), shards.size()), prec);
    if (partials.size() != G) {
        std::fprintf(stderr, "[%s] %zu partials, expected G=%zu\n", tag, partials.size(), G);
        return false;
    }

    // ---- 1: SUB-VIEW + LOCAL-ID MATH (per device) ----------------------------
    for (std::size_t g = 0; g < G; ++g) {
        const DeviceShard& sh = shards[g];
        const FakeBackend* fb = rig.backends[g];
        if (fb->calls != 1) {  // every device driven exactly once (incl. empty shards)
            std::fprintf(stderr, "[%s] device %zu called %d times, expected 1\n",
                         tag, g, fb->calls);
            return false;
        }
        if (!fb->local_id_ok) {
            std::fprintf(stderr, "[%s] device %zu got a non-dense/out-of-range local id\n",
                         tag, g);
            return false;
        }
        const long  M_local = sh.s1 - sh.s0;
        const int   n_block_local = sh.b1 - sh.b0;
        const double* expect_base = qd.data() + static_cast<std::size_t>(P) *
                                    static_cast<std::size_t>(sh.s0);
        if (fb->seen_q_data != expect_base) {
            std::fprintf(stderr, "[%s] device %zu sub-view base != Q.data + P*s0\n", tag, g);
            return false;
        }
        if (fb->seen_M != M_local) {
            std::fprintf(stderr, "[%s] device %zu saw M=%ld, expected s1-s0=%ld\n",
                         tag, g, fb->seen_M, M_local);
            return false;
        }
        if (fb->seen_n_block != n_block_local) {
            std::fprintf(stderr, "[%s] device %zu saw n_block=%d, expected b1-b0=%d\n",
                         tag, g, fb->seen_n_block, n_block_local);
            return false;
        }
    }

    // ---- 2: FAN-OUT + COMBINE REASSEMBLY (bit-exact vs the reference) ---------
    const F2BlockTensor full = combine_f2_partials_host(
        std::span<const F2BlockTensor>(partials.data(), partials.size()),
        std::span<const DeviceShard>(shards.data(), shards.size()), P, n_block);
    if (full.P != P || full.n_block != n_block) {
        std::fprintf(stderr, "[%s] combined P=%d n_block=%d, expected %d/%d\n",
                     tag, full.P, full.n_block, P, n_block);
        return false;
    }
    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    for (int gb = 0; gb < n_block; ++gb) {
        for (std::size_t e = 0; e < slab; ++e) {
            const std::size_t idx = slab * static_cast<std::size_t>(gb) + e;
            if (!bits_equal(full.f2[idx], f2_value(gb, e))) {
                std::fprintf(stderr, "[%s] f2 mismatch at global block %d elem %zu\n",
                             tag, gb, e);
                return false;
            }
            if (!bits_equal(full.vpair[idx], vpair_value(gb, e))) {
                std::fprintf(stderr, "[%s] vpair mismatch at global block %d elem %zu\n",
                             tag, gb, e);
                return false;
            }
        }
        if (full.block_sizes[static_cast<std::size_t>(gb)] != block_size_value(gb)) {
            std::fprintf(stderr, "[%s] block_size mismatch at global block %d\n", tag, gb);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

// G==2 even split over 6 uniform blocks: device 0 owns [0,3), device 1 [3,6); the
// combine must place each device's run at its GLOBAL offset. Pins the sub-view/local-id
// transform AND the reassembly on the common balanced case.
[[nodiscard]] bool test_two_device_even() {
    return run_and_check("G=2 even 6x4", /*P=*/3, {4, 4, 4, 4, 4, 4}, /*G=*/2);
}

// G==1 identity: the single full shard [0,n_block) over all columns; local ids == global
// ids; the lone partial reconstructs the whole tensor. The structural single-GPU
// invariance (the public entry point's G==1 fast path), proven GPU-free on the core.
[[nodiscard]] bool test_single_device_identity() {
    return run_and_check("G=1 identity 4x5", /*P=*/4, {5, 5, 5, 5}, /*G=*/1);
}

// G==4 over UNEVEN block sizes: distinct per-device shard spans + a real local-id
// shift on every device. Exercises the concurrent fan-out across 4 worker threads
// writing 4 distinct partials[g] slots (race-freedom under TSan), and the combine
// reassembling four differently-sized runs.
[[nodiscard]] bool test_four_device_uneven() {
    return run_and_check("G=4 uneven", /*P=*/2, {3, 7, 1, 5, 2, 9, 4, 6}, /*G=*/4);
}

// n_block < G: 3 blocks over G=5 — trailing devices get EMPTY shards (b0==b1 ⇒
// M_local==0/n_block_local==0). The fake returns an empty partial for them (driven once
// each, dense-local-id vacuously ok), the combine places nothing, and the non-empty
// prefix still reassembles bit-exact.
[[nodiscard]] bool test_n_block_lt_g() {
    return run_and_check("n_block<G (3 blocks / G=5)", /*P=*/3, {4, 4, 4}, /*G=*/5);
}

// Single block over G=2: device 0 owns the one block, device 1 gets an empty shard.
[[nodiscard]] bool test_single_block_two_device() {
    return run_and_check("1 block / G=2", /*P=*/2, {6}, /*G=*/2);
}

// EXCEPTION PROPAGATION: a worker whose backend throws surfaces as a normal throw out
// of compute_multigpu_partials (the exception_ptr rethrow), NOT std::terminate. Drive
// G=3 with device 1's fake set to throw; expect a std::runtime_error.
[[nodiscard]] bool test_exception_propagates() {
    const BlockPartition bp = partition_from_sizes({4, 4, 4, 4, 4, 4});
    const long M = static_cast<long>(bp.block_id.size());
    const int  n_block = bp.n_block;
    constexpr int P = 2;

    FakeRig rig = make_rig(3);
    const std::vector<DeviceShard> shards = plan_multigpu_shards(bp, M, n_block, 3);
    for (std::size_t g = 0; g < 3; ++g) rig.backends[g]->expected_b0 = shards[g].b0;
    rig.backends[1]->throw_on_call = 1;  // device 1 faults

    std::vector<double> qd(static_cast<std::size_t>(P) * static_cast<std::size_t>(M), 0.5);
    const MatView Q = make_view(qd, P, M);
    const MatView Vv = make_view(qd, P, M);
    const MatView Nv = make_view(qd, P, M);
    const Precision prec{Precision::Kind::EmulatedFp64, 40};

    try {
        const std::vector<F2BlockTensor> partials = compute_multigpu_partials(
            rig.resources, Q, Vv, Nv, bp,
            std::span<const DeviceShard>(shards.data(), shards.size()), prec);
        (void)partials;
    } catch (const std::runtime_error&) {
        return true;  // surfaced as a normal throw, not std::terminate
    } catch (...) {
        std::fprintf(stderr, "[exception propagation] wrong exception type\n");
        return false;
    }
    std::fprintf(stderr, "[exception propagation] no throw from a faulting worker\n");
    return false;
}

struct NamedTest {
    const char* name;
    bool (*fn)();
};

}  // namespace

int main() {
    const NamedTest tests[] = {
        {"G=2 even split: sub-view/local-id + reassembly", test_two_device_even},
        {"G=1 identity: single full shard, local==global", test_single_device_identity},
        {"G=4 uneven: 4-thread fan-out + reassembly", test_four_device_uneven},
        {"n_block<G: trailing empty shards, prefix exact", test_n_block_lt_g},
        {"single block / G=2: one owner + one empty shard", test_single_block_two_device},
        {"exception propagates (not std::terminate)", test_exception_propagates},
    };

    int failures = 0;
    for (const NamedTest& t : tests) {
        const bool ok = t.fn();
        std::fprintf(stderr, "[%s] %s\n", ok ? "PASS" : "FAIL", t.name);
        if (!ok) ++failures;
    }
    if (failures == 0) {
        std::fprintf(stderr, "test_f2_blocks_multigpu: ALL %zu cases passed\n",
                     sizeof(tests) / sizeof(tests[0]));
    } else {
        std::fprintf(stderr, "test_f2_blocks_multigpu: %d case(s) FAILED\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
