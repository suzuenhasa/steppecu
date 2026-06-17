// tests/reference/test_f2_blocks_gather.cu
//
// B25 OBJECTIVE GATE — a FOCUSED gather/scatter unit test for the M4-specific
// index math (cleanup f2_blocks_kernel F24, backlog B25; architecture.md §13
// "thin kernels get a launch-and-compare test"). The M4 path's gather
// (launch_gather_group) and scatter (launch_assemble_blocks_group) index math is
// only TRANSITIVELY covered today:
//   * test_f2_blocks_equivalence.cu's `single-block == M0` arm forces all SNPs into
//     block 0 (block_ids_in_group = {0}), so it exercises ONLY the IDENTITY scatter
//     (slab k == block id == 0) and never a pad column (s_pad == the block size).
//   * test_f2_blocks_group.cu (the B24 gate) drives a reordered 2-block layout
//     end-to-end, but it diffs only the FINAL assembled f2/Vpair tensors — it does
//     NOT read back the intermediate gather slabs, so the pad-column-zero and
//     real-column-copy contracts are inferred (via "Vpair is the real size, not
//     s_pad"), not asserted directly on the slab bytes.
//
// THIS test pins the three M4-specific properties DIRECTLY, against hand-computed
// expected values, on a synthetic REORDERED multi-block layout (no real data):
//
//   (i)   PAD-COLUMN ZERO. After launch_gather_group, every pad column c >= sz of
//         every slab of dQg, dVg, and dSg (both the Qsq and the Hc half of the
//         [2P × s_pad] S slab) is EXACTLY 0.0 — the bit pattern, not "close to 0".
//         This is the contract that makes the padded GEMM identical to the unpadded
//         one (V=0 ⇒ zero rows/cols; f2_blocks_kernel.cuh:40-41).
//
//   (ii)  GATHERED COLUMN == FEEDER COLUMN. For every real column c < sz of slab k
//         (the global block block_ids_in_group[k]), the gathered slab entries equal
//         the feeder columns at the block's contiguous offset block_offsets[id]+c —
//         Qg/Vg from the [P × M] feeder, and BOTH rows of Sg (Qsq at row i, Hc at
//         row P+i) from the [2P × M] feeder — bit-for-bit (a pure memory copy, no
//         arithmetic). This pins the gather's source/destination index arithmetic
//         (i + P·c + P·s_pad·k vs i + P·src; the 2P stack for S).
//
//   (iii) SCATTER LANDS AT BLOCK id'S SLAB, NOT SLAB k. With a REORDERED
//         block_ids_in_group = {1, 0} (slab 0 -> global block 1, slab 1 -> global
//         block 0) and two blocks of DIFFERENT size (2 vs 3) hence DIFFERENT f2, the
//         assemble kernel must write each slab's result to the resident [P×P×n_block]
//         offset i + P·j + P·P·block_ids_in_group[k] — i.e. block id's slab, not the
//         group-local slab k. We assert each resident slab matches ITS OWN block's
//         oracle AND, as a discriminating NEGATIVE CONTROL, that it does NOT match
//         the OTHER block's oracle (so a bug that scattered to slab k instead of id
//         — the identity that single-block==M0 cannot distinguish — would fail).
//
// Data-free, synthetic, control-flow/index-math (NOT a precision claim — ROADMAP
// §0), so it runs on every lane with no real AADR. Native Fp64 (the gather/scatter
// index math is precision-neutral). Built by CMake/CTest as the `f2_blocks_gather`
// test (tests/CMakeLists.txt), linking steppe::device for the device-private
// wrappers + the CublasHandle/DeviceBuffer RAII, and steppe::core_internal for the
// shared host/device f2 primitive (the scatter oracle).
// Run:  ./test_f2_blocks_gather     (no data needed)
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/internal/f2_estimator.hpp"   // het_correction, assemble_f2_numerator, finalize_f2
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK
#include "device/cuda/device_buffer.cuh"    // DeviceBuffer<T> (RAII device memory)
#include "device/cuda/f2_blocks_kernel.cuh" // launch_gather_group, run_f2_gemms_group, launch_assemble_blocks_group
#include "device/cuda/handles.hpp"          // CublasHandle (RAII cuBLAS handle)

using steppe::Precision;
using steppe::core::assemble_f2_numerator;
using steppe::core::finalize_f2;
using steppe::core::het_correction;
using steppe::device::CublasHandle;
using steppe::device::DeviceBuffer;

namespace {

// ===========================================================================
// Synthetic feeder-output layout (what the gather READS), all column-major:
//   Q_all [P × M]  : masked allele freq q (the feeder output)
//   V_all [P × M]  : validity mask in {0,1}
//   S_all [2P × M] : top P rows = Qsq (= v·q²), bottom P rows = Hc (het corr)
// The gather selects each block's contiguous columns [off, off+sz) into a padded
// [P × s_pad] (Qg/Vg) / [2P × s_pad] (Sg) slab; pad columns (c >= sz) are
// zero-filled. We pick tiny, fully-determined, ALL-DISTINCT values so a copy of
// the wrong column is detectable and the host oracle reproduces every sum exactly.
// ===========================================================================
constexpr int kP = 2;       // populations
constexpr int kSpad = 4;    // bucket pad width (a power of kBlockGroupPadBase=2)
constexpr int kNblock = 2;  // resident tensor slabs

// Two blocks of DIFFERENT size: block 0 spans SNP columns [0, 2) (size 2), block 1
// spans [2, 5) (size 3). M = 5 contiguous feeder columns. Both pad to s_pad = 4, so
// block 0 has 2 pad columns and block 1 has 1 pad column.
constexpr long kM = 5;
const long kOffsets[kNblock] = {0, 2};
const int kSizes[kNblock] = {2, 3};

// Per (population i, SNP column src): a fully-determined, ALL-DISTINCT q and N. Made
// distinct across BOTH i and src so (ii) detects a transposed or off-by-one column
// (a copy of the wrong source column would change the value).
double q_at(int i, long src) {
    return 0.13 + 0.07 * static_cast<double>(i) + 0.11 * static_cast<double>(src);
}
double n_at(int /*i*/, long src) {
    return 4.0 + static_cast<double>(src);  // non-missing haploid count, >= 2
}

// The three feeder-output host arrays. Indexers mirror the gather kernel's source
// index math exactly: Q/V at i + P·src; S at i + 2P·src (Qsq) and (P+i) + 2P·src (Hc).
std::size_t qv_idx(int i, long src) {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(kP) * static_cast<std::size_t>(src);
}
std::size_t s_qsq_idx(int i, long src) {
    return static_cast<std::size_t>(i) + 2u * static_cast<std::size_t>(kP) * static_cast<std::size_t>(src);
}
std::size_t s_hc_idx(int i, long src) {
    return (static_cast<std::size_t>(kP) + static_cast<std::size_t>(i)) +
           2u * static_cast<std::size_t>(kP) * static_cast<std::size_t>(src);
}

void build_feeder(std::vector<double>& Qall, std::vector<double>& Vall,
                  std::vector<double>& Sall) {
    const std::size_t pm = static_cast<std::size_t>(kP) * static_cast<std::size_t>(kM);
    Qall.assign(pm, -7.0);       // sentinel; every real entry overwritten below
    Vall.assign(pm, -7.0);
    Sall.assign(2u * pm, -7.0);
    for (long s = 0; s < kM; ++s) {
        for (int i = 0; i < kP; ++i) {
            const double q = q_at(i, s);
            const double n = n_at(i, s);
            Qall[qv_idx(i, s)] = q;     // masked q (valid => q)
            Vall[qv_idx(i, s)] = 1.0;   // valid
            Sall[s_qsq_idx(i, s)] = q * q;  // Qsq = v·q² (v=1)
            Sall[s_hc_idx(i, s)] = het_correction(q, n, /*valid=*/true);  // Hc
        }
    }
}

// ---- The gather slab index math (the destination side, mirrored for read-back) --
//   Qg/Vg [P × s_pad × n]  : element (i, c, k) at i + P·c + P·s_pad·k
//   Sg     [2P × s_pad × n] : Qsq at i + 2P·c + 2P·s_pad·k; Hc at (P+i) + 2P·c + 2P·s_pad·k
std::size_t qg_idx(int i, int c, int k) {
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(kP) * static_cast<std::size_t>(c) +
           static_cast<std::size_t>(kP) * static_cast<std::size_t>(kSpad) * static_cast<std::size_t>(k);
}
std::size_t sg_qsq_idx(int i, int c, int k) {
    return static_cast<std::size_t>(i) +
           2u * static_cast<std::size_t>(kP) * static_cast<std::size_t>(c) +
           2u * static_cast<std::size_t>(kP) * static_cast<std::size_t>(kSpad) * static_cast<std::size_t>(k);
}
std::size_t sg_hc_idx(int i, int c, int k) {
    return (static_cast<std::size_t>(kP) + static_cast<std::size_t>(i)) +
           2u * static_cast<std::size_t>(kP) * static_cast<std::size_t>(c) +
           2u * static_cast<std::size_t>(kP) * static_cast<std::size_t>(kSpad) * static_cast<std::size_t>(k);
}

// Host oracle: f2(i,j) and Vpair(i,j) for one block (REAL columns [off, off+sz)),
// reproducing the three GEMM sums and the shared assemble/finalize primitives. The
// gather pads to s_pad with zeros, which MUST be inert (V=0 ⇒ no contribution), so
// summing only [off, off+sz) is the ground truth for (iii).
void oracle_block(long off, int sz, int i, int j, double& f2_out, double& vpair_out) {
    double G = 0.0, Vpair = 0.0, sumsq_i = 0.0, sumsq_j = 0.0, hsum_i = 0.0, hsum_j = 0.0;
    for (int c = 0; c < sz; ++c) {
        const long src = off + c;
        const double qi = q_at(i, src), qj = q_at(j, src);
        const double ni = n_at(i, src), nj = n_at(j, src);
        G += qi * qj;
        Vpair += 1.0;  // every entry valid
        sumsq_i += (qi * qi) * 1.0;
        sumsq_j += (qj * qj) * 1.0;
        hsum_i += het_correction(qi, ni, true) * 1.0;
        hsum_j += het_correction(qj, nj, true) * 1.0;
    }
    const double num = assemble_f2_numerator(sumsq_i, sumsq_j, G, hsum_i, hsum_j);
    f2_out = finalize_f2(num, Vpair);
    vpair_out = Vpair;
}

// ===========================================================================
// The whole test: ONE gather -> 3 GEMMs -> ONE scatter on a reordered 2-block
// layout, then read back (a) the gather slabs to pin (i)+(ii), and (b) the resident
// tensors to pin (iii). Returns true on PASS.
// ===========================================================================
bool run_gather_scatter() {
    std::vector<double> Qall, Vall, Sall;
    build_feeder(Qall, Vall, Sall);

    // A genuine PERMUTATION: slab 0 -> global block 1, slab 1 -> global block 0.
    // (The single-block==M0 arm can only ever use the identity {0}.)
    const int n_in_group = kNblock;
    const std::vector<int> ids_in_group = {1, 0};

    // ---- Upload feeder outputs + per-block metadata -----------------------
    const std::size_t pm = static_cast<std::size_t>(kP) * static_cast<std::size_t>(kM);
    DeviceBuffer<double> dQall(pm), dVall(pm), dSall(2u * pm);
    STEPPE_CUDA_CHECK(cudaMemcpy(dQall.data(), Qall.data(), pm * sizeof(double), cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(dVall.data(), Vall.data(), pm * sizeof(double), cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(dSall.data(), Sall.data(), 2u * pm * sizeof(double), cudaMemcpyHostToDevice));

    DeviceBuffer<int> dIds(static_cast<std::size_t>(n_in_group));
    DeviceBuffer<long> dOff(static_cast<std::size_t>(kNblock));
    DeviceBuffer<int> dSizes(static_cast<std::size_t>(kNblock));
    STEPPE_CUDA_CHECK(cudaMemcpy(dIds.data(), ids_in_group.data(),
                                 static_cast<std::size_t>(n_in_group) * sizeof(int), cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(dOff.data(), kOffsets, sizeof(kOffsets), cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(dSizes.data(), kSizes, sizeof(kSizes), cudaMemcpyHostToDevice));

    // ---- Group slabs: dQg/dVg [P×s_pad×n], dSg [2P×s_pad×n] ----------------
    // Pre-fill the slabs with a NONZERO sentinel so the gather's pad-column zero-fill
    // (i) is a real write the kernel must perform — not a residue of zero-init memory.
    const std::size_t psp_n = static_cast<std::size_t>(kP) * static_cast<std::size_t>(kSpad) *
                              static_cast<std::size_t>(n_in_group);
    DeviceBuffer<double> dQg(psp_n), dVg(psp_n), dSg(2u * psp_n);
    {
        const std::vector<double> sentinelQ(psp_n, 99.0);
        const std::vector<double> sentinelS(2u * psp_n, 99.0);
        STEPPE_CUDA_CHECK(cudaMemcpy(dQg.data(), sentinelQ.data(), psp_n * sizeof(double), cudaMemcpyHostToDevice));
        STEPPE_CUDA_CHECK(cudaMemcpy(dVg.data(), sentinelQ.data(), psp_n * sizeof(double), cudaMemcpyHostToDevice));
        STEPPE_CUDA_CHECK(cudaMemcpy(dSg.data(), sentinelS.data(), 2u * psp_n * sizeof(double), cudaMemcpyHostToDevice));
    }
    const std::size_t pp = static_cast<std::size_t>(kP) * static_cast<std::size_t>(kP);
    const std::size_t pp_n = pp * static_cast<std::size_t>(n_in_group);
    DeviceBuffer<double> dGg(pp_n), dVpairg(pp_n), dRg(2u * pp_n);

    // ---- Resident [P×P×n_block] f2 + Vpair tensors ------------------------
    const std::size_t total = pp * static_cast<std::size_t>(kNblock);
    DeviceBuffer<double> dF2all(total), dVpairAll(total);
    STEPPE_CUDA_CHECK(cudaMemset(dF2all.data(), 0, total * sizeof(double)));
    STEPPE_CUDA_CHECK(cudaMemset(dVpairAll.data(), 0, total * sizeof(double)));

    // ---- The handle: stream + workspace bound once (the §12 invariant) -----
    // Native Fp64 here, so the emulated-FP64 workspace is not load-bearing, but bind
    // one to mirror the production handle setup (CublasHandle owns (stream, workspace)).
    CublasHandle blas;
    DeviceBuffer<unsigned char> dWs(64u * 1024u * 1024u);  // a real 64 MiB workspace
    blas.set_workspace(dWs.data(), dWs.bytes());
    blas.set_stream(/*default stream*/ nullptr);

    const Precision prec{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};

    // ---- (A) The gather under test -----------------------------------------
    steppe::device::launch_gather_group(dQall.data(), dVall.data(), dSall.data(),
                                        dIds.data(), dOff.data(), dSizes.data(),
                                        kP, kSpad, n_in_group,
                                        dQg.data(), dVg.data(), dSg.data(), /*stream=*/nullptr);
    STEPPE_CUDA_CHECK(cudaDeviceSynchronize());

    // Read back the gather slabs and check (i) pad-zero + (ii) real-column-copy.
    std::vector<double> Qg(psp_n), Vg(psp_n), Sg(2u * psp_n);
    STEPPE_CUDA_CHECK(cudaMemcpy(Qg.data(), dQg.data(), psp_n * sizeof(double), cudaMemcpyDeviceToHost));
    STEPPE_CUDA_CHECK(cudaMemcpy(Vg.data(), dVg.data(), psp_n * sizeof(double), cudaMemcpyDeviceToHost));
    STEPPE_CUDA_CHECK(cudaMemcpy(Sg.data(), dSg.data(), 2u * psp_n * sizeof(double), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int k = 0; k < n_in_group; ++k) {
        const int id = ids_in_group[k];
        const long off = kOffsets[id];
        const int sz = kSizes[id];
        for (int c = 0; c < kSpad; ++c) {
            for (int i = 0; i < kP; ++i) {
                const double gq = Qg[qg_idx(i, c, k)];
                const double gv = Vg[qg_idx(i, c, k)];
                const double gqsq = Sg[sg_qsq_idx(i, c, k)];
                const double ghc = Sg[sg_hc_idx(i, c, k)];
                if (c >= sz) {
                    // (i) PAD COLUMN — exactly 0.0 (bit pattern) in Q, V, and BOTH S rows.
                    if (gq != 0.0 || gv != 0.0 || gqsq != 0.0 || ghc != 0.0) {
                        std::fprintf(stderr,
                                     "  [FAIL] (i) pad column NOT zero: slab k=%d (id=%d) c=%d i=%d "
                                     "-> Qg=%.17g Vg=%.17g Sg.qsq=%.17g Sg.hc=%.17g (expected 0)\n",
                                     k, id, c, i, gq, gv, gqsq, ghc);
                        ok = false;
                    }
                } else {
                    // (ii) REAL COLUMN — bit-for-bit equal to the feeder column at off+c.
                    const long src = off + c;
                    const double eq = Qall[qv_idx(i, src)];
                    const double ev = Vall[qv_idx(i, src)];
                    const double eqsq = Sall[s_qsq_idx(i, src)];
                    const double ehc = Sall[s_hc_idx(i, src)];
                    if (gq != eq || gv != ev || gqsq != eqsq || ghc != ehc) {
                        std::fprintf(stderr,
                                     "  [FAIL] (ii) gathered != feeder: slab k=%d (id=%d, src=%ld) c=%d i=%d\n"
                                     "         Qg=%.17g feeder=%.17g | Vg=%.17g feeder=%.17g\n"
                                     "         Sg.qsq=%.17g feeder=%.17g | Sg.hc=%.17g feeder=%.17g\n",
                                     k, id, src, c, i, gq, eq, gv, ev, gqsq, eqsq, ghc, ehc);
                        ok = false;
                    }
                }
            }
        }
    }
    std::printf("  (i) pad-column-zero + (ii) gathered==feeder (reordered ids={1,0}, "
                "sizes {2,3} padded to %d) -> %s\n", kSpad, ok ? "PASS" : "FAIL");

    // ---- (B) The GEMMs + the scatter under test ----------------------------
    steppe::device::run_f2_gemms_group(blas.get(), prec, kP, kSpad, n_in_group,
                                       dQg.data(), dVg.data(), dSg.data(),
                                       dGg.data(), dVpairg.data(), dRg.data());
    steppe::device::launch_assemble_blocks_group(dGg.data(), dVpairg.data(), dRg.data(),
                                                 dIds.data(), kP, n_in_group,
                                                 dF2all.data(), dVpairAll.data(), /*stream=*/nullptr);
    STEPPE_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> f2(total), vpair(total);
    STEPPE_CUDA_CHECK(cudaMemcpy(f2.data(), dF2all.data(), total * sizeof(double), cudaMemcpyDeviceToHost));
    STEPPE_CUDA_CHECK(cudaMemcpy(vpair.data(), dVpairAll.data(), total * sizeof(double), cudaMemcpyDeviceToHost));

    // (iii) The scatter must land each slab at block id'S resident offset (not slab
    // k's). The two blocks differ in size (2 vs 3) so their f2/Vpair DIFFER; a bug
    // that wrote to slab k instead of id would swap the two blocks' results. Assert
    // each resident slab b == block-b oracle AND (negative control) that it is NOT
    // the OTHER block's value — proving the test discriminates the id-vs-k offset.
    const double tol = 1e-12;  // exact-arithmetic synthetic; GEMM reorder is far below this
    bool scatter_ok = true;
    for (int b = 0; b < kNblock; ++b) {
        const int other = 1 - b;  // the only other block (kNblock == 2)
        for (int j = 0; j < kP; ++j) {
            for (int i = 0; i < kP; ++i) {
                double ef2 = 0.0, evp = 0.0;
                oracle_block(kOffsets[b], kSizes[b], i, j, ef2, evp);
                double of2 = 0.0, ovp = 0.0;  // the OTHER block's oracle (must NOT match)
                oracle_block(kOffsets[other], kSizes[other], i, j, of2, ovp);

                const std::size_t idx = static_cast<std::size_t>(i) +
                                        static_cast<std::size_t>(kP) * static_cast<std::size_t>(j) +
                                        pp * static_cast<std::size_t>(b);
                const double gf2 = f2[idx], gvp = vpair[idx];

                if (std::fabs(gf2 - ef2) > tol || std::fabs(gvp - evp) > tol) {
                    std::fprintf(stderr,
                                 "  [FAIL] (iii) slab %d (i=%d,j=%d) != its own oracle: "
                                 "f2 gpu=%.15g want=%.15g | vpair gpu=%.15g want=%.15g\n",
                                 b, i, j, gf2, ef2, gvp, evp);
                    scatter_ok = false;
                }
                // Negative control: the resident slab must NOT be the OTHER block's
                // result (Vpair 2 vs 3 always differs, so the offset is genuinely
                // discriminated — a slab-k scatter would land the WRONG block here).
                if (std::fabs(gvp - ovp) <= tol && std::fabs(evp - ovp) > tol) {
                    std::fprintf(stderr,
                                 "  [FAIL] (iii) slab %d holds the OTHER block's Vpair (%.15g) — "
                                 "scatter used slab-k offset, not block-id\n", b, gvp);
                    scatter_ok = false;
                }
            }
        }
    }
    std::printf("  (iii) scatter lands block-id slab at i + P·j + P·P·id (reordered, "
                "sizes 2 vs 3 discriminate) -> %s\n", scatter_ok ? "PASS" : "FAIL");

    return ok && scatter_ok;
}

}  // namespace

int main() {
    // Fail fast (not a silent skip) if there is no usable device: this is a GPU
    // gate. The box always has a device.
    int dev_count = 0;
    const cudaError_t derr = cudaGetDeviceCount(&dev_count);
    if (derr != cudaSuccess || dev_count < 1) {
        std::fprintf(stderr, "RESULT: FAIL — no CUDA device available (%s); this is a GPU gate.\n",
                     cudaGetErrorString(derr));
        return EXIT_FAILURE;
    }

    std::printf("\nB25 M4 gather/scatter focused index-math test (synthetic, no data)\n");

    const bool ok = run_gather_scatter();

    std::printf("\n");
    if (!ok) {
        std::fprintf(stderr,
            "RESULT: FAIL — the M4 gather did not zero pad columns / copy the feeder\n"
            "        columns bit-for-bit, or the scatter did not land each slab at its\n"
            "        block-id [P×P] offset (architecture.md §13; cleanup f2_blocks_kernel\n"
            "        F24, B25).\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "RESULT: PASS (gather pad-zero + feeder-copy, scatter to block-id "
                         "slab — the M4 index math directly verified)\n");
    return EXIT_SUCCESS;
}
