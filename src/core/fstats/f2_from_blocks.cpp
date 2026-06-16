// src/core/fstats/f2_from_blocks.cpp
//
// Host orchestration of the f2 assembly (architecture.md §5 S2 "assembled by
// core/fstats/f2_from_blocks.cpp"; ROADMAP §2, M0). This is the `core`-owned
// entry point that drives the f2 computation THROUGH the `ComputeBackend` seam —
// it never issues a GEMM, allocates device memory, or includes a CUDA header
// itself (architecture.md §2, §4: `core` is pure host C++20 and reaches the GPU
// only via the CUDA-free ComputeBackend interface). The same orchestration runs
// unchanged against `CpuBackend` (the reference oracle) and `CudaBackend` (the
// GPU 3-GEMM reformulation), which is exactly the dependency-injection property
// that makes the pipeline GPU-free-testable (architecture.md §8).
//
// M0 scope: a single SNP block. The Q/V/N contract (views.hpp) for one block
// goes in; the bias-corrected f2 matrix + the retained pairwise-valid count
// (Vpair, the S4 jackknife weight) come out via F2Result. The per-block →
// [P × P × n_block] f2_blocks tensor assembly (batched over the block axis) and
// the SNP→block partition are later milestones (ROADMAP M3/M4); this is the
// host-side composition root the structure-lift needs so steppe_core is a real,
// layering-correct target rather than an empty stub.
//
// LAYERING: includes only the CUDA-free seam (device/backend.hpp), the shared
// host-pure views (core/internal/views.hpp), and the public config — no CUDA.
#include "core/fstats/f2_from_blocks.hpp"

#include "device/backend.hpp"       // steppe::ComputeBackend, steppe::F2Result
#include "core/internal/views.hpp"  // steppe::core::MatView (Q/V/N contract)
#include "steppe/config.hpp"        // steppe::Precision

namespace steppe::core {

F2Result compute_f2_block(ComputeBackend& backend, const MatView& Q, const MatView& V,
                          const MatView& N, const Precision& precision) {
    // Pure dispatch through the injected backend. The orchestration owns the
    // policy (which block, which precision); the backend owns HOW the f2 is
    // computed (scalar oracle vs 3-GEMM GPU). `core` issues no device call
    // directly — this single forwarding line IS the layering contract
    // (architecture.md §2, §5).
    return backend.compute_f2(Q, V, N, precision);
}

}  // namespace steppe::core
