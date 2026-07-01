# The OPT-IN device-resident P2P f2 combine (CUDA-free declaration)

> Source: `src/device/p2p_combine.hpp` — the CUDA-free DECLARATION of the single-node
> multi-GPU (SPMG) fast-path combine. Cross-refs: architecture.md §11.4 (capability-tiered
> combine), §12 (parity law). See also `docs/design/multigpu-combine-gate.md`.

The OPT-IN device-resident P2P f2 combine — DECLARATION (CUDA-FREE) of the single-node
multi-GPU (SPMG) fast-path combine (architecture.md §11.4 "Capability-tiered combine ...
GPU 0 pulls each peer's partial via `cudaMemcpyPeer` (a byte-exact DMA copy) and PLACES
them in the same fixed `g = 0..G-1` order on-device — BIT-IDENTICAL to the host-staged
combine and the single-GPU reference", §12 PARITY LAW; design §4, §5). GPU 0 (the combine
root, `gpus[0]`) pulls each peer device's COMPACT partial via `cudaMemcpyPeer` (MEASURED
55.6 GB/s, `canAccessPeer==1` both directions on rtxbox) and PLACES it on-device into its
disjoint slice in the SAME FIXED `g=0..G-1` device order the host-staged baseline uses.

## CUDA-FREE BY CONTRACT

CUDA-FREE BY CONTRACT, exactly like `device/backend_factory.hpp`: it names only the public
CUDA-free `F2BlockTensor` (the host result it returns) and the CUDA-free `DeviceShard` plan
+ std types — NO `<cuda_runtime.h>`. This split (CUDA-free decl here, CUDA definition in
`cuda/p2p_combine.cu`) is what lets the CUDA-free core entry point
`compute_f2_blocks_multigpu` (steppe::core, src/core/fstats) reach the device-resident
combine WITHOUT pulling the CUDA toolkit into steppe_core — the same pattern
`make_cuda_backend` uses (a CUDA-free factory decl in `backend_factory.hpp`, the `new
CudaBackend` body in `cuda_backend.cu`; design §5 "Rename note"). The `.cu` definition
`#include`s this CUDA-free decl, mirroring `cuda_backend.cu` including
`backend_factory.hpp`.

## WHY THIS IS BIT-IDENTICAL TO THE HOST-STAGED BASELINE (architecture.md §11.4, §12)

The transport (`cudaMemcpyPeer`) only MOVES BYTES — the exact f2/vpair doubles each
device's `compute_f2_blocks` produced — and the on-device step PLACES each partial in the
SAME FIXED `g=0..G-1` ORDER into its disjoint result slice (a raw byte copy, each slab
written exactly once — NO sum, NO `cudaMemset`; see the `combine_f2_partials_resident`
doc-comment), the identical fixed-order placement `combine_f2_partials_host` performs on
the host. "the transport only moves bytes; software fixes the order" — so the two combine
tiers are parity-NEUTRAL siblings, and both equal the single-GPU reference. NEVER an NCCL
AllReduce (its order varies with G; §12).
