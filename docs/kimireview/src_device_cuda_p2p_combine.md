I read through `src/device/cuda/p2p_combine.cu` carefully. This is **not slop**, but it's also not the polished, single-responsibility translation unit a senior would hold up as showcase code. The core idea is sound, the ownership model is mostly right, and the parity reasoning is careful — but the file is weighed down by copy-pasted scaffolding, an under-validated placement contract, and comment bloat that verges on self-justification.

## What's genuinely good

- **The architecture is basically correct.** Keeping the CUDA-free declaration in `device/p2p_combine.hpp` and the CUDA definition here is the right seam; it keeps `<cuda_runtime.h>` out of `steppe_core` and mirrors the `backend_factory`/`cuda_backend` split.
- **RAII ownership is used well.** `DeviceBuffer<double>` owns the result buffers, `Stream` owns the non-blocking root stream, `RegisteredHostRegion` pins the D2H destinations, and the `DeviceGuard` restores the previous CUDA device. There are no raw `cudaMalloc`/`cudaFree` pairs leaking out of this file.
- **Error handling is consistent.** Everything that should throw (`STEPPE_CUDA_CHECK`) does, and the recoverable peer-enable "already enabled" case is handled through the non-throwing `STEPPE_CUDA_WARN` with an explicit sticky-error clear. That's the correct CUDA idiom.
- **The P2P strategy is the right one.** One `cudaMemcpyPeerAsync` per peer into a disjoint slice, one final D2H, no host staging, no zero-and-accumulate. The bit-identity argument in the header comment is over-long but fundamentally correct: a byte copy into a disjoint tile preserves `-0.0` and avoids the reduction-order nondeterminism of an AllReduce.
- **Validation happens before allocation.** `validate_resident_partials` is called before any `cudaMalloc`, which is the fail-fast order you want.

## What a senior developer would flag

**Massive duplication between the two public entry points.**

`combine_f2_partials_resident` (lines 142–238) and `combine_f2_partials_resident_device` (lines 249–318) repeat the same validation, device guard, stream creation, slab/total math, `place_partials_into` call, and final sync. The file loudly notes that it "deduplicated" the placement loop, but it left the rest of the scaffolding copy-pasted. A senior reviewer would ask why these two aren't implemented as thin wrappers around a shared helper that does everything except the final D2H step.

**The `DeviceGuard` is defined identically inside both functions.**

```cpp
// lines 163–178
struct DeviceGuard {
    int dev;
    explicit DeviceGuard(int d) noexcept : dev(d) {}
    ~DeviceGuard() { (void)STEPPE_CUDA_WARN(cudaSetDevice(dev)); }
    DeviceGuard(const DeviceGuard&) = delete;
    ...
} restore{prev_device};
```

And again at lines 262–277, with the same multi-line C++20 standard comment. This is a classic copy-paste drift hazard: if someone fixes a bug in one guard, the other silently diverges. This should be a small reusable RAII type in a device utility header.

**The placement loop trusts `b0` offsets without a bounds check.**

```cpp
// lines 96–107
for (int lb = 0; lb < part.n_block_local; ++lb) {
    out_block_sizes[static_cast<std::size_t>(part.b0 + lb)] =
        part.block_sizes[static_cast<std::size_t>(lb)];
}
...
const std::size_t dst_off = slab * static_cast<std::size_t>(part.b0);
double* dst_f2 = dst_f2_base + dst_off;
```

`validate_resident_partials` checks that `part.b0 == shards[g].b0` and that the shards tile `[0, n_block_full)` by total span, but it does **not** verify that each `b0` is non-negative, that `b0 + n_block_local <= n_block_full`, or that the shards are disjoint in offset order. A malformed plan could write out of bounds on the root result buffer. The validator is the right place to close that gap.

**The size computation can wrap before `DeviceBuffer`'s overflow guard sees it.**

```cpp
// lines 187–190
const std::size_t slab =
    static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
const std::size_t total = slab * static_cast<std::size_t>(n_block_full);
```

`P` and `n_block_full` are `int`. The casts prevent signed overflow, but the unsigned `size_t` product can still wrap. `DeviceBuffer` then checks `total * sizeof(T)`, but `total` is already wrapped, so the guard is looking at the wrong value. Same issue in `place_partials_into` at line 102 (`part_elems = slab * n_block_local`). In practice genomics sizes won't hit this, but a senior would flag that the overflow guard is only as good as the number fed into it.

**Over-commenting and repeated prose.**

The file opens with a 40-line comment block, the placement helper carries a 20-line doc comment, and even the local `DeviceGuard` gets a paragraph citing P1008R1. Some of this is valuable (the parity argument, the peer-access rationale), but a lot of it is restating what the code already says or what belongs in `architecture.md`. The repeated "bit-identical / −0.0 / NO place-add" refrain starts to read like a defense brief rather than code.

**Minor const-correctness nit.**

`place_partials_into` takes `std::span<DevicePartial>` and binds a non-const `DevicePartial&` at line 91, but it only reads from the partial. It mirrors the non-const span in the public API, which itself doesn't modify the partials either. A senior would suggest taking `std::span<const DevicePartial>` through the call chain.

## The "slop" test

**Not slop.** Slop is magic numbers, unchecked errors, copy-paste drift with stale comments, or obviously wrong algorithms. This file has none of those at the headline level. The algorithms are correct, the errors are checked, and the comments — though excessive — are accurate. The copy-pasted scaffolding is a maintainability problem, not a sloppiness problem.

## What it actually looks like

This looks like **competent, defensive CUDA plumbing written by someone who understands the parity and P2P constraints but hasn't fully internalized the "don't repeat yourself" discipline.** The core P2P combine is well-reasoned and uses the right ownership primitives. The surrounding code, however, is two near-identical functions held together by a very long comment blanket. A senior CUDA engineer would trust the math and the DMA pattern but would immediately ask for the duplicated guard and entry-point scaffolding to be collapsed.

## Verdict

**B+** — correct, safe, and production-viable, but the duplicated `DeviceGuard` and near-cloned entry points keep it out of A-tier. Refactor the shared scaffolding into one helper and tighten the validator's offset bounds, and it becomes showcase-clean.
