Review: `src/device/cuda/transpose_canonical_kernel.cu`

I read this carefully. It is **not slop** — it is a deliberately correct, well-explained CUDA transpose/gather that mirrors the host format-reader contract. A senior reviewer would respect the correctness work, but would immediately flag the **memory-access pattern** and a couple of API-extensibility/style issues.

## What's genuinely good

- **The one-thread-per-output-byte design is the right call.** Lines 77–81 and 119 make the race-freedom obvious: each `(gathered individual g, output byte b)` pair owns exactly one output byte, so no atomics, no partial-bit ORs, no synchronization.
- **Bit-exact parity with the host reader.** The kernel pulls in `core::genotype_code`, `core::kCodesPerByte`, and `core::kBitsPerCode` (lines 47–48, 96–98, 111–112), so the bit-extraction cannot silently diverge from the CPU oracle or `decode_af`.
- **The encoding hook is forward-looking.** `apply_encoding` (lines 61–68) is a switch on `TransposeEncoding` even though only `Identity` exists today, which is the correct extension point for EIGENSTRAT/PLINK later.
- **A narrow launch wrapper.** The `<<< >>>` is isolated in one place (lines 139–143), there is an explicit empty-tile early return (line 132), and a debug assertion catches an over-sized grid (lines 135–138).
- **The header comment is a real correctness document.** Lines 9–39 explain the `rlen`-floor padding, the gather/reorder mapping, and the precision-policy exception (no FP64 emulation for format readers). This is the kind of context that saves the next maintainer hours.

## What a senior developer would flag

**The global-memory read pattern is almost certainly not coalesced.**

```cuda
const std::size_t g = tid / out_bytes_per_record;
const std::size_t b = tid % out_bytes_per_record;
...
const std::uint8_t src_byte =
    snp_major[s * src_bytes_per_record + src_byte_in_snp];
```

Consecutive threads differ in `b` (the output byte), so they read SNP records `s = 4b` that are `src_bytes_per_record` bytes apart (lines 89–90, 109–110). The *writes* to `out` are coalesced, but the *reads* from `snp_major` are strided across the warp. For typical genomics tile sizes this is a bandwidth disaster. A senior CUDA reviewer would expect either a block-level shared-memory transpose or a thread mapping that keeps the loads SNP-contiguous.

**`apply_encoding` silently returns identity for any unknown enum value.**

```cuda
switch (enc) {
    case TransposeEncoding::Identity:
    default:
        return code;
}
```

Lines 63–66 fold the only valid case and `default` together. That means when someone adds `TransposeEncoding::Eigenstrat` or `Plink` and forgets to update this switch, the kernel will **produce wrong output without a compile warning**. A senior dev would drop the `default` so `-Wswitch` catches the missing enumerator, or put an explicit `assert`/`unreachable` path for unexpected values.

**The launch grid math goes through a signed `long` narrowing path.**

```cuda
const long grid_x =
    core::cdiv(static_cast<long>(total), static_cast<long>(kTransposeBlock));
STEPPE_ASSERT(grid_x >= 0 &&
              static_cast<unsigned long long>(grid_x) <= core::kMaxGridX, ...);
```

Lines 133–138 cast `total` (a `std::size_t`) to `long` before computing the grid. If `total` ever exceeds `LONG_MAX`, that cast is implementation-defined/undefined and the assertion is checking a corrupted value. A cleaner launch wrapper would keep the arithmetic unsigned and fail-fast on the `size_t` bound before any narrowing cast.

**`STEPPE_CUDA_CHECK_KERNEL()` is stream-oblivious and, in debug, device-wide.**

Lines 139–143 use the project-wide macro. As defined in `check.cuh`, it calls `cudaGetLastError()` plus a debug-only `cudaDeviceSynchronize()`. That is fine for the default stream, but if this wrapper is ever called on a non-blocking stream, the debug sync serializes the whole device. It is the project convention, so it is not a local bug, but a senior reviewer would note the implicit contract.

**The comments are sometimes denser than the code they describe.**

Lines 100–118, for example, spend five lines explaining that packing four 2-bit codes into a byte uses shifts of `6, 4, 2, 0`. The intent is good, but the signal-to-noise ratio is low. In a job-application review, over-commenting can read as compensation rather than clarity.

## The "slop" test

**Not slop.** It has no unexplained magic numbers, no stale copy-pasted comments, no missing error checking, and no obviously-wrong algorithm. The comments, while verbose, are accurate and explain *why* decisions were made, not just *what* the code does. The correctness argument is coherent and the bit-exact host parity is intentional.

## What it actually looks like

This looks like **solid research-engineering CUDA written by someone who understands the genomics format-reader problem cold.** The author clearly gets the domain invariants (SNP-major vs individual-major, `rlen` padding, selection/reorder, MSB-first packing) and knows enough CUDA to avoid the obvious correctness pitfalls. What it does *not* look like is code written by a GPU-performance specialist: the strided read pattern is the kind of thing a senior CUDA engineer fixes in the next iteration. It would pass a correctness review with minor edits, but it would not pass a performance review without reworking the memory access pattern.

## Verdict

**B+ / A- if the coalescing is addressed.**

**Bottom line:** A strong, correct showcase of domain-aware CUDA — but the strided gather reads are a real performance footgun that would embarrass the author if this kernel is meant to demonstrate GPU expertise.
