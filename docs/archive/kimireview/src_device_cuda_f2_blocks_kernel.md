I read through `f2_blocks_kernel.cu` and `kimiexample.md`. This is a solid piece of CUDA work, not slop, but it has some senior-dev eyebrow-raisers — mostly around integer-type discipline and portability.

## What's genuinely good

- **The grouped-batched GEMM design is well-motivated.** The header comment (lines 9–17) gives measured speedups (1.9× over per-block loop, 7.2×/8.9× over native FP64), explains the VRAM ceiling that killed the naive global approach, and justifies the bucket trade-off. That's competence, not cargo-culting.
- **Shared precision policy with the CPU path.** Lines 47, 264, and the use of `core::assemble_f2_numerator` / `core::finalize_f2` mean the GPU and oracle can't drift on the cancellation formula. That's the right way to keep a numeric kernel honest.
- **CUDA error checking is consistent and appropriate.** `STEPPE_CUDA_CHECK_KERNEL`, `CUBLAS_CHECK`, and the `grid_z_extent` / `grid_for` centralization (lines 46, 225–227, 307–310) show the author knows not to scatter ad-hoc `cudaGetLastError` calls everywhere.
- **The assemble kernel coalescing analysis is honest.** Lines 134–150 explicitly call out the uncoalesced transpose, explain why it's mathematically required, and state the accepted cost. A junior would have hidden that.
- **`__launch_bounds__` matches the actual launch configuration** (lines 73, 155) and the comments correctly note why it won't under-launch.

## What a senior developer would flag

**Mixed integer-width types for indexing — this is the biggest portability gotcha.**

The gather kernel uses `long` for strides:

```cuda
const long Pl = static_cast<long>(P);          // line 89
const long dQidx = static_cast<long>(i) + Pl * c + Psp * k;  // line 98
```

The assemble kernel uses `size_t`:

```cuda
const size_t Pz = static_cast<size_t>(P);      // line 168
const size_t pp_off = si + sj * Pz;            // line 175
```

And `run_f2_gemms_group` uses `long` again for cuBLAS strides:

```cuda
const long Psp = static_cast<long>(P) * s_pad;  // line 268
```

`cublasGemmStridedBatchedEx` takes `long long int` for strides. On Linux LP64, `long` happens to be 64-bit, but on Windows/LLP64 it is 32-bit. A senior dev would flag this as a latent overflow/ABI issue and demand `long long` (or `int64_t`) throughout the cuBLAS interface. It's the kind of bug that passes tests on the dev box and explodes on a Windows build or with large P.

**The `twoP` computation is `int`, not `int64_t`:**

```cuda
const int twoP = kF2StackedBlocks * P;  // line 267
```

P is a population count so this likely won't overflow, but it's inconsistent with the rest of the index math and with the `long twoP` used in the gather kernel (line 90). Pick a width and stick to it.

**The "caller already set the stream" assumption is correct but fragile.**

```cuda
// No cublasSetStream here: the handle's stream + emulated-FP64 workspace are
// bound ONCE at backend construction via CublasHandle::set_stream ...  // lines 255–263
```

The comment is right, but `run_f2_gemms_group` has no stream parameter and no runtime guard. If someone calls it from a different backend, the mistake is silent. A senior would want either a stream argument or a `STEPPE_ASSERT` that the handle's stream matches expectations.

**The `__launch_bounds__` expression is correct but indirectly computed:**

```cuda
__launch_bounds__(steppe::kCdivBlock * steppe::kCdivBlock)  // lines 73, 155
```

This evaluates to 256. Fine, but `__launch_bounds__(256)` would be clearer and harder to break if someone changes `kCdivBlock` independently.

**Comment density is high even by this codebase's standards.** The 20-line coalescing justification in the assemble kernel (lines 134–150) is good engineering archaeology, but a senior skimming the file would mutter "compensating for something?" It's accurate, just verbose.

**Internal cleanup-ticket references (F6/B24, X-7/B6, 20.1/MED)** are useful if you live in the issue tracker, but they're opaque to an outside reviewer. That's not a code bug, it's a "this file will age poorly" concern.

## The "slop" test

**Not slop.** There are no unexplained magic numbers, no copy-pasted kernels with stale comments, no missing error checks, and no hand-wavy numeric shortcuts. The comments explain *why*, not just *what*. The math is tied to a CPU oracle. Even the warts are the kind you get from someone thinking carefully, not from someone phoning it in.

## What it actually looks like

This looks like **competent research-engineering CUDA written by someone who understands the genomics algorithm and knows enough GPU programming to avoid the classic pitfalls** — coalescing is considered, shared memory isn't abused, batched GEMMs are used where they belong, and the precision policy is factored out. It's not the work of a career GPU performance engineer (the integer-type discipline and launch-occupancy tuning aren't razor-sharp), but it's also not a naive port. It reads like code that has been through a real cleanup pass and is meant to last.

## Verdict

**B+, ship after tightening integer-type portability.** Fix the `long` vs `long long` vs `size_t` inconsistency in the cuBLAS call paths, make `twoP` consistently 64-bit, and consider passing or asserting the stream in `run_f2_gemms_group`. After that, it's production-worthy.