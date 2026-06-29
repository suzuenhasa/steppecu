I read through this carefully. This is **not slop** — it's a clean, focused CUDA TU written by someone who understands the domain constraint (bit-exact parity with the host ploidy detector) and keeps the kernel deliberately simple. A senior developer would find it competent and shippable, with a few minor cleanliness notes.

## What's genuinely good

- **Bit-exact host parity is the right goal and it's explicitly documented.** The file header explains why the device prepass must match `io::detect_sample_ploidy` (line 3-21), and the implementation sticks to integer/bit ops (`core::genotype_code`, `kHeterozygousGenotypeCode`) so there is no precision or reduction-order drift. That's careful engineering.
- **Appropriate kernel granularity.** One thread per individual is exactly the right choice here — the work per thread is tiny (`kPloidyDetectSnps` is presumably small), and anything fancier (shared-memory reduction, warps) would be over-engineered.
- **Clean launch wrapper.** `launch_detect_ploidy()` hides the `<<<>>>` syntax from host code, takes a `cudaStream_t`, guards `n_individuals == 0`, caps the detection window, and asserts the grid bound before launching (lines 76-99).
- **Coalesced output and good pointer hygiene.** The `ploidy[g] = pl` write is contiguous across threads, and both device pointers carry `__restrict__` (lines 47, 50).
- **Edge cases are thought through.** The `window == 0` comment (lines 87-90) explains why no special-case branch is needed — the kernel still writes the default pseudo-haploid value, matching the host loop's init.

## What a senior developer would flag

**Header hygiene for `STEPPE_ASSERT`:**

```cuda
STEPPE_ASSERT(grid_x >= 0 &&
                  static_cast<unsigned long long>(grid_x) <= core::kMaxGridX,
              ...);
```

`STEPPE_ASSERT` is used at line 93 but is not directly included by this TU. It likely arrives transitively via `launch_config.hpp` or `decode_af.hpp`, but relying on transitive includes for a macro you call is brittle. A senior reviewer would want either a direct `#include` of the assert header or a comment noting the dependency.

**Mixed signed/unsigned arithmetic around the grid calculation:**

```cuda
const long grid_x =
    core::cdiv(static_cast<long>(n_individuals), static_cast<long>(kPloidyBlock));
...
detect_ploidy_kernel<<<static_cast<unsigned>(grid_x), kPloidyBlock, 0, stream>>>(...);
```

It works because the assertion bounds `grid_x`, but the `long` → `unsigned long long` → `unsigned` chain is awkward. Since `gridDim.x` is `unsigned int`, it would be cleaner to express the grid directly in that type and avoid the signed intermediate.

**No validation of `bytes_per_record` against the scan window:**

The kernel computes `byte_in_rec = s / kCodesPerByte` up to `window - 1` and indexes `rec[byte_in_rec]`. If a caller passes a `bytes_per_record` smaller than `ceil(window / kCodesPerByte)`, this reads out of bounds. The host loop presumably has the same contract, but a defensive check in the launch wrapper (or a documented precondition) would make the CUDA path less fragile.

**The per-iteration division/modulo is fine but not optimal:**

```cuda
const std::size_t byte_in_rec = s / static_cast<std::size_t>(core::kCodesPerByte);
const int pos_in_byte = static_cast<int>(s % static_cast<std::size_t>(core::kCodesPerByte));
```

For a small fixed `kPloidyDetectSnps` window this is negligible, but a senior CUDA reviewer might note that you could step one byte at a time and decode all four codes per byte. That would be a genuine optimization if this kernel ever became a hot path; as written, it prioritizes literal parity with the host loop.

**Comment density is high.** The header comment is 24 lines for a 102-line file. The comments are accurate and useful, but some seniors will find them verbose. The example references to `architecture.md §13`, `§4`, and `§7` are helpful today but are the first things to go stale if the docs are reorganized.

## The "slop" test

**Not slop.** Slop is magic numbers, copy-pasted drift, missing error checks, or algorithms that only accidentally pass tests. This file has none of that. The constants are named, the logic is trivially verifiable, and the error handling (`STEPPE_CUDA_CHECK_KERNEL`, the grid bound assertion) is present.

## What it actually looks like

This looks like **solid production CUDA written by a developer who values correctness and traceability over micro-optimization.** The author clearly understood that the real requirement was "match the host detector bit-for-bit" and resisted the urge to make the kernel clever. It's the kind of code you'd be happy to see in a code review: short, scoped, well-explained, and free of obvious footguns.

A senior CUDA specialist would probably say: "Ship it — and if profiling shows this prepass matters, we can unroll the byte decode." A senior C++ person would say: "Tighten the includes and the integer types, then it's done."

## Verdict

**B+.** Correct, focused, and well-documented. Minor demerits for header hygiene and signed/unsigned type mixing. Easily A- with those cleaned up.
