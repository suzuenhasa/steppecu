I read through this carefully. This is **not slop** — the header is clean, the API intent is clear, and the documentation shows serious domain expertise. But a senior C++ reviewer would have a mixed reaction: the public interface is well-shaped, yet it leaks implementation details like a design memo and carries a few API warts that would need cleanup before shipping.

## What's genuinely good

- **The interface is CUDA-free and minimally dependent.** Lines 42-55 include only `<string>`, `<vector>`, and two project headers, then forward-declare `device::Resources`. For a GPU-accelerated project, keeping the public header free of CUDA headers is exactly the right seam.
- **`[[nodiscard]]` on `run_dates`** (line 118) is a nice touch — the result is the whole point of the call, and the compiler should warn if it's discarded.
- **Degenerate outcomes are surfaced as `Status`, not exceptions** (lines 98-100, 116-117). This matches a sane error model for numerical tools: I/O or contract faults throw, but "the data don't support a date" is a result.
- **Reproducibility signals are everywhere.** Default values are explicitly tied to the reference `par.dates` (lines 60-77), and the top comment pins the upstream version, exact source-file line ranges, and a golden test value (lines 29-35). That's the habit of someone who actually validates against a reference implementation.
- **The math write-up is competent.** The exponential-decay model, the ALDER FFT autocorrelation trick, and the `v12/sqrt(v11*v22)` normalization (lines 9-27) are all correct and clearly stated.

## What a senior developer would flag

**The header is doing double duty as a design document.** The first 41 lines explain the internal algorithm in detail — grid scatter, cuFFT batched autocorrelation, `IFFT(|FFT|^2)`, the `ComputeBackend` seam, etc. That's valuable, but most of it belongs in `docs/design/` or next to the implementation, not in a public header that other translation units will parse on every build. A public header should state *what* the caller can rely on; this one explains *how* the engine works. It's likely to drift out of sync with the `.cpp` and `.cu` files.

**Mixed distance units in `DatesOptions` are a footgun.** Lines 61-70 use Morgans for `binsize_morgans`, `maxdis_morgans`, and the grid spacing, but `lovalfit_cm` is in centiMorgans:

```cpp
    double binsize_morgans = 0.001;
    double maxdis_morgans = 1.0;
    double lovalfit_cm = 0.45;
```

The comment explains the conversion, but an API that silently mixes units invites bugs. A senior reviewer would push for one unit throughout, or a strongly-typed wrapper (`Morgans`, `CentiMorgans`) so the compiler catches unit errors.

**Platform-sensitive type for the seed.** Line 76:

```cpp
    long seed = 77;
```

`long` is 32-bit on Windows and 64-bit on Linux/macOS. If the seed is meant to pin "bit-comparability with the reference" (line 74), `std::int64_t` is the safer choice. As written, the same source code can produce different integer widths depending on the host toolchain.

**The error contract is internally inconsistent in the comment.** Line 116 says an I/O fault "PROPAGATES as an exception for the app to map to a nonzero exit," but the function returns `DatesResult` with a `Status`. So the caller must handle *both* exceptions and `Status` outcomes. That's a reasonable design, but the header should make the boundary explicit: which failures throw and which populate `result.status`?

**`Precision::Kind` field is redundant today.** Line 103:

```cpp
    Precision::Kind precision_tag = Precision::Kind::Fp64;
```

The comment immediately above it says the arithmetic "is always Fp64." If it's always Fp64, the field is dead weight in this struct; if it's there for future expansion, the comment is already misleading.

**Upstream line-number references will go stale.** Lines 29-35 cite `dates.c:604`, `:620-630`, `:655-665`, etc., plus "Version 750; built on box5090." That kind of provenance is gold for parity debugging, but it's fragile in a header — the next upstream release or reformat will silently invalidate the line numbers. Better kept in a separate `DATES-parity.md` note.

## The "slop" test

**Not slop.** There's no copy-pasted code with stale comments, no unexplained magic numbers, no obviously wrong algorithm, and no missing error handling. The comments are dense, but they're mostly accurate and explain *why*, not just *what*.

## What it actually looks like

This looks like a **public API written by a domain expert who cares deeply about reproducing a published method.** The interface itself is sound: a clear options struct, a result struct, and one entry point. But the header file is being asked to carry too much — it's part interface contract, part scientific notebook, part build/environment audit trail. A senior C++ reviewer would say: "Good bones, but trim the public header to the contract and move the algorithmic prose to the implementation or design docs."

## Verdict

**B+.** Clean, competent, and clearly maintained by someone who understands the science. Points off for leaking implementation detail into the public header and for the unit/type nits that a production API should have caught.
