I read through this carefully. This is a **clean, well-scoped public header** — clearly written by someone thinking hard about API layering — but a senior developer would flag a few design tensions and some over-declared surface area.

## What's genuinely good

- **The architectural intent is explicit and correct.** The comments clearly justify why this CUDA-free library seam exists (binding contract, exception-based errors, value returns) and how it relates to the CLI wrapper. That kind of "why" documentation in a public header is rarer than it should be.
- **Forward declarations keep the surface light.** `device::Resources` and `io::PopSelection` are forward-declared (lines 38, 42), so consumers of this header don't transitively pull CUDA headers or I/O implementation details. That's good layering discipline.
- **The result type is a proper value type.** `F2ExtractResult` is aggregate-friendly, with sensible defaults, and the function is marked `[[nodiscard]]` (line 91). Returning the full result by value is the right contract for a Python-binding-facing C++ API.
- **Ploidy and tier are mirrored into the public namespace.** Decoupling `ExtractPloidy` and `ExtractTier` from internal `config::PloidyMode` and `device::OutputTier` is a defensible public-API choice — it keeps the surface stable even if internals rename or refactor.

## What a senior developer would flag

**The `Status` field inside `F2ExtractResult` (line 76):**

```cpp
Status status = Status::Ok;             ///< Ok on success (a fault throws, never a status).
```

This is dead weight in the success path. The comment admits it: faults throw, so `status` is always `Ok` on any returned result. A senior reviewer would ask why the caller needs a status field that can never vary. Either return `Status` for domain outcomes, or drop it from the result. Including it "because `error.hpp` exists" is not a reason.

**Type-width inconsistency for SNP counts (lines 70-71):**

```cpp
long n_snp_total = 0;                   ///< SNPs READ from the .snp (pre-filter).
long n_snp_kept = 0;                    ///< SNPs kept after the filters.
```

`long` is 64-bit on Linux64 but 32-bit on Windows. For a cross-platform genomics library, this is a portability footgun. If these counts can exceed 2 billion (they can in whole-genome data), `std::int64_t` is the honest type. The adjacent `std::size_t` fields (lines 72-73) already got this right.

**Mirrored enums that duplicate internal enums:**

```cpp
enum class ExtractPloidy { Auto, PseudoHaploid, Diploid };
enum class ExtractTier { Resident, HostRam, Disk };
```

These are reasonable for API stability, but they are also duplication magnets. If someone adds `Haploid` or a new tier internally, this header silently drifts out of date. There is no static mapping visible here (no `to_device_ploidy(ExtractPloidy)`), so the conversion logic lives elsewhere and can diverge. A senior dev would want either a single source of truth with an adapter, or a compile-time assertion that the enumerators stay in sync.

**The default precision claim vs. the default value (line 74):**

```cpp
Precision::Kind precision_tag = Precision::Kind::EmulatedFp64;  ///< the ENGAGED precision.
```

The comment on the function (line 86) says `precision` "governs the f2 GEMMs (default EmulatedFp64 40-bit)." That's fine, but the struct's default initializer hardcodes the same value. A result from a failed call is never returned (it throws), so this default only matters for the caller's own default-constructed temporaries — which is a bit confusing. The default belongs in the function's default argument or in the caller, not in the result type.

**The comment tone is loud and defensive:**

Lines 3-7 and throughout use a lot of ALL CAPS ("VERBATIM", "REPLACED", "PARITY-NEUTRAL", "THROWS", "RETURNS", "READ"). The content is good, but the shouting makes the header read like a legal disclaimer. A senior reviewer would quietly trim most of the caps and trust the code to speak for itself.

**"Byte-identical" parity claims (lines 5-6, 19-24):**

```cpp
// compute_f2_blocks_multigpu_tiered->to_host chain, lifted VERBATIM out of
// src/app/cmd_extract_f2.cpp (the math is byte-identical — the goldens are untouched)
```

This is a strong claim in a header comment. If it's true, great — but header comments are not test assertions. A senior dev would want a parity test or a reference to the golden suite in `tests/`, not just a comment promise. Comments like this rot fast when the implementation inevitably diverges.

**The function signature is long and mostly positional:**

```cpp
[[nodiscard]] F2ExtractResult run_extract_f2(const std::string& geno,
                                             const std::string& snp,
                                             const std::string& ind,
                                             const io::PopSelection& pops,
                                             const FilterConfig& filter,
                                             const Precision& precision,
                                             double blgsize_morgans,
                                             ExtractPloidy ploidy,
                                             device::Resources& resources);
```

Nine parameters, several of which (`filter`, `precision`, `ploidy`) are configuration bundles that could be grouped. A builder or options struct would make this easier to call correctly and extend without breaking the ABI on every new flag. As it stands, adding one more option means another parameter shuffle.

## The "slop" test

**Not slop.** Slop is unexplained magic numbers, copy-paste drift, missing error handling, or obviously wrong public contracts. This header has none of that. The surface is deliberate, the comments explain intent, and the defaults are mostly reasonable. The worst sins here are over-commenting, a redundant status field, and a slightly too-long function signature — all fixable in a single pass.

## What it actually looks like

This looks like **solid public-API design by someone who cares about binding contracts and header hygiene.** The author clearly understands the difference between a CLI command (prints, exits) and a library function (throws, returns values), and they've taken pains to keep CUDA out of the public header. That is genuinely competent systems thinking.

At the same time, it reads like a first public draft: enthusiastic documentation, a few API decisions that haven't been tightened yet, and some duplication that a more senior maintainer would refactor into a smaller, more stable surface. A senior C++ reviewer would say: "Good bones — now delete the status field, freeze the SNP-count types, and quiet the comments down."

**Verdict:** B+. A respectable public seam with a couple of design warts that are easy to clean up before it hardens into long-term ABI debt.
