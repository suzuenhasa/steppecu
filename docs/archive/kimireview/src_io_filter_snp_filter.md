I read through this carefully. This is **not slop** — it's clearly written by someone who has been burned by silent data-corruption bugs and has decided to make every fail-fast contract explicit. A senior developer would respect the intent but would flag the comment density and a few C++ hygiene issues.

## What's genuinely good

- **Fail-fast philosophy is correct and well-explained.** The ploidy check (lines 36-40), `pop_individuals` length check (lines 52-57), null-pointer check (lines 65-68), and `SnpTable` length checks (lines 128-136) all reject wiring errors with context instead of producing plausible-but-wrong masks. This is the right trade for an `io`-leaf TU.
- **Single source of truth for shared decisions.** Delegating the per-SNP predicate to `snp_keep_decision` (line 152) and the pooled reduction to `derive_pooled_summary_one` (line 86) means the host path and any future device path cannot drift on filter semantics. The comment at lines 151-153 calling out the "drop-not-flip" contract is a good invariant.
- **RAII and no manual cleanup.** Outputs are `std::vector<PerSnpSummary>` and `std::vector<bool>`; there's no `new`/`delete`, no raw ownership in the local scope, and exception safety is straightforward.
- **Membership is pre-boiled to a bool before the decision primitive** (lines 149-153). This keeps `snp_keep_decision` pure and free of `SnpMembership` dependency, which is a clean API boundary for the future M4.5 device kernel.

## What a senior developer would flag

**Comment bloat that outpaces the code.**

Lines 25-40 are a 15-line comment for a 5-line `throw`:

```cpp
// Fail-fast on a nonsensical ploidy (architecture.md §2 fail-fast; cleanup
// B10/F5/X-11). Ploidy is METADATA, never auto-detected, and {1, 2} are the
// only meaningful values (decode_af.hpp). A non-positive / out-of-range ploidy
// is a config/wiring error: the previous silent clamp-to-1 produced a
// plausible-but-wrong missing fraction (it doubles nonmissing_indiv vs the
// diploid truth) rather than surfacing the bug. This RECONCILES the illegal-
// ploidy contract with the decode primitive: the device-side finalize_af has
// no throw path on the GPU so it masks the cell OUT ({0,0,0}); this host `io`
// leaf has a throw path, so it rejects up front — both refuse to fabricate a
// trustworthy N from a bad ploidy. std::invalid_argument is the io-leaf idiom
// (cf. include_exclude.cpp's runtime_error on a bad read).
if (in.ploidy != 1 && in.ploidy != 2) {
    throw std::invalid_argument(
        "snp_filter: ploidy must be 1 (pseudo-haploid) or 2 (diploid); got " +
        std::to_string(in.ploidy));
}
```

The historical context is useful once; here it repeats for every guard. A senior reviewer would ask: "If the code and the exception message are this clear, do we need a paragraph of cleanup ticket references?" The same pattern repeats at lines 42-51 and 59-64. Dense comments that explain *why* are good; dense comments that narrate every past bug are a maintenance liability.

**`std::vector<bool>` for the keep mask.**

```cpp
std::vector<bool> keep(static_cast<std::size_t>(M < 0 ? 0 : M), false);
```

It works for a bitmask, but `std::vector<bool>` is a well-known specialization with reference-proxy iterators and occasional surprises (e.g., `auto&` on an element is a proxy, not a `bool&`). For a new mask that later code likely converts to something else, `std::vector<uint8_t>` or a dedicated bitset would be more honest and less surprising.

**Type-width soup.**

`P` is `int`, `M` is `long`, loop indices flip between `long s` and `std::size_t si`, and `pop_individuals` holds some integer type summed into `std::size_t`. This is mostly harmless for genomics sizes, but it forces repeated defensive casts like:

```cpp
std::vector<PerSnpSummary> out(static_cast<std::size_t>(M < 0 ? 0 : M));
```

and

```cpp
for (int p = 0; p < P; ++p) {
    total_indiv += in.pop_individuals[static_cast<std::size_t>(p)];
}
```

A senior would push for a single size type (e.g., `std::size_t` with `std::ssize` where signed is needed) or a small helper for the `M < 0 ? 0 : M` dance, which appears twice in this file alone.

**Raw pointers in `DecodedTileSummaryInput` with no ownership signal here.**

```cpp
if (in.q == nullptr || in.n == nullptr) {
    throw std::invalid_argument(
        "snp_filter: q and n must be non-null when P>0 && M>0");
}
```

The null check is good, but the reader of this file has no idea whether `q` and `n` are borrowed views, owned arrays, or GPU pointers. That's a struct-design issue more than a bug in this file, but this TU is where the contract surfaces. A `std::span<const T>` or a small non-owning view type would make the lifetime contract explicit.

**The `require_at_least` lambda captures `[&]` out of habit.**

```cpp
const auto require_at_least = [&](const char* what, std::size_t have, bool active) {
    if (active && have < Mu) { ... }
};
```

It only needs `Mu` and `M`. Capturing everything by reference is fine in a short function, but it's a minor smell — explicit capture `[&Mu, &M]` would be clearer and would not break if someone later adds a non-const local.

## The "slop" test

**Not slop.** Slop is magic numbers without explanation, copy-pasted drift, missing error handling, or wrong algorithms that happen to pass. This file has none of that. The comments are overgrown, but they explain real invariants and real past bugs rather than stating the obvious.

## What it actually looks like

This looks like **solid defensive C++ written by an engineer who prioritizes correctness and auditability over brevity.** The architecture is clean: shared predicates, shared reduction, fail-fast validation, membership decoupled from the decision primitive. The main flaw is that the author cannot stop explaining the history of every guard clause. A senior reviewer would trust the logic quickly and then spend the rest of the review asking for shorter comments and a more consistent size-type story.

A C++ specialist would say: "Competent, safe, ship it — but trim the comments and stop using `std::vector<bool>`." A genomics colleague would say: "Finally, someone who refuses to silently corrupt the missing-fraction denominator."

**Verdict:** B+ — strong engineering, slightly over-explained, a few conventional C++ rough edges.
