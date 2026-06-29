I read through this carefully. This is **not slop** — it's clearly written by someone who understands the Python oracle they're porting and who can write modern C++ when they choose to. But a senior developer would flag a handful of semantic/documentation rough edges and one real API usability trap.

## What's genuinely good

- **The oracle-parity intent is front and center.** Comments cite `build_tgeno_matrix.py`, `Counter.most_common`, `sorted(sel)`, and the "same 50-pop set" contract. That kind of provenance-aware thinking is exactly what you want in a genomics compute project where the GPU output is validated bit-for-bit against a host oracle.
- **Clean `io`-leaf layering.** Pure host C++20, only standard-library headers, no CUDA, no upward dependency into `core`/`device`. `std::ifstream` gives RAII file handling; no raw `FILE*` or manual `fclose` anywhere.
- **The selection logic is mostly correct and well-factored.** Extracting the shared Explicit/MinN filter loop into `filter_into` (lines 97–101) removes copy-paste drift, and using `std::stable_sort` keyed on `(count desc, first_seen asc)` (lines 117–122) faithfully reproduces Python `Counter.most_common` tie-breaking.
- **Good const-correctness and ownership hygiene.** `selected` holds `const RawGroup*` into the local `groups` vector; `RawGroup` lives in the anonymous namespace so it never leaks across the ABI. Public API returns a plain value type with `[[nodiscard]]` on the entry point.
- **No raw pointers, no manual memory management, no global state.** For host I/O that is mostly a parser, that's the baseline — and this file meets it.

## What a senior developer would flag

**`n_individuals_total` semantics don't match its contract (lines 58–72, 84).**

```cpp
if (row >= n_records_present) {
    // ... rows ignored ...
    continue;   // line 71
}
// ...
++row;          // line 80
```

The header (`.hpp:71`) says `n_individuals_total` is "total .ind rows == TGENO records". But `row` is only incremented on *parseable* lines (blank/short lines are skipped at line 63), **and** it keeps incrementing past the `n_records_present` cap (the `++row` at line 80 runs inside the skip branch). So the field can be neither the total file rows nor the capped genotype-axis length. Since no downstream caller currently uses it, it's a latent footgun rather than an active bug — but a doc/contract mismatch like this will silently bite the first person who trusts the comment.

**The default-constructed `PopSelection` is guaranteed to throw (`.hpp:45,48`; `.cpp:123–135`).**

```cpp
struct PopSelection {
    Mode mode = Mode::AutoTopK;
    std::size_t k = 0;       // AutoTopK with k == 0
    std::size_t min_n = 1;
    // ...
};
```

`read_ind(PopSelection{})` always hits the empty-selection throw at line 133–135. That's a hostile default. Worse, the Python oracle's fall-through default when no flags are set is the MinN branch (`min_n == 1`), not AutoTopK with `k == 0`, so C++ defaults diverge from the oracle's default path. A senior reviewer would ask: "Is the default intentionally invalid, or did you just forget to set it?"

**`RawGroup::first_seen` is redundant with its own vector index (line 76).**

```cpp
index_of.emplace(pop, groups.size());
groups.push_back(RawGroup{pop, groups.size(), {row}});
```

Because `groups` is never reordered, `first_seen` is always equal to the element's index in `groups`. The adjacent double-use of `groups.size()` also couples lines 75–76; a future edit that computes the slot differently on one line would introduce a subtle off-by-one. Either compute `slot` once, or drop `first_seen` entirely and rely on the stable-sort input order for the tie-break.

**The AutoTopK tie-break is doubly specified (lines 117–122).**

The comparator orders ties by `first_seen`, but the input is already in first-appearance order and the sort is `stable_sort`. Belt-and-suspenders works, but it makes it unclear *which* mechanism is actually load-bearing. Pick one.

**The "matches Python's `sorted()`" claim is overstated for non-ASCII labels (line 138).**

```cpp
// std::string < is byte/lexicographic order,
// matching Python's sorted() on str.
```

`std::string::operator<` orders by unsigned byte value; Python `sorted()` on `str` orders by Unicode code point. They agree on pure ASCII, but a UTF-8 population label can sort differently in the two languages. Since this ordering determines the Q/V/N row order, that's a real parity hazard on a non-ASCII dataset — and the comment falsely reassures the reader.

**There are no unit tests for the selection logic itself.** `read_ind` is covered only indirectly by end-to-end reference tests exercising AutoTopK. Explicit mode, MinN mode, the cap, empty-selection throws, blank-line handling, and the label-sort order are all untested. For a pure-host function with no GPU dependencies, that's a missed opportunity.

## The "slop" test

**Not slop.** Slop would be magic numbers, copy-pasted branches with stale comments, ignored errors, or a hand-rolled parser that gets the oracle semantics subtly wrong. This file has none of that. The comments are dense and mostly accurate; the code mirrors the Python reference deliberately. The issues above are craft nits and contract imprecision, not sloppiness.

## What it actually looks like

This looks like **solid, domain-aware host-side C++ written by someone who understands the Python oracle better than they understand C++ API ergonomics.** The parser and selection policies are correct, the layering is textbook clean, and the author clearly cares about bit-for-bit parity. But the default-mode trap, the `n_individuals_total` contract drift, and the over-strong "matches Python" claims are exactly the kind of things a senior reviewer circles in red before approving a merge. A few hours of tightening contracts, adding a focused host-only unit test file, and fixing the default `PopSelection` would turn this into a showcase piece.

**Verdict:** B+ — competent, oracle-faithful host I/O with a few contract and usability rough edges that should be cleaned up before it's held up as production-perfect.
