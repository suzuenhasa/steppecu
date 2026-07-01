I read through `include/steppe/fstats.hpp` carefully. It's a tiny file, but the review request was specific about what would impress vs. embarrass in a job-application showcase. My take: **this is competent, well-informed code, but it's more "carefully documented internal struct" than "polished public API."**

## What's genuinely good

- **It's actually CUDA-free and minimal.** Lines 24–25 use a plain include guard, and the only includes are `<cstddef>` and `<vector>` (lines 27–28). No `cuda_runtime.h`, no hidden device dependency. That matches the architectural promise in the header comment.
- **The layout is explicitly documented.** Lines 35–37 give the exact flat-index formula `i + P·j + P·P·b`, including the column-major convention and the outer block axis. A reader doesn't have to guess.
- **RAII and no raw pointers.** `std::vector<double>` and `std::vector<int>` own their storage. No manual `new`/`delete`, no `cudaMallocHost` leaks, no ownership ambiguity.
- **`[[nodiscard]]` and `noexcept` on `size()`** (line 74) are correct modern C++ idioms for a pure observer.
- **The diagonal convention is pinned in a comment** (lines 40–46). That's a real invariant that could silently change downstream behavior, and the author flags it explicitly.

## What a senior developer would flag

**Signed `int` for dimensions and sizes.**

```cpp
int P = 0;                       // line 68
int n_block = 0;                 // line 71
std::vector<int> block_sizes;    // line 65
```

In modern C++, sizes are `std::size_t` (or at least `std::int64_t`). Using `int` for SNP counts and population counts is a latent footgun: `block_sizes` could overflow on large panels, and `size()` (lines 74–77) will wrap a negative `P` or `n_block` into a gigantic `std::size_t`. The casts are careful, but they can't fix bad input.

**`vpair` is integer data stored as `double`.**

```cpp
std::vector<double> vpair;       // line 60
```

The comment acknowledges this ("Integer-valued (carried as double)"), but a senior reviewer would ask *why*. Counts above 2^53 lose integer precision, and any code consuming this has to remember that these are logically discrete counts. If the fit engine needs `double`, convert at the boundary; don't store counts as FP.

**No invariant enforcement.**

`F2BlockTensor` is a public struct with four independent public fields. Nothing prevents:

```cpp
F2BlockTensor t;
t.P = 100;
t.n_block = 50;
// forgot to resize f2
auto idx = i + t.P * j + t.P * t.P * b;  // out-of-bounds
```

For an installed public API, a senior C++ dev would expect either private members with accessors, a constructor/factory that resizes consistently, or at least a `valid()` / `check_invariants()` method. Right now the "convenience" `size()` is the only guard, and it doesn't check that `f2.size() == size()`.

**Over-commented with internal ticket noise for a public header.**

Lines 4–23 are dense with `architecture.md §...`, `ROADMAP §3 M4`, `cleanup X-13/B26`, etc. The intent is good, but in an *installed* public header this reads like internal scaffolding. A showcase header should explain the contract to external consumers, not trace every design decision back to roadmap tickets.

**Magic number 8 in the storage-budget comment.**

```cpp
// ... each `P² · n_block · 8` bytes ...   // line 20
```

That's `sizeof(double)`, but the comment hard-codes `8`. It happens to be true on every platform the project targets, but a senior reviewer would flag it.

**The header promises more than it delivers.**

The comment calls this "PUBLIC f-statistics API surface" (line 3), but the file defines exactly one data struct and zero functions. There's no serialization, no validation, no f3/f4 accessors, no `operator==`, no move semantics discussion. That's fine for an internal interchange struct, but calling it a public API surface oversells it.

## The "slop" test

**Not slop.** Slop is magic numbers without context, copy-pasted drift, uninitialized memory, or "it passed the tests so it's fine." None of that is here. The comments are heavy, but they're *accurate* and explain *why* things are laid out the way they are. The code itself is small, correct, and free of obvious leaks.

## What it actually looks like

This looks like **competent research/engineering code written by someone who understands the genomics domain and is trying to build a clean seam between a CUDA backend and a host-side fit engine.** It's careful about the math and the memory layout, but it's still thinking like an internal data structure rather than a public consumer-facing API. A senior C++ reviewer would say: "Good bones, but fix the integer types and add some invariant enforcement before you ship this as a public header."

## Verdict

**B+.** Ship it as an internal interchange struct now; tighten the integer types, add a constructor or invariant check, and tone down the roadmap archaeology before calling it a stable public API.