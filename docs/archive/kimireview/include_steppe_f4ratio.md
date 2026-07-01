I read through this carefully. This is **not slop** — it is a clean, professional public API header written by someone who understands both the domain (admixtools-style f4-ratio statistics) and modern C++ interface design. A senior developer would generally approve, though a few things would get flagged.

## What's genuinely good

- **The API surface is tight and correct.** Two overloads — one for device-resident `DeviceF2Blocks`, one for host `F2BlockTensor` — give you GPU-first execution and host-oracle parity through the same public contract. That is exactly the right shape for this kind of compute library.
- **Modern C++ choices.** `std::span<const std::array<int, 5>>` for the 5-tuples (line 75), `[[nodiscard]]` on both entry points (lines 74, 82), `std::vector` result storage, and forward declarations instead of dragging CUDA headers into a "CUDA-free" public header. These are competence signals.
- **Status-based error handling.** `Status status = Status::Ok;` (line 64) and the explicit "NEVER an exception" contract align with a project-wide error model. That is much better than throwing from a genomics compute routine.
- **The math documentation is precise.** The comments spell out the AT2 `qpf4ratio` convention, the fact that the ratio is the block-jackknife `$est` rather than `tot/tot`, and why the num/den quartets share one assemble call. A reader who knows admixtools can verify the design without opening the `.cpp`.

## What a senior developer would flag

**Stale/magic line-number references in comments.**

Line 24 says:

```cpp
// F1/OQ-12 survivor-block compaction (backend.hpp:113-126) is IDENTICAL for num and den —
```

This is the kind of cross-file line reference that rots immediately. A senior reviewer would flag it: if `backend.hpp` gets refactored, the comment becomes actively misleading. If you need to cite internals, use a stable symbol name or remove the line numbers.

**Parallel-array invariant is not enforced by the type system.**

```cpp
struct F4RatioResult {
    std::vector<int>    p1, p2, p3, p4, p5;
    std::vector<double> alpha;
    std::vector<double> se;
    std::vector<double> z;
    ...
};
```

Nine vectors that must all have the same length. The comment says "one parallel-array slot per input 5-tuple" (line 51), but the struct does nothing to guarantee that. A single malformed construction or resize bug produces silently mismatched rows. A senior C++ reviewer would suggest either a `std::vector<Row>` of small structs or, at minimum, a factory/helper that assembles a valid result in one place.

**The `p1..p5` fields invite copy-paste drift.**

Five identically-typed population-index vectors are easy to transpose (`p3` written where `p4` should be). The input is a `std::array<int, 5>`, so the output could just as easily echo the whole tuple in a single `std::vector<std::array<int, 5>> tuples_in`. That would remove a whole class of off-by-one population bugs.

**`precision_tag` is hard-wired to `Fp64` and the explanation is convoluted.**

Line 66-68:

```cpp
/// Which arithmetic produced this (the assemble honored tag; assemble_f4_quartets is
/// the cancellation carve-out and stays native, so alpha is always native FP64).
Precision::Kind precision_tag = Precision::Kind::Fp64;
```

A reader has to parse two lines of comment to learn "this is always FP64." If it is genuinely always FP64, either remove the field or make it a `constexpr`/`static`. If it exists for forward-compatibility, a one-line comment saying so is enough.

**Awkward comment continuation on the `qpadm.hpp` include.**

Lines 46-47:

```cpp
#include "steppe/qpadm.hpp"   // steppe::QpAdmOptions (the shared per-call options) +
                              // the device::DeviceF2Blocks / device::Resources fwd-decls
```

The trailing `+` continuation in a comment looks like a formatting accident. It is harmless, but it is the kind of untidiness that makes a senior reviewer pause and wonder what else was rushed.

**`QpAdmOptions` as the options type for `run_f4ratio`.**

`run_f4ratio` takes `const QpAdmOptions& opts` (lines 76, 84). Naming suggests this options struct is shared with `qpadm`, which is fine if true, but a reader of the header cannot tell whether `f4ratio` silently ignores QpAdm-specific fields or whether QpAdm carries unrelated settings. A small `F4RatioOptions` alias or a clearly-shared base would make the contract less surprising.

## The "slop" test

**Not slop.** Slop is magic numbers, stale copy-paste, silent error swallowing, and obvious algorithmic laziness. None of that is present. The comments are dense but accurate, the API is deliberate, and the error model is consistent. The issues above are polish and design-tradeoff questions, not incompetence.

## What it actually looks like

This looks like **solid systems-API code written by a domain expert who respects the public interface.** The author clearly thought about header hygiene (CUDA-free, forward declarations, span-based inputs), parity testing (host overload), and result semantics (status + NaN sentinels). The weaknesses are mostly in type-level safety and comment discipline — the kind of things a senior C++ reviewer would flag in a second pass, not in a first-round rejection.

A senior reviewer would likely say: "Good header. Tighten the result struct invariants, delete the stale line-number references, and consider whether `QpAdmOptions` is the right name here."

**Verdict:** B+ — production-ready public API with minor design and documentation rough edges.
