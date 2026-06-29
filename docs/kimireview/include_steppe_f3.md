I read through this carefully. This is **not slop** — it's a well-thought-out public API header written by someone who understands both the genomics math and the project's architectural constraints. A senior reviewer would find a lot to like, but would also flag the coupling and the comment density.

## What's genuinely good

- **The architectural contract is clean and explicit.** Marking the header "CUDA-FREE BY CONTRACT" (line 24) and using forward-declared `device::DeviceF2Blocks` / `device::Resources` while the `.cpp` pays the include cost is the right way to keep a public header lightweight and host-toolchain friendly.
- **It reuses existing seams instead of inventing new infrastructure.** The comment (lines 7–10) is explicit that `assemble_f3_triples` and `jackknife_cov` are reused, and only one new math seam (the 3-slab combine) is added. That's good API hygiene.
- **The math identity is documented and correct.** The comment on lines 12–16 gives the AT2 mapping `est = 0.5*( f2(C,A) + f2(C,B) - f2(A,B) )` and explains the apex/arg roles. This shows real domain competence, not just code-shoveling.
- **`F3Result` uses `Status` for degenerate outcomes instead of exceptions.** Lines 48–59 follow the project's `record-and-continue` discipline (`architecture.md §10`), which is the correct choice for a batch compute API.
- **Modern C++ surface.** `std::span<const std::array<int, 3>>` for input triples, `[[nodiscard]]` on the entry points, and value-initialized default members are all solid signals.
- **Dual overload design is principled.** The device-resident overload (lines 69–72) is the production path, and the `F2BlockTensor&` overload (lines 77–80) is the host-oracle/parity door. The symmetry is easy to understand.

## What a senior developer would flag

**The `qpadm.hpp` include is a coupling smell.**

```cpp
#include "steppe/qpadm.hpp"   // steppe::QpAdmOptions (the shared per-call options; fudge) +
                              // the device::DeviceF2Blocks / device::Resources fwd-decls
```

The comment admits this is a grab-bag: `QpAdmOptions` plus the forward declarations. A senior reviewer would ask why an f3 runner depends on a qpAdm options struct. `QpAdmOptions` almost certainly contains fields that f3 ignores (ALS settings, rank-test knobs, the `fudge` ridge that the comment explicitly says f3 does *not* use). Either extract a shared `FstatsOptions` base, or at least forward-declare `QpAdmOptions` and include only what this header actually needs. Right now every translation unit that includes `f3.hpp` gets the qpAdm surface dragged in.

**The `f4.hpp` include is only for a p-value helper.**

```cpp
#include "steppe/f4.hpp"      // steppe::f4_two_sided_p (REUSED — the SAME z->p convention)
```

This is pragmatic, but it telegraphs that `f4_two_sided_p` is misnamed: if f3 also uses it, it's just a generic two-sided normal tail function. A senior dev would suggest moving it to a neutral math header (`steppe/math.hpp` or similar) rather than making f3 depend on f4 for a p-value utility.

**Hardcoded GPU 0.**

```cpp
/// Routes through resources.gpus[0].
```

Line 68 says the device path always routes through `resources.gpus[0]`. For a project that talks about "GPU-first production envelope, design-for-scale" (line 17), baking in device index 0 is a mismatch. At minimum this should be an option, or the comment should explain why multi-GPU selection is intentionally out of scope for this entry point.

**The `F3Result` echoes the input triples back as three `std::vector<int>` fields.**

```cpp
std::vector<int>    p1, p2, p3;  ///< the P-axis indices of each triple (len N); p1=C apex.
```

The comment justifies this ("echoed for the emitter/binding to label the rows"), but it's still redundant data that the caller already has. For large triple counts it's not free. A leaner design would return only the computed columns and let the binding keep the input labels. This is a mild API design concern, not a bug.

**Comment density is very high — and some of it is design rationale that belongs elsewhere.**

The file is 84 lines, and roughly 60 of them are comments. Lines 1–28 are essentially a design document. It's accurate and useful, but a senior reviewer would note that headers should state *contracts*, not narrate the entire design history. Phrases like "fit-engine §6 / the standalone-f3 design, mirroring f4" and "architecture.md §4" suggest the author is compensating for documentation that lives elsewhere, or is trying to make the header self-contained. Neither is wrong, but the result is close to over-commented.

**`precision_tag` defaulting to `Fp64` is a potential lie.**

```cpp
Precision::Kind precision_tag = Precision::Kind::Fp64;
```

Line 63 defaults the tag to FP64, and the comment says "the est is always native FP64." But if `Precision::Kind` is meant to reflect the actual runtime arithmetic, a default value in the result struct is a footgun: a caller who forgets to set it will report FP64 even if the backend ran FP32. The header should either omit the default or require the implementation to set it explicitly.

## The "slop" test

**Not slop.** Slop is magic numbers without explanation, copy-pasted code with stale comments, no error handling, or obviously wrong algorithms that happen to pass. This header has none of that. The comments are dense but they explain *why*, not just *what*. The API surface is small and coherent.

## What it actually looks like

This looks like **careful, architecture-aware work by a developer who knows the genomics domain and is trying to keep the public surface clean.** The header does one thing and does it transparently: it exposes a CUDA-free f3 entry point, documents the math, and reuses existing backend seams. The C++ is modern enough (span, array, nodiscard, Status) that it wouldn't embarrass anyone in a showcase.

A senior reviewer would say: "Good header, but why is it married to `qpadm.hpp` and `f4.hpp`? Decouple those and trim the novella, and it's close to pristine." A senior C++ person would also raise an eyebrow at the redundant `p1/p2/p3` echo and the hardcoded `gpus[0]`, neither of which is broken but both suggest a slightly too-comfortable coupling to the caller's convenience.

**Verdict:** Solid production header. A-/B+ depending on how much the reviewer cares about include-graph purity and over-commenting. The code behind it still has to earn the grade, but the public contract is competent.
