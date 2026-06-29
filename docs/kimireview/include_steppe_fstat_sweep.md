I read through this carefully. This is **not slop** — it's a thoughtfully designed public API header written by someone who understands the domain. But a senior developer would flag the tension between "clean public interface" and "implementation manifesto dumped into a header."

## What's genuinely good

- **The separation of concerns is right.** A CUDA-free public header (`include/steppe/fstat_sweep.hpp`) that fronts a GPU-only implementation is the correct seam for a genomics library. The caller doesn't see CUDA types, kernels, or device pointers.
- **The data structures are clear and modern C++.** `enum class SweepFilter`, `std::array<int, 4>`, `std::vector<double>`, `[[nodiscard]]` on the entry points — this is idiomatic, not C-with-classes.
- **The safety cap is surfaced honestly.** `sure` lifting `kFstatMaxComb` and `capped` reporting it back in `SweepResult` is a clean contract (lines 53, 70).
- **The result struct carries enough metadata.** `enumerated`, `survivors`, `capped`, `status`, and `precision_tag` give the caller the full story without forcing them to poke at internals.
- **Defaults are defensible.** `min_z = 3.0` with an inline "AT2-style significance cut" note and `top_k = 100` are reasonable starting points, and the comments explain the intent (lines 50–51).

## What a senior developer would flag

**The header file is doing too much narrative work.** Lines 2–24 are a 22-line implementation essay in what should be a concise public contract. A senior reviewer would expect that level of detail in `docs/design/` or in the `.cpp`/`.cu` implementation, not in the header that external users include. Public headers should say *what* the API promises; this one explains *how* CUB stream-compaction works. That will rot the moment the implementation changes.

**A factual error in the comments:**

```cpp
// THE GPU-ONLY PIPELINE (the fix for the CPU-bound host-enumeration disaster; design-verified
// against CUDA 13.x docs)
```

Line 12: there is no CUDA 13.x. The latest released CUDA toolkit is 12.x. This looks like a typo or a forward-guess, but in a job-application showcase it reads as either sloppiness or resume inflation.

**TopK semantics contradict the headline "filter on-device":**

```cpp
TopK,  ///< keep the K items with the largest |z| (device keeps all; host ranks to K).
```

Line 42 says TopK keeps *all* items and ranks on the host, while lines 5–6 and 17 claim the sweep "FILTERS by |z| (or top-K) ON THE GPU" and uses "cub::DeviceSelect::Flagged stream-compacts them on-device." A senior reviewer would ask: does TopK actually stream-compact, or does it materialize the full multi-TB table after all? The comment in the `SweepFilter` enum is more specific and therefore more believable, but it undermines the architectural pitch in the file header.

**Hardcoded single-GPU routing is buried in a doc comment:**

```cpp
// Routes through resources.gpus[0].backend (the CUDA backend's f4_sweep virtual).
```

Line 78 documents that the API silently uses `gpus[0]`. That's fine for a "MULTI-GPU PARKED" feature, but it belongs in the function contract or as a note on `Resources`, not as an inline parenthetical. A caller with two GPUs needs to know this is not a multi-GPU dispatch.

**The `keys` array wastes space for f3:**

```cpp
std::vector<std::array<int, 4>> keys;  ///< survivor P-axis index tuples (p4 unused for f3).
```

Line 62: using a 4-int tuple for both f3 and f4 is pragmatic (one result type), but a senior reviewer would note the unused fourth int for every f3 survivor. At TopK scale it's negligible; at large survivor counts it's real memory overhead for no benefit.

**No validation contract for `pop_subset`.** The request accepts `std::vector<int> pop_subset` (line 52) but the header says nothing about what happens with duplicates, out-of-range indices, negative values, or an empty vector. A defensive public API would either document preconditions or expose a builder/validator.

**Precision default may be optimistic:**

```cpp
Precision::Kind precision_tag = Precision::Kind::Fp64;
```

Line 73 defaults the result to FP64, but the comment block talks about "emulated-FP64 policy inherited." If the backend may return emulated or reduced precision, a default of `Fp64` in the result struct is a potential lie unless the implementation always promotes to FP64 before return.

## The "slop" test

**Not slop.** Slop is undocumented magic numbers, stale copy-paste comments, and no error path. This file has none of that. The structures are coherent, the defaults are explained, and the API surface is small. The problem here is *over*-documentation in the wrong place, not absence of care.

## What it actually looks like

This looks like **competent systems-engineering API design by someone who knows the genomics domain and is trying to protect users from a foot-gun (the multi-TB full-sweep materialization).** The author clearly thought about the public/private split and the safety cap, but they couldn't resist turning the header into a design document. A senior reviewer would say: "Good bones — now move 80% of the prose into a design doc and tighten the function-level contracts."

A senior C++ person would also want to see unit tests or at least a note on thread-safety/reentrancy of `run_f4_sweep`/`run_f3_sweep` when called with the same `Resources` object, but that concern lives outside this single file.

**Verdict:** Solid, above-average API header with a documentation-drift risk. **B+** — would be A- with the manifesto trimmed and the CUDA 13.x typo fixed.
