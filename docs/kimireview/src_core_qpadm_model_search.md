I read through this carefully. This is **not slop** — it's clearly written by someone who understands both the qpAdm domain and modern C++ concurrency, but a senior developer would flag a few architectural and API-contract warts.

## What's genuinely good

- **The small-path / large-path partition is the right shape.** Routing same-shape models through a batched virtual while falling back to per-model fits for the tail avoids the classic "force everything into one kernel" trap. Keeping the gate (`model_in_small_path`) single-sourced in `qpadm_bounds.hpp` so the host partition and device kernel arrays cannot drift is thoughtful defensive design.
- **Template dedup of `fit_one_model`.** Using one templated body for both `DeviceF2Blocks` and `F2BlockTensor` is clean, and the comment explaining *why* (line 24-36) is accurate: both overloads of `assemble_f4` exist and both expose `.block_sizes`.
- **Deterministic re-sort via pre-sized slots.** Instead of sorting after the fact, `results` is pre-sized to `n` and each worker writes its result to the slot given by `model_index`. The `scatter_into_slots` helper single-homes the bounds check so the G==1 fast path and G>=2 worker paths enforce it identically (lines 204-213).
- **Good multithreading hygiene.** `std::jthread` gives join-on-scope-exit for free, and worker exceptions are captured in `std::exception_ptr` and rethrown on the main thread (lines 290-321). That's the correct way to surface errors out of a thread pool without data races.
- **Honest documentation of the multi-GPU limitation.** The giant TODO block (lines 167-191) is unusually verbose, but it accurately identifies the root cause (no P2P on consumer RTX 5090s), gives measured numbers, and proposes the real fix (per-device precompute) rather than a band-aid. The replication code itself (`replicate_f2`, lines 215-248) is careful: it materializes the host tensor only once, reuses the resident handle where possible, and keeps borrowed/owned ownership explicit.

## What a senior developer would flag

**The `std::runtime_error` sentinel contract for `fit_models_batched`:**

```cpp
117 try {
118     small_results = be.fit_models_batched(...);
119 } catch (const std::runtime_error&) {
120     // No batched override on this backend (the CpuBackend oracle sentinel) —
121     small_results = fit_models_batched_default(...);
122 }
```

Catching *all* `std::runtime_error`s to detect "this backend doesn't implement batched" is fragile. A genuine `runtime_error` from inside the batched implementation (OOM, CUDA error, bad model) gets silently reinterpreted as "fall back to per-model" and may hide real failures or produce inconsistent results. A dedicated exception type, a capability query, or returning an empty optional would be cleaner.

**Inconsistent "never throw" contract.** The header comment on `fit_models_batched_default` says domain outcomes "ride in `results[i].status`, NEVER a throw" (line 69). But `scatter_into_slots` *does* throw on an out-of-range `model_index` (lines 207-210), and the public `run_qpadm_search` throws for `G == 0` (line 258). Those are invariant violations, not domain outcomes, but the distinction isn't called out clearly and a caller reading "NEVER a throw" will be surprised.

**Dead store in `fit_shard`:**

```cpp
116 const Precision prec = default_fit_precision();  // single-homed ([7.2]/[9.1] dedup)
```

`prec` is never used in this function; `fit_models_batched_default` does not take a precision argument, and `be.fit_models_batched` receives one on line 135. The comment suggests this was once passed through and is now leftover.

**Unchecked result-vector sizes after batch calls.** Neither the `small_results` nor the `large_results` path asserts that the returned vector length matches the input span length before scattering. If a backend implementation is buggy, `scatter_by_pos` will walk off the end of `results` or leave slots unfilled. A defensive `assert(small_results.size() == small.size())` (or a real check) would be cheap.

**Potential underflow in `model_in_small_path`:**

```cpp
90 const int nr = static_cast<int>(model.right.size()) - 1;
91 const int r = (opts.rank < 0) ? (nl - 1) : opts.rank;
92 return model_fits_small_path(nl, nr, r);
```

If `model.right` is ever empty, `nr` underflows to `INT_MAX`. The caller may guarantee non-emptiness, but a senior reviewer would want either a precondition comment or an explicit guard. Also, `nl - 1` can underflow if `left` is empty.

**Inconsistent bounds checking for `resources.gpus`:**

```cpp
271 ComputeBackend& be = *resources.gpus[0].backend;          // G==1 path
333 ComputeBackend& be = *resources.gpus.at(0).backend;       // oracle overload
```

Both are safe in context (G==1 is checked; oracle assumes at least one GPU), but mixing `operator[]` and `.at()` in the same file looks sloppy. Pick one style.

**Redundant `std::span` temporaries:**

```cpp
135 be.fit_models_batched(f2, std::span<const QpAdmModel>(small), opts, prec);
139 fit_models_batched_default(be, f2, std::span<const QpAdmModel>(small), opts);
145 fit_models_batched_default(be, f2, std::span<const QpAdmModel>(large), opts);
```

`std::span` is implicitly constructible from a `std::vector`; the explicit temporaries add noise without value.

**The enormous TODO block probably belongs in a design doc.** The measured performance narrative (8.72 GB, 3.8 s, 1.21x scaling) is valuable, but a 25-line TODO embedded in source is hard to keep current and distracts from the code. A short TODO comment plus a link to `docs/design/multigpu.md` would be more maintainable.

**Worker exception store is technically safe but invites scrutiny.** `worker_errors[g]` is written by thread `g` without synchronization. Because each thread writes a distinct index, this is data-race-free, but a reader has to reason about it. A comment noting the per-index isolation would help.

## The "slop" test

**Not slop.** Slop is magic numbers without explanation, copy-pasted code with stale comments, no error checking, or algorithms that happen to pass tests. This file has none of that. The comments are dense but they explain *why*, not just *what*, and the architecture (partition, batched virtual, per-model fallback, deterministic slot scatter) is coherent.

## What it actually looks like

This looks like **solid production orchestration code written by a domain expert who is competent at modern C++.** The author clearly understands the qpAdm rotation problem, the difference between the deliverable device path and the host oracle, and how to structure multithreaded GPU dispatch without races or leaks. The biggest weakness is API fragility: using a generic `std::runtime_error` as a feature-detection sentinel, saying "NEVER a throw" while throwing on invariant violations, and a few dead/unnecessary details that suggest refactoring residue.

A senior reviewer would probably say: "Ship it after tightening the exception contract and deleting the dead precision variable." A junior reviewer might copy the `catch (const std::runtime_error&)` pattern without realizing it's a loaded gun.

**Verdict:** Respectable production code. B+ to A- depending on tolerance for the exception-based sentinel and the oversized TODO.
