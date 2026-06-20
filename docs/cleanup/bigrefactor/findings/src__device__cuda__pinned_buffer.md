# Review findings — src__device__cuda__pinned_buffer

Files: /home/suzunik/steppe/src/device/cuda/pinned_buffer.cuh

## Group 4 — Type & numeric

- [4.7][LOW] src/device/cuda/pinned_buffer.cuh:102-103,158,260 — `PinnedBuffer<T>::data()` returns raw `T*` and `RegisteredHostRegion`/`PinnedRegistryCache::ensure` take raw `const void*`; nothing in the type system marks these as HOST pointers, so a device pointer could be passed in (it would fail the `cudaHostRegister`/`cudaHostAlloc` API at runtime rather than at compile time). Not a numeric bug — these are correctly host-only allocations by design and the failure path degrades gracefully. Suggested: optional project-wide host-vs-device pointer wrapper; no change needed for correctness here.

Notes (no defect):
- 4.1 N/A — no float/double math in this unit (pure pointer/byte RAII).
- 4.2 Clean — all sizes/indices are `std::size_t` (`size_`, `n`, `bytes`, `kSlots`, `next_`); no `int` global index into the f2 tensor or genotype matrix.
- 4.3 Clean — line 80 `cudaHostAlloc(..., n * sizeof(T), ...)` and line 105 `size_ * sizeof(T)` both include the element size; `cudaHostRegister` (line 164) takes a caller-supplied byte count.
- 4.4/4.5 Clean — the only loop (line 262) is a `size_t` range-for; no unsigned countdown, no signed/unsigned bound compare.
- 4.6 Clean — line 74 guards `n * sizeof(T)` with `n > numeric_limits<size_t>::max() / sizeof(T)` BEFORE the multiply (lines 80, 105), so the byte product cannot wrap; line 272 `(next_ + 1) % kSlots` is `size_t` arithmetic.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

Notes (no defect):
- 2.1 N/A — header-only RAII; no CMake arch lists / `sm_*` build flags in this unit.
- 2.2 Clean — no `texture<...>`/`cudaBindTexture*`/surface references; only the host-memory family (`cudaHostAlloc` L80, `cudaFreeHost` L112, `cudaHostRegister` L164, `cudaHostUnregister` L201) + `cudaGetLastError` L173 — all current in CUDA 13, none deprecated/removed.
- 2.3 N/A — no device kernel code; no warp intrinsics (sync or non-sync).
- 2.4 Clean — no `cudaThreadSynchronize` (no synchronization calls at all).

## Group 3 — Dead / commented-out code

No Group 3 issues found.

Notes (no defect):
- 3.1 Clean — every `//` block (file header L1-35, the doc comments on each class/method) is explanatory prose or Doxygen, not stashed code; no commented-out statements.
- 3.2 Clean — no `#if 0`. The two early `return;` statements (L159 null/zero range in `RegisteredHostRegion` ctor, L261 null/zero range in `ensure`) are guard clauses, not dead code after a return; nothing follows an unconditional return/break.
- 3.3 Clean — all includes are used: `<array>` (L285 `std::array`), `<cstddef>` (`std::size_t`), `<limits>` (L74 overflow guard), `<source_location>` (L77 typed throw), `<utility>` (L86/91/92/178/183 `std::exchange`), `<cuda_runtime.h>` (CUDA API), `log.hpp` (L114/203 `STEPPE_LOG_WARN`), `check.cuh` (`STEPPE_CUDA_CHECK`/`STEPPE_CUDA_WARN`/`CudaError`). `PinnedBuffer<T>::bytes()` (L105) and the const `data()` overload (L103) are unreferenced inside this unit but are legitimate public API accessors on a reusable RAII type.
- 3.4 Clean — every assignment is later read: `Slot::ptr`/`Slot::bytes` (L270-271) feed the cache-hit compare at L263; `ptr_`/`size_`/`registered_`-via-`ptr_` are all read by accessors / `reset()`; `next_` (L272) indexes L268 on the next call.

## Group 5 — Hardcoded values / magic numbers

- [5.2][LOW] src/device/cuda/pinned_buffer.cuh:279 — `kSlots = 3` hardcodes the pinned-registry capacity to the count of "exactly the Q/V/N H2D inputs the f2 path stages per call". It is a named `static constexpr` and thoroughly documented (L276-278), and the eviction logic is bounded/correct for any value, so this is hygiene only. But the `3` is an implicit coupling to the backend's current input count: if the f2/M5 path ever stages a 4th stable H2D source, the steady state would self-evict and thrash (the register tax stops amortizing) with no compile-time signal. Suggested: leave as-is (correct + documented), or optionally derive/assert the value against the actual H2D-input count where the cache is constructed so a 4th input fails loudly rather than silently thrashing.

Notes (no defect):
- 5.1 Clean — no unnamed float/int literals: no `0.001f`/`1024`/`0.5`. The `0`/`nullptr` values (L86, L91-92, L118-119, L207, L281-282, L286) are RAII null/zero-size sentinels, not magic numbers; `1` in `(next_ + 1) % kSlots` (L272) is the standard round-robin increment.
- 5.3 Clean — no duplicated/drift-prone constant: `kSlots` (L279) is the single source of truth for BOTH the `std::array<Slot, kSlots>` array size (L285) and the round-robin modulus `(next_ + 1) % kSlots` (L272); there is no separately-written literal `3` that could drift from the array bound (the exact DRIFT class this task targets).
- 5.4 Clean — no hardcoded paths, IDs, or device ids; no device index, stream id, or filesystem path appears (header-only host-memory RAII).
- 5.5 N/A — no `32` (warp-size or otherwise) anywhere; this is host-side RAII with no kernel/launch config and no warp intrinsics. CUDA flag constants are named enums (`cudaHostAllocDefault` L81, `cudaHostRegisterDefault` L164), not magic numbers.

## Group 6 — Naming

No Group 6 issues found.

Notes (no defect):
- 6.1 Clean — every single-letter name is an idiomatic, tight-scope local: `o` for the rvalue "other" in move ctor/assign (L85,88,177,180); `e` for the `cudaError_t` in `reset()` (L112,201); `s` for the register status `cudaError_t` (L163) and the loop `const Slot&` (L262); `p` for the `const_cast` host pointer (L162). None is a long-lived/opaque variable. No `tmp`/`data2`/`arr`/`flag`-style names anywhere.
- 6.2 Clean — no misleading names: `registered()` (L194) truly reports whether the pin took effect; `ensure()` (L260) ensures a registration; `next_` (L286) is genuinely the next round-robin index, not a count; `kSlots`/`size_`/`bytes` are real counts/sizes, not indices; `Slot::ptr`/`Slot::bytes` (L281-282) hold exactly that.
- 6.3 Clean — one consistent convention throughout: data members carry a trailing underscore (`ptr_`, `size_`, `next_`), compile-time constants use the `k` prefix (`kSlots`), types are PascalCase (`PinnedBuffer`, `RegisteredHostRegion`, `PinnedRegistryCache`, `Slot`), methods/locals are lowercase. The byte count is uniformly spelled `bytes` (the `bytes()` accessor L105, the ctor param L158, the `ensure` param L260, the `Slot::bytes` member L282); element count is uniformly `n`/`size_`. No `nElements` vs `num_elements` vs `n` drift.
- 6.4 Clean — only standard, widely-recognized abbreviations: `ptr` (pointer), `dst` (destination), `reg` (the RegisteredHostRegion member, L283), `o` (other). No nonstandard/opaque abbreviations.

## Group 7 — Duplication

- [7.1][LOW] src/device/cuda/pinned_buffer.cuh:108-120,197-208 — `PinnedBuffer::reset()` and `RegisteredHostRegion::reset()` are the same copy-pasted teardown block differing only by the free function (`cudaFreeHost` vs `cudaHostUnregister`) and the warning message string: `if (ptr_) { e = <freefn>(ptr_); if (e != cudaSuccess) STEPPE_LOG_WARN("<msg>: %s", cudaGetErrorString(e)); } ptr_ = nullptr;`. Two distinct types so no shared base, but the never-throw-warn-on-nonzero idiom is duplicated. Suggested: a small `device::warn_on_cuda_teardown(e, "<context>")` free helper folding the `if (e != cudaSuccess) STEPPE_LOG_WARN(... cudaGetErrorString(e))` so the message format lives in one place.
- [7.2][LOW] src/device/cuda/pinned_buffer.cuh:270-271 — in the cache-miss path `dst.reg.registered()` is evaluated twice back-to-back (`dst.ptr = dst.reg.registered() ? ptr : nullptr;` then `dst.bytes = dst.reg.registered() ? bytes : 0;`). The call is a trivial `ptr_ != nullptr` so this is purely hygiene, not perf, but it is a repeated invariant within a single fast-path. Suggested: hoist `const bool ok = dst.reg.registered();` once and reuse for both assignments.

Notes (no defect):
- 7.1 (other) Clean — the move ctor / move-assign pairs in `PinnedBuffer` (L85-95) and `RegisteredHostRegion`/`PinnedRegistryCache` are the §7 canonical move-only RAII shape (`std::exchange` swap + `reset()`-then-steal in assign); identical-by-design across RAII owners, and extracting them into a CRTP/base would obscure the deliberately-simple per-type RAII rather than reduce real duplication — not flagged.
- 7.3 Clean — `sizeof(T)` recurs at L74 (overflow guard divisor), L80 (alloc byte product), L105 (`bytes()` accessor), but each is a distinct, correct use of the type's element size, not a hoistable repeat; the two casts (`reinterpret_cast<void**>` L80, `const_cast<void*>` L162) appear once each.
- 7.4 Clean — apart from the 7.1 teardown idiom, no other macro/helper-foldable boilerplate: the two early `return;` guard clauses (L159, L261) and the ALLOWLISTED-TU comment markers are intentional per-call-site annotations, not collapsible duplication.

## Group 8 — Comments

- [8.2][LOW] src/device/cuda/pinned_buffer.cuh:266-267 — the cache-miss comment says "The assignment runs the OLD RegisteredHostRegion's dtor first (unregisters the evicted range), then move-assigns the freshly-registered region in." The move-assign at L269 (`dst.reg = RegisteredHostRegion(ptr, bytes);`) does NOT run the old object's destructor; it runs `RegisteredHostRegion::operator=` (L180-186), which calls `reset()` (L197, the unregister) on the old held range, then steals. Net effect (old range unregistered before the new one is held) is exactly as described, so the documented behavior is correct — only the mechanism ("dtor") is imprecise; the dtor runs at cache teardown (L254), not here. Suggested: reword to "runs the OLD RegisteredHostRegion's reset() (unregisters the evicted range)" to match the move-assign path actually taken.

Notes (no defect):
- 8.1 Clean — no restating-the-code comments. Every `//`/`///` block is WHY-rationale or Doxygen API prose (file header L1-35 on pinning/graceful-degrade/§12 parity-neutrality; the per-class/per-method doc comments; the inline ALLOWLISTED-TU markers L79,112,163,201 which annotate the §2 DRY grep-gate allowance, not the obvious call). No `x = 0; // set x to zero`-style narration.
- 8.2 (other) Clean — comment claims match the code: L145/L170-173 "sticky last-error CLEARED" matches the `(void)cudaGetLastError();` at L173; L210 "non-null ⇔ we registered and must unregister" matches the L165-166 assignment + L194 `registered()`; L254 "RegisteredHostRegion dtors unregister all" matches the `std::array<Slot>` of RAII members; L270-271 "only remember a real pin" matches the `registered() ? ... : nullptr/0` guard; the L138-143 degrade list (`cudaErrorMemoryAllocation`/`cudaErrorHostMemoryAlreadyRegistered`/"any other status") matches the catch-all non-throwing `STEPPE_CUDA_WARN` at L164. The graceful-degrade narrative (pin failure → pageable, never throw) matches the actual non-throwing ctor (L158-175).
- 8.3 Clean — rationale present wherever the code is non-obvious: `kSlots = 3` is justified against the Q/V/N input count + self-eviction reasoning (L276-278); the measured perf numbers (~44%, ~50-360ms register cost, ~51ms vs ~109ms/iter, ~2x) carry "MEASURED on rtxbox"/perf-discovery citations (L211-223); the `const_cast<void*>` on a const H2D source is justified as sound because `cudaHostRegister` only changes page state, not bytes (L148-150, L161-162); the overflow guard's rationale (modular unsigned wrap → silent under-allocation) is spelled out (L64-70); pinning-is-perf-not-correctness and §12 parity-neutrality are repeatedly motivated (L25-33, L136-145, L242-245). No bare magic value left unexplained.
- 8.4 Clean — no TODO/FIXME/HACK/XXX/WORKAROUND/TBD markers anywhere in the unit (grep confirmed); nothing orphaned or owner-less.

## Group 9 — Constants & configuration

- [9.1][LOW] src/device/cuda/pinned_buffer.cuh:162 — `void* p = const_cast<void*>(ptr);` is a single-use local: on the success path it is assigned to `ptr_` (L166) and otherwise unread; it is never re-pointed, so it could be `void* const p` (the const-cast strips constness of the pointee, but the pointer variable itself is a write-once local). Pure hygiene — no behavioral effect. Suggested: optionally declare `void* const p = const_cast<void*>(ptr);` to signal the local is fixed after init; or leave as-is (single-use, tightly scoped).

No further Group 9 issues found.

Notes (no defect):
- 9.1 (const/constexpr) otherwise Clean — the genuinely-immutable knob `kSlots` is already `static constexpr` (L279); the `cudaError_t` teardown/register statuses are already declared `const` (L112 `const cudaError_t e`, L163 `const cudaError_t s`, L201 `const cudaError_t e`); the loop binding `for (const Slot& s : slots_)` is already const (L262). The mutable members `ptr_`/`size_`/`next_`/`Slot::ptr`/`Slot::bytes` are deliberately mutable RAII/round-robin state (reassigned across calls), not should-be-const.
- 9.2 (tangled config) Clean — the only tunable knob in the unit is `kSlots` (L279), surfaced as a single named `static constexpr` with full rationale (L276-278) rather than buried as an inline literal in the eviction logic; it is the lone source of truth feeding both the `std::array<Slot, kSlots>` bound (L285) and the `% kSlots` modulus (L272). CUDA behavior selectors are named enum flags (`cudaHostAllocDefault` L81, `cudaHostRegisterDefault` L164), not magic ints embedded in logic. (The kSlots-vs-input-count coupling is already captured under 5.2; not re-flagged here.)
- 9.3 (positional booleans) Clean — no function in this unit takes a bool parameter, so there is no `foo(true,false,true)` opaque-call-site pattern: the ctors take `(std::size_t n)` / `(const void* ptr, std::size_t bytes)`, `ensure` takes `(const void* ptr, std::size_t bytes)`, and the CUDA calls pass named enum flags, not raw booleans.

## Group 10 — Initialization

No Group 10 issues found.

Notes (no defect):
- 10.1 Clean — every local is declared at its first use and initialized in the same statement: `void* p = const_cast<void*>(ptr)` (L162), `const cudaError_t s = ...` (L163), `const cudaError_t e = cudaFreeHost(ptr_)` (L112) / `cudaHostUnregister(ptr_)` (L201), the `const Slot& s` loop binding (L262), and `Slot& dst = slots_[next_]` (L268). No declared-far-from-use locals, no uninitialized-then-assigned: in `PinnedBuffer(std::size_t n)` (L72) `size_` is set in the init list and `ptr_` retains its NSDMI `nullptr` until the conditional `cudaHostAlloc` writes it at L80 (so `n == 0` leaves a well-defined null/zero state, not a garbage pointer).
- 10.2 Clean — no reliance on implicit zero-init that does not hold. Every data member has an explicit in-class initializer: `PinnedBuffer::ptr_ = nullptr` / `size_ = 0` (L122-123), `RegisteredHostRegion::ptr_ = nullptr` (L210), `Slot::ptr = nullptr` / `bytes = 0` / `reg{}` (L281-283), `next_ = 0` (L286); `slots_{}` (L285) value-initializes the array and each `Slot`'s NSDMIs make the members well-defined regardless of array value-init. The `cudaHostAlloc` region (L80) is intentionally NOT zeroed — it is a staging slot the host/device fills before transfer, and no code in this unit reads those bytes assuming a zero prefill, so there is no broken zero-init assumption (and the comment at L52-57 documents the fill-then-transfer contract).
