# Code Review (adversarial 2nd pass, verified) — `src/core/internal/views.hpp` (core-internal-views)

Unit under review: the `MatView` struct — the host-pure, CUDA-free Q/V/N column-major `[P × M]` contract that is the seam between the data front-end and the f2 kernel. 73 lines, header-only, one struct (`data`/`P`/`M`), one member function (`element`).

This is the adversarial verification pass over the prior draft. Every prior finding was re-checked line-by-line against the actual source, and every claim that turns on documented behavior was checked against a primary source:

- **MSVC `long` width** — verified against Microsoft's *Data Type Ranges* page: `long` = **4 bytes**, range −2,147,483,648 … 2,147,483,647; `long long` = 8 bytes. (https://learn.microsoft.com/en-us/cpp/cpp/data-type-ranges)
- **`STEPPE_DEBUG_ONLY` / `STEPPE_ASSERT` existence** — `grep -rn` across all `src/`, `include/`, `tests/` returns **zero** hits; the tokens appear only in `docs/architecture.md` prose (§7). Confirmed absent.
- **`span_view.hpp` existence** — `ls src/core/internal/` shows only `views.hpp`, `f2_estimator.hpp`, `decode_af.hpp`. Confirmed absent (architecture.md §8 lists it as the single "Views" home, but it is unbuilt).
- **ROADMAP §4 `size_t` disposition** — `docs/ROADMAP.md:98`: `| (size_t)i + (size_t)j*P | index cast | keep size_t (free, mandatory above P≈32k); not removed |`. Confirmed verbatim.
- **The Q/V/N contract single-source** — `docs/ROADMAP.md:52`: `element (pop i, snp s) at i + P·s`. Matches `views.hpp` exactly.
- **Consumers** — read `src/device/cpu/cpu_backend.cpp`, `src/device/cuda/cuda_backend.cu`, `src/device/cuda/f2_block_kernel.cu`, `src/device/cuda/f2_blocks_kernel.cu`, `src/device/backend.hpp`, `src/io/filter/snp_filter.hpp`, `src/core/domain/block_partition_rule.hpp`, `src/core/internal/f2_estimator.hpp`, `src/core/internal/decode_af.hpp`, `include/steppe/config.hpp`, and the `tests/`.

I confirm the prior draft's three corrections (the `STEPPE_ASSERT` false premise, the `span_view.hpp` absence, the `colmajor_index` type-collision against the kernel's `size_t` output indexing and ROADMAP §4) — all hold against the actual tree. I add several findings the prior draft missed (a `static_assert` ABI pin for the POD claim that the draft *asserts* but the file doesn't enforce; an imprecise doc-comment about which factor promotes; the `STEPPE_HD` macro-leak hazard that the recommended `colmajor_index` placement would inherit; the cross-file dependence on `long double` width that the `long` decision rhymes with) and reject a couple of new candidates. Net verdict below.

---

## Role & layering

`MatView` is the single host-side anchor for the Q/V/N contract (architecture.md §4 repo-layout line for `views.hpp` line 133; §8 "Views" DRY row line 527). Verified consumers, by grep + read:

- **CPU oracle** (`cpu_backend.cpp:127–140` for `compute_f2`, `248–254` for `compute_f2_blocks`): the *only* production path that calls `element()`. Confirmed by `grep -rn '\.element('`: hits are in `cpu_backend.cpp` and three `tests/reference/*.cu` only. It walks the (upper-)triangle calling `V.element`/`Q.element`/`N.element`.
- **CUDA backend** (`cuda_backend.cu:59–61, 76–80, 119–126, 196–204`): reads `.P`, `.M`, `.data` on the host and `cudaMemcpyAsync`s the raw buffer (`Q.data`, byte count `pm * sizeof(double)`); never calls `element()`. The device kernels recompute the index inline (`f2_block_kernel.cu:99–101`; `f2_blocks_kernel.cu:82–94`).
- **Backend interface** (`backend.hpp:135–166`): every f2 entry point takes `const core::MatView&` by reference; `views.hpp` is `#include`d at `backend.hpp:31`.
- **Tests**: aggregate-init `MatView{ptr,P,M}` + `element()` in scalar oracles (`test_f2_equivalence.cu:483–491`, `test_f2_blocks_equivalence.cu:149`).
- **`io` leaf deliberately does *not* use it** (`snp_filter.hpp:14–16,39`: "it takes plain double pointers, not core::MatView … so the `io` leaf does not depend on core::MatView") — the documented §4 layering decision, respected on both sides.
- **Downstream `long` copier** (`block_partition_rule.hpp:111`): "the loop index is `long` to match views.hpp `MatView::M`" — a *cross-file* consumer that copied the `long` choice, which elevates the `long`-width question (7.3) from a local nit to a contract issue.

Layering verdict: **clean and correct.** The header includes nothing, names no CUDA type, owns no storage, lives in `core/internal/` exactly where §4 and §8 place it. No upward dependency, no cycle.

The header is small and disciplined; several review categories are genuinely **N/A** and are marked as such rather than padded. The substantive findings cluster in: (a) a factually-wrong portability rationale for `long` (confirmed against MSVC docs); (b) a real but *refined* DRY gap — the column-major index is duplicated in two kernel files, but the fix is a *type-generic* helper, not a single-`long` one (the kernel output index is `size_t`, kept by ROADMAP §4); (c) the unresolved `span_view.hpp` relationship (file absent); (d) hygiene/contract gaps (no debug tripwire — but the assert facility it would use does not exist yet; no element-count helper; no `valid()`; no ABI `static_assert`); (e) an internal type mismatch (`M` is `long`, every count derived from it is `std::size_t`).

---

## Score: 8.5/10 — solid, correct, well-documented contract anchor; held at 8.5

Zero ownership, zero CUDA, zero includes, correct layering, dense load-bearing comments, `[[nodiscard]] … const noexcept` accessor, arithmetically-correct hot index. Held below 9.5–10 by: (1) the `long`-for-overflow rationale being **false on LLP64/MSVC** where `long` is 32-bit — *confirmed against Microsoft's Data Type Ranges* (`long` = 4 bytes); (2) the indexing rule duplicated verbatim in `f2_block_kernel.cu`/`f2_blocks_kernel.cu` rather than single-sourced — real, but the fix is a **type-generic** `colmajor_index<T>`, not a single-`long` one (the kernel output index is `size_t`, and ROADMAP §4 keeps that cast); (3) the unresolved §8 `span_view.hpp` convergence (the file does not exist); (4) hygiene gaps whose recommended fix (debug assert) depends on a `STEPPE_DEBUG_ONLY`/`STEPPE_ASSERT` facility that **does not exist** in the codebase. Not raised above 8.5 because (1) and (2) are real and cheap and ride before relying further on the oracle; not lowered because nothing here is a live bug and several candidate defects are rejected below.

---

## Findings

### (1) Correctness & bugs

**1.1 — `element()` index math is correct and consistent across all three paths. [severity: none — verification, CONFIRMED]**
`element(int i, long s)` returns `data[static_cast<long>(i) + static_cast<long>(P) * s]` (lines 66–68). Cast order verified: `static_cast<long>(P) * s` is `long*long` (no `int` intermediate that could overflow before promotion); the `static_cast<long>(i)` addend keeps the sum `long`. Cross-checked bit-for-bit against:
- CUDA feeder `f2_block_kernel.cu:99–100`: `const long Pl = static_cast<long>(P); const long idx = i + Pl * s;` (here `i`,`s` are already `long` per lines 95–96);
- batched feeder `f2_blocks_kernel.cu:82–83`: same `long Pl`/`twoP` form;
- CPU oracle `cpu_backend.cpp:127–140` / `248–254`: `V.element(i,s)`/`Q.element(i,s)` with the same column-major `i + P·s`.
Internally consistent. **Kept as evidence the hot index was actually checked.**

**1.2 — Doc-comment is imprecise about *which* factor promotes to `long`. [severity: low; effort: S; before-M4.5: no] — NEW (prior draft missed).**
Lines 49–50 say: "the flat index promotes to `long` before multiplying by `P`." But the code does **not** multiply the flat index by `P`, nor does it widen `i` and multiply *that* by anything — it widens **`P`** and multiplies `static_cast<long>(P) * s` (the `s` axis), then adds `static_cast<long>(i)`. The sentence as written reads as "widen `i`, then `× P`," which is the wrong mental model (it would suggest `i*P`, i.e. row-major). The arithmetic is correct; the *prose describing it* is misleading for a file whose entire job is to be the precise indexing contract. *Fix:* "the leading dimension `P` and the column index `s` are each widened to `long` before the `P·s` multiply, so the offset never overflows 32-bit even when `P·M > 2^31`." **Confirmed against lines 67; small but it is in the contract header, so it matters.**

**1.3 — `element()` returns `double` by value; document it as a read accessor. [severity: low; effort: S; before-M4.5: no]**
`data[…]` on `const double*` yields a `double` prvalue, not `const double&`. For a scalar this is preferable (register return, no aliasing). The only risk is a future maintainer "fixing" it into a reference and accidentally opening a write path on a read-only view. *Fix:* one sentence in the `element()` doc ("read-only by-value accessor; `MatView` is a read view"). **Confirmed; doc nit.**

**1.4 — No null check on a partially-constructed non-empty view. [severity: low; effort: S; before-M4.5: no]**
`data` defaults to `nullptr` (line 54); `P`/`M` default to `0`. A *default* `MatView{}` is the consistent empty view and `element()` is never called on it (callers guard `if (P <= 0 || M <= 0)` — `cpu_backend.cpp:111,207`; `cuda_backend.cu:62,127` early-return / zero-allocate). The hazard is `MatView{nullptr, P>0, M>0}`, which `element()` would dereference. A plain aggregate cannot prevent this (see 2.1). The doc already says "no bounds check (hot path)" (line 65); extend to "and no null check — a non-empty view must have non-null `data`." **Confirmed; doc nit. Runtime guard belongs in the (nonexistent) debug-assert path, see 2.4.**

### (2) Edge cases & failure modes

**2.1 — Aggregate cannot enforce `V≠0⟺N>0` or `P,M ≥ 0`. [severity: low; effort: M; before-M4.5: no]**
The header documents a hard producer invariant (line 33, `V != 0 ⟺ N > 0`) and implicit `P,M ≥ 0`, `data` non-null when non-empty. As a public-member aggregate, nothing rejects `MatView{p, -1, -5}`. §2 "fail-fast" and §9 "validated once" favor early rejection. *But* (why only low): the cross-view `V⟺N` invariant spans three separate `MatView`s and is **not checkable from a single one**; and `MatView` is intentionally a "minimal host-side anchor" (line 10). Right middle ground: a free `[[nodiscard]] constexpr bool valid() const noexcept { return P >= 0 && M >= 0 && (M == 0 || P == 0 || data != nullptr); }` plus a `tests/`-side `expect_qvn_consistent(Q,V,N)` asserting the cross-view invariant once at the seam. Keep the struct an aggregate. **Confirmed; correct scope.**

**2.2 — `P·M` / `P·P` element counts widened to `std::size_t` by *callers*, repeatedly. [severity: low; effort: S; before-M4.5: no] — CONFIRMED, count re-measured.**
`P` is `int` (correct; see rejected note on widening `P`). Every consumer recomputes the count by hand. Verified by `grep -cn 'static_cast<std::size_t>'`: **28** occurrences in `cpu_backend.cpp`, **24** in `cuda_backend.cu` (not all are `P·M`, but the `static_cast<std::size_t>(P) * static_cast<std::size_t>(...)` element-count pattern dominates — e.g. `cpu_backend.cpp:108,109,165,167,203,265,267,295,326`; `cuda_backend.cu:61,63,125,126,179`), plus the `* sizeof(double)` byte form at `cuda_backend.cu:76,78,80,96,99,196,198,200`. `MatView` is the natural single home for `size()` (= `P·M`) — see 6.1. **Confirmed; duplication is real and larger than "~6". No overflow today (each site widens before multiplying).**

**2.3 — `element(int i, long s)` accepts negative indices silently (UB). [severity: low; effort: S; before-M4.5: no]**
`i=-1`/`s=-1` ⇒ negative offset ⇒ `data[negative]` is UB. The doc states the precondition (line 65); for a hot accessor that is the right *contract*, but there is zero enforcement even in debug. Note `block_partition_rule.hpp` produces "negative positions get their own block" only for *genetic positions*, never for SNP **indices**, so `s` is non-negative in practice (the oracle loops `for (long s = s0; s < s1; ++s)` with `s0 ≥ 0`). **Confirmed as a latent gap; remedy is the debug assert (2.4), whose facility does not yet exist.**

**2.4 — No debug-only bounds/null assert — AND the macro the prior draft proposed does not exist. [severity: med (value), effort M; before-M4.5: no] — CONFIRMED CORRECTION.**
`grep -rn "STEPPE_DEBUG_ONLY\|STEPPE_ASSERT"` across all `.hpp/.cuh/.cu/.cpp` returns **zero hits** — the tokens appear *only* in `docs/architecture.md` prose (§7's `DeviceBuffer`/`STEPPE_CUDA_CHECK_KERNEL` snippets, lines 415, 443, 460). `src/core/internal/` contains only `views.hpp`, `f2_estimator.hpp`, `decode_af.hpp` — there is **no `log.hpp`, no `expected.hpp`, no `nvtx.hpp`, no `launch_config.hpp`** (architecture.md §8 lists all of these as planned single-homes; none are built). Only `check.cuh` exists, and it is CUDA-only, so a host POD cannot use it. So guarding `element()` is **not** a 1-line add: it first requires *introducing* a host debug-assert facility in `core/internal/` (the cross-cutting `STEPPE_DEBUG_ONLY`/`STEPPE_ASSERT` architecture.md §7 already specifies), then consuming it here. The *value* stands — `element()` is the single chokepoint of the trust-anchor oracle; a silent OOB read in the triangular loop (`cpu_backend.cpp:122–173`, `244–273`) could pass an equivalence test if the stray read happens to be benign, exactly what §13's sanitizer sweep exists to catch — but it is **med value, M effort, not before-M4.5** because the prerequisite facility is unbuilt and is a separate unit's concern. *Corrected fix:* (a) land a host `STEPPE_DEBUG_ONLY`/`STEPPE_ASSERT` in `core/internal/` (cross-cutting, tracked elsewhere); (b) then `STEPPE_DEBUG_ONLY(STEPPE_ASSERT(data && i >= 0 && i < P && s >= 0 && s < M))` in `element()`. **Finding kept; effort/ordering fixed; the false "already implied by §7" claim stays removed.**

**2.5 — Empty `M==0`/`P==0` views handled by convention, correctly. [severity: none — verification, CONFIRMED]**
Default member initializers make `MatView{}` the consistent empty view; callers early-return on `P<=0||M<=0` (`cpu_backend.cpp:111,207`; `cuda_backend.cu:62,127` allocate zero-length `DeviceBuffer`s). `element()` is never called on an empty view. **Correct as written.**

### (3) Numerical / precision vs §12

**3.1 — The view makes no numerical decision; no precision loss or reordering. [severity: none — verification, CONFIRMED]**
`MatView` carries `const double*` (line 54); `f2_blocks` storage is FP64 in every precision mode (config.hpp: `Precision` is an operation mode, not a storage type). `element()` is a pure load — no rounding, no reordering, no accumulation. The catastrophic-cancellation locus is `assemble_f2_numerator` (`f2_estimator.hpp:97–101`, held native FP64 per its own header), and the cross-SNP accumulation order lives in the oracle's `pairwise_sum` (`cpu_backend.cpp:71–81`) — both **outside** this view. **N/A — a view should make no numerical decision, and this one doesn't.**

*Cross-reference (out of scope for this unit, flag for the `cpu_backend` review):* the MSVC docs page just consulted also lists `long double` = "same as `double`" on MSVC. The oracle's cancellation-free property (`cpu_backend.cpp` `pairwise_sum` in `long double`) silently degrades to FP64 on Windows. That is the same LLP64-platform footgun family as 7.3, but it lives in `cpu_backend.cpp`, not `views.hpp` — noting it only so the two are tracked together.

### (4) CUDA idioms / RAII / stream & async / launch config / occupancy vs §7

**4.1 — RAII/streams/launch/occupancy N/A by design; the "shared indexing rule" DRY claim is real, but the fix must be type-generic. [severity: med; effort: M; before-M4.5: no] — CONFIRMED gap, FIX CORRECTED.**
Host-pure, CUDA-free (lines 6–10), so RAII-of-CUDA, streams, launch config, occupancy are **N/A** here — they belong to `DeviceBuffer`/`Stream`/`launch_config.hpp` (§4, §7). Correct separation.

The real gap: the doc calls `element()` "the one indexing rule shared by the CPU reference and the GPU feeder" (lines 62–65), but the **GPU re-derives the index inline** — `f2_block_kernel.cu:99–101` (`const long Pl = static_cast<long>(P); const long idx = i + Pl * s; const long sidx = i + (2 * Pl) * s;`) and `f2_blocks_kernel.cu:82–94` both open-code `i + Pl * c` etc. and never call `element()` (the kernels take raw `double*`, which is correct — `MatView` is host-only). So the column-major *formula* is duplicated in source: change the layout (row-major, or a padded `lda ≠ P`) and you must edit `views.hpp` + two kernel files in lock-step or the oracle and GPU silently diverge — the hazard §2/§8 warn about, and the one `f2_estimator.hpp` solves for the *formula* but no one solves for the *layout/index*.

**Why a single-`long` `colmajor_index` is wrong (corrected fix):**
1. The kernel **output** indexing is deliberately `size_t`, not `long`: `f2_block_kernel.cu:131` ("`size_t` indexing is mandatory above P≈32k"); `f2_blocks_kernel.cu:144–160` index the `[P×P]`/`[2P×P]`/`[P×P×n_block]` slabs in `size_t` (`Pp*Pp*k`, `twoP*Pp*k`, etc.). The kernel **input/feeder** index is `long` (`f2_block_kernel.cu:99`, `f2_blocks_kernel.cu:82`). A single fixed-return helper cannot serve both.
2. ROADMAP §4 line 98 explicitly keeps `(size_t)i + (size_t)j*P` with disposition **"keep `size_t` (free, mandatory above P≈32k); *not* removed."** Routing everything through a `long` `element()` would regress that.

*Corrected fix:* add a **type-generic** `template<class I> [[nodiscard]] STEPPE_HD constexpr I colmajor_index(I i, I s, I lda) noexcept`, have `element()` call it (`long`/`int64_t`), and have the kernels call it with `long` (input) / `size_t` (output, preserving the §4-mandated cast); the stacked-`S` `lda=2P` case is the same primitive with a doubled leading dimension. **Finding kept; fix is type-correct and §4-consistent.**

**4.2 — `M`-is-`long` vs the cuBLAS `int` boundary; doc caveat warranted. [severity: low; effort: S; before-M4.5: no]**
`MatView::M` is `long`, the kernel `s` axis is `long` (`f2_block_kernel.cu:96`) — consistent. cuBLAS v2 GEMM extents (`m`/`n`/`k`/`lda`) are `int` (the `int64_t` extents are only in the `*_64` API). `MatView::P` (`int`) is fine for `lda`/`m`. `MatView::M` as a GEMM `n` could in principle exceed `INT_MAX`, but the M4 design batches *per block* and pads each block's column count to a small bucket (`cuda_backend.cu:161` `ceil_bucket`; `f2_blocks_kernel.cu` `s_pad`), so per-GEMM extents are small and the `int` re-narrowing is legitimate. Worth one line in the `M` doc pointing at where the narrowing happens, so "M is wide" isn't read as "M is wide everywhere." **Confirmed; doc nit.**

### (5) Magic numbers & hardcoded values vs §4 / ROADMAP §4

**5.1 — Zero magic numbers. [severity: none — verification, CONFIRMED]**
The only literals are `nullptr`, `0`, `0L` (default member initializers) — true defaults. `P`-as-leading-dimension and `i + P·s` are mathematical layout facts, which ROADMAP §4 exempts ("no literal may survive … except true mathematical constants"; §4 line 98 even *names* this index expression as a kept cast). Cross-checked against config.hpp: nothing in `views.hpp` belongs there (`kHetCorrDenomFloor`, `kCdivBlock` live there and are consumed by `f2_estimator.hpp`, not here). **Fully ROADMAP-§4-compliant.**

### (6) Decomposition / single-responsibility / function size vs §2

**6.1 — SRP exemplary; the struct is *under*-provisioned, pushing count-widening to ~50 caller sites. [severity: low; effort: S; before-M4.5: no] — CONFIRMED.**
`MatView` does one thing; `element()` is a 1-liner. The flip side (2.2): the `static_cast<std::size_t>(P)*M` / `* sizeof(double)` widening is re-derived at dozens of sites. `[[nodiscard]] constexpr std::size_t size() const noexcept` (returns `static_cast<std::size_t>(P)*static_cast<std::size_t>(M)`) is the high-value addition; `bytes()` (= `size()*sizeof(double)`) would let the `cudaMemcpyAsync` calls read `Q.bytes()` (`cuda_backend.cu:76–80,196–200`). `std::span`/`mdspan` both expose `.size()`, so the helper is idiomatic, not scope-creep. *Caveat (§4-consistent):* `size()` returning `std::size_t` is exactly the widening §4 keeps — it **centralizes** the cast rather than removing it, composing with the §4 disposition rather than fighting it. **Confirmed; `size()` is the high-value one.**

**6.2 — Internal type inconsistency: `M` is `long`, everything derived from it is `std::size_t`. [severity: low; effort: S; before-M4.5: no] — CONFIRMED.**
`M` is `long` (line 60) and `element`'s `s` is `long`, but the *count* `P·M` is `std::size_t` at every call site (2.2), and the kernel output index is `size_t` (4.1). So the contract type for the SNP axis is `long`, while the contract type for everything sized *by* that axis is `size_t`. Latent mismatch: on LP64 (Linux — the actual targets) `long` and `size_t` are both 64-bit so invisible today; but the file's own "M won't overflow 32-bit" promise is carried by `long` (32-bit on LLP64, 7.3) while the actual counts use the always-≥64-bit `size_t`. Resolving 7.3 (`M → std::int64_t`) + adding `size()` (6.1) makes the axes coherent: a 64-bit signed extent and a `size_t` count derived from it. **Reinforces 7.3 + 6.1.**

### (7) Readability, naming, const-correctness, [[nodiscard]]/noexcept, comment density

**7.1 — `element()` attributes are correct; const-correct. [severity: none — verification, CONFIRMED]**
`const`, `[[nodiscard]]`, `noexcept` (line 66) — right for a pure read (no side effects; dropping the result is a bug; cannot throw). `data` is `const double*` (read-only view). Public members on an aggregate by design. **Good.**

**7.2 — Comment density is high and purposeful. [severity: none — verification, CONFIRMED]**
The contract block (lines 1–36) pins Q/V/N semantics, the `V≠0⟺N>0` invariant, the AT2 allele-count convention (haploid count = 2×diploids or 1×pseudo-haploids), and the relationship to `span_view.hpp`/`block_partition_rule`. Matches `f2_estimator.hpp`/`decode_af.hpp`/`backend.hpp` density; the comments are load-bearing (they pin AT2 conventions consumed by the oracle and kernel). **Good — not over-commented.**

**7.3 — The `M`-is-`long` rationale is factually wrong on LLP64 — CONFIRMED against Microsoft's Data Type Ranges. [severity: med; effort: S; before-M4.5: no]**
Lines 49–50 + 59–60: "`M` is `long` so a [P × M] view over a large SNP block does not overflow a 32-bit count." On **LP64** (Linux — the `steppebox5090` / vast / RTX PRO 6000 targets) `long` is 64-bit, so it works. The *stated reason* is false on **LLP64 (Windows/MSVC)**: Microsoft's *Data Type Ranges* page lists **`long` = 4 bytes, −2,147,483,648 … 2,147,483,647** (i.e. 32-bit), `long long` = 8 bytes — verified directly, not from memory. So on MSVC `long` is exactly the 32-bit type the comment claims to escape, and CUDA 13 supports the MSVC host compiler, so a Windows build would silently revert `MatView::M` (and `block_partition_rule.hpp:111`'s copied `long` loop index, which back-references this very choice) to 32-bit. The portable ≥64-bit type is `std::int64_t` (signed; keeps `s < 0` debug checks meaningful) or `std::ptrdiff_t`. *Fix:* `std::int64_t M = 0;`, `element`'s `s` → `std::int64_t`, add `#include <cstdint>`, and rewrite the comment to "`M` is `std::int64_t` so the count is 64-bit on every platform (on LLP64/Windows `long` is only 32-bit — see MSVC Data Type Ranges)." Effort genuinely **S** (one include + a type swap; `<cstdint>` is already pulled into this directory by `decode_af.hpp:39`, so the include is uncontroversial). **Confirmed by primary source; the single highest-value correctness-of-intent fix.** Severity stays **med** (operationally Linux-only today) — but the file's entire job is to be the precise contract, and it documents a wrong reason.

**7.4 — `MatView` name doesn't signal host-only/FP64-only vs the planned `span_view.hpp`. [severity: low; effort: S; before-M4.5: no]**
Lines 7–10 promise a "full `span_view.hpp` (architecture.md §8)" over `cuda::std::span`/`mdspan`, with `MatView` the "minimal host-side anchor." So two view vocabularies are planned. `MatView` doesn't telegraph "host-only, FP64-only, raw-pointer"; it would compile in a `.cu` and a reader might reach for it in device code. *Fix:* add one line ("`MatView` is the **host-side** anchor; device code uses the `cuda::std::mdspan` views from `span_view.hpp` (§8)") or rename to `HostMatView`/`MatViewF64`. See 9.1 for the deeper convergence question. **Confirmed; doc/naming nit.**

**7.5 — Header-guard, namespace, naming all consistent with siblings. [severity: none — verification, CONFIRMED]**
`STEPPE_CORE_INTERNAL_VIEWS_HPP` matches the `STEPPE_CORE_INTERNAL_*_HPP` convention of `f2_estimator.hpp`/`decode_af.hpp`; `namespace steppe::core` matches; the file-path banner comment matches the sibling style. **Good.**

### (8) Performance

**8.1 — `element()` is optimal for the path it serves; zero abstraction penalty. [severity: none — verification, CONFIRMED]**
`inline noexcept` returning a single indexed load by value ⇒ compiles to the same code as the open-coded index. The CPU oracle's `O(P²·M)` triangular loop (`cpu_backend.cpp:122–173`, `244–273`) calls it in the inner loop, but inlining makes it free. The GPU hot path never calls it (memcpy + inline index), so zero device cost. The repeated `static_cast<long>(P)*s` per call is hoisted/strength-reduced by any optimizer, and the oracle is the *reference*, not the perf path. **Performance-correct; N/A for a fix.**

### (9) Layering / API / ABI vs §4

**9.1 — Two parallel view vocabularies; `span_view.hpp` does NOT exist yet. [severity: med; effort: M; before-M4.5: no] — CONFIRMED (absence verified).**
§8's DRY table (line 527) lists a single "Views" home: `internal/span_view.hpp` → views over `cuda::std::span`/`mdspan`. `ls src/core/internal/` shows **only** `views.hpp`, `f2_estimator.hpp`, `decode_af.hpp` — `span_view.hpp` is **not built**. So today: `views.hpp::MatView` (host, raw `const double*`, exists) vs a *planned* `span_view.hpp` (the §8 single-home, absent). That is two view families for "the same concern," with no recorded decision on whether `MatView` becomes (a) a thin alias over the `span_view` templates, (b) a permanent host-only sibling, or (c) absorbed. Left unresolved this is a future §8 DRY violation. *Fix:* record the intended end-state in a one/two-line note or a ROADMAP/ADR line ("`MatView` is the FP64 host anchor; `span_view.hpp` is the device/typed generalization; `MatView` becomes `using MatView = HostMatrixView<const double>` at M-N, or stays the deliberate minimal host sibling"). **Confirmed; the file *acknowledges* the tension (lines 7–10) but does not resolve it.**

**9.2 — Header is self-contained and trivially-copyable, but the POD/ABI property is asserted only in prose, not in code. [severity: low; effort: S; before-M4.5: no] — NEW (prior draft missed).**
`MatView` is passed by `const MatView&` across the `core ↔ device` C++ seam (`backend.hpp:135–166`), never across the public C ABI (which uses `steppe_status_t` per §10/§16). It *is* a trivially-copyable standard-layout POD (one pointer + `int` + `long`), so by-value passing is cheap and ABI-safe within the library — but nothing in the file *enforces* that. A future field that breaks triviality (e.g. a `std::function`, a smart pointer, a non-trivial default) would silently invalidate every assumption that `MatView` is cheap to copy and stable to pass between TUs compiled by the host compiler and by nvcc. *Fix:* one line after the struct — `static_assert(std::is_trivially_copyable_v<MatView> && std::is_standard_layout_v<MatView>);` (requires `#include <type_traits>`). Pins the ABI claim the prior draft merely asserted. Low cost, prevents a real regression class on a contract type crossing a compiler boundary. **New; confirmed against `backend.hpp` usage.**

**9.3 — Empty include set is correct *today*; the future fixes add includes. [severity: low; effort: S; before-M4.5: no] — CONFIRMED, refined.**
The header uses only `int`/`long`/`double*`/`nullptr` (all builtin), so it legitimately includes nothing and is IWYU-clean as written — a point in its favor (architecture.md §14 runs IWYU in CI). The `std::int64_t` swap (7.3) **requires** `#include <cstdint>`; a future `size()`/`bytes()` returning `std::size_t` (6.1) requires `#include <cstddef>`; the `static_assert` (9.2) requires `#include <type_traits>`. `<cstdint>` is already used by sibling `decode_af.hpp:39`, `<cstddef>` by `config.hpp:26`, so they are uncontroversial. **Confirmed; flag the includes *with* the fixes that need them, not before.**

### (10) Testability vs §13

**10.1 — Trivially host-testable, exercised indirectly, but no direct unit test. [severity: low; effort: S; before-M4.5: no] — CONFIRMED.**
`element()` is pure `__host__` (no CUDA, no `STEPPE_HD` — it never runs on device), so it is CPU-unit-testable per §13. It *is* used inside `test_f2_equivalence.cu:483–491`, `test_f2_blocks_equivalence.cu:149` — indirectly covered. But `ls tests/unit/` shows `test_block_partition.cpp`, `test_f2.cpp`, `test_filters.cpp` — **no `test_views.cpp`** directly pinning `element(i,s) == data[i + P*s]` at the corners (`element(0,0)`, `element(P-1, M-1)`), the empty-view convention, or (when 2.4 lands) the debug tripwire. Given `element()` is the single source of the indexing rule the whole oracle trusts, a ~10-line direct test is cheap insurance and matches §13's "pure `__host__ __device__` numerics tested in plain `.cpp`." More valuable once the type-generic `colmajor_index` (4.1) lands. *Fix:* `tests/unit/test_views.cpp`. **Confirmed.**

### (11) Capability tiers (PRO-6000 vs budget-5090) — TODO.md CAPABILITY-TIER

**11.1 — Tier-agnostic, and correctly so. [severity: none — verification, CONFIRMED]**
TODO.md's capability matrix (GDS/cuFile, `ncu --set full`, P2P/`cudaMemcpyPeer`, RLIMIT_MEMLOCK probes) is all data-movement/observability, parity-neutral. `MatView` describes a host memory layout; it touches none of those levers. The capable-vs-budget split lives in the allocator, the §11.4 multi-GPU combine, and the device probes — **not** in a layout descriptor. So this unit needs no capability probe / tagged-fallback log line; adding one would be misplaced. **Correctly tier-neutral.**

*Forward-looking note (argument for 6.1, not a finding):* TODO.md M4.5 adds the `canAccessPeer`-gated P2P device-combine pulling per-device partial `f2_blocks` (`[P×P×n_block]` slabs — column-major, exactly this kind of view) via `cudaMemcpyPeer`. If/when that lands, `size()`/`bytes()` (6.1) makes the peer-copy byte count single-sourced rather than re-derived at the capable-path call site. No tier-specific code belongs *in* `views.hpp`.

---

## Considered & rejected

- **"`P` should be `long`/`std::int64_t` for symmetry with `M`."** *Rejected (agree with prior draft).* `P` is the population count (≤ ~4266 real AADR per §5/§12; ≤ ~3,331 single-GPU on the 96 GB PRO 6000 per TODO.md) and is the cuBLAS leading dimension / GEMM `m`, which is `int` in the v2 API. `int P` is correct; widening forces narrowing casts at every cuBLAS call. The `int P` / wide `M` asymmetry is intentional.

- **"Make `MatView` a class with private members + a validating constructor (full RAII per §2)."** *Rejected (agree).* It is a non-owning hot-path descriptor brace-initialized in tests (`MatView{ptr,P,M}` — C++20 aggregate init). Aggregate-ness is a feature (cheap, trivially-copyable, designated-init-friendly). The right validation is a free `valid()` + a debug assert (2.1, 2.4), not constructor encapsulation, which would break every `MatView{ptr,P,M}` call site for no real safety gain (the cross-view `V⟺N` invariant isn't checkable from one view anyway).

- **"`element()` should return `const double&`."** *Rejected (agree).* By-value `double` is strictly better for a scalar read (register return, no aliasing); the view is read-only. Doc clarification only (1.3).

- **"The `2` in the `[2P × M]` stacked-`S` / `lda=2P` is a magic number."** *Rejected (agree).* It's the `[Qsq;Hc]` stacking factor (`2P` rows; `f2_block_kernel.cu:100` `sidx = i + (2 * Pl) * s`), a structural constant of the reformulation (architecture.md §5 S2; same exemption ROADMAP §4 grants the `2` in `a²−2ab+b²`, mirrored in `assemble_f2_numerator`'s `2.0*cross`, `f2_estimator.hpp:100`). Not a `views.hpp` literal anyway.

- **"Add `<cstddef>`/`<type_traits>` now."** *Rejected as standalone defects (agree).* The header uses no `std::size_t`/`<type_traits>` today; it is IWYU-clean. The includes become required only with 6.1/7.3/9.2 (flagged in 9.3).

- **"No-bounds-check `element()` is a bug."** *Rejected as a release bug (agree); accepted as a debug gap.* No-bounds-check on a documented hot accessor is the correct release design (inner loop of the `O(P²·M)` oracle). The ask is a *debug-only* assert (2.4) — gated on a facility that does not yet exist.

- **"Add the debug assert to `element()` now, effort S, before M4.5."** *Rejected as stated; downgraded.* Verified false premise: `grep` returns zero occurrences of `STEPPE_DEBUG_ONLY`/`STEPPE_ASSERT`; no host assert/log facility exists in `core/internal/`. So the change depends on first building the cross-cutting host debug-assert helper. Re-scoped to **med value, M effort, not before-M4.5** (2.4).

- **"Route the kernel indices through a single `long`-returning `colmajor_index`."** *Rejected as stated; corrected to a type-generic helper.* A `long` return is wrong for the kernel's `size_t` output indexing (`f2_block_kernel.cu:131`, `f2_blocks_kernel.cu:144–160`), and ROADMAP §4 line 98 explicitly *keeps* the `size_t` index cast ("not removed"). The DRY gap is real; the fix must be `colmajor_index<T>` (or two named helpers) preserving `size_t` at the output sites (4.1).

- **"`element()` should be `STEPPE_HD` to be reusable on device."** *Rejected (agree).* `MatView` is the deliberate host-only anchor (lines 7–10); the device path indexes raw pointers and would use `span_view.hpp`, not `MatView`. Making `element()` `__host__ __device__` would invite exactly the device-side `MatView` use that 7.4 warns against. The type-generic `colmajor_index` (4.1) — *not* `element()` — is the thing that should be `STEPPE_HD` and shared.

- **NEW candidate — "Put `colmajor_index<T>` in `f2_estimator.hpp` (as the prior draft suggests) — but note the `STEPPE_HD` macro leaks."** *Accepted as a caveat, not a `views.hpp` finding.* `f2_estimator.hpp:35–39` and `decode_af.hpp:41–45` both `#define STEPPE_HD` and **neither `#undef`s it**, so it leaks into every TU that includes them. Co-locating `colmajor_index<T>` there inherits that leak. This is a real hygiene issue but it belongs to the `f2_estimator.hpp`/`decode_af.hpp` reviews, not this one (`views.hpp` defines no macro). Flagged here only so the 4.1 fix doesn't blindly worsen it; the helper's home should `#undef STEPPE_HD` at end-of-header or live in the planned `launch_config.hpp` with a single guarded definition.

- **NEW candidate — "`MatView` should be `final` / its members `const`."** *Rejected.* Making members `const` would delete the implicit copy-assignment and break the aggregate's reseat-ability (callers build a fresh `MatView` per block); `final` on a POD descriptor buys nothing (no inheritance is intended or possible to misuse cheaply). The struct is correctly a plain mutable-by-reassignment aggregate.

- **NEW candidate — "`element` should take `std::size_t` indices to match the count type."** *Rejected.* Signed `long`/`int64_t` indices keep the negative-index debug check (2.3/2.4) meaningful; an unsigned index would make `i = -1` wrap to a huge positive value and defeat the bounds assert. Signed extent + `size_t` *count* (6.2) is the right split; the *index* parameters stay signed.

---

## What it takes to reach 10/10

1. **Fix the `long` rationale (7.3, confirmed vs MSVC docs).** `M` and `element`'s `s` → `std::int64_t`; add `#include <cstdint>`; rewrite the comment to name LLP64 (`long` = 32-bit on Windows, per MSVC Data Type Ranges). Removes a documented falsehood; makes the overflow guarantee real on every platform and consistent with the `size_t` counts (6.2). *(med, S — the one to ride in before relying further on the contract.)*
2. **Single-source the column-major index — type-generically (4.1, corrected).** Add `template<class I> STEPPE_HD constexpr I colmajor_index(I i, I s, I lda) noexcept` (beside `cdiv`/`grid_for` in `f2_estimator.hpp`, `#undef`-ing the leaked `STEPPE_HD`, or in the planned `launch_config.hpp`); have `element()` call it with `int64_t`, and the kernels call it with `long` (input) / `size_t` (output, preserving the §4-mandated cast). Kills the duplicated layout rule across `views.hpp` + both kernels. *(med, M)*
3. **Add `size()` (6.1) — and optionally `bytes()`.** `[[nodiscard]] constexpr std::size_t size() const noexcept` collapses the ~50 `static_cast<std::size_t>(P)*M` / `*sizeof(double)` sites (2.2) into `Q.size()`/`Q.bytes()`; *centralizes* the §4-kept `size_t` cast rather than removing it. *(low, S)*
4. **Pin the ABI/POD property with a `static_assert` (9.2).** `static_assert(std::is_trivially_copyable_v<MatView> && std::is_standard_layout_v<MatView>);` (+ `#include <type_traits>`). Turns the "trivially-copyable, safe across the nvcc/host seam" claim from prose into a compile-time guard. *(low, S)*
5. **Add `valid()` + a `tests/`-side `expect_qvn_consistent` (2.1).** Cheap host checkers for `P,M ≥ 0` / non-null-when-non-empty and the cross-view `V≠0⟺N>0`, at the seam; struct stays an aggregate. *(low, M)*
6. **Land a host debug-assert facility, then guard `element()` (2.4, re-scoped).** First introduce `STEPPE_DEBUG_ONLY`/`STEPPE_ASSERT` in `core/internal/` (does not exist — cross-cutting, tracked outside this unit); then `STEPPE_DEBUG_ONLY(STEPPE_ASSERT(data && i>=0 && i<P && s>=0 && s<M))` in `element()`. Zero-cost in release; turns silent OOB in the trust-anchor oracle loud under §13's sanitizer/debug runs. *(med value, M effort — depends on the facility.)*
7. **Resolve `MatView` ↔ `span_view.hpp` in writing (9.1).** `span_view.hpp` does not exist; decide alias/sibling/absorb and record it so §8's single Views home isn't quietly violated. Add the host-only/FP64-only one-liner or rename (7.4). *(med, M)*
8. **Add `tests/unit/test_views.cpp` (10.1).** Pin `element()` at the corners + empty-view + (once 6 lands) the debug tripwire; also unit-test the type-generic `colmajor_index` from (2). *(low, S)*
9. **Doc touch-ups:** fix the "promotes to `long` before multiplying by `P`" prose (1.2); read-by-value-by-design (1.3); the legitimate `int` re-narrowing at the per-block cuBLAS boundary (4.2); no-null-check on non-empty views (1.4); note the `long`/`size_t` axis coherence after 7.3 (6.2). *(low, S)*

Before M4.5: none is strictly required, but (1) is trivial + high-value-of-correctness, (2) closes a real divergence hazard in the trust seam, and (4) is a 2-line ABI guard — all three are good to ride in. The rest is quality polish; (6) is gated on infrastructure this unit does not own.

## Good patterns to keep

- **Zero ownership, zero CUDA, zero includes** — a textbook non-owning host view; the §4/§8 shape exactly. The `io` leaf's deliberate refusal to depend on it (`snp_filter.hpp:14–16` raw `double*`) shows the boundary is understood on both sides.
- **`[[nodiscard]] … const noexcept` accessor** — correct attributes for a pure read; verified.
- **`long` SNP axis / `int` population axis — the *intent* is exactly right** (wide SNP count; narrow pop count matching cuBLAS `int` dims). Only the *spelling* (`long` → `int64_t`) and its stated reason need fixing.
- **Dense, load-bearing contract comments** — Q/V/N semantics, `V≠0⟺N>0`, the AT2 allele-count convention, the pointer to `block_partition_rule` for block membership. The right place to pin the contract; matches `f2_estimator.hpp`/`decode_af.hpp` density.
- **Default member initializers ⇒ `MatView{}` is the consistent empty view** — composes cleanly with callers' `P<=0||M<=0` early-returns (`cpu_backend.cpp:111,207`); no half-constructed default.
- **No magic numbers** — fully ROADMAP-§4-compliant; only true defaults / mathematical layout facts (ROADMAP §4 line 98 even names the index expression as a kept cast).
- **The contract is genuinely single-sourced for the consumers that take `MatView`** (CPU oracle, orchestration, tests all funnel through `element()`); the only leak is the GPU kernels, which legitimately can't take a host view (4.1).
