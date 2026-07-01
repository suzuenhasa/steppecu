# Review — `io/filter/snp_filter` (`snp_filter.hpp` + `snp_filter.cpp`)

Reviewer: ADVERSARIAL second pass (single-unit deep dive). Quality bar: 9.5–10/10
(architecture.md §2/§4/§7/§8/§9/§11/§12/§13, ROADMAP §4/§5/§6, TODO capability tiers).
Read line-by-line and re-verified against: `snp_filter.{hpp,cpp}`, `filter_decision.hpp`,
`include_exclude.{hpp,cpp}`, `mind_prepass.hpp`, `filter_plan.hpp`, `snp_reader.hpp`,
`include/steppe/config.hpp`, `device/backend.hpp` (`DecodeResult`/`DecodeTileView` contract),
`core/internal/decode_af.hpp` (`finalize_af`), `tests/unit/test_filters.cpp`,
`tests/reference/test_filter_oracle.cu`, plus architecture.md §2/§13 and TODO §"Keeping it
GPU-dominant" / capability-tier table. External standard-library facts cited inline were looked up
(Microsoft Learn `vector<bool>`, SEI CERT STR37-C); cppreference returned 403 so the equivalent
normative sources were used.

This pass **re-verified all 20 first-pass findings** (confirmed 18, downgraded/merged 0, moved 0 to
rejected — they all hold), **added 8 new findings** the first pass missed (F21–F28, incl. a real
API-contract smell, an `int`-truncation tie-in, a sharpening of the F19 bit-identity claim into a
*correctness precondition*, a determinism/reduction-order analysis the first pass omitted entirely,
and a more severe reading of the test gap against §13), and **expanded the Considered & rejected
ledger**. Net verdict moves slightly DOWN from the first pass: the same defensive-input debt, plus a
newly-identified contract smell and a stronger §13 reading, against a unit with *no production caller*
that the roadmap is about to rewrite.

## Role & layering

`snp_filter` is the per-tile **SNP-level keep-mask producer** (architecture.md §5 S0′ "cheap filters
decidable from one tile"; ROADMAP M2). Two host-pure functions: `derive_per_snp_summary` (pooled
folded MAF + per-SNP missing fraction + pooled allele count, each a SNP-global reduction across the
kept pops) and `build_snp_keep_mask` (applies the shared `filter_decision.hpp` predicates +
`SnpMembership` + the flag-gated monomorphic/transversion/autosome options into one
`std::vector<bool>`). It is an **`io`-leaf TU** (architecture.md §4): pure host C++20, no CUDA, no
`core`/`device` dependency. It deliberately takes plain `const double*` Q/V/N (the M1
`DecodeResult.q/.v/.n` contract) rather than `core::MatView`, keeping the leaf free of
`core/internal` — verified correct and explicitly justified in the header (lines 38–39).

Layering is **clean and verified**: the include set is `<cstddef>`, `<vector>`, `<string>`,
`include_exclude.hpp`, `snp_reader.hpp`, `config.hpp`, `filter_decision.hpp` — all `io`-leaf or
public CUDA-free config; nothing upward, no CUDA header. The decision *logic* lives once in
`filter_decision.hpp` (the §8 single-source rule); this TU only orchestrates. `DecodedTileSummaryInput`
mirrors `device/backend.hpp::DecodeResult` field-for-field WITHOUT including `backend.hpp` (correct
decoupling — the `app`/test layer bridges them).

**Material context for scoring (re-verified by grep):** this unit has **NO production caller**.
`grep -rn 'build_snp_keep_mask|derive_per_snp_summary'` across `src/` + `tests/` returns ONLY
`tests/reference/test_filter_oracle.cu` (lines 259, 288, 321, 400, 417, 471). `FilterPlan::snp_keep`
exists but **nothing in `src/` populates it from this function** — confirmed. And TODO §"Keeping it
GPU-dominant" item 3 names *this very file* verbatim as "the clearest host liability in
`snp_filter.cpp` today", slated for M4.5 device fusion. Both facts shape the findings.

## Score: 8.0/10 — disciplined, oracle-pinned, layering-exemplary host code, but the 9.5–10 bar is missed on (a) four documented preconditions silently absorbed instead of fail-fast (§2), (b) an API-contract smell the first pass missed (an already-folded MAF re-folded by `is_monomorphic`), (c) a §13 test gap that is *more* severe than first scored — the unit's ONLY coverage requires the GPU/data box, violating the "host, no GPU" testability principle, and (d) an unaddressed capability-tier/device-fusion seam in the explicitly-named M4.5 fusion target, plus a `vector<bool>` return shape that the fusion cannot copy to the device.

The core math matches the independent scalar oracle integer-exactly across 7 configs (`test_filter_oracle.cu`
(c)), and the drop-equals-mask f2 identity holds bit-for-bit (test (b)), so **correctness on the
validated AADR path is not in question** — verified by reading both the production reduction
(`snp_filter.cpp` lines 36–56) and the oracle reduction (`test_filter_oracle.cu` lines 147–157): they
are the SAME arithmetic in the same order, which is *why* (c) is integer-exact. The deductions are
for: fail-fast gaps that turn caller-contract violations into silent wrong answers or segfaults; a
dead `v` field; the `is_monomorphic` double-fold contract smell; a double-pass/`vector<bool>`
performance-and-seam issue; and the absence of any capability-tier story in the named M4.5 fusion
target. None is a correctness bug on real AADR; several are latent wrong-answers on adversarial input,
and all are cheap to fix. I score 8.0 (not the first pass's 8.5): the new contract smell (F21) and the
sharpened §13 reading (F23) are real, and a 9.5–10 unit does not silently absorb four documented
preconditions.

## Findings

### (1) Correctness & bugs

**F1 [med] CONFIRMED — `pop_individuals` shorter than P silently truncates `total_indiv` →
understated missing fraction; `missing_frac` can go negative and SPURIOUSLY PASS the geno filter.**
`derive_per_snp_summary` lines 26–28 guard the *denominator* loop with `p < P && p <
static_cast<int>(in.pop_individuals.size())`. But the *numerator* loop (lines 39–48) runs `p < P`
unconditionally, summing `npn/ploidy_d` for **all** P pops. If `pop_individuals.size() < P` (a caller
contract violation — the header documents `pop_individuals` as "length P", line 52), `nonmissing_indiv`
includes pops never counted in `total_indiv`, so `missing_frac = 1 - nonmissing/total` is computed
against a too-small denominator. Re-verified the failure direction: with `nonmissing > total`,
`missing_frac < 0`, and `snp_passes_geno(negative, geno_max_missing)` returns `negative <=
geno_max_missing` = **true** (filter_decision.hpp line 94) — a missing-heavy SNP is KEPT. Silent
wrong-answer, not a diagnosable error. *Why it matters:* architecture.md §2 fail-fast verbatim
("invalid arch lists, missing devices, non-SPD covariance … surface immediately with file/line
context, not as silent corruption"); §11.4 (the partition supplies these sizes — a sharding bug here
would silently corrupt the geno filter). *Fix:* at function entry, throw `std::invalid_argument` if
`in.pop_individuals.size() != static_cast<std::size_t>(P)` (when `P > 0`). The `io` leaf already
surfaces bad input as exceptions — `include_exclude.cpp` line 17 throws `std::runtime_error` on a bad
prune.in path, so throwing here is idiomatic for this layer. Do NOT clamp. Severity med, effort S,
before-M4.5: **yes** (the M4.5 fusion moves this denominator onto the device — pin the precondition
first).

**F2 [low] CONFIRMED — `pooled_ref_af` is unclamped; a Q outside [0,1] would yield a negative folded
MAF and mis-drop. No bug on contracted input.** Lines 52–53 compute `pooled_ref_count /
pooled_allele_count`. Re-verified the decode contract: `finalize_af` (decode_af.hpp lines 96–109)
guarantees `Q = AC/N` with `AC ≤ N` (AC = Σ codes over non-missing individuals, each code ≤ ploidy;
N = ploidy·AN), so `Q ∈ [0,1]` exactly on the contracted path and `Q·N` re-derives the integer AC.
**No bug on contracted input.** But the function takes raw `const double*` with no range contract; a
future imputation/mean-fill `2p` producer (§12) passing Q outside [0,1] would yield `folded_maf < 0`
(filter_decision.hpp line 74–77: `min(q, 1-q)` with `q>1` gives `1-q<0`) and `snp_passes_maf` would
mis-decide. *Fix:* document the `Q ∈ [0,1]` precondition on `DecodedTileSummaryInput::q` (cheapest;
do NOT clamp, which would mask a wiring bug). Severity low, effort S, before-M4.5: no.

**F3 [low → med if a real pipeline mismatches] CONFIRMED — default `'N'`/`0`/`kEmptyId` fallback on a
short `SnpTable` silently DROPS SNPs (or alters membership) instead of erroring.** `build_snp_keep_mask`
lines 76–77, 94, 101 read `snps.ref/alt/chrom/id` with bounds guards substituting `'N'`/`0`/`kEmptyId`
when `SnpTable` is shorter than M. Re-verified the consequences against `filter_decision.hpp`:
`normalize_allele('N')` → `'\0'` (line 125), so `is_multiallelic('N','N')` → true (line 177) → the
SNP is DROPPED; `chrom == 0` fails `is_autosome` (line 211, `0 < kAutosomeChromMin==1`); an empty id
fails a non-empty include set (`passes` line 59). So a too-short `SnpTable` **silently zeroes the tail
of the tile**. The header says "`snps.count` must be >= `in.M`" (line 99) — a precondition silently
absorbed. *Why it matters:* §2 fail-fast. A mismatched `.snp`/decode length is a wiring/data bug that
must abort with context. *Fix:* once at entry, check `snps.ref.size() >= M && snps.alt.size() >= M`
(and `chrom`/`id` size when those filters/membership are active), throw on violation; the per-element
ternaries then disappear (perf + clarity, see F18, F26). Severity low (med if a pipeline ever
mismatches), effort S, before-M4.5: no.

**F4 [low — VERIFIED SAFE] CONFIRMED — `is_multiallelic` || `is_strand_ambiguous` ordering is
correct.** Re-verified: `is_strand_ambiguous` returns false for any non-ACGT char
(filter_decision.hpp line 164), so the `||` is order-independent for correctness; `is_multiallelic`
first is the right perf order (common case "not multiallelic" on a clean panel, ambiguous check is
the cheap second test). No bug. Both predicates re-`normalize_allele` the same chars — ties into the
redundant-normalize perf note (F26).

**F5 [low] CONFIRMED — `ploidy <= 0` is silently coerced to 1, not rejected.** Line 30: `in.ploidy >
0 ? in.ploidy : 1`. A non-positive ploidy is nonsensical metadata; coercing to 1 produces a
plausible-but-wrong missing fraction (doubles `nonmissing_indiv` vs diploid truth) rather than
aborting. Re-verified the ROADMAP/decode contract: ploidy is "a METADATA parameter, NEVER
auto-detected" (decode_af.hpp lines 23–25, 90–91; backend.hpp lines 76–80), with `{1, 2}` the only
meaningful values. *Why it matters:* §2 fail-fast — a 0/negative ploidy is a config error. *Fix:*
validate `in.ploidy == 1 || in.ploidy == 2` (or at least `> 0`) at entry, throw. The defensive `?:`
masks a wiring bug. Severity low, effort S, before-M4.5: no.

**F21 [low, NEW — first pass missed] `is_monomorphic(sm.pooled_minor_af)` passes an ALREADY-FOLDED
MAF into a predicate that re-folds it — idempotent today, but a latent contract trap.** Line 91 calls
`is_monomorphic(sm.pooled_minor_af)`. `sm.pooled_minor_af` is already `folded_maf(pooled_ref_af)`
(line 54), i.e. it is in `[0, 0.5]`. But `is_monomorphic` (filter_decision.hpp lines 110–112) does
`return folded_maf(pooled_minor_af) == 0.0;` — it FOLDS AGAIN. Re-verified the math: `folded_maf` of
a value already in `[0,0.5]` is the identity (`min(x, 1-x) == x` for `x ≤ 0.5`), so the double-fold is
idempotent and **the result is correct today**. BUT: (a) the `is_monomorphic` parameter is *named*
`pooled_minor_af` yet its body treats it as a *ref_af* (it folds it) — the predicate's own contract is
self-contradictory, and a caller reading the signature cannot tell whether to pass the folded or the
unfolded value; (b) if `is_monomorphic`'s contract is ever "corrected" to expect the unfolded ref_af
(so it folds once), THIS call site silently breaks for any SNP with `ref_af > 0.5` (it would pass
`pooled_minor_af` = `min(ref_af, 1-ref_af)` and the predicate would re-fold a value already ≤ 0.5,
still giving the wrong answer only if the contract flips). This is exactly the kind of "two places
each assume a different convention" trap §8 single-sourcing is meant to prevent. *Why it matters:*
§8 (one home for the rule), §2 (clarity/contracts). The drop_monomorphic config doc (config.hpp lines
236–240) says monomorphic ≡ "pooled MAF exactly 0" — so the *intent* is "folded MAF == 0", and the
cleanest contract is `is_monomorphic(double pooled_minor_af) { return pooled_minor_af == 0.0; }`
(no re-fold), called with the already-folded `sm.pooled_minor_af`. *Fix:* either (i) drop the re-fold
in `is_monomorphic` and rename nothing (it then matches its parameter name and this call site), or
(ii) pass `sm.pooled_ref_af` here and keep the re-fold — but pick ONE so the parameter name and the
body agree. Severity low, effort S, before-M4.5: no (but do it when M4.5 touches `filter_decision.hpp`
to make predicates `__host__ __device__`, since the fold count must be pinned for the parity gate).

### (2) Edge cases & failure modes

**F6 [med] CONFIRMED — null `in.q`/`in.n` with `P>0 && M>0` dereferences null; no guard.**
`derive_per_snp_summary` line 21 returns early only on `P <= 0 || M <= 0`. With `P=50, M=100000` but
`q`/`n` left at their `nullptr` struct defaults (header lines 41, 43), lines 42/45/46 dereference
null → SIGSEGV. Re-verified `v` is never read (F9), so a null `v` is silently fine. *Why it matters:*
§2 fail-fast wants a diagnosable error, not a segfault three frames deep; the struct's `= nullptr`
defaults make this an *easy* caller mistake. *Fix:* at entry, when `M>0 && P>0`, require
`in.q && in.n`, else throw `std::invalid_argument`. Severity med, effort S, before-M4.5: **yes** (the
fusion path passes device pointers; pin the precondition contract now).

**F7 [low — VERIFIED SAFE] CONFIRMED — `p + P·s` cannot realistically overflow `size_t`.** Lines
40–41 cast each factor to `size_t` BEFORE the multiply, so the arithmetic is 64-bit unsigned. For the
contracted `P ≤ ~4266`, `M ≤ 100k`, `P·M ≈ 4.3e8` ≪ 2⁶⁴. The per-factor cast correctly avoids the
`int*long → int` overflow trap the naive form would hit. No realistic bug; a *good pattern*.

**F8 [low] CONFIRMED — empty/degenerate inputs handled cleanly; one comment owed.** `M ≤ 0` → empty
mask (lines 66–67); `P ≤ 0` → all-zero summaries (line 21); `total_indiv == 0` → `missing_frac = 1.0`
(line 56) which correctly *fails* an active geno filter and *passes* the default `geno_max_missing ==
1.0`; `pooled_allele_count == 0` (all-missing SNP) → `pooled_ref_af = 0` → folded MAF 0 → dropped by
any `maf_min > 0`, kept at default. All consistent with the oracle and the no-op-default property.
**Subtlety worth a one-line comment:** an all-missing SNP at default config is **KEPT** (MAF 0 ≥ 0,
missing_frac 1 ≤ 1) — intentional and oracle-matching, but it surprises readers who expect all-missing
to drop. *Fix:* add the comment "all-missing ⇒ kept at default is intentional (no-op property)".
Severity low, effort S, before-M4.5: no.

**F9 [med] CONFIRMED — `in.v` is declared, documented, filled by the caller, but NEVER read.**
`DecodedTileSummaryInput::v` (header line 42) is documented "validity mask (1 valid / 0 missing)" and
the oracle fills `fin.v = dec.v.data()` (test line 245). The `.cpp` never touches `in.v`: the
reduction uses `N == 0 ⇔ missing` instead (comment lines 43–44). Re-verified this is CORRECT: the
decode contract makes V and N==0 redundant (`finalize_af` sets both `v=0` and `n=0` when `an==0` —
decode_af.hpp lines 103–106), and using N is right because the pooled counts need N anyway. But a dead
input field is a maintenance trap: a reader cannot tell whether V is load-bearing, and a future change
to V semantics would silently not affect this function. *Why it matters:* §2 separation/clarity.
*Fix:* either drop `v` from `DecodedTileSummaryInput`, OR add the explicit comment "`v` is
intentionally unused: N==0 ⇔ V==0 ⇔ missing by the decode contract (decode_af.hpp finalize_af), so V
is redundant for these reductions." I lean toward the comment (the struct mirrors `DecodeResult` and a
future filter might consume V). Severity med (clarity), effort S, before-M4.5: no.

**F10 [low] CONFIRMED — no NaN/Inf guard.** If a Q or N were NaN (impossible on the contracted
integer-derived decode, possible on a future float producer), `pooled_ref_count`/`missing_frac`
propagate NaN and every comparison (`>=`, `<=`) is **false** (IEEE-754), so a NaN SNP is *dropped* —
the safe failure direction, but silent. Tie to F2: document the finite-`Q`/`N` precondition. Severity
low, effort S, before-M4.5: no.

**F22 [low, NEW — first pass missed] `static_cast<int>(in.pop_individuals.size())` truncates if the
vector exceeds INT_MAX, and the denominator-loop guard becomes UB-adjacent.** Line 26:
`p < static_cast<int>(in.pop_individuals.size())`. `std::vector::size()` is `std::size_t`; casting a
value `> INT_MAX` to `int` is implementation-defined (since C++20, modular wrap to a possibly-negative
value), and the loop guard `p < <negative>` would then be immediately false → `total_indiv` stays 0 →
`missing_frac = 1.0` everywhere → every SNP fails an active geno filter. This is purely theoretical
(`pop_individuals.size()` == P ≤ ~4266 on real data), and the F1 fix (`size() == P` precondition)
eliminates it entirely. Noted because the right fix (F1) subsumes it — do NOT add a second cast guard;
fix F1 and this evaporates. Severity low (theoretical), effort S (folds into F1), before-M4.5: with F1.

### (3) Numerical / precision vs §12

**F11 [low] CONFIRMED — plain-double `+=`; the oracle uses the SAME naive double accumulation in the
SAME order, so parity holds; latent 1-ULP-on-boundary risk vs a raw-integer-count AT2 reference.**
Re-verified the two reductions side by side: `snp_filter.cpp` lines 45–47 do `pooled_ref_count +=
in.q[off]*npn; pooled_allele_count += npn; nonmissing_indiv += npn/ploidy_d` in pop-major order;
`test_filter_oracle.cu` lines 150–153 do the identical accumulation in the identical order. That
exact match is *why* test (c) is integer-exact — it is parity by shared arithmetic, NOT independent
verification of the rounding. This is a *filter decision*, not a reported statistic: §12's precision
law governs the f2 GEMMs and the reported est/se/z, not the keep/drop boundary, so plain double is
appropriate here. Two real subtleties (both confirmed):
  - The MAF/geno **boundary** is a hard `>=`/`<=`. `pooled_ref_count = Σ Q·N` re-multiplies `Q=AC/N`
    back toward `AC`; `AC/N` is generally not exactly representable, so `Q·N = AC ± ε` and the pooled
    ratio can differ from the *exact* pooled frequency by a few ULP. On the threshold this could flip
    a single SNP's keep/drop vs a *different* reference (ADMIXTOOLS 2 computing from raw integer
    counts). It does NOT flip vs *this project's* oracle (same arithmetic), so the M2 gate passes —
    but the AT2-golden comparison (ROADMAP §6) could see a 1-SNP boundary difference. *Fix candidate:*
    integer-accumulate `Σ AC` via `AC = llround(Q*N)` (exact: `AC ≤ N` fits in `long`) and `Σ N`
    (already integral), then `pooled_ref_af = double(ΣAC)/double(ΣN)` — the exact pooled rational.
    This is the *same* "integer-accumulate, divide once" discipline `decode_af.hpp` uses and praises
    (lines 92–95). Severity low (no current-gate failure; latent AT2-boundary risk), effort M,
    before-M4.5: no — flag in the M2 convention report.
  - `missing_frac = 1.0 - nonmissing/total`: benign subtraction (both operands in [0,1], no
    cancellation of large like-magnitude values). Fine.

**F27 [low, NEW — first pass omitted §12 DETERMINISM entirely] The reduction order is fixed and
deterministic (a positive), but the M4.5 GPU segmented reduction must replicate THIS order or risk a
boundary flip — pin it now.** §12's law is two-part: precision AND **determinism / reduction order**
(architecture.md §12 "Bit-reproducible single-stream", §7/§12 CCCL `run_to_run`). The first pass
covered precision (F11) but never analyzed reduction-order determinism. Re-verified: the host
reduction (lines 39–48) sums pop-major (`p = 0..P-1`) for each SNP, a single fixed serial order — so
it is deterministic run-to-run on the host, and the oracle uses the identical order, which is part of
why (c) is bit-exact. **The forward-looking risk:** TODO M4.5 item 3 will reimplement this reduction
*inside `decode_af`* on the GPU as a segmented reduction. A GPU segmented reduction does NOT sum in
host pop-major order by default; with plain-double accumulation (F11), a different summation order can
produce a different last-ULP `pooled_ref_af`, which on the hard `>=`/`<=` boundary can FLIP one SNP's
keep/drop vs this host oracle — breaking the "bit-for-bit parity vs `test_filter_oracle`" gate the
roadmap *explicitly requires*. The F11 integer-accumulate fix neutralizes this (integer `Σ AC`/`Σ N`
is order-independent because integer addition is associative and exact under 2⁵³). *Why it matters:*
§12 determinism + the M4.5 parity gate. *Fix:* resolve F11 (integer-accumulate) BEFORE the GPU kernel
re-derives the reduction, so host and device agree by construction regardless of reduction order; or,
if plain-double is kept, the kernel must use a `run_to_run`/fixed-order reduction AND match host
pop-major order — fragile. Severity low (latent, M4.5-gated), effort M (same fix as F11), before-M4.5:
**yes** (it is the parity-gate design for the fusion).

### (4) CUDA idioms / RAII / stream & async / launch / occupancy vs §7

**N/A for the code as written** — this is a host-pure `io`-leaf TU with no CUDA, no kernels, no
streams, no device allocations; §7 does not apply to the current implementation. **But see §11
(capability tiers): TODO M4.5 item 3 explicitly schedules this logic to become `__host__ __device__`
and run inside the `decode_af` kernel as a validity-mask multiply.** When that lands, §7 applies in
full (the fused predicate must keep the bit-for-bit parity gate vs `test_filter_oracle` — see F27).
The current host implementation is the correct *baseline/oracle* to keep, mirroring how
`decode_af.hpp` is the shared `__host__ __device__` primitive both paths call. One design note for
that future: `build_snp_keep_mask` returns `std::vector<bool>` (packed bits, **no `data()`**,
`operator[]` returns a proxy — confirmed against Microsoft Learn `vector<bool>`: "`bool* pb = &vb[1];`
// conversion error — do not use"), so it is **not** a buffer you can `cudaMemcpyAsync` to the device.
See F14.

### (5) Magic numbers vs §4 / ROADMAP §4

**F12 [low] CONFIRMED — the `'N'`/`0`/`1`-ploidy/`M<0?0` literals are local sentinels, and the right
fix DELETES them (fail-fast), not promotes them.** Re-verified against ROADMAP §4 ("no literal may
survive except true mathematical constants") and config.hpp (the §8 named-constant home):
  - `'N'` (lines 76, 77) and `0` (line 94) as short-`SnpTable` fallbacks: sentinels for a precondition
    violation. The F3 fail-fast fix removes them entirely — they should NOT be promoted to named
    constants, they should not exist.
  - `1` in `in.ploidy > 0 ? in.ploidy : 1` (line 30): a defensive floor F5 says should be a
    validation.
  - `0.0`/`1.0` (lines 53, 56): true mathematical identities (no-data ref-AF is 0; no-data missing
    fraction is 1) — legitimately bare per §4.
  - `M < 0 ? 0 : M` (lines 20, 66): defensive clamp of a signed length; fine as a guard, though F13
    notes a negative M is better treated as a precondition error (it cannot occur from `DecodeResult.M`
    which the backend sets ≥ 0, but the struct field is a raw `long`).
  *Net:* no magic-number *home* is missing; the literals are either legitimate math constants or
  sentinels to be deleted by fail-fast (F3/F5). The autosome range correctly lives in config.hpp
  (`kAutosomeChromMin/Max`) and is read via `is_autosome` — verified, no bare 22 here. Severity low,
  effort S, before-M4.5: no.

### (6) Decomposition / single-responsibility / function size vs §2

**Good — CONFIRMED.** Two functions, each one responsibility, ~44 and ~48 lines — well under any
size smell. `build_snp_keep_mask` delegates *every* decision to `filter_decision.hpp` predicates and
`SnpMembership` (the §8 single-source rule) and contains no inline filter logic that could diverge
from the oracle. `derive_per_snp_summary` is exposed precisely so the oracle can pin it (§13 — good).
No god-function, no mixed concerns. **One observation (→ F13):** `build_snp_keep_mask` recomputes the
*full* per-SNP summary even when the only active filters are class/membership filters that do not need
MAF or missing fraction (e.g. `transversions_only` alone, `autosomes_only` alone) — the reductions are
then pure waste.

**F28 [low, NEW — first pass missed] The per-SNP decision body is INLINED in `build_snp_keep_mask`'s
loop, not extracted — so the M4.5 fusion will have to re-extract it, and the host/device cannot share
ONE primitive today.** Lines 79–105 inline the full keep/drop sequence (class drops → MAF → geno →
flag-gated → membership) directly in the loop. This is readable today, but it is NOT the
`decode_af.hpp` pattern (a single shared `__host__ __device__` per-element primitive both the CPU
oracle and the GPU kernel call). TODO M4.5 item 3 requires exactly that fusion; the inlined orchestration
will have to be lifted into a `snp_keep_decision(const PerSnpSummary&, char ref, char alt, int chrom,
const FilterConfig&, bool membership_bit)` free function that returns one bool, callable from both the
host mask-builder and the future device kernel. *Why it matters:* §8 (single source for the host and
device paths so they cannot diverge), and it is the explicit M4.5 design. Doing the extraction NOW
(while the only caller is the test) is the clean move — it is free of churn risk. *Fix:* extract the
loop body into a pure `[[nodiscard]]` per-SNP decision function; `build_snp_keep_mask` becomes a thin
loop over it; the M4.5 kernel calls the same function. Severity low (forward-looking; it is the M4.5
design), effort M, before-M4.5: **yes**.

### (7) Readability, naming, const-correctness, [[nodiscard]]/noexcept, comments

**F13 [low] CONFIRMED — `build_snp_keep_mask` always computes the heavy per-SNP summary even when no
threshold filter is active.** Line 69 unconditionally calls `derive_per_snp_summary` (an O(P·M)
reduction). When `cfg.maf_min == 0 && cfg.geno_max_missing >= 1 && !cfg.drop_monomorphic`, NONE of the
summary fields is consulted — only class predicates (lines 82, 92–96) and membership (line 102) run.
On the no-op default path (the parity path ROADMAP says must stay fast), this is O(P·M) of dead
arithmetic per tile. *Why it matters:* §11.1 (the precompute stream is throughput-sensitive, this runs
per tile); ROADMAP M2 (the no-op-default path must not regress). *Fix:* gate the call —
`const bool needs_summary = cfg.maf_min > 0.0 || cfg.geno_max_missing < 1.0 || cfg.drop_monomorphic;`
and only `derive_per_snp_summary` then. Severity low (real, constant-factor), effort S, before-M4.5:
**yes** — it also partly anticipates the M4.5 fusion (which deletes the host reduction entirely).

**F14 [med, design] CONFIRMED + CITATION VERIFIED — `std::vector<bool>` return type is space-clever
but a poor seam for the device path and for `FilterPlan`.** `build_snp_keep_mask` returns
`std::vector<bool>`, and `FilterPlan::snp_keep` (filter_plan.hpp line 43) is also `std::vector<bool>`.
Verified against Microsoft Learn (`vector<bool>` class): the specialization packs one bool/bit, has
**no `data()`**, `operator[]` returns the proxy `vector<bool>::reference`, and "the address of the
`vector<bool>::reference` object can't be taken" — `bool* pb = &vb[1];` is a compile error.
Consequences:
  (a) it cannot be `cudaMemcpyAsync`'d to the device as-is (no contiguous `bool*`), directly colliding
      with TODO M4.5 item 3's "apply the keep-mask as a validity-mask multiply inside `decode_af`" —
      that fusion needs an addressable mask buffer (`std::vector<std::uint8_t>` or a device buffer);
  (b) the proxy `operator[]` and per-bit reads are slower per-element than a byte vector for the tight
      per-SNP loop a device-staging copy would do;
  (c) no `std::span<bool>` over packed bits — the §7-preferred view type is unavailable.
The space win (1 bit/SNP vs 1 byte) is ~100 KB for M=100k — negligible against the per-tile Q/V/N
working set (P·M doubles ≈ 3.4 GB-class). *Why it matters:* §7 prefers span/contiguous views; the
explicit M4.5 plan needs a device-copyable mask. *Fix:* consider returning
`std::vector<std::uint8_t>` (0/1) — contiguous, `data()`-able, `std::span<const std::uint8_t>`-viewable,
directly device-stageable — and likewise type `FilterPlan::snp_keep`. (Trade-off: `vector<bool>` is
the idiomatic "mask" type and the test uses it happily; this is a judgment call, but the device-fusion
roadmap tips it.) Severity med (design, forward-looking), effort M, before-M4.5: **yes** (decide before
the fusion calcifies the type — the §"pay first … M4.5 replicates/calcifies these" principle).

**F15 [low] CONFIRMED — `[[nodiscard]]` present (good); neither function `noexcept` (correct);
const-ref inputs (good).** Both functions are `[[nodiscard]]` in the header (lines 80, 101) — correct
(a discarded keep mask is a bug). Neither can be `noexcept`: they allocate `std::vector` (may throw
`bad_alloc`), and the F1/F3/F5/F6 fail-fast additions would throw `std::invalid_argument` — so they
remain (correctly) non-`noexcept`. All inputs are const-ref or by-const-pointer. Do NOT add `noexcept`.
No action.

**F16 [low] CONFIRMED — `static const std::string kEmptyId` inside the loop (line 100) is correct but
reads oddly.** The function-local `static const std::string kEmptyId` is initialized once (thread-safe
since C++11) and used as the short-`SnpTable` fallback id. Correct and zero-cost after first call, but
its placement *inside* the per-SNP loop body (guarded by `!mem_noop`) is surprising. If F3's fail-fast
removes the short-table fallback, this disappears entirely. Otherwise hoist to function/namespace
scope. Severity low, effort S, before-M4.5: no.

**F17 [low] CONFIRMED — comment density is high and accurate; one panel-coupled doc-comment is a
stale-risk.** The headers and inline comments are excellent and correctly cite arch §/ROADMAP and pin
conventions (the MAF pooling, the drop-not-flip ordering, the no-branch-because-N==0 justification at
lines 43–44). This is a *good* pattern (see below). The one stale-risk: the header (lines 92–97)
asserts "for the real AADR HO panel [default] keeps EVERY SNP … the oracle test confirms it drops zero
on the HO panel" — coupling a doc comment to an empirical data property. Currently true (pinned by test
(a)), but if the panel changes the comment silently becomes wrong. *Fix:* soften to "the oracle test
asserts this on the current HO panel". Severity low, effort S, before-M4.5: no.

### (8) Performance

**F18 [low] CONFIRMED — double pass over the SNP axis + a ~3.2 MB intermediate.** `build_snp_keep_mask`
reduces all P·M in `derive_per_snp_summary` (line 69), materializing `std::vector<PerSnpSummary>`
(M structs × 32 bytes = 3.2 MB for M=100k — verified: `PerSnpSummary` is 4 doubles = 32 bytes), then
loops M again to apply predicates. The summary is exposed for testing (a real §13 benefit).
*Recommended:* keep the public `derive_per_snp_summary` for the oracle, but have `build_snp_keep_mask`
compute the per-SNP summary inline in its single loop (via the shared per-SNP helper of F28), avoiding
the intermediate and the second pass. Combined with F13 (skip the summary when no threshold filter is
active), this removes both the intermediate and the dead work. *Why it matters:* §11.1 per-tile
throughput. Note: the M4.5 device fusion supersedes this on the GPU path, so the value is bounded to
the CPU-backend / pre-fusion window. Severity low, effort M, before-M4.5: optional.

**F19 [low] CONFIRMED + SHARPENED — `nonmissing_indiv` is a redundant accumulator; the inner-loop
divide is avoidable — but the bit-identity is a CORRECTNESS PRECONDITION, not a free given.** Line 47
does `nonmissing_indiv += npn / ploidy_d` — a divide per (pop,SNP) cell. Re-verified the algebra
against `finalize_af`: `npn = N_pop = ploidy·AN_pop` with `AN_pop` a non-negative integer
(decode_af.hpp lines 96–99), and `ploidy ∈ {1,2}`. So per cell `npn/ploidy_d = AN_pop` **exactly**
(dividing an exact even integer by 2, or by 1, is exact in IEEE-754). Therefore
`nonmissing_indiv = Σ(npn/ploidy) = Σ AN_pop` and `pooled_allele_count/ploidy_d = (Σ npn)/ploidy =
(ploidy·Σ AN_pop)/ploidy = Σ AN_pop` — **both reduce to the same integer `Σ AN_pop`, hence
bit-identical** — BUT ONLY because (i) `npn` is exactly `ploidy·integer`, (ii) `ploidy ∈ {1,2}`
(division exact), and (iii) `Σ AN_pop < 2⁵³` (always true for AADR ≈ 4000 individuals). The first
pass asserted this was "correctness-neutral" without stating the preconditions; for a general FLOAT
producer (F2) where `npn` is not `ploidy·integer`, `Σ(a/c) ≠ (Σ a)/c` in general (FP addition is not
associative and per-term rounding differs), so the simplification would FLIP a boundary SNP and break
the (c) parity gate. *Fix:* drop the `nonmissing_indiv` accumulator and compute `missing_frac = 1 -
(pooled_allele_count / ploidy_d) / total_indiv_d` — removes one accumulator and P divides per SNP —
**and add a comment that this identity holds because N = ploidy·(integer) exactly** (so the M4.5 GPU
kernel, which will rely on the same relationship, knows the precondition). Severity low, effort S,
before-M4.5: **yes** (cleaner, and it pins the relationship the M4.5 fusion will rely on — but verify
against the (c) parity gate after the change, since it is only exact under the stated preconditions).

### (9) Layering / API / ABI vs §4

**Excellent and explicitly reasoned — CONFIRMED.** The TU is an `io`-leaf, takes plain `const double*`
instead of `core::MatView` (header lines 38–39 justify this precisely: avoids the `core/internal`
dependency), and includes nothing upward. `DecodedTileSummaryInput` mirrors `DecodeResult` (the M1
seam) field-for-field without depending on `device/backend.hpp` — correct decoupling (the `app` layer
wires them, §4). The filter logic is single-sourced in `filter_decision.hpp`. The only API notes are
F14 (the `vector<bool>` seam), the dead `v` field (F9), and the `is_monomorphic` double-fold contract
(F21). No ABI surface (host C++ types). No layering violation.

### (10) Testability vs §13

**F23 [med, CONFIRMED + UPGRADED severity — first pass under-weighted this against §13] No pure-host
unit test of `derive_per_snp_summary`/`build_snp_keep_mask`; the ONLY coverage is the `.cu` oracle that
requires the GPU/data box — which directly contradicts §13's "host, no GPU" principle.** Re-verified by
reading `tests/unit/test_filters.cpp` (285 lines): it includes `filter_decision.hpp`,
`include_exclude.hpp`, `mind_prepass.hpp` and tests the *predicates* and membership — but it NEVER
includes `snp_filter.hpp` and NEVER calls `derive_per_snp_summary` or `build_snp_keep_mask`. The only
caller of these two functions is `tests/reference/test_filter_oracle.cu`, which (a) is a `.cu` requiring
nvcc, (b) calls `make_cpu_backend()->decode_af` on REAL AADR data (`read_ind`/`read_snp`/`read_tile`,
test lines 193–242), so it needs the data box. Architecture.md §13 is explicit: "Pure `__host__
__device__` numerics … tested in plain `.cpp` on the host — **no GPU**", and reserves `tests/reference/`
for GPU-vs-CPU *equivalence*. These two functions are pure host C++ — they BELONG in a host unit test,
and currently have zero host coverage. This is *more* severe than the first pass's "med": it means the
F1/F3/F5/F6 edge cases (short `pop_individuals`, short `SnpTable`, `ploidy<=0`, null pointers) are
**completely untested** AND cannot be tested without the data box, and the M4.5 fusion's parity gate
will rest on a `.cu` oracle with no host regression net beneath it. *Fix:* add host cases to
`tests/unit/test_filters.cpp` (one-line `#include "io/filter/snp_filter.hpp"`): a tiny hand-built
`DecodedTileSummaryInput` (2 pops × 3 SNPs, one all-missing SNP, one boundary-MAF SNP, one
monomorphic SNP, `ploidy=1` and `ploidy=2`), a no-op-default keep-all assertion, and — after the
fail-fast fixes — short-`pop_individuals`/short-`SnpTable`/null-pointer/`ploidy=0` cases asserting the
throw. Severity med, effort S, before-M4.5: **yes** — this host net is the prerequisite regression
harness for the M4.5 fusion parity gate.

### (11) Capability tiers (PRO-6000-capable vs budget-5090 path)

**F20 [high, forward-looking] CONFIRMED VERBATIM — this unit is the named M4.5 device-fusion target,
and it currently has NO capability-tier seam.** TODO §"Keeping it GPU-dominant" item 3 states verbatim:
*"[M4.5] Fuse the cheap in-tile filter into the decode device pass — make `filter_decision.hpp`
predicates `__host__ __device__`, apply the keep-mask as a validity-mask multiply inside `decode_af`;
deletes a D2H + an O(P·M) host reduction + an O(M) host predicate loop per tile (the clearest host
liability in `snp_filter.cpp` today). Gate on bit-for-bit parity vs `test_filter_oracle`."* Re-verified
against the capability-tier table (TODO lines 126–137: PRO-6000 capable path vs budget-5090 fallback,
and "cross-cutting — a capability probe + capability-tagged results … every run records which path it
took + why it degraded"). So:
  - The **current host implementation is the budget/baseline + oracle path** and is correct to KEEP as
    the parity reference (mirroring how `decode_af.hpp` is the shared `__host__ __device__` primitive).
    The *right* architecture — do not delete it when the fusion lands.
  - **What is missing for the tiered story:** (a) the predicates this TU calls (`filter_decision.hpp`)
    are NOT yet `__host__ __device__`, so they cannot be shared with the kernel — and the per-SNP
    decision should be extracted into ONE shared primitive (F28) both the host mask-builder and the
    future device kernel call (the decode_af.hpp pattern); the current code inlines the orchestration,
    which would have to be *re-extracted* at M4.5 — do it now. (b) the mask transport type
    (`vector<bool>`, F14) is not device-copyable; the capable path will compute the mask on-device and
    never materialize a host `vector<bool>`, while the budget/CPU-backend path keeps a host vector —
    the type should anticipate both. (c) there is **no capability-tag plumbing** here: when the fusion
    exists, a run should record "filter applied on-device (fused)" vs "filter applied host-side (CPU
    backend / fusion disabled)" — none of that is wired (and there is no seam for it).
  - **Parity neutrality:** the fusion is explicitly gated "bit-for-bit parity vs `test_filter_oracle`";
    F11/F19/F27 above note that the integer-accumulate refinement makes the host reduction
    order-independent and exact, so the GPU segmented reduction and the host agree by construction —
    worth resolving BEFORE the device kernel re-implements the reduction.
  *Fix (staged):* (1) extract the `__host__ __device__`-able per-SNP decision primitive shared by host
  + future kernel (F28); (2) decide the mask transport type (F14); (3) integer-accumulate (F11/F27);
  (4) leave a seam/TODO for the capability tag. Severity high (it is the explicit M4.5 plan and the
  current shape needs rework), effort M–L, before-M4.5: **yes** (this is the M4.5 design).

## Considered & rejected

- **"`std::toupper` in `normalize_allele` is UB on a negative `char`."** Rejected — that helper is in
  `filter_decision.hpp`, not this unit, and it ALREADY casts to `unsigned char` before `std::toupper`
  (line 124), the documented-correct guard. Verified against SEI CERT STR37-C: the argument "shall be
  representable as an `unsigned char` or shall equal EOF; otherwise the behavior is undefined", and the
  cast to `unsigned char` makes it representable. No bug; the dependency is safe.
- **"`is_strand_ambiguous` then `is_multiallelic` ordering could misclassify."** Rejected — verified
  `is_strand_ambiguous` returns false for non-ACGT input (filter_decision.hpp line 164), so the `||` is
  order-independent for correctness; `is_multiallelic` first is also the better perf order (F4).
- **"The `v` field being unread is a correctness bug (missing SNPs not masked)."** Rejected — the
  decode contract (`finalize_af` lines 96–106) guarantees `N == 0 ⇔ V == 0 ⇔ missing`, so using `N ==
  0` is equivalent and is what the pooled counts need anyway. It is a *clarity* smell (F9), not a bug;
  the oracle test (b) drop-equals-mask passes, proving the missing handling is correct.
- **"Plain-double accumulation violates the §12 precision law."** Rejected — §12 governs the f2 GEMMs
  and the reported statistics, not the filter keep/drop boundary. The filter is a threshold decision;
  double is appropriate. The residuals are a *latent* 1-ULP-on-boundary risk vs a raw-integer-count AT2
  reference (F11) and a determinism/reduction-order pin for the M4.5 GPU fusion (F27) — refinements,
  not violations.
- **"The `is_monomorphic` double-fold is a correctness bug."** Rejected as a *bug* (it is idempotent —
  `folded_maf` of a value in [0,0.5] is the identity, verified), but CONFIRMED as a contract/clarity
  smell (F21): the predicate's parameter is named `pooled_minor_af` yet its body re-folds it, so the
  signature lies about whether to pass the folded or unfolded value.
- **"`build_snp_keep_mask` should be `noexcept`."** Rejected — it allocates a `std::vector` and (after
  the fail-fast additions) may throw `std::invalid_argument`; `noexcept` would be wrong (F15).
- **"`std::vector<bool>` is always a bug."** Rejected as overstated — it is *functionally* correct and
  idiomatic for a mask; the objection (F14) is specifically about the device-fusion seam and span
  interop (verified: no `data()`, proxy `operator[]`, address-not-takeable), a forward-looking design
  call, not a present defect.
- **"`p + P·s` offset overflows `size_t`."** Rejected for realistic P/M (≈4.3e8 ≪ 2⁶⁴); the explicit
  per-factor `size_t` casts are the correct discipline and prevent the `int*long` trap (F7).
- **"`derive_per_snp_summary` should clamp Q to [0,1]."** Rejected as the default — on the contracted
  integer-derived decode, `Q ∈ [0,1]` exactly (`AC ≤ N`); clamping would MASK a wiring bug. Documenting
  the precondition (F2) is the right move.
- **"`missing_frac = 1.0 - …` is catastrophic cancellation."** Rejected — both operands are in [0,1];
  no subtraction of large like-magnitude values (F11).
- **"The `keep` vector should be a bitset / the default-false-then-set-true is wrong."** Rejected — the
  default-drop + `continue`-on-fail + `keep[si]=true` on survival is a clean, correct structure that
  enforces drop-not-flip by shape (a SNP is kept ONLY if it falls through every filter). No bug; a
  *good pattern*.
- **"F19's accumulator-fold breaks the (c) parity gate."** Partially rejected, partially CONFIRMED as a
  caveat — verified the fold is bit-exact ON THE CONTRACTED INPUT (N = ploidy·integer, ploidy ∈ {1,2},
  Σ < 2⁵³), so it does NOT break (c) on AADR; but it is NOT a free identity for a general float producer
  (F19/F2), so the fix must carry the precondition comment and be re-checked against (c).

## What it takes to reach 10/10

1. **Fail-fast on the documented preconditions (F1, F3, F5, F6, F22):** at function entry, validate
   `pop_individuals.size() == P`, `snps.{ref,alt}.size() >= M` (and `chrom`/`id` when those filters /
   membership are active), `ploidy ∈ {1,2}`, and `q && n` non-null when `P>0 && M>0`; throw
   `std::invalid_argument` with context (the `io`-leaf idiom — cf. `include_exclude.cpp` line 17).
   Delete the silent `'N'`/`0`/`1`/truncation fallbacks. Largest single jump — turns 4 silent-wrong-
   answer / segfault paths into diagnosable errors.
2. **Add pure-host unit tests** of `derive_per_snp_summary` and `build_snp_keep_mask` in
   `tests/unit/test_filters.cpp` (it has the harness; one `#include` away): all-missing SNP, boundary
   MAF, monomorphic SNP, `ploidy=1`/`ploidy=2`, no-op-default keep-all, and — post-fix — short
   `pop_individuals` / short `SnpTable` / null pointers / `ploidy=0` (expect throw). Closes the §13 gap
   where the ONLY coverage is the GPU/data-box `.cu` (F23).
3. **Fix the `is_monomorphic` contract (F21):** make the predicate's parameter name and its fold-count
   agree (drop the re-fold and pass the already-folded `sm.pooled_minor_af`, or rename + pass ref_af) —
   pin it before M4.5 makes the predicate `__host__ __device__`.
4. **Extract the per-SNP decision into one shared primitive** (`snp_keep_decision(...)`, F28) that
   `build_snp_keep_mask` and the future M4.5 `decode_af` kernel both call, and make
   `filter_decision.hpp` predicates `__host__ __device__`. The decode_af.hpp pattern; the M4.5 design;
   doing it now avoids a re-extraction (F20).
5. **Decide the mask transport type** (F14): `std::vector<std::uint8_t>` (contiguous, span-able,
   device-stageable) vs `vector<bool>`; align `FilterPlan::snp_keep`.
6. **Lazy summary + fold the redundant accumulator** (F13, F19): skip `derive_per_snp_summary` when no
   threshold filter is active; drop the `nonmissing_indiv` accumulator (it equals `pooled_allele_count/
   ploidy` — exactly, under the stated preconditions; comment them). Optionally fuse the two passes
   (F18) while keeping the public derive function for the oracle.
7. **Integer-accumulate `Σ AC` / `Σ N`** (F11) so the pooled frequency is the exact rational and both
   the host path AND the M4.5 GPU segmented reduction match a raw-count AT2 reference by construction —
   neutralizing the determinism/reduction-order risk (F27) before the kernel re-derives it.
8. **Clarify the dead `v` field** (F9), soften the panel-coupled doc comment (F17), add the "all-missing
   ⇒ kept at default is intentional" note (F8), and leave a seam/TODO for the M4.5 capability tag (F20).

## Good patterns to keep

- **Single-source decision logic.** The TU contains *zero* inline filter logic — every keep/drop goes
  through `filter_decision.hpp` predicates and `SnpMembership`. The §8 DRY rule is honored exactly; the
  host path and the oracle cannot diverge on a boundary (verified: (c) is integer-exact *because* they
  share the arithmetic).
- **Oracle-pinnable derivation exposed as its own `[[nodiscard]]` function** (`derive_per_snp_summary`)
  — the §13 testability move that makes test (c) integer-exact. (The remaining gap is HOST coverage, F23.)
- **Layering discipline: plain `const double*` instead of `core::MatView`,** with an explicit comment
  justifying why the `io` leaf does not reach into `core/internal` (§4). Mirrors `DecodeResult` without
  depending on it.
- **No-branch reduction justified by the decode contract** (lines 43–44: "Q is zero-filled where
  invalid, and N is 0 there") — verified true against `finalize_af`. The comment cites the invariant it
  relies on — exactly the documentation density the rest of the codebase praises.
- **Explicit per-factor `size_t` casts on the column-major offset** (lines 40–41) — avoids the
  `int*long` overflow trap; the right idiom.
- **No-op-when-default property preserved** (MAF≥0, missing≤1, flags off, empty membership) — the
  parity-path guarantee, confirmed bit-identical by test (a) and monotonicity (d).
- **Drop-not-flip enforced by structure** — the unconditional class drop is *first*, one bool per SNP,
  never per-(pop,SNP); the §1/§5-S2 invariant is enforced by shape, not by convention. The
  default-false-then-set-true-on-survival loop is a clean expression of "drop unless it passes
  everything".
- **Const-correct, `[[nodiscard]]`, correctly non-`noexcept`** throughout.

---
Sources for the standard-library facts cited above:
- [vector<bool> Class — Microsoft Learn](https://learn.microsoft.com/en-us/cpp/standard-library/vector-bool-class):
  the bool specialization "provides space optimization by storing one bool value per bit"; `operator[]`
  "Returns a simulated reference … the proxy class `vector<bool>::reference`"; and "`bool* pb =
  &vb[1];` // conversion error — do not use" (the address of a `vector<bool>::reference` cannot be
  taken) — no contiguous `bool*`/`data()`. Basis for F14.
- [SEI CERT STR37-C](https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/rules/characters-and-strings-str/str37-c):
  arguments to `toupper`/`isspace`/etc. "shall be representable as an `unsigned char` or shall equal
  EOF; otherwise the behavior is undefined"; cast to `unsigned char` first. Confirms the rejected
  `normalize_allele` candidate (the helper already casts).
- decode contract verified directly against `core/internal/decode_af.hpp::finalize_af` (N = ploidy·AN,
  Q = AC/N, V = AN>0; AC ≤ N) — basis for F2, F9, F19's exactness.
