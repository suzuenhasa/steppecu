# Review — `io/filter/mind_prepass` (the conditional S-1 `--mind` streaming pre-pass)

Unit under review:
- `/home/suzunik/steppe/src/io/filter/mind_prepass.hpp`
- `/home/suzunik/steppe/src/io/filter/mind_prepass.cpp`

Context read in full: `io/eigenstrat_format.hpp` (the byte/decode helpers it reuses),
`io/filter/filter_decision.hpp` (the shared predicate it calls), `include/steppe/config.hpp`
(`FilterConfig`), `io/filter/snp_filter.{hpp,cpp}` (the sibling SNP-side filter, for
API/convention consistency), `io/filter/filter_plan.hpp` (the downstream consumer struct),
`io/genotype_tile.hpp` + `io/geno_reader.hpp` (the tile/stream producer), the two tests
(`tests/unit/test_filters.cpp`, `tests/reference/test_filter_oracle.cu`), and
`docs/architecture.md` §1/§4/§5 (S-1, S0', S2)/§7/§11.1/§11.4/§12, `docs/ROADMAP.md` §4,
`docs/TODO.md` capability-tier section.

## Role & layering

`mind_prepass` is the host-pure realization of architecture.md §5 **S-1 "QC pre-pass
(conditional)"**: `--mind` is the one cheap filter that is *not* decidable from a single
SNP tile (it needs every SNP for a sample), so per architecture §1/§5 it gets its own light
streaming pass that emits a kept-**sample** index set, folded into the `FilterPlan`
(`filter_plan.hpp` `kept_samples`). The unit takes a CUDA-free `MindPrepassInput` view over
individual-major TGENO packed bytes, accumulates per-sample non-missing counts, and resolves
the kept set via the single shared predicate `sample_passes_mind` (filter_decision.hpp).

Layering is **correct and exemplary**: it is an `io`-leaf TU (architecture §4) that depends
only on `steppe/config.hpp`, `io/eigenstrat_format.hpp`, and `io/filter/filter_decision.hpp`
— no CUDA, no `core`, no `device`, no upward dependency. It deliberately reuses the io-side
byte path (`io::code_in_byte`, `io::kMissingCode`) rather than `core/internal/decode_af.hpp`,
which would be an upward dependency; the header comment states this and it checks out. The
DRY discipline is genuine: the keep decision is delegated to the shared predicate, not
re-spelled, so the in-tile path, this pre-pass, and the oracle test cannot diverge on a
boundary (architecture §8, the explicit goal of filter_decision.hpp).

The substantive tension — and the one issue that keeps this from a top score — is that the
unit is described as a **streaming** pre-pass (architecture §5 S-1 "streams tiles", "streams
via the same tiler"; §11.1 "we never load the genotype matrix whole") but its actual API
consumes one monolithic `MindPrepassInput` covering all `n_snp` SNPs in one resident buffer.
That is a real layering/scalability gap on the production out-of-core path, detailed below.

## Score: 8.5/10 — clean, correct, well-documented host code with one genuine streaming-API gap and a cluster of small hardening/idiom misses

The arithmetic and control flow are correct, the boundary semantics match the shared
predicate and the oracle test exactly, the layering is textbook, and the documentation
density is high and accurate. It is held below the 9.5–10 bar by: (1) a streaming-shape
mismatch with the very §5/§11.1 mechanism the file header cites — the API cannot accumulate
across tiles; (2) missing fail-fast/defensive handling of internally-inconsistent inputs
(`bytes_per_record` too small for `n_snp`, `packed != nullptr` with `bytes_per_record == 0`);
(3) `[[nodiscard]]`/`noexcept` documentation and a few const/decomposition niceties; (4) no
capability-tier or tagged-log hook even though this is squarely on the M5 ingest path the
TODO flags. None of these are correctness bugs on the inputs the current tests feed; they are
robustness, scale, and polish gaps that a demanding reviewer flags before M4.5.

## Findings

### (1) Correctness & bugs

**1.1 — `bytes_per_record` is never validated against `n_snp`; an out-of-bounds read is
possible from a malformed-but-plausible input. [med, S, before-M4.5: yes]**
Location: `mind_prepass.cpp:32-48` (the streaming loop), specifically `rec[s / 4u]` at line 41.
The loop reads byte `s/4` of each record for `s` in `[0, n_snp)`, i.e. it touches bytes
`[0, packed_bytes(n_snp))` of every record (`packed_bytes` defined in eigenstrat_format.hpp:67).
Nothing checks that `bytes_per_record >= packed_bytes(n_snp)`, nor that
`packed` actually spans `n_individuals * bytes_per_record` bytes. The struct doc
(`mind_prepass.hpp:39-44`) states the *contract* (`bytes_per_record` = stride, `n_snp` =
prefix length) but the code trusts it blindly. If a caller passes `n_snp` larger than the
record actually packs (e.g. a `GenotypeTile` whose `bytes_per_record == ceil(tileSnps/4)` but
`n_snp` set to the full dataset SNP count by mistake — exactly the kind of slip the streaming
gap in 9.1 invites), every record overruns into the next record's bytes, or past the buffer
on the last record. This is a silent OOB read, not a crash, so it would corrupt the
missingness counts rather than fail.
Why it matters: architecture §2 "fail-fast" and the senior-engineer bar. A leaf that takes a
raw pointer + sizes must either assert its size invariant or document that the caller owns it
and the function is precondition-checked. snp_filter's sibling code guards its index
math (`si < snps.ref.size() ? ... : 'N'`, snp_filter.cpp:76-77); mind_prepass guards nothing.
Fix: add `assert(in.bytes_per_record >= io::packed_bytes(n_snp))` (and document that `packed`
must span `n_individuals * bytes_per_record`), or—better—accept a `std::span<const std::uint8_t>`
for `packed` (see 4.3) so the byte extent is carried with the pointer and the last-record
overrun is checkable. At minimum the contract violation should be a hard precondition, not a
silent OOB.
Adversarial check: is it actually reachable? The two current callers (test_filters.cpp,
test_filter_oracle.cu) both pass `bytes_per_record` and `n_snp` from the same `GenotypeTile`,
where `bytes_per_record == ceil(n_snp/4)` holds by construction, so it is not triggered today.
But the function is a public `io`-leaf entry point with a pointer-and-sizes signature; "no
current caller hits it" is not the standard for a defensive leaf. Confirmed real.

**1.2 — `packed != nullptr` with `bytes_per_record == 0` silently misbehaves. [low, S,
before-M4.5: no]**
Location: `mind_prepass.cpp:32` (the `if (in.packed != nullptr && n_snp > 0)` guard) and 38.
The guard checks `packed != nullptr && n_snp > 0` but not `bytes_per_record > 0`. With
`bytes_per_record == 0` and `n_snp > 0`, every record pointer `in.packed + g*0` aliases byte 0,
so `rec[s/4]` reads the same first `ceil(n_snp/4)` bytes for *all* `n_individuals` — producing
identical (wrong) non-missing counts rather than failing. This is an inconsistent-input state
(stride 0 but claiming SNPs) that the no-op default would never produce but a bug upstream
could.
Why it matters: §2 fail-fast; an internally inconsistent view should be rejected, not
quietly produce plausible-looking garbage.
Fix: include `in.bytes_per_record > 0` in the streaming-path guard (and assert it when
`n_snp > 0`), routing the inconsistent case to the no-data branch or a hard precondition.
Adversarial check: legitimate? `bytes_per_record == 0` with `n_snp == 0` is the genuine
no-data case and is already handled by the `n_snp > 0` half of the guard. Only the
`bytes_per_record == 0 && n_snp > 0` combination is the bug, and it is unambiguously
inconsistent. Confirmed (low severity because no current path constructs it).

**1.3 — No-data branch (`packed == nullptr || n_snp == 0`) reports `missing_frac = 0` for
every sample, including under an active filter. [low/by-design, S, before-M4.5: no]**
Location: `mind_prepass.cpp:49-54` and the resolve loop 59-64.
When there are no SNPs, the code sets `missing_frac[g] = 0` for all `g` and (because the
resolve loop uses `frac = active ? missing_frac[g] : 0.0`, and `missing_frac` is 0 here) keeps
everyone even under an active `mind_max_missing < 1`. The header (mind_prepass.hpp:58-60) and
the .cpp comment (50-52) both document this as deliberate: "missing fraction is undefined;
treat as 0 missing." This is a *defensible* convention but it is arguably the wrong fail-safe:
with zero SNPs and an active mind filter, treating every sample as 0%-missing (keep all)
rather than 100%-missing (drop all, "no evidence the sample has data") is a judgment call, and
the ADMIXTOOLS 2 analogue (`--mind` on an empty SNP set) is undefined. The snp_filter sibling
takes the *opposite* convention for its analogous denominator-zero case: `missing_frac` defaults
to `1.0` (PerSnpSummary, snp_filter.hpp:66; snp_filter.cpp:56 `total_indiv_d > 0 ? ... : 1.0`),
i.e. all-missing when the denominator is zero. So the two sibling filters disagree on the
empty-denominator fail-safe direction.
Why it matters: consistency across the §8 single-home filter family, and predictability of
the degenerate case. It is documented, so it is not a bug; but the inconsistency with
snp_filter is a real wart worth reconciling explicitly.
Fix: either align with snp_filter (frac = 1.0 ⇒ drop-all under active filter, keep-all only at
the no-op default), or add a one-line note in *both* files that the conventions intentionally
differ and why. Right now the divergence is silent.
Adversarial check: is it actually inconsistent? snp_filter's denominator-zero ⇒ 1.0 path is
reached when `total_indiv == 0` (no kept individuals); mind_prepass's ⇒ 0.0 path is reached
when `n_snp == 0` (no SNPs). Different axes, so not strictly the "same" case — but they are the
*same shape* of degenerate fail-safe ("the fraction has no denominator, what do we assume?")
and they pick opposite directions. The wart stands as a documentation/consistency item.

**1.4 — `missing_frac` is computed for ALL samples even when the filter is inactive, but only
*used* when active. Minor wasted work, not a bug. [low, S, before-M4.5: no]**
Location: `mind_prepass.cpp:32-48` runs the full O(n_ind · n_snp) streaming scan whenever
`packed != nullptr && n_snp > 0`, *regardless of `active`*. The header (mind_prepass.hpp:55-60)
explicitly promises this ("We still report the missing fractions if we have data") so it is
intentional — the no-op path still reports observed fractions so a caller can inspect them.
But note the resolve loop then *discards* those fractions when `!active` (`frac = active ?
out.missing_frac[g] : 0.0`, line 60): the no-op keep set is computed from a constant 0.0, not
from the just-computed fractions. So the expensive scan's result is reported but not used for
the decision in the no-op case. That is consistent with the documented "report but keep all"
intent — flagging only because it is a subtle "compute-then-ignore" that a reader must reconcile
against the header promise. Not a defect; see 8.1 for the performance angle (an active-only
fast path).

### (2) Edge cases & failure modes

**2.1 — Integer width / overflow: safe. [N/A — verified]**
`nm` (per-sample non-missing count) is `std::size_t` and bounded by `n_snp` (also `std::size_t`),
so it cannot overflow short of `n_snp == SIZE_MAX` (impossible — it is a byte count / 4). The
record offset `g * in.bytes_per_record` is `std::size_t * std::size_t` and is the same offset
the producer (`GenotypeTile`) uses; for a real AADR record (`ceil(584131/4) ≈ 146 KB`) ×
27,594 individuals ≈ 4 GB, which fits `std::size_t` comfortably on the 64-bit target. No 32-bit
truncation (everything is `std::size_t`). Verified — no width issue.

**2.2 — `missing_frac` can be a hair outside `[0, 1]` from FP rounding, but harmlessly.
[low, S, before-M4.5: no]**
Location: `mind_prepass.cpp:46-47`, `1.0 - static_cast<double>(nm)/static_cast<double>(n_snp)`.
`nm <= n_snp` always, so `nm/n_snp ∈ [0,1]` exactly in real arithmetic; in IEEE-754 double the
quotient of two exactly-representable integers (both ≤ ~6e5, well within 2^53) is correctly
rounded, and for `nm == n_snp` the quotient is exactly 1.0, so `1.0 - 1.0 == 0.0` exactly, and
for `nm == 0` it is exactly `1.0 - 0.0 == 1.0`. So at the two boundaries the value is exact; in
between it can round but stays in `(0,1)`. The comparison `frac <= mind_max_missing` is robust to
this (a frac that rounds to slightly >1.0 cannot occur because `nm < n_snp ⇒ nm/n_snp < 1 ⇒
1 - that ∈ (0,1)`). So no spurious keep/drop at the boundary.
Why it flags at all: the oracle test (test_filter_oracle.cu:379) recomputes the *identical*
expression, so any rounding is shared and the integer-exact set comparison still holds — which
is good, but it means the parity relies on the two sites using byte-identical float expressions.
That is a single-source argument currently maintained by *duplication* of the formula
(see 3.1), not by sharing it. Fix: see 3.1 (hoist the frac formula into a shared helper so the
"identical expression" guarantee is structural, not coincidental).

**2.3 — `n_individuals == 0`: handled correctly. [N/A — verified]**
With `n_ind == 0`, all three `assign(0, ...)`/`reserve(0)` are no-ops, both loops iterate zero
times, and an empty `MindSummary` is returned. The downstream `FilterPlan.kept_samples` would
be empty, which is the correct "no samples" result. No special-casing needed; correct by
construction.

**2.4 — `mind_max_missing` negative or NaN: under-specified at this layer. [low, S,
before-M4.5: no]**
`active = cfg.mind_max_missing < 1.0` (line 30). If `mind_max_missing` is negative (nonsensical
but representable), `active` is true and `sample_passes_mind(frac, neg)` returns
`frac <= neg`, which is false for every nonnegative frac ⇒ **drops every sample**. If it is
`NaN`: `NaN < 1.0` is false ⇒ `active == false` (the "keep all" intent), but the resolve loop
then calls `sample_passes_mind(0.0, NaN)` = `0.0 <= NaN` = false ⇒ keeps *no one* despite
`!active`. That NaN path is an inconsistency: the "no-op because inactive" intent (keep all) is
silently violated because the predicate sees NaN. This is really a `FilterConfig` validation gap
(the config should reject NaN/negative `mind_max_missing` at construction/`ConfigBuilder::build()`),
not strictly mind_prepass's job — but mind_prepass is where the surprising behavior surfaces.
Why it matters: §9 typed-immutable-config + fail-fast — a nonsensical threshold should be
rejected up front with a clear error, not produce a silent drop-all/keep-none.
Fix: validate `mind_max_missing` (and the other thresholds) in the config builder, OR add an
`assert(!std::isnan(cfg.mind_max_missing))` here. Document that thresholds are pre-validated.
Adversarial check: is the NaN keep-none real? `active = (NaN < 1.0) = false`; resolve loop:
`frac = active ? ... : 0.0 = 0.0`; `sample_passes_mind(0.0, NaN) = (0.0 <= NaN) = false` ⇒
`kept` stays empty. Yes — NaN threshold yields an empty kept-sample set while reporting
`!active`, a genuinely surprising self-inconsistency. Low severity (no path produces NaN today)
but real.

**2.5 — Degenerate `n_snp == 0` with `n_individuals > 0` under an active filter: see 1.3.**
Covered above; the keep-everyone-on-no-SNPs choice is the documented but debatable fail-safe.

### (3) Numerical / precision vs §12

**3.1 — The missing-fraction formula is DUPLICATED between mind_prepass and the oracle test
(and conceptually parallels snp_filter's). [med, S, before-M4.5: yes]**
Location: `mind_prepass.cpp:46-47` vs `test_filter_oracle.cu:379` — both compute
`1.0 - static_cast<double>(nm)/static_cast<double>(n_snp)` for the per-sample missing fraction,
and the non-missing accumulation loop (mind_prepass.cpp:40-44) is *also* duplicated verbatim in
the oracle (test_filter_oracle.cu:355-362). §12 parity here is "integer-exact set comparison",
which only holds because the two expressions are byte-identical. But that identity is currently
guaranteed by copy-paste, not by a shared function. The whole point of filter_decision.hpp
(its header: "the rule lives ONCE so the in-tile path, the pre-pass, and the tests cannot
diverge") is to make these guarantees structural. The *predicate* (`sample_passes_mind`) is
shared; the *frac derivation* (`1 - nonmissing/n_snp`) and the *non-missing accumulation over
packed bytes* are not.
Why it matters: architecture §8 DRY single-source; §12 determinism-by-construction. snp_filter
solved exactly this for its side by exposing `derive_per_snp_summary` (snp_filter.hpp:80) as the
single SNP-side derivation the oracle pins against. mind_prepass has no analogous shared
`per_sample_missing_frac(nonmissing, n_snp)` helper, so the oracle reimplements the loop and the
formula. If anyone "improves" one site (e.g. guards `n_snp == 0` differently, or reorders), the
parity guarantee silently weakens.
Fix: add a tiny `[[nodiscard]] inline double per_sample_missing_frac(std::size_t nonmissing,
std::size_t n_snp) noexcept` to filter_decision.hpp (next to `sample_passes_mind`) and have
both mind_prepass and the oracle call it; optionally expose the non-missing accumulation as a
shared `count_nonmissing_in_record(...)` helper so the byte loop is single-sourced too.
This is the same single-home discipline snp_filter already follows.
Adversarial check: is the test "supposed" to reimplement independently? Yes — an *oracle* test
should recompute *independently* to be a real check (test_filter_oracle.cu's header: "recomputed
independently, NOT via snp_filter"). So the *accumulation* being independent is arguably correct
test design. BUT the *frac formula* `1 - nm/n_snp` is a definition, not a computation to
independently verify — and snp_filter's oracle does reuse the shared
`is_multiallelic`/`is_strand_ambiguous`/`is_transversion` predicates (test_filter_oracle.cu:163,
167) while recomputing the *numeric* pooling itself. The right line: the byte accumulation may
stay independent in the oracle (a genuine recompute), but the no-SNP fail-safe and the
frac definition should be single-homed. Partially upheld — the frac/no-data-convention sharing
is the load-bearing part of this finding.

**3.2 — Accumulation order / determinism: deterministic by construction. [N/A — verified]**
`nm` is an integer running count over `s = 0..n_snp-1` in fixed order; integer addition is
exact and order-independent. `missing_frac` is one division per sample, no summation, so no
accumulation-order concern. The kept-set is built in ascending sample order (`g = 0..n_ind-1`,
push_back), so `kept` is sorted ascending as the header promises (mind_prepass.hpp:51,
filter_plan.hpp:50-53 "ascending"). Fully deterministic, single-stream, host-side — exactly the
§12 fixed-order property. No Ozaki/native-FP64 concern here (this is a count + a division, not a
GEMM). Verified clean.

### (4) CUDA idioms / RAII / stream & async / launch config vs §7

**4.1 — No CUDA here, correctly. [N/A — host-pure leaf]**
This unit is host-pure by design (architecture §4 io-leaf; the file header states it). There is
no allocation, no stream, no kernel, no device pointer — so the §7 RAII/stream/launch-config
catalog does not apply. That is the *right* call: per architecture §11.1 the pre-pass "streams
via the same tiler" on the host and emits a small set; the heavy work is the decode/GEMM path,
not this. Correctly N/A.

**4.2 — But it sits on the M5 ingest path and ignores it (see capability-tier §11).**
The non-missing scan is a full sequential pass over the packed genotype bytes — the same bytes
the M5 pinned/double-buffered ingest pipeline streams (TODO §"M5 pinned double-buffered ingest").
The current API forces the *whole* genotype matrix resident (one `MindPrepassInput.packed`),
which directly contradicts §11.1 "we never load the genotype matrix whole." This is the
streaming gap; see 9.1 and 11.1.

**4.3 — Raw `const std::uint8_t*` + sizes instead of `std::span`. [low, S, before-M4.5: no]**
Location: `mind_prepass.hpp:40-41` (`const std::uint8_t* packed; std::size_t bytes_per_record`).
Architecture §7 explicitly prefers "span/mdspan views not raw pointers." A
`std::span<const std::uint8_t> packed` would carry the byte extent, making the 1.1 OOB check
trivial (`packed.size() >= n_individuals * bytes_per_record`) and the `.subspan(g*stride, stride)`
per-record slice bounds-checkable. The sibling `DecodedTileSummaryInput` (snp_filter.hpp) also
uses raw `const double*`, so this is a *consistent* project-wide choice for these io plain-view
structs — but architecture §7 names spans as the target idiom, so it is a standing low-severity
modernization for the whole io-filter family, not unique to this file.
Adversarial check: is a span feasible at this layer? `<span>` is C++20 std (the project is
C++20 per every header), CUDA-free, so yes. The reason it is low and not med: the *whole* io
plain-struct family uses raw pointers consistently, and changing one file in isolation would
make it the odd one out. Best done as a family-wide sweep.

### (5) Magic numbers & hardcoded values vs §4 / ROADMAP §4

**5.1 — Bare `4u` for the codes-per-byte divisor/modulus, twice, despite `kCodesPerByte`
existing. [med, S, before-M4.5: yes]**
Location: `mind_prepass.cpp:41-42`: `rec[s / 4u]` and `code_in_byte(byte,
static_cast<int>(s % 4u))`. The named constant for exactly this is
`io::kCodesPerByte == 4` (eigenstrat_format.hpp:54, documented as "Genotype codes packed per
byte ... 4 ... 2 bits each. A true structural constant of the 2-bit packing"), and the
single-home byte helpers already encapsulate it: `packed_bytes(n_codes)` does the `/ 4` and
`code_in_byte(byte, k)` does the `% 4`-equivalent shift internally. ROADMAP §4 is explicit:
"No literal may survive ... except true mathematical constants." 4 here is the packing radix,
not a math constant; it has a named home one header away that this very file includes.
Why it matters: §4 single-source — the `4` is the SAME number `kCodesPerByte`,
`packed_bytes`, and `code_in_byte`'s internal shift all derive from. Open-coding `4u` here is
precisely the "re-spell the number at a decode call site" the eigenstrat_format.hpp header
warns against ("No decode site may re-spell these numbers"). And it is a *correctness coupling*:
if the packing radix ever changed, `code_in_byte` and `packed_bytes` would update but these two
`4u`s would not.
Fix: use the helper that already hides the radix. The cleanest is to index via the byte and the
position derived from `kCodesPerByte`:
`const std::uint8_t byte = rec[s / static_cast<std::size_t>(io::kCodesPerByte)];`
and `code_in_byte(byte, static_cast<int>(s % static_cast<std::size_t>(io::kCodesPerByte)))`.
Better still, add a one-liner `io::code_at(rec, s)` to eigenstrat_format.hpp that does
`code_in_byte(rec[s / kCodesPerByte], s % kCodesPerByte)` so neither the divisor, the modulus,
nor the MSB-first order is ever re-spelled at a call site — mind_prepass, the oracle test
(test_filter_oracle.cu:359, same bare `4u`), and any future decoder would share it.
Adversarial check: is `4` a "true mathematical constant" exempt under ROADMAP §4? No — the
exemption example is the `2` in `a²−2ab+b²` (intrinsic to the algebra). 4 codes/byte is a
*format* constant (2 bits × 4 = 8 bits), which ROADMAP §4 itself lists as belonging in io
format constants ("TGENO header 48 / ceil(nsnp/4) belong in io format constants"). And
`kCodesPerByte` was created *specifically* to be that home. So the `4u` is a clear (if minor)
§4 violation. Confirmed; the same bare `4u` in the oracle test should be swept too.

**5.2 — `1.0` no-op threshold sentinel is open-coded. [low, optional, before-M4.5: no]**
Location: `mind_prepass.cpp:30` `cfg.mind_max_missing < 1.0`. The `1.0` is the "filter
inactive / keep-everything" sentinel (a frac of 1.0 = 100% missing allowed = no drop). It
matches `FilterConfig`'s documented default (`mind_max_missing = 1.0`, config.hpp:222) and the
predicate's no-op property (filter_decision.hpp:97-103). It is a *mathematical* maximum
(a fraction's ceiling is 1.0), so it is arguably an exempt true-constant under ROADMAP §4 —
unlike the `4u`. Flagging only for completeness: if you want zero bare numerics, a named
`kMaxMissingFraction = 1.0` (or, better, a `FilterConfig::mind_active()` accessor encapsulating
the `< 1.0` test) would centralize the "is this filter active?" question (snp_filter implicitly
relies on the same no-op property for maf=0/geno=1). I lean toward leaving the literal 1.0
(it is the closed-form max of a fraction) but encapsulating the *activeness test* as a config
method is the higher-value version of this.
Adversarial: this is borderline; the 1.0 is a genuine fraction ceiling, so I would *not* fail
the review on it. Listed as low / optional.

### (6) Decomposition / single-responsibility / function size vs §2

**6.1 — `run_mind_prepass` is one ~50-line function doing three distinct jobs. [low, S,
before-M4.5: no]**
Location: `mind_prepass.cpp:17-66`. The function (a) allocates/zeroes the summary, (b) does the
streaming non-missing accumulation + frac derivation, (c) resolves the kept set. Each is a
clean, separable step. At 50 lines it is not egregious, and the linear top-to-bottom structure
is readable, but §2 single-responsibility would favor splitting (b) into a free
`accumulate_nonmissing(in, nonmissing, missing_frac)` (which is *also* what 3.1 wants — the
shared accumulation the oracle could call) and (c) into `resolve_kept(missing_frac, cfg)`.
That would (i) make the accumulation independently testable without going through the resolve,
(ii) let the oracle pin the accumulation against a shared function, and (iii) make the
active/no-op decision a one-liner. Low severity — the current function is comprehensible — but
the decomposition is the natural way to also fix 3.1.

**6.2 — `MindSummary` carries `nonmissing` purely for the test. [low, won't-fix, before-M4.5:
no]**
Location: `mind_prepass.hpp:48-52`. The header is candid that `nonmissing` is "exposed so the
oracle test can recompute and compare exactly" (mind_prepass.hpp:47, 49). Exposing internal
state solely for a test is mildly against §13 "test through the public contract" purism, but
here it is *useful* output (a caller may legitimately want per-sample non-missing counts), it is
documented, and it mirrors snp_filter's `PerSnpSummary` exposure for the same reason. Acceptable
as-is; noting only that the justification is test-driven, which §13 tolerates when the exposed
data is genuine output (it is).

### (7) Readability, naming, const-correctness, [[nodiscard]]/noexcept, comments

**7.1 — `run_mind_prepass` is `[[nodiscard]]` (good) but its exception contract is undocumented.
[low, S, before-M4.5: no]**
Location: `mind_prepass.hpp:61` / `mind_prepass.cpp:17`. The function can only throw via the
`std::vector` allocations (`assign`/`reserve`/`push_back`) — i.e. `std::bad_alloc` on OOM. It is
not marked `noexcept`, which is *correct* (it can throw bad_alloc) and consistent with
snp_filter's `build_snp_keep_mask` (also not noexcept). So this is *not* a defect — but the
file header's layering note says "the `io` leaf never throws across the layer boundary on a
format probe" (referring to eigenstrat_format.hpp's `parse_geno_header`, which *is* `noexcept`).
The contrast deserves a one-line doc: the *probe* is noexcept, the *pre-pass* may throw
bad_alloc (it allocates O(n_ind) vectors). Currently a reader has to infer the exception
contract. Fix: document it (e.g. "/// Throws only std::bad_alloc on allocation failure; never
throws on input shape."). The `[[nodiscard]]` is correctly present and appropriate (the summary
is the whole point of the call). Pure-doc nit.

**7.2 — Sizing/zeroing idioms are correct and good. [N/A — verified good]**
`assign(n_ind, 0)` / `assign(n_ind, 0.0)` correctly size *and* zero-initialize
`nonmissing`/`missing_frac` (so the no-data branch's explicit `missing_frac[g]=0.0` loop at
line 53 is in fact redundant — see 8.2). `reserve(n_ind)` on `kept` is the right call (kept ≤
n_ind, single allocation, no over-allocation). Const-correctness is clean: `n_ind`, `n_snp`,
`active`, `rec`, `byte`, `code`, `frac` are all `const`. Naming (`nm` for non-missing, `g` for
the gathered-individual index matching the rest of io, `s` for SNP) matches the surrounding io
convention (genotype_tile.hpp, geno_reader). Comment density is high and accurate. This is good
code.

**7.3 — `static_cast<int>(s % 4u)` mixes `std::size_t` modulus with `unsigned` literal and a
cast to `int`. [low, S, before-M4.5: no]**
Location: `mind_prepass.cpp:42`. `code_in_byte(std::uint8_t, int k)` (eigenstrat_format.hpp:76)
takes `int k`; `s % 4u` is `std::size_t` (since `s` is `std::size_t`, `4u` promotes to
`std::size_t`), so the cast to `int` is needed and the result `∈ {0,1,2,3}` is always in `int`
range — *safe*. But the `4u` (unsigned int) literal interacting with a `std::size_t` operand is
a slightly muddy mix; combined with 5.1, the clean form is
`static_cast<int>(s % static_cast<std::size_t>(io::kCodesPerByte))`. Folds into 5.1's fix.

### (8) Performance

**8.1 — The full O(n_ind · n_snp) scan runs even when the filter is a no-op. [med, M,
before-M4.5: yes]**
Location: `mind_prepass.cpp:32-48`. Architecture §5 S-1 is explicit: the pre-pass is "Skipped
entirely when no aggregate filter is requested," and the file header (mind_prepass.hpp:4-7,
.cpp:3-4) repeats "ONLY runs when mind_max_missing < 1.0; otherwise it is skipped entirely."
But the implementation does NOT skip the scan when inactive — it computes `active` (line 30) and
then *unconditionally* runs the streaming scan if `packed != nullptr && n_snp > 0` (line 32),
ignoring `active`. The header rationalizes this ("We still report the missing fractions if we
have data") but that contradicts the architecture's "skipped entirely" — on the real AADR panel
that is a ~4 GB / 27k-sample × 584k-SNP scan executed *for nothing* at the no-op default that
the parity path uses (the default `FilterConfig`). For a milestone whose whole premise is
"defaults are no-ops so the parity path is untouched" (config.hpp:197-199), doing a full
genotype sweep at the default is a real and avoidable cost on the hottest path.
Why it matters: §11.1 the precompute pass is bandwidth-bound; a wasted full-matrix read at the
no-op default is the opposite of "skipped entirely." The "report fractions anyway" feature is
nice-to-have, not free.
Fix: gate the scan on `active` (only stream when `cfg.mind_max_missing < 1.0`). If reporting
no-op-path fractions is genuinely wanted, make it opt-in (a flag) rather than the default —
the default must match §5 S-1 "skipped entirely." This is the single most impactful perf/spec
item in the file.
Adversarial check: is the unconditional scan actually intended over the architecture's
"skipped entirely"? The header author clearly *chose* to always-report; but architecture §5 S-1
is the governing spec and it says skip. The two disagree, and the spec wins. Note the *caller*
(the filter front-end, not yet written) is supposed to decide whether to even call this — the
architecture's "conditional" lives at the call site. So arguably mind_prepass should *assume*
it is only called when active and *assert* it (or skip internally). Either way the always-scan-
at-default behavior is wrong vs §5. Confirmed med.

**8.2 — Redundant re-zero of `missing_frac` in the no-data branch. [low, trivial, before-M4.5:
no]**
Location: `mind_prepass.cpp:53` (`for (...) out.missing_frac[g] = 0.0;`). `missing_frac` was
already value-initialized to `0.0` by `assign(n_ind, 0.0)` at line 23, so this loop writes 0.0
over 0.0. Harmless but dead. Fix: drop the loop (or replace the no-data branch body with a
comment, since the vectors are already zero). Trivial.

**8.3 — Per-byte re-fetch vs per-byte unpack. [low, recommend-against-inline, before-M4.5: no]**
Location: `mind_prepass.cpp:40-44`. The inner loop fetches `rec[s/4]` and shifts out one code
per `s` — so each packed byte is loaded 4 times (once per code). A micro-optimization would
load each byte once and test all 4 codes (e.g. a precomputed "non-missing count per byte" 256-
entry table, or a bit trick since "missing" is code 3 == both bits set). On the bandwidth-bound
real panel this *might* matter (it is the same data M5 ingest cares about), but: (a) the byte is
almost certainly in L1 across its 4 reads, and (b) doing this *here* would re-implement bit
logic that belongs in the shared eigenstrat_format helpers, undercutting the single-source byte
path. So I would NOT micro-optimize in mind_prepass; if a fast "non-missing codes in a byte"
primitive is wanted, it belongs in eigenstrat_format.hpp (a `nonmissing_in_byte(byte)`
table-or-bittrick helper) and both this and the oracle would call it. Listed as a perf
*possibility*, explicitly recommended against doing inline. (And it is moot until 8.1 stops the
no-op scan.)

### (9) Layering / API / ABI vs §4

**9.1 — The API is non-streaming, contradicting the §5/§11.1 "streams tiles" mechanism the
file itself cites. [high, L, before-M4.5: yes]**
Location: the whole `MindPrepassInput`/`run_mind_prepass` shape (`mind_prepass.hpp:39-62`).
The header says (lines 3-4, 9-12) this is the S-1 *streaming* pre-pass and that `--mind` "needs
every SNP for a sample, so it gets its own light streaming pre-pass." Architecture §5 S-1
(architecture.md:240) is explicit: "io/filter/prepass (host, **streams tiles**)" ... "**streams
via the same tiler**; emits plain sets." And §11.1 (architecture.md:662-669): "we **never load
the genotype matrix whole**." But `run_mind_prepass` takes ONE `MindPrepassInput` whose `packed`
must contain *all* `n_snp` SNPs for *all* `n_individuals` simultaneously — i.e. the entire
genotype matrix resident in one host buffer. For the real AADR panel that is the ~4 GB packed
matrix the architecture says must never be loaded whole. So the *API shape* cannot stream; the
function can only run after the very load §11.1 forbids.
Why it matters: this is the core architectural promise of the unit (its name, its header, the
§5 row) and the API does not deliver it. A streaming pre-pass must accumulate *across tiles*:
the natural shape is a stateful accumulator (`MindAccumulator` holding the per-sample
`nonmissing` running counts and `n_snp_seen`, with `add_tile(const MindPrepassInput& partial)`
called per SNP tile, then `finalize(cfg) -> MindSummary`). The current one-shot API is fine for
the *tests* (which use a 100k-SNP capped tile that fits) but cannot be the production S-1 path.
Note the input is even named/documented in tile terms ("the same layout as io::GenotypeTile",
mind_prepass.hpp:35), which makes the single-tile-only limitation more surprising.
Fix: introduce a streaming accumulator: a small struct holding `std::vector<std::size_t>
nonmissing`, `std::size_t n_snp_total = 0`, `n_individuals`; an `add_tile(const MindPrepassInput&)`
that folds one SNP tile's per-sample non-missing into the running counts (reusing the shared
accumulation from 3.1/6.1); and a `finalize(const FilterConfig&) -> MindSummary` that divides by
`n_snp_total` and resolves the kept set. Keep `run_mind_prepass` as a thin convenience wrapper
over add_tile+finalize for the single-buffer case (so the tests and oracle compile unchanged).
This is the change that moves the unit from "passes the M2 tests" to "is the production S-1 pass
§5/§11.1 describes."
Adversarial check: is the one-shot API maybe intentional for M2, with streaming deferred? The
docs label filter/ as "[BUILT, M2]" (architecture.md:165) and S-1 as part of M2's scope, and
the file header presents itself as the streaming pre-pass *now*, not a stub. There is no
"streaming TODO" note in the file. So the gap is real and unflagged. It is plausibly *acceptable
to defer the multi-tile accumulator to M5* (when the streaming ingest pipeline lands — TODO M5),
but then the file/header should say so explicitly ("single-tile for M2; multi-tile accumulator
arrives with M5 streaming ingest") rather than claim to be the streaming pass. Either implement
the accumulator or add the deferral note. Rated high because it is the unit's central contract
and it is currently *mis-stated*, not merely incomplete.

**9.2 — ABI: plain POD structs cross the layer boundary cleanly. [N/A — verified good]**
`MindPrepassInput` (raw pointer + 3 `std::size_t`) and `MindSummary` (3 std::vectors) are plain
data, no CUDA type, consistent with `FilterPlan`/`GenotypeTile`/`DecodedTileSummaryInput`. They
cross the io→consumer boundary unchanged (architecture §4). Correct.

### (10) Testability vs §13

**10.1 — Well-covered at the boundary; missing the degenerate/inconsistent cases. [low, S,
before-M4.5: no]**
The unit IS tested two ways: a host unit test (`test_filters.cpp::test_mind_prepass`, lines
194-242) exercising the no-op, an active cap (0.4 keeps only sample 0), and the exact-boundary
(0.5 keeps 0 and 1, the `<=` side) over a hand-packed 3×4 tile with a `code_in_byte` bit-order
sanity check; and the real-AADR oracle (`test_filter_oracle.cu` block (c'), lines 345-389)
asserting per-sample non-missing counts match an independent recompute exactly and the kept set
matches at a biting cap, plus the no-op keeps all. That is solid §13 coverage of the *happy*
paths and boundaries. **Not** covered: `n_individuals == 0`, `n_snp == 0` (the 1.3 fail-safe),
`bytes_per_record == 0` (1.2), the OOB-on-undersized-record case (1.1), and a multi-byte record
(all current unit-test cases use `bytes_per_record == 1`, so the `s/4` byte-stride across bytes
is never exercised by the unit test — the oracle covers it via the real multi-byte tile, but the
unit test's 4-SNP single-byte case can't catch a `s/4` indexing bug locally). Fix: add unit
cases for the empty-sample, no-SNP, and ≥5-SNP (multi-byte record) inputs; once the accumulator
(9.1) exists, test add_tile across two tiles summing to the one-shot result.
Adversarial check: is the multi-byte gap real given the oracle covers it? The oracle does
exercise multi-byte records on real data, so a `s/4` bug *would* be caught by CI — but only on
the box with data, and only as part of a large test. A cheap host unit case with `n_snp = 5`
(2 bytes/record) would catch it locally with no data. Worth adding; low severity since the
oracle backstops it.

**10.2 — The unit is highly testable by construction (host-pure, no CUDA, plain inputs).
[N/A — verified good]**
No GPU, no I/O, no globals — a `MindPrepassInput` is hand-buildable in a few lines (the test
does exactly this). This is the §2 testability goal achieved. Good.

### (11) Capability tiers (PRO-6000-capable vs budget-5090) vs TODO

**11.1 — No capability-tier hook, no tagged log line, on a unit that sits on the M5 ingest
path. [med, M, before-M4.5: partial]**
Where it touches the tiers: TODO's capability table (TODO.md:125-130) and the M5 ingest line
(TODO.md:106) put the genotype byte stream squarely in the capability-tiered ingest path — the
capable path (full-host RTX PRO 6000: true GDS cuFile DMA NVMe→VRAM) vs the budget-5090 fallback
(POSIX pread into pinned double-buffer), "every run records which path it took + why it
degraded" (TODO.md:137 "cross-cutting — a capability probe + capability-tagged results"). The
`--mind` pre-pass reads exactly those bytes. Today mind_prepass is a pure CPU loop over an
already-resident buffer — it has no notion of how those bytes arrived, no capability probe, and
emits no tagged log line. That is *acceptable as long as it is purely host-side and someone else
owns ingest*, but the unit is silent about it.
Why it matters: TODO's cross-cutting requirement is that every stage that touches ingest records
its path/degradation tag. When 9.1's streaming accumulator lands (fed by the M5 pinned
double-buffer / optional GDS lane), the `--mind` pass IS an ingest consumer and should (a) be
fed tiles by the capability-tiered ingester (so it inherits the tag) and (b) at minimum log a
single line stating it ran (`[S-1 mind] streamed N tiles, M SNPs, dropped K/Ni samples at
mind<=X`) so a run's audit trail shows the conditional pass fired. Parity is unaffected either
way (the byte values are identical regardless of how they were ingested — TODO.md:133 "GDS only
changes how bytes reach VRAM"), so this is observability/integration, not a parity lever.
Fix: (a) when the accumulator (9.1) is added, take tiles from the shared capability-tiered
ingester rather than a self-loaded buffer, inheriting the ingest tag; (b) add a single
structured log line on completion (count of tiles/SNPs streamed, samples dropped, the active
threshold) consistent with the rest of steppe's tagged logging; (c) record in the run's
capability summary that S-1 fired (or was skipped). No probe of its own is needed — mind_prepass
should *consume* the ingest tier, not detect it.
Adversarial check: should a host-pure leaf know about capability tiers at all? No — and it
shouldn't *probe*. The right design keeps mind_prepass capability-agnostic and pushes the
probe/tag to the ingester (which §4 layering supports: io owns ingest). So the finding is really
"this unit must be wired to the tiered ingester and emit its audit line," not "add a probe here."
Rated med because it is an integration requirement TODO calls cross-cutting, and partial-before-
M4.5 because the actual GDS/pinned wiring is M5, but the audit-log line is cheap now.

## Considered & rejected

- **"`code_in_byte`'s `k` could be out of range."** Rejected: `s % 4u ∈ {0,1,2,3}`, and
  `code_in_byte` (eigenstrat_format.hpp:76-79) computes a shift `(4-1-(k%4))*2 ∈ {0,2,4,6}` and
  masks `& 0x3`, so any in-range `k` is correct, and the passed `k` is always valid. No bug.
- **"`missing_frac` division by zero when `n_snp == 0`."** Rejected: the `n_snp > 0` guard
  (line 32) ensures the division at line 47 only runs when `n_snp > 0`; the no-data branch (49)
  does not divide. No divide-by-zero.
- **"`kept` may not be ascending."** Rejected: built by `push_back` over `g = 0..n_ind-1` in
  order, so it is ascending by construction, matching the header/FilterPlan contract. Verified.
- **"FP equality / boundary instability at `frac == mind_max_missing`."** Rejected for the
  count-derived fracs: at the integer-exact endpoints (`nm==0`→1.0, `nm==n_snp`→0.0) the value is
  exact, and the predicate uses `<=` consistently with the oracle's `<=` (test_filter_oracle.cu:
  380) and the unit test's exact-0.5 case — the boundary is pinned and reproducible. (The frac
  *definition* duplication is the real issue — see 3.1 — not the comparison itself.)
- **"Should use `std::accumulate`/`std::count_if` for the inner loop."** Rejected: the inner
  loop unpacks 2-bit codes from a byte, not a flat range; `std::count_if` would need a custom
  iterator over codes and would be *less* clear than the explicit loop. The raw loop is the
  right call here.
- **"`reserve(n_ind)` then conditionally fewer push_backs wastes memory."** Rejected: `kept ≤
  n_ind`, the over-reserve is at most n_ind `std::size_t`s (≤ ~220 KB for the real panel),
  freed with the vector, and it avoids reallocation in the common keep-most case. Correct
  trade-off.
- **"Marking `run_mind_prepass` `noexcept` would be better."** Rejected: it allocates O(n_ind)
  vectors and can throw `std::bad_alloc`; `noexcept` would convert that to `std::terminate`,
  which is worse for a library entry point. Correctly left non-noexcept (the doc should *say*
  so — 7.1).
- **"The two-pass structure (scan, then resolve) should be fused into one loop."** Rejected:
  the resolve loop needs `active` (a single config read) and is O(n_ind), trivially cheap;
  fusing it into the scan would not help (the scan is O(n_ind·n_snp) and dominates) and would
  entangle the no-data branch with the resolve. The current separation is cleaner; the real
  perf win is 8.1 (skip the scan when inactive), not fusion.
- **"`out.missing_frac` should be `float` to halve memory."** Rejected: §12 wants the frac
  compared on the same FP type as the oracle (double); the oracle uses double; and O(n_ind)
  doubles is ~220 KB — negligible. Keep double.

## What it takes to reach 10/10

1. **(9.1, high) Make it actually streaming OR explicitly defer.** Add a `MindAccumulator`
   (`add_tile` per SNP tile + `finalize(cfg)`), keep `run_mind_prepass` as a thin one-shot
   wrapper. If multi-tile is deferred to M5, say so in the header instead of claiming to be the
   streaming pass. This is the unit's central contract.
2. **(8.1, med) Honor "skipped entirely."** Gate the full scan on `active` (`mind_max_missing <
   1.0`); make no-op-path fraction reporting opt-in, not the default. No wasted ~4 GB sweep on
   the parity path.
3. **(3.1 + 6.1, med) Single-home the frac formula + accumulation.** Add
   `per_sample_missing_frac(nonmissing, n_snp)` (and ideally a shared non-missing-in-record
   helper) to filter_decision.hpp/eigenstrat_format.hpp; have mind_prepass and the oracle call
   them, so §12 parity is structural, not copy-paste.
4. **(5.1, med) Kill the bare `4u`.** Use `kCodesPerByte` (or a new `io::code_at(rec, s)` helper)
   for the `/4`/`%4`; sweep the same `4u` in test_filter_oracle.cu:359.
5. **(1.1/1.2/2.4, med→low) Fail fast on inconsistent input.** Assert/validate
   `bytes_per_record >= packed_bytes(n_snp)`, `bytes_per_record > 0` when `n_snp > 0`, and
   non-NaN/non-negative `mind_max_missing` (the last at config build); document `packed`'s
   required extent. Prefer `std::span` for `packed` (4.3).
6. **(1.3, low) Reconcile the empty-denominator fail-safe** with snp_filter (frac 1.0 vs 0.0)
   or document the intentional divergence in both files.
7. **(11.1, med) Wire to the tiered ingester + emit an audit line.** When streaming lands,
   consume capability-tiered tiles (inherit the tag) and log one structured S-1 line; keep the
   unit itself capability-agnostic (no probe here).
8. **(7.1, low) Document the exception contract** ("throws only std::bad_alloc").
9. **(10.1, low) Add degenerate/multi-byte unit cases** (empty samples, 0 SNPs, ≥5-SNP record;
   add_tile-across-two-tiles once the accumulator exists).
10. **(8.2, trivial) Drop the redundant `missing_frac = 0.0` re-zero loop** (already
    value-initialized).

## Good patterns to keep

- **Textbook layering.** A genuinely host-pure `io`-leaf with no CUDA and no upward dependency,
  reusing the io-side byte path (`code_in_byte`/`kMissingCode`) rather than reaching up into
  `core/internal/decode_af.hpp` — and the header *explains why* (shared pinned bit order).
- **DRY on the decision.** The keep/drop decision is delegated to the single shared predicate
  `sample_passes_mind`, not re-spelled — exactly the single-source discipline filter_decision.hpp
  exists to enforce.
- **Deterministic by construction.** Integer counts + one division per sample + ascending
  push_back; no accumulation-order or FP-reduction concern, matching §12's fixed-order property,
  with the boundary pinned to the `<=` side consistently with the oracle and the unit test.
- **High, accurate comment density** that matches the surrounding io files: the header states
  the spec section, the layering rationale, the SAMPLE-GLOBAL invariant, and the no-op semantics;
  the .cpp comments justify each branch.
- **Correct value-init + reserve.** `assign(n, 0)` for the sized/zeroed outputs and `reserve(n)`
  for the grow-by-push_back `kept` — the right std::vector idioms.
- **Real two-tier testing.** A host unit test for boundary logic + a real-AADR oracle that
  recomputes non-missing independently and asserts integer-exact — the §13 "host logic test +
  real-data oracle" split, done right.
- **Honest convention flags.** Where a choice is debatable (the no-SNP fail-safe), the code says
  so rather than hiding it — the same candor the sibling filter_decision.hpp shows on the
  strand-ambiguous convention.
