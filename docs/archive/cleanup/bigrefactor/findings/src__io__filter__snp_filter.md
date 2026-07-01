# Review findings — src__io__filter__snp_filter

Files: /home/suzunik/steppe/src/io/filter/snp_filter.cpp, /home/suzunik/steppe/src/io/filter/snp_filter.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Notes (why clean), for the record:
- 4.2/4.6 (index width / overflow): the only global index into the P*M arrays is
  snp_filter.cpp:87-88 `off = (size_t)p + (size_t)P * (size_t)s` — every operand is
  widened to std::size_t BEFORE the multiply and add, so P*s and the sum are 64-bit.
  At scale (P=2500, M=584131 ⇒ P*M ≈ 1.5e9, larger tiles can exceed 2^31) this is
  correctly computed in size_t. SNP loop counters `s`/`M` are `long`; `total_indiv`,
  `Mu`, `si` are `std::size_t`. No 32-bit index/offset into a >2^31 array.
- 4.1 (float-vs-double): all arithmetic is FP64 by design (pooled AF, missing_frac,
  npn/ploidy_d) — intentional, parity-load-bearing; no wrong narrowing.
- 4.3 (allocation sizing): only std::vector element-count constructors
  (snp_filter.cpp:21, 113) — no cudaMalloc/new, no byte-vs-element confusion.
- 4.4 (unsigned countdown): no reverse loops.
- 4.5 (signed/unsigned compares): size_t-vs-size_t comparisons use explicit casts
  (snp_filter.cpp:51, 126, 132, 137); loop bounds are same-signedness (long<long,
  int<int).
- 4.7 (host/device pointer typing): raw `const double*` q/v/n (snp_filter.hpp:42-44)
  are host memory by contract in this host-pure `io`-leaf TU (is_cuda=false); no
  device memory is touched, so no host/device space-confusion risk here.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Notes (why clean), for the record:
- This is a host-pure `io`-leaf TU (is_cuda=false): plain C++20, no CUDA. No
  __global__/__device__ kernels, no nvcc-compiled code. The only "__host__
  __device__" / "cuda" / "geno" string hits (snp_filter.hpp:106, snp_filter.cpp:48,61)
  are in comments/prose (a reference to the decode_af device primitive pattern,
  and the substrings in "geno"/"surface"), not code.
- 2.1 (dropped archs Maxwell/Pascal/Volta, sm_50/60/70): no arch flags, no CMake
  arch lists, no compute_*/sm_* in this unit (CMake is out of scope of these files).
- 2.2 (texture/surface references removed in CUDA 12): no texture<...>, surface<...>,
  cudaBindTexture*, or texture-object API anywhere in the unit.
- 2.3 (non-_sync warp intrinsics): no __shfl/__ballot/__any/__all/__activemask —
  no warp-level code at all (host TU).
- 2.4 (cudaThreadSynchronize -> cudaDeviceSynchronize): no CUDA runtime sync calls
  (no cudaThreadSynchronize, no cudaDeviceSynchronize) — no CUDA runtime use here.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Notes (why clean), for the record:
- 3.1 (commented-out blocks "just in case"): none. Every comment in both files is
  documentation/rationale prose (file headers snp_filter.cpp:1-6 / .hpp:1-21, the
  cleanup-reference rationale at .cpp:24-67/120-141, the doc comments .hpp:35-170) —
  no commented-out statements or disabled code.
- 3.2 (unreachable code): no `#if 0`. No code after a return/break/throw. The early
  returns (snp_filter.cpp:22 `if (P<=0||M<=0) return out;`, :114 `if (M<=0) return
  keep;`) are guards; every `return false`/`throw` is on a conditional guard path,
  all reachable.
- 3.3 (unused symbols): all includes are used — .cpp `<cstddef>` (std::size_t),
  `<stdexcept>` (std::invalid_argument), `<string>` (std::to_string),
  filter_decision.hpp (folded_maf); .hpp `<cstddef>`, `<vector>`,
  filter_decision.hpp (predicates in snp_keep_decision), include_exclude.hpp
  (SnpMembership), snp_reader.hpp (SnpTable), config.hpp (FilterConfig). No unused
  params/locals/helpers. The struct field DecodedTileSummaryInput::v (.hpp:43) is
  documented as intentionally unread (N==0 ⇔ V==0 ⇔ missing by the decode contract,
  .hpp:98-99, .cpp:62-63) — it is a public INPUT-CONTRACT field mirroring
  DecodeResult.q/.v/.n, not a dead local; intentional, not flagged.
- 3.4 (computed but unread): every assignment is read. total_indiv -> total_indiv_d
  -> missing_frac; ploidy_d -> numerator; pooled_ref_count/pooled_allele_count/
  nonmissing_indiv all feed sm.* outputs; mem_noop read twice (:137,:154); chrom &
  membership_ok feed snp_keep_decision (:157). No write-only variable.
-->

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

<!--
Notes (why clean), for the record:
- 5.1 (unnamed literals): the only literals are domain constants and math identities,
  not nameless tuning knobs. The ploidy values 1/2 (snp_filter.cpp:35, .hpp:62/57)
  are self-documenting (haploid/diploid), validated by a fail-fast throw whose message
  spells them out ("ploidy must be 1 (pseudo-haploid) or 2 (diploid)", :36-38), and
  pinned to the decode_af.hpp contract; naming them would not add clarity. 0/0.0/1.0
  (guards P<=0||M<=0 :22, pooled_allele_count>0.0 :100, total_indiv_d>0.0 :103,
  1.0-nonmissing/total :103, the inert chrom placeholder 0 :150, false init :113) are
  mathematical identities / empty-sum & complement defaults, not configurable values.
  All actual THRESHOLDS (maf_min, geno_max_missing) come from cfg (snp_keep_decision
  .hpp:128-134) — none are hardcoded here.
- 5.2 (hardcoded sizes/bounds): none. Every size derives from runtime inputs — out/keep
  sized from M (.cpp:21,113), loop bounds P and M, Mu/si/off all from P,M,s. No capacity,
  tile size, or array-length constant baked in.
- 5.3 (duplicated constants / drift): the [P x M] column-major offset
  (size_t)p + (size_t)P*(size_t)s appears once (.cpp:87-88); the ploidy set {1,2} is
  used consistently in the condition and its throw message (:35-38). No value duplicated
  across sites that could drift.
- 5.4 (hardcoded paths/IDs/device ids): none — host-pure io-leaf TU, no filesystem paths,
  no device IDs, no CUDA stream/device handles.
- 5.5 (ambiguous 32 / warp size): no 32 anywhere; no warp/block dims (no CUDA in this TU).
-->

## Group 6 — Naming

No Group 6 issues found.

<!--
Notes (why clean), for the record:
- 6.1 (cryptic names): the short locals are either domain-standard math notation or
  tight-loop counters, each documented at its definition. `P`/`M` (snp_filter.cpp:18-19,
  .hpp:45-46) are the canonical pop/SNP dimensions (used project-wide). `p`/`s`
  (.cpp:73,79,86,145) are pop-axis / SNP-axis loop counters. `off` (.cpp:87) is the
  column-major offset, documented inline ("element (pop i, snp s) at i + P*s"). `npn`
  (.cpp:89) carries an explicit same-line comment ("non-missing haploid count for
  (pop, snp)"). `sm` (.cpp:97) is the PerSnpSummary reference. `in`/`cfg`/`mem`/`snps`
  (.cpp:17,108-111) are conventional struct-arg short names, each a documented typed
  parameter. None are anonymous `tmp`/`data2`/`arr`/`flag`.
- 6.2 (misleading names): names match contents. `total_indiv`/`nonmissing_indiv`
  (.cpp:72,85) are individual counts (denominator/numerator of the missing fraction),
  `pooled_ref_count`/`pooled_allele_count` (.cpp:83-84) are the pooled sums, `keep`
  (.cpp:113) is the keep MASK, `membership_ok`/`mem_noop` (.cpp:125,154) are bools.
  No count-that-is-an-index or list-that-is-a-map.
- 6.3 (inconsistent conventions in one file): consistent snake_case for locals/fields
  (`pooled_ref_count`, `total_indiv_d`, `missing_frac`, `pop_individuals`,
  `nonmissing_indiv`) with the deliberate uppercase math dims `P`/`M`; the `_d` suffix
  ("as double") is applied consistently (`total_indiv_d`, `ploidy_d` .cpp:76-77). No
  nElements-vs-num_elements-vs-n clash within either file.
- 6.4 (nonstandard abbreviations): the abbreviations present (`npn`, `sm`, `Mu`, `si`,
  `off`, `_d`) are each defined/documented where introduced — `Mu`/`si` are the size_t
  forms of M/s (.cpp:116,146), `off` the offset, `npn` documented inline. Borderline
  but not opaque; none risk a wrong reading. Public API names (DecodedTileSummaryInput,
  PerSnpSummary, derive_per_snp_summary, build_snp_keep_mask, snp_keep_decision,
  pooled_minor_af, missing_frac) are fully spelled out.
-->

## Group 7 — Duplication

- [7.2][LOW] snp_filter.cpp:87-88 — inside the hot inner pop loop the offset computes `static_cast<std::size_t>(P) * static_cast<std::size_t>(s)` every iteration, but both factors are invariant w.r.t. `p`; only `(size_t)p` varies. `(size_t)P * (size_t)s` is a loop-invariant base recomputed P times per SNP (P*M multiplies total). Suggested: hoist `const std::size_t base = static_cast<std::size_t>(P) * static_cast<std::size_t>(s);` to the outer SNP loop and index `base + (std::size_t)p` (semantically identical, keeps the 64-bit widening).
- [7.4][LOW] snp_filter.cpp:126-141 — the three SnpTable size-preconditions repeat the same boilerplate shape (`if (active && snps.FIELD.size() < Mu) throw std::invalid_argument("snp_filter: ... >= M (" + std::to_string(M) + "); got ... " + std::to_string(snps.FIELD.size()))`) differing only by the field + active-flag + label. Suggested: a small local lambda `require_at_least(const char* what, std::size_t have, bool active)` (or guard helper) would fold the three throws; low priority since each message is distinct and the file is short.
- [7.4][LOW] snp_filter.cpp:36-38,52-55,65-66 — the four `derive_per_snp_summary` precondition throws share the `throw std::invalid_argument("snp_filter: " + ... + std::to_string(...))` idiom; the `"snp_filter: "` prefix is repeated at every site (also .cpp:127,133,138). Suggested: optional — factor the prefix or a `throw_invalid(msg)` helper; the distinct diagnostics make this borderline and extraction would slightly obscure each message, so leaving as-is is defensible.

## Group 8 — Comments

No Group 8 issues found.

<!--
Notes (why clean), for the record:
- 8.1 (restating code): no comment merely paraphrases its statement. Every inline
  comment carries information not in the code: snp_filter.cpp:89 (`npn` = "non-missing
  haploid count for (pop, snp)" — units), :91 (WHY no missing-branch is needed: Q
  zero-filled + N==0 at missing cells), :83-85 (the pooled scalars + their AT2-oracle
  equivalences `== Σ ssum_pop` / `== Σ 2·scnt_pop`), :70-71/:80-81 (the SNP-global
  reduction invariant), :148-149 (chrom read only when autosomes_only active). None
  are `i++; // increment i`.
- 8.2 (stale comments): every comment matches the current code. .cpp:71 "`pop_individuals.size()
  == P` is pinned above, so this loop is unconditional" is true given the throw at
  :51-56. .cpp:62-63 / .hpp:98-99 "v is NOT required / intentionally unused" matches —
  `in.v` is never dereferenced (the reduction reads only q/n, .cpp:89-94). .hpp:131
  "is_monomorphic takes the UNFOLDED pooled ref-af" matches .hpp:132 (passes
  `sm.pooled_ref_af`, the unfolded value). .hpp:152-153 / .cpp:151-153 refer to a
  "future M4.5 device kernel" sharing `snp_keep_decision` — forward-looking design
  intent, not a claim about removed behavior. The .cpp:41-50/120-124 "the prior code
  …" passages describe the REPLACED behavior as the rationale for each fail-fast and
  are explicitly historical, not stale descriptions of the present code.
- 8.3 (missing rationale): every non-obvious choice already carries its WHY. The ploidy
  ∈ {1,2} fail-fast (.cpp:24-34) explains the silent-clamp-to-1 bug it prevents and
  reconciles host-throw vs device-mask-out. The pop_individuals.size()==P check
  (.cpp:41-50) explains the negative-missing_frac spurious-keep bug. The null q/n check
  (.cpp:58-63) explains the segfault-three-frames-deep alternative AND why v is exempt
  (N==0 ⇔ V==0 ⇔ missing by decode_af.hpp). The SnpTable size checks (.cpp:120-124)
  explain the is_multiallelic('N','N')-drops-the-tail silent bug. The unfolded-ref-af
  to is_monomorphic note (.hpp:130-131, "ends the double-fold") and the precomputed-
  membership rationale (.cpp:151-153) are present. Constants (1/2 ploidy, 0.0/1.0
  empty-sum/complement defaults) are math identities, not tuning knobs needing
  rationale (cf. Group 5 notes).
- 8.4 (orphan TODO/FIXME/HACK): none — grep for TODO|FIXME|HACK|XXX|WIP over both files
  returns nothing. No dangling markers, owner-less or otherwise.
-->

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Notes (why clean), for the record:
- 9.1 (should-be-const/constexpr left mutable): all locals that CAN be const ARE.
  derive_per_snp_summary: P/M (snp_filter.cpp:18-19), total_indiv_d/ploidy_d (:76-77),
  off/npn (:87,89) are const; total_indiv (:72) and the inner accumulators
  pooled_ref_count/pooled_allele_count/nonmissing_indiv (:83-85) are intentionally
  mutable (loop accumulators). build_snp_keep_mask: M (:110), Mu (:116), mem_noop
  (:125), summary (:143), si (:146), chrom (:150), membership_ok (:154) are all const;
  only the output `keep` mask (:113) is mutated by design. No mutable-but-should-be-
  const. No compile-time constant to promote to constexpr — every value derives from
  runtime inputs (cf. Group 5).
- 9.2 (tangled config): all tunable knobs are surfaced in a config struct, none buried
  in logic. The thresholds + flags (maf_min, geno_max_missing, drop_monomorphic,
  transversions_only, autosomes_only) come from FilterConfig& cfg (snp_keep_decision
  reads cfg.* at .hpp:128-134); ploidy and pop sizes come from the input struct
  DecodedTileSummaryInput (.hpp:41-63). The ploidy {1,2} set is a domain-validated
  METADATA contract (decode_af.hpp), fail-fast checked (.cpp:35-39), not a buried
  tuning knob. No literal threshold or behavior switch is hardcoded inside the loops.
- 9.3 (positional booleans foo(true,false,true)): no positional boolean literals at any
  call site. snp_keep_decision(...) (snp_filter.cpp:157-158) passes the bool last arg as
  the NAMED local `membership_ok`, not a literal. The only bool literals are the
  std::vector<bool> fill value `false` (.cpp:113) and the `true` in the
  `mem_noop ? true : ...` ternary (:154) — a container fill-element and a ternary
  branch, not a multi-flag positional config call.
-->

## Group 10 — Initialization

No Group 10 issues found.

<!--
Notes (why clean), for the record:
- 10.1 (late/distant decl, uninitialized-then-assigned): no scalar is declared
  uninitialized and assigned later. In derive_per_snp_summary every local is
  initialized at its declaration AND adjacent to first use: `out` (snp_filter.cpp:21),
  `total_indiv = 0` right before its loop (:72), `total_indiv_d`/`ploidy_d` (:76-77),
  the SNP-loop accumulators `pooled_ref_count`/`pooled_allele_count`/`nonmissing_indiv`
  = 0.0 at the top of the loop body (:83-85), `off`/`npn` at use (:87-89). `sm` (:97)
  is a reference bound at use whose four fields are ALL unconditionally assigned
  (:98-103) before any read. In build_snp_keep_mask `M`/`keep`/`Mu`/`mem_noop`/`summary`
  (:110-143) and the per-iteration `si`/`chrom`/`membership_ok` (:146-154) are each
  initialized at declaration at point of use. No declare-here-assign-far pattern.
- 10.2 (zero-init assumptions that do not hold): both std::vectors use EXPLICIT
  construction, not silent zero-reliance. `out(M)` (:21) value-constructs PerSnpSummary,
  whose in-class member initializers (snp_filter.hpp:69-72: pooled_ref_af=0.0,
  pooled_minor_af=0.0, missing_frac=1.0, pooled_allele_count=0.0) set the "no data"
  defaults — note missing_frac defaults to 1.0 (NOT 0), so it does not rely on
  zero-init; and every field is then unconditionally overwritten in the loop (:98-103).
  `keep(Mu, false)` (:113) passes the fill value explicitly. The accumulators are
  explicitly `= 0.0` (:83-85), not zeroed implicitly. The input structs
  DecodedTileSummaryInput (.hpp:42-62: pointers=nullptr, P=0, M=0, ploidy=2) and
  PerSnpSummary carry default member initializers, so no field silently depends on a
  caller value-initializing the struct. No memset/calloc; no POD trusting zero-init.
-->

