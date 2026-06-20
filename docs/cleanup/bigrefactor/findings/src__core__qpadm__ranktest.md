# Review findings — src__core__qpadm__ranktest

Files: /home/suzunik/steppe/src/core/qpadm/ranktest.cpp, /home/suzunik/steppe/src/core/qpadm/ranktest.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Notes (not findings — context for why this is clean):
- 4.1: All math is intentional FP64 (weights/chisq/p/dof are double; NaN via std::numeric_limits<double>::quiet_NaN; popdrop_feasible compares against 0.0/1.0). No float temps, no narrowing in a parity-critical path. N/A per FP64 context.
- 4.2/4.6: This orchestrator operates on the ALREADY-REDUCED f4 covariance (m = nl*nr, population-scale), NOT the P*P*n_block f2 tensor or the M*P genotype matrix. The one large index — the Qinv/Q sub-block address at ranktest.cpp:70-72 (src = ind[a] + m_full*ind[b]) — is correctly widened: each operand is cast to std::size_t BEFORE the multiply/add (static_cast<std::size_t>(m_full) * static_cast<std::size_t>(ind[b])). The dst index (73-74) is likewise size_t-widened. The smaller indices (j + nr*ii at lines 49/53-54/62) stay within m_red ≤ m_full and fit int at qpAdm population scale.
- 4.3: No cudaMalloc/new; all allocation is std::vector::assign sized in element count (correct for std::vector). The Qinv/Q assign at lines 66-67 uses static_cast<std::size_t>(m_red)*static_cast<std::size_t>(m_red) — size_t multiply, no overflow before widening.
- 4.4: No unsigned countdown loops. The descending drop loop (148) uses signed int drop with drop >= 0 — terminates correctly.
- 4.5: Loop bounds are int-vs-int (lines 50/52/59/61/68-69) or size_t-vs-size_t (line 120 s < surv.size()); no signed/unsigned mismatch.
- 4.7: Host-pure file (is_cuda=false), CUDA-free per the header contract; no device pointers — 4.7 N/A.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Notes (not findings — context for why this is clean):
- 2.1: No CMake arch lists or build/compile flags in this unit (ranktest.cpp/.hpp are source-only); no sm_50/60/70 references. N/A.
- 2.2: No texture/surface references (no `texture<...>`, no `cudaBindTexture*`, no surface refs) — host-pure file with no CUDA at all. N/A.
- 2.3: No warp intrinsics (no `__shfl*`/`__ballot*`/`__any*`/`__all*`), sync or non-sync. N/A.
- 2.4: No CUDA runtime calls at all (no `cudaThreadSynchronize`/`cudaDeviceSynchronize`). Includes are only <cmath>/<cstddef>/<limits>/<string>/<vector> + project headers; all work is host index arithmetic + std::vector, fit routed through the ComputeBackend seam. N/A.
-->

## Group 3 — Dead / commented-out code

- [3.4][LOW] ranktest.cpp:44 — `(void)m_full;` is a dead no-op suppression: `m_full` (declared line 42) IS read at line 71 (`static_cast<std::size_t>(m_full) * ...`), so the unused-variable cast is leftover and misleading (likely from a prior version where m_full was unused). Suggested: delete line 44.

<!--
Notes (not findings — context for why the rest is clean):
- 3.1: No commented-out CODE kept "just in case". Every comment block (file header 1-8, reduce_rows doc 34-37/57, popdrop_one doc 79-81, the AT2 rank-fit rationale 94-105, the inline weight/full-row notes 115/137/142-146) is explanatory contract/design documentation, not disabled source.
- 3.2: No unreachable code. No `#if 0`. No statements after a return/break; the `continue` at line 25 and the early `return false` at line 27 are normal control flow within loops.
- 3.3: No unused symbols. All includes are used: <cmath> (std::isnan L25), <cstddef> (std::size_t), <limits> (std::numeric_limits L117), <string> (row.pat is std::string, L86-87), <vector>, steppe/error.hpp (Status::Ok L119). All function params are used (be/x/cov/opts/precision threaded through; nl_full/drop/surv consumed). The anonymous-namespace helpers reduce_rows + popdrop_one are both called (L91, L140/L152).
- 3.4: Only the L44 (void)m_full above. Every other local is read: nr, nl_red, m_red, ind, r_fit, ri, rs, gw, row fields all consumed downstream.
-->

## Group 5 — Hardcoded values / magic numbers

- [5.3][MED] ranktest.cpp:110 — the dof fallback `(xr.nl - r_fit) * (xr.nr - r_fit)` duplicates the chi-square degrees-of-freedom formula `(nl-r)*(nr-r)` that the backend `rank_sweep` computes internally (rs.dof). If the backend's dof convention ever changes (e.g. a different parameter count), this local copy silently DRIFTS, producing a row.dof that disagrees with the rest of the sweep on the fallback path (taken when ri >= rs.dof.size()). Suggested: factor the dof formula into one shared helper (e.g. qpadm_dof(nl,nr,r)) used by both the backend and this fallback, so the two cannot diverge.

<!--
Notes (not findings — context for why the rest is clean):
- 5.1: No unnamed magic literals. The `0.0`/`1.0` at ranktest.cpp:27 are the AT2 weight-feasibility bounds [0,1] (the allow_negative_weights=false predicate) — semantic, self-documenting, parity-load-bearing per the header contract (.hpp:44-46), not a tunable. The pattern chars '0'/'1' (L86-88) and wt=1/0 (L88) are the AT2 res$popdrop pat-string / weight-column encoding (verbatim contract). The 0.0 fills (L49/66/67) are zero-initialization of vectors. quiet_NaN (L117) is the documented dropped-slot sentinel. None are anonymous constants to be named.
- 5.2: No hardcoded sizes or bounds. Every dimension is derived from the inputs: nr=x.nr (L40), nl_red=surv.size() (L41/92), m_full=x.nl*nr (L42), m_red=nl_red*nr (L43), r_fit=nl_red-1 (L106), the reserve nl_full+1 (L135). No fixed capacities, no clamps, nothing that should be a parameter.
- 5.4: No hardcoded paths, IDs, or device ids (host-pure file, no devices/files referenced).
- 5.5: No literal 32 (or any warp-size / block-dim constant) anywhere in the unit — host orchestrator, no kernel launches or shared-mem sizing. N/A.
-->

## Group 6 — Naming

- [6.1][LOW] ranktest.cpp:90 — the reduced-fit locals are named `xr`/`cr` (and the helper out-params `xr`/`cr` at ranktest.cpp:39), two-char opaque names for "reduced F4Blocks" / "reduced JackknifeCov". Outside their tight loops they read as noise next to the otherwise descriptive `nl_red`/`m_red`. Suggested: rename to `x_reduced`/`cov_reduced` (matches the `_red` suffix already used for the dims).
- [6.1][LOW] ranktest.cpp:22-23 — in `popdrop_feasible`, the accumulator flag is named `any` (a generic flag-ish name) for "at least one surviving (non-NaN) weight was seen". Suggested: rename to `has_surviving` to state the predicate it carries.

<!--
Notes (not findings — context for why the rest is clean):
- 6.1: The single-letter math-matrix names are intentional/acceptable: `x` = the F4Blocks f4-estimate matrix (the standard AT2 design-matrix symbol, used consistently across the qpadm fit seam), `cov` = the JackknifeCov. The abbreviations `be` (backend), `gw` (GlsWeights), `rs` (RankSweep), `ri` (rank index into rs arrays), `surv` (surviving sources), `ind` (index map) are each used in a tight local scope with the full type named at the declaration and/or documented in the adjacent contract comment (e.g. `ind` is defined inline at L57). The loop counters `i`/`ii`/`j`/`a`/`b`/`s`/`w` are all tight-loop indices (matrix row/col/element walk), the canonical exception to 6.1.
- 6.2: No misleading names. `row.wt` (L88, set to 1/0) is NOT a misnamed count — it is the AT2 res$popdrop "weight column" encoding reproduced verbatim per the header contract (.hpp). `surv` is genuinely the surviving-source list, `drop` the dropped index, `ind` an index map (correctly named a map of positions, not a count).
- 6.3: Conventions are consistent within the file: snake_case lowercase throughout (nl_full, nl_red, m_full, m_red, r_fit, nl_red recomputed identically at L92). The terse two-char locals (xr/cr/gw/rs) are the only departure from the descriptive `_full`/`_red` style — flagged under 6.1 above rather than as a separate convention split. No nElements-vs-num_elements-vs-n mix.
- 6.4: No nonstandard abbreviations. `qinv`/`Qinv`/`Q` are the AT2 fudged-inverse-covariance symbols (domain-standard, in the contract). `dof` (degrees of freedom), `chisq` (chi-square), `nl`/`nr` (n-left / n-right), `pat` (pattern string) are all established qpAdm/AT2 vocabulary documented in the header.
-->

## Group 7 — Duplication

- [7.1][LOW] ranktest.cpp:50-55,59-63 — two adjacent loop nests in `reduce_rows` share identical iteration structure (`ii` over `nl_red`, `i = surv[ii]`, inner `j` over `nr`, same `j + nr*ii` destination index): the first fills `xr.x_total`, the second fills the `ind` index map. They are copy-pasted scaffolding differing only by the written target. Suggested: fuse into one pass that writes both `xr.x_total[j+nr*ii]` and `ind[j+nr*ii]` in the same inner body (computing `i`, the dest index, and the source index `j+nr*i` once each).
- [7.2][LOW] ranktest.cpp:70-74 — in the Qinv/Q copy nest, the row operand `static_cast<std::size_t>(ind[a])` and `static_cast<std::size_t>(a)` are recomputed on every inner-`b` iteration though both are invariant in `b`; only the column term varies. Suggested: hoist `const std::size_t ia = static_cast<std::size_t>(ind[a]);` and `const std::size_t a_sz = static_cast<std::size_t>(a);` to the outer-`a` loop body so the inner loop only re-evaluates the `b`-dependent column term.

<!--
Notes (not findings — context for why the rest is clean):
- 7.2: nl_red IS recomputed in two scopes (reduce_rows:41 `static_cast<int>(surv.size())` and popdrop_one:92 `static_cast<int>(surv.size())`), but these are in SEPARATE functions with no shared cheap-to-pass binding (popdrop_one passes `surv` by ref into reduce_rows; reduce_rows already has its own copy), and `surv.size()` is O(1) — not a loop-invariant recompute inside a hot loop. Left un-flagged as it is two independent locals, not a repeated expression in one expression context. No other loop-invariant arithmetic survives: m_full/m_red/nr/nl_red are each computed once (L40-43) and reused.
- 7.3: No `sizeof` anywhere. The pervasive `static_cast<std::size_t>(...)` index widening is the §4 scale-safety idiom (Group 4 confirmed the Qinv/Q address is correctly widened before multiply) — necessary, not redundant casts to fold; flagging them would be noise against the FP64/scale context. The two genuinely-hoistable repeats are the 7.2 inner-loop operands above.
- 7.4: No collapsible macro/helper boilerplate. The two table-row builders (the full "0..0" row via popdrop_one(...,-1,all,...) at L140 and the per-drop rows at L152) already share the SAME helper (popdrop_one) — the duplication is already factored. row.pat init (L86-87) and row.weight NaN-fill (L116-117) are single small per-row initializers, not repeated blocks. The run_popdrop full-vs-drop split (L138-140 vs L147-154) differs in real logic (the `all` set vs the per-drop `surv` set + the row-order contract), not boilerplate that a helper would fold.
-->

## Group 8 — Comments

- [8.3][LOW] ranktest.cpp:88 — `row.wt = (drop >= 0) ? 1 : 0;` has no comment, yet `wt` is the AT2 res$popdrop "weight column" = the COUNT of dropped sources (0 for the full model, 1 for each single-drop), which collides semantically with the adjacent `row.weight` (the mixture weights, set NaN/[0,1] at L116-122). The non-obvious AT2-column mapping is documented in the .hpp contract but not at the assignment, where a reader most needs it to tell `wt` apart from `weight`. Suggested: add a one-line comment, e.g. `// AT2 popdrop "weight" column = number of dropped sources (0/1), NOT mixture weight`.

<!--
Notes (not findings — context for why the rest is clean):
- 8.1: No restating-the-code comments. Every comment carries meaning the code does not: L25 `// a dropped slot — not a constraint` (WHY NaN is skipped), L29 `// at least one surviving weight, all in [0,1]` (the predicate's contract), L57 (defines the ind[] index-map semantics), the reduce_rows/popdrop_one doc-comments (L34-37/L79-81) state the AT2 drop_pops contract. None paraphrase the statement they sit on.
- 8.2: No stale comments. The big AT2 rank-fit rationale (L94-105) matches the code exactly: r_fit = nl_red-1 (L106), run_rank_sweep still issued (L107), row.f4rank = r_fit / the fitted rank (L109), dof = (nl_red-r)*(nr-r) with r=nl_red-1 (the L110 fallback uses (xr.nl-r_fit)*(xr.nr-r_fit), and xr.nl IS nl_red per L47), gls_weights at the fitted rank r_fit (L118). The "Earlier this used rs.f4rank ... DIVERGED on nl>=3" passage describes a fixed past bug and is correct as history, not a description of current behavior. The row-order note (L142-146: drop higher index first, "01" then "10") matches the descending `for (int drop = nl_full-1; drop >= 0; --drop)` at L148. Header (.hpp:10-21) drop_pops contract matches reduce_rows + popdrop_one. All current.
- 8.3: The genuinely non-obvious decisions ARE documented: the rank-fit-vs-rank-decision distinction (L94-105, the load-bearing AT2 parity rationale), the single-source nl_full==1 no-valid-drop carve-out (L143-146), the qinv[ind,ind] sub-block subsetting (header + L34-37/L57), the NaN dropped-slot sentinel (L115). The 0.0/1.0 feasibility bounds are explained in the .hpp:44-46 contract. Only the L88 `wt` column lacks a local rationale (flagged above). The `(void)m_full;` (L44) is a no-op, not a comment, and is already a Group 3 finding.
- 8.4: No orphan TODO/FIXME/HACK/XXX/WIP markers anywhere in either file (grep confirmed). No dangling owner-less work notes.
-->

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Notes (not findings — context for why this is clean):
- 9.1: No should-be-const/constexpr left mutable. Every read-only local is ALREADY const: ranktest.cpp nr/nl_red/m_full/m_red (L40-43), i (L51/60), r_fit (L106), rs (L107), ri (L108), gw (L118), src/dst (L70-74). The mutable locals are all genuinely mutated: `any` (popdrop_feasible L23, set true at L26), `xr`/`cr` (L90, filled by reduce_rows out-params), `row` (L85, fields assigned), `rows`/`all`/`surv` (vectors built incrementally), and the i/ii/j/a/b/s/w loop counters. No file-scope or static values that should be constexpr — there are no named constants in the unit (the 0.0/1.0 bounds and '0'/'1' pat chars are inline AT2-contract literals, addressed under Group 5). The header `run_rank_sweep` is correctly `[[nodiscard]] inline` returning by value; `precision`/`opts`/`cov`/`x` params are passed by const ref where read-only.
- 9.2: No tangled config / buried tunable knobs. The one tunable, the rank-decision significance, is NOT hardcoded in the logic — it is surfaced through QpAdmOptions and threaded explicitly as `opts.rank_alpha` (ranktest.cpp:107, header L36-41). Precision policy likewise arrives via the `Precision&` parameter, not an in-function literal. There are no thresholds, capacities, or behavioral flags embedded in the orchestration that should be lifted to a config struct.
- 9.3: No positional booleans. No call in either file passes a bare bool literal. popdrop_one (L140 with -1, L152 with `drop`) takes an int source index, not a flag; run_rank_sweep takes `alpha`/`opts`/`precision`; gls_weights takes `r_fit` (an int rank). There is no foo(true,false,...) pattern to convert to named flags/enums.
-->

## Group 10 — Initialization

No Group 10 issues found.

<!--
Notes (not findings — context for why this is clean):
- 10.1: No late/distant or uninitialized-then-assigned declarations. popdrop_feasible's `any` is declared-with-init at L23 and used immediately (L24-29). In reduce_rows the dims nr/nl_red/m_full/m_red are init at L40-43 right before use; `ind` is declared at L58 and fully populated at L59-63 (declared adjacent to its fill); src/dst (L70-74) are init at their inner-loop declaration. In popdrop_one `row`/`xr`/`cr` (L85/L90) are populated on the next lines, and nl_red/r_fit/rs/ri/gw (L92/106/107/108/118) are all declared-with-init at point of use. In run_popdrop nl_full/rows/all/surv (L132/133/138/149) are each declared next to where filled. No "declare at top, assign far below" pattern.
- 10.2: No reliance on implicit/default zero-init that does not hold. Every vector is explicitly value-filled via .assign() with an explicit value BEFORE any read: xr.x_total (L49, 0.0), ind (L58, 0), cr.Qinv/cr.Q (L66-67, 0.0), row.weight (L116-117, quiet_NaN), row.pat (L86, '0') — none depend on std::vector's default-construct zeroing. cr.Q is written only conditionally (L76, `if (!cov.Q.empty())`), but the unwritten case is already the explicit 0.0-fill from the L67 assign, not an uninitialized read. The dropped-slot NaN in row.weight (L116-117) is an intentional explicit sentinel, not a missing init. All read POD/struct fields are explicitly assigned: xr.nl/nr/n_block (L47-49), cr.m/status (L64-65), row.wt/f4rank/dof/chisq/p/status (L88/109-113). No object is read while still default/garbage-valued.
-->
