# Review findings — include__steppe__qpadm

Files: /home/suzunik/steppe/include/steppe/qpadm.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Rationale (read-only confirmation, not findings):
- 4.1: Every floating field is `double` (fudge 1e-4 L60; rank_alpha/p thresholds
  L81,92; weight/se/z/p/chisq L125-132,137,147; QpWaveResult L216). Intentional
  per the FP64-by-design parity law — correct, not narrowing. N/A.
- 4.2: Population indices are `int` (QpAdmModel.target/left/right/model_index
  L109,113,116,118; std::span<const int> left/right L226-228,234-235). These are
  per-population indices in 0..P-1 (P<=~2500), NOT global offsets into the
  P*P*n_block f2 tensor (that offset is computed in the .cpp). `int` is correct.
- 4.3: No allocations — header of CUDA-free value types only. N/A.
- 4.4/4.5: No loops. N/A.
- 4.6: No index arithmetic in this header. N/A.
- 4.7: No raw pointers; device::DeviceF2Blocks / device::Resources are passed by
  ref via CUDA-free fwd-decls (L37-38,169,177,225,232). Host/device seam is clean.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Rationale (read-only confirmation, not findings):
- Unit is a PUBLIC, deliberately CUDA-FREE, standard-C++20 header (stated L11-15);
  no CUDA toolkit symbols by design (architecture.md §4 layering rule).
- 2.1 Dropped archs: header carries no CMake arch lists / sm_XX build flags. N/A.
- 2.2 Texture/surface references: no texture<...>, cudaBindTexture*, or surface
  refs anywhere (device handles are CUDA-free fwd-decls L36-39). N/A.
- 2.3 Non-_sync warp intrinsics: no warp intrinsics (__shfl/__ballot/__any/__all);
  host-only value types + free-function decls. N/A.
- 2.4 cudaThreadSynchronize: no CUDA runtime API calls at all. N/A.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Rationale (read-only confirmation, not findings):
- 3.1 Commented-out blocks: none. All comments are explanatory documentation —
  the header banner (L1-22), field/enum doc-comments (e.g. L41-51, L56-99,
  L121-162), and entry-point docs (L164-236). No code is commented out.
- 3.2 Unreachable code: no `#if 0`; the only preprocessor directives are the
  include guard (L23-24 `#ifndef`/`#define`, L240 `#endif`). Declarations only —
  no function bodies, so no code after return/break. N/A.
- 3.3 Unused symbols: this is a PUBLIC API header — all enums/structs/free-function
  decls are the intended exported surface (JackknifePolicy L47, QpAdmOptions L56,
  QpAdmModel L106, QpAdmResult L124, run_qpadm L169/177, run_qpadm_search L191/200,
  QpWaveResult L211, run_qpwave L225/232). Every include is used: <span> (L191/200/
  226-228/234-235), <string> (std::string L149), <vector> (pervasive), config.hpp
  (Precision::Kind L159/220), error.hpp (Status L156/219), fstats.hpp (F2BlockTensor
  L177/200/232). No unused params (declarations carry no bodies). N/A.
- 3.4 Computed but unread: no statements in a header — only member default
  initializers (the intentional API defaults, e.g. L60/64/68/88/92/99/L118-161).
  Nothing is computed-then-discarded. N/A.
-->

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

<!--
Rationale (read-only confirmation, not findings):
- 5.1 Unnamed literals: every numeric literal is a NAMED default-member-initializer
  with a doc-comment, not a bare expression literal — fudge=1e-4 (L60), als_iterations
  =20 (L64), rank=-1 (L68, the AT2 "default nl-1" sentinel), rank_alpha=0.05 (L81),
  p_se_threshold=0.05 (L92), the AT2 parity constants are deliberately surfaced as
  named fields (header states this L41,L54-55,L80) and recorded in golden metadata.
  The JackknifePolicy enumerators 0/1/2 (L48-50) are the FROZEN user-facing
  --jackknife mapping, explicitly named. model_index=-1 (L118,L161), p/chisq/dof=0,
  est_rank/f4rank=0 are zero-default sentinels on value-type fields. Not magic.
- 5.2 Hardcoded sizes/bounds: none — CUDA-free value-type header, no array dims,
  capacities, or launch bounds. N/A.
- 5.3 Duplicated constants: 0.05 appears at L81 (rank_alpha) and L92
  (p_se_threshold) but they are SEMANTICALLY DISTINCT, independently configurable
  options (rank-decision alpha vs feasible-only p-gate); not one concept duplicated,
  no drift-correctness coupling. No shared block-dim/array-size pairing exists here. N/A.
- 5.4 Hardcoded paths/IDs/device ids: no paths, no literal device ids. "gpus[0]" /
  "resources.gpus[0].backend" appear only in DOC-COMMENTS (L168,L171,L199) describing
  the .cpp routing, not as code in this header. N/A.
- 5.5 Ambiguous 32 (warp vs other): no 32 / warp-size literal anywhere; CUDA-free
  header has no launch geometry. N/A.
-->

## Group 6 — Naming

No Group 6 issues found.

<!--
Rationale (read-only confirmation, not findings):
- 6.1 Cryptic names: no opaque identifiers. Single-letter `r` appears ONLY in
  doc-comments as the rank symbol (L67,132,137,139); `w`/`E` appear only in math
  doc-comments mirroring AT2 notation, not as code identifiers. The short
  parameter names `f2`/`f2_host` (L169,177,191,200,225,232) name the f2-statistics
  tensor — an established domain term of art documented in the banner (L4-10), not
  opaque. Fields are descriptive (als_iterations, rank_alpha, p_se_threshold,
  rankdrop_*, popdrop_*). No tmp/data2/arr/flag.
- 6.2 Misleading names: model_index IS an index/stable identity (L118,161);
  est_rank/f4rank/rank are ints holding rank values; popdrop_feasible is a
  vector<char> documented as 0/1 (L152, intentionally vector<bool>-free). Names
  match semantics — no count-that-is-an-index / list-that-is-a-map.
- 6.3 Inconsistent conventions: internally consistent steppe convention —
  snake_case fields/free-functions (als_iterations, rank_alpha, p_se_threshold,
  se_require_p, allow_negative_weights, model_index, rank_chisq, rankdrop_dofdiff,
  run_qpadm/run_qpwave/run_qpadm_search), PascalCase types + enumerators
  (JackknifePolicy/None/FeasibleOnly/All, QpAdmModel/QpAdmResult/QpWaveResult/
  QpAdmOptions). No nElements-vs-num_elements-vs-n mixing.
- 6.4 Nonstandard abbreviations: fudge, als, dof, chisq, se, z, nl/nr, f4rank,
  qpadm/qpwave, f2 are all established admixtools2 / statistics domain terms
  (documented, parity-load-bearing per §12); opts is conventional. None are
  project-invented nonstandard abbreviations. N/A.
-->

## Group 7 — Duplication

- [7.1][LOW] qpadm.hpp:137-147,211-221 — QpAdmResult and QpWaveResult repeat an identical rank-sweep field cluster: rank_chisq, rank_dof, rank_p, rankdrop_f4rank, rankdrop_dof, rankdrop_dofdiff, rankdrop_chisq, rankdrop_p, rankdrop_chisqdiff, rankdrop_p_nested, f4rank, plus est_rank/status/precision_tag. The comment at L146 ("mirrors RankSweep.rd_*") implies a RankSweep aggregate already exists; the two public result types could embed it instead of restating all ~14 fields, removing the drift risk where one struct gains/renames a rankdrop column and the other does not. Suggested: factor the shared block into a `struct RankSweep` member embedded by both result types (keeps the public field names via the member, or via a documented accessor).

<!--
Rationale (read-only confirmation, not findings):
- 7.1 (the one real candidate is above). The run_qpadm/run_qpadm_search/run_qpwave
  DeviceF2Blocks-vs-F2BlockTensor overload PAIRS (L169-180, L191-204, L225-236) are
  NOT copy-paste differing by a constant — they are intentional overloads on
  distinct first-parameter types (device-resident primary vs host-oracle parity
  door, banner L7-10/L196-204/L231-232); declarations only, no body logic to share.
- 7.2 Repeated/loop-invariant expressions: none — a CUDA-free value-type + free-
  function-declaration header with no executable statements, no loops, no
  expressions to hoist. N/A.
- 7.3 Repeated sizeof/casts: no sizeof, no casts, no reinterpret anywhere. N/A.
- 7.4 Collapsible boilerplate: the shared rank-sweep cluster (7.1) is the only fold;
  the enum/option/model declarations carry no repeated macro-foldable boilerplate.
-->

## Group 8 — Comments

- [8.2][MED] qpadm.hpp:17-22 — The SCOPE banner states "this header freezes only the single-model shapes M(fit-1) needs ... The batched/search forms (run_qpadm_search, the rank sweep) are later milestones," but the header now ALSO declares the full later-milestone surface: run_qpadm_search (L191,200), run_qpwave + QpWaveResult (L211-221,225,232), the M(fit-2) rank-test/rankdrop/popdrop fields (L141-152), and the S8 JackknifePolicy/jackknife options (L47-99). The banner understates the actual declared API — stale relative to the file's current contents. Suggested: update the SCOPE paragraph to state the header now also freezes the M(fit-2) qpWave/rank-sweep and M(fit-6) search/S8 shapes (or scope the M(fit-1)-only wording to run_qpadm).

<!--
Rationale (read-only confirmation, not findings):
- 8.1 Restating code: none. Declaration-only header; every comment adds intent/AT2
  rationale (e.g. L41-51 JackknifePolicy, L57-99 options), never narrates trivial
  mechanics. No "i++; // increment i" style.
- 8.3 Missing rationale: well-covered. Non-obvious constants carry their AT2 source
  and intent — fudge=1e-4 (L57-60, "Matches AT2 qpadm.R exactly"), als_iterations=20
  (L62-64, OQ-1), rank=-1 sentinel (L66-68), rank_alpha=0.05 (L79-81), the
  vector<char> popdrop_feasible choice is justified (L152, "to keep ... vector<bool>-
  free"), se.empty() sentinel explained (L127-129), per-model status vs exceptions
  cites architecture.md §10 (L121-123,156). No bare magic workaround lacking a why.
- 8.4 Orphan TODO/FIXME/HACK: grep over the file returns NONE. Forward-milestone
  notes (M(fit-2)/M(fit-6)) are scoped, owned references to design docs, not orphan
  markers. N/A.
-->

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Rationale (read-only confirmation, not findings):
- 9.1 Should-be-const/constexpr left mutable: declaration-only header — no file-scope
  constants and no mutable globals. The AT2 parity constants (fudge=1e-4 L60,
  als_iterations=20 L64, rank=-1 L68, rank_alpha=0.05 L81, p_se_threshold=0.05 L92)
  are deliberately PER-CALL tunable `QpAdmOptions` struct members with default
  initializers, NOT compile-time constants — a caller overrides them per fit, so
  they MUST remain mutable fields (cannot be const/constexpr). JackknifePolicy is
  already a scoped `enum class : int` (L47). N/A.
- 9.2 Tangled config: the OPPOSITE of tangled. The AT2 parity knobs are explicitly
  surfaced as NAMED, documented fields in one `QpAdmOptions` config struct (L56-100),
  with the banner (L41 "named; OQ-1/OQ-4 — not magic numbers", L54-55) calling out
  that these are deliberately de-magic-numbered and recorded in golden metadata. No
  tunable buried in logic (there is no logic — declarations only). Exemplary. N/A.
- 9.3 Positional booleans: no function CALLS in a declaration-only header. The free
  functions take a named-field `const QpAdmOptions&` (L171,179,193,202,228,235), never
  positional bool args. The bool members (constrained L72, allow_negative_weights L76,
  se_require_p L99) are set by named field assignment, and the SE-policy knob is a
  proper `enum class JackknifePolicy` (L47-51, L88) rather than a bare bool. N/A.
-->

## Group 10 — Initialization

- [10.2][LOW] qpadm.hpp:109 — In QpAdmModel, the `int target` member has NO in-class default initializer, unlike `model_index = -1` directly below it (L119) and every scalar member in the three sibling structs (QpAdmOptions L60-99, QpAdmResult L131-161, QpWaveResult L217-220 are all explicitly initialized). A value-initialization that omits target (`QpAdmModel{}`, or any aggregate-init missing the first member) leaves it indeterminate, and target is then used as an f2_blocks P-axis index in the .cpp. No current call site is wrong (the caller always sets target), so this is a defensive/consistency gap, not an active bug. Suggested: give it a sentinel default (`int target = -1;`) to match model_index and the rest of the value-type convention.

<!--
Rationale (read-only confirmation, not findings):
- This unit is a PUBLIC, declaration-only, CUDA-free value-type header: no function
  bodies, no local variables, no executable statements, no loops.
- 10.1 Late/distant declarations / uninitialized-then-assigned: N/A — there are no
  locals or statements to declare late or to assign after an uninitialized decl.
- 10.2 Zero-init assumptions: every scalar/enum struct member that needs a default
  carries an explicit in-class initializer rather than relying on implicit zero-init:
  QpAdmOptions (fudge=1e-4 L60, als_iterations=20 L64, rank=-1 L68, constrained=false
  L72, allow_negative_weights=true L76, rank_alpha=0.05 L81, jackknife=All L88,
  p_se_threshold=0.05 L92, se_require_p=false L99); QpAdmResult (p=0.0 L131, chisq=0.0
  L132, dof=0 L133, est_rank=0 L139, f4rank=0 L144, status=Ok L156, precision_tag=Fp64
  L159, model_index=-1 L161); QpWaveResult (f4rank=0 L217, est_rank=0 L218, status=Ok
  L219, precision_tag=Fp64 L220). std::vector / std::string members are
  default-constructed (empty), which is well-defined and intended (se.empty() is the
  documented "not computed" sentinel, L127-129). The ONLY member lacking a defensive
  default is QpAdmModel::target (the 10.2 LOW above).
-->

