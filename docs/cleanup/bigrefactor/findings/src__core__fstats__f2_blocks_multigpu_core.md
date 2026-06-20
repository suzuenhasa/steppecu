# Review findings — src__core__fstats__f2_blocks_multigpu_core

Files: src/core/fstats/f2_blocks_multigpu_core.cpp, src/core/fstats/f2_blocks_multigpu_core.hpp

## Group 4 — Type & numeric

- [4.7][LOW] src/core/fstats/f2_blocks_multigpu_core.cpp:246 — compute_multigpu_partials_into takes the result buffers as raw `double* dst_f2, double* dst_vpair, int* block_sizes_dst` with no host-vs-device space typing; they are genuinely HOST (pinned) D2H destinations here (comment :261 "into one host buffer"), so there is no wrong-space hazard in this TU, but the bare pointers can't statically prevent a device pointer being passed by a future caller. Suggested: optionally a thin host-pointer wrapper at the seam; not load-bearing, file is correct as-is.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

## Group 6 — Naming

No Group 6 issues found.

## Group 7 — Duplication

- [7.1][MED] src/core/fstats/f2_blocks_multigpu_core.cpp:117-152, 190-229, 269-306 — the three fan-out functions (compute_multigpu_partials / _resident / _into) share a copy-pasted jthread launch + worker body that differs ONLY in the trailing backend seam call (compute_f2_blocks vs compute_f2_blocks_resident vs compute_f2_blocks_into) and the result-slot write. The comments themselves admit it ("The EXACT same concurrent fan-out as compute_multigpu_partials", :175, :251). The shared prologue (shard fields s0/s1/M_local/n_block_local, col_off, Qg/Vg/Ng sub-views, block_id_local build loop, try/catch into worker_errors[g], reserve(G) launch loop) is triplicated. Suggested: extract a single fan-out helper templated on a per-worker callable, e.g. `fan_out_shards(resources, shards, [&](g, sh, Qg,Vg,Ng, blk_local.data(), n_block_local){ ... seam call ... })`, that builds the sub-views + local block_id + jthreads + worker_errors rethrow once; the three public fns become thin wrappers supplying only the seam lambda and result handling.
- [7.4][LOW] src/core/fstats/f2_blocks_multigpu_core.cpp:157-161, 232-236, 311-315 — the deterministic "rethrow FIRST worker failure (lowest g)" loop over worker_errors is identical boilerplate in all three functions. Suggested: fold into the same fan-out helper (7.1) or a tiny `rethrow_first_error(worker_errors)` free fn so the join-barrier + first-error contract lives in one place.
- [7.2][LOW] src/core/fstats/f2_blocks_multigpu_core.cpp:139-142, 211-214, 290-293 — the dense zero-based local block_id build loop (`block_id_local[k] = partition.block_id[s0+k] - sh.b0`) is repeated verbatim three times. Suggested: hoist into a small helper (`make_local_block_id(partition, s0, M_local, b0)` returning the std::vector<int>) reused by the extracted worker body.

## Group 8 — Comments

- [8.2][LOW] src/core/fstats/f2_blocks_multigpu_core.cpp:3-12 — the top-of-file header narrates the TU as "the block-aligned shard PLAN and the per-device concurrent FAN-OUT that produces each device's compact partial" (singular fan-out) and ":10 "the body was lifted verbatim from f2_blocks_multigpu.cpp ... same plan, same fan-out, same partials". This predates the two later additions: the file now defines THREE fan-out variants (compute_multigpu_partials, _resident :166, _into :241). The header reads as if only the one original fan-out lives here. Suggested: update the header to enumerate the three fan-out variants (host-staged / device-resident / host-staged-direct) so it reflects the current TU.
- [8.2][LOW] src/core/fstats/f2_blocks_multigpu_core.hpp:11-26 — the "WHY THIS SPLIT EXISTS" block likewise frames the host-pure heart as "These two steps — the block-aligned shard PLAN and the per-device concurrent FAN-OUT" (two steps), but the unit now declares four public fns (plan + three fan-out variants :105/:138/:160). The per-fn doc-comments (:71-166) are current and accurate; only this top narrative lags. Suggested: say "the shard plan plus the three fan-out variants" so the file-level rationale matches the declarations below it.
- [8.3][LOW] src/core/fstats/f2_blocks_multigpu_core.cpp:129-130, 202-203, 281-282 — the col_off guard `s0 < 0 ? 0 : s0` (and the matching `M_local < 0 ? 0 : M_local` at :138, :210, :289) defends against a NEGATIVE s0/M_local, but the adjacent comments only justify the EMPTY shard (b0==b1 ⇒ M_local==0, s0 a valid non-negative offset) — they never explain why a negative branch exists, given block-aligned shards tile [0,n_block) with valid non-negative ranges (shard_plan.hpp). The defensive `< 0` clamp thus reads as unexplained. Suggested: add a half-line noting the clamp is belt-and-suspenders against a malformed (negative) shard range so unsigned col_off can't wrap, or drop it if the shard contract already forbids negatives.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.
