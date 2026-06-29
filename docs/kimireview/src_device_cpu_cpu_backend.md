I read through this carefully. This is **not slop** — it's a working correctness oracle written by someone who knows the genomics math, but a senior C++ reviewer would have **mixed reactions**. Some parts are genuinely excellent; others would raise eyebrows.

## What's genuinely good

- **The numerical oracles are principled.** The cancellation-free long-double f2 path (`f2_pair_over_range`, lines 247-266), the pairwise summation (lines 90-105), and the long-double accumulators in the jackknife code show real attention to why the reference must be more accurate than the GPU path.
- **Memory ownership is basically sane.** No raw owning pointers; outputs are `std::vector`, the factory returns `std::unique_ptr<ComputeBackend>` (line 2244), and scratch buffers are caller-owned or local.
- **Shared primitives are used consistently.** `het_correction`, `f2_term`, `finalize_f2`, `block_ranges`, `genotype_code`, etc. live in `core::` headers, so the oracle and the GPU feeder share the same formula details.
- **The f2 diagonal convention is handled explicitly** (lines 306-318). That parity-level detail is exactly what an oracle needs.
- **It's pure host C++20.** No CUDA headers, no `__syncthreads`, no raw device mallocs — appropriate for a CPU reference backend, so the CUDA footguns the user asked about simply don't apply here.

## What a senior developer would flag

**Comment/contract drift in the survivor-block rule.** `survivor_blocks` (lines 195-229) drops a block only when it is *partially* missing (`any_missing && any_present`). But `assemble_f4` says the rule is "KEPT iff NO loaded pop pair has Vpair == 0" (line 1012). Those are not equivalent; an all-zero Vpair slab is kept. One of those comments is wrong.

**Silent fail-soft on malformed input.** `survivor_blocks` keeps every block if `vpair` is mis-sized (lines 202-204). A reference oracle should assert or return `Status`, not silently ignore a programming error.

**Hidden mutable state across the public API.** `jackknife_cov` and `jackknife_diag` read `tot_line_` (line 1760), which is set only by `compute_loo_and_total` inside the `assemble_f4*` methods. If you call `jackknife_cov` on an `F4Blocks` you built yourself, you get stale zeros with no diagnostic.

**Inconsistent NaN spelling.** Most of the file uses `std::nan("")` (lines 943, 1329-1332, 1392, 1424, etc.), but `rank_sweep` uses `std::numeric_limits<double>::quiet_NaN()` (lines 1635-1636). Both work, but mixing them in one TU is sloppy.

**Unprotected divisions in `ratio_block_jackknife`.** The f4ratio path divides `at(num,kk,b) / at(den,kk,b)` at lines 1372 and 1385 without checking whether `den` is zero. `xblk_den` is thresholded, but the per-replicate `den` is not.

**Error handling is uneven — and can be unsafe.** `jackknife_cov`/`als_weights` propagate `Status::NonSpdCovariance`/`RankDeficient`, but `rank_test` (lines 1556-1569) ignores `cov.status` and always returns `Status::Ok`. Worse, `chisq_of` (lines 2077-2085) blindly indexes into `qinv` assuming it is `m×m`; if `cov.Qinv` is empty, that's an out-of-bounds read.

**Copy-paste boilerplate in the f4/f3 assemblers.** `assemble_f4`, `assemble_f4_quartets`, and `assemble_f3_triples` (lines 1001-1191) repeat the same survivor-block setup, the same `f2at` lambda (lines 1033, 1101, 1167), the same compaction loop, and the same `compute_loo_and_total` call. A helper would remove ~80 lines and eliminate drift risk.

**Raw-pointer APIs in a C++20 codebase.** `dates_curve` (lines 584-589) and `dstat_block_reduce` (lines 524-527) take several raw pointers with lengths implied by other parameters. `std::span` is already used in this file; these should be spans.

**The `RatioJackArray null{nullptr,0,0,0}` sentinel (line 1547).** It works only because the `tot_mode==1` branch happens not to dereference it. That's a fragile contract.

**The file is doing too much.** 2248 lines in a single `.cpp` implementing f2, decode, DATES, qpfstats, qpAdm, qpGraph, etc. Even as an oracle, splitting by algorithm would make testing and review far easier.

**Comment density.** Almost every statement has a multi-line comment citing a cleanup ID or roadmap section. The explanations are mostly accurate, but the noise-to-signal ratio is high and it makes the code harder to scan than it should be.

## The "slop" test

**Not slop.** Slop would be unexplained magic numbers, copy-pasted code with stale comments, no error checking, or obviously wrong algorithms that happen to pass. This file has *some* stale-comment drift and duplicated structure, but the math is careful, memory is managed, and the comments are mostly accurate.

## What it actually looks like

This looks like **competent research/engineering code written by a domain expert who is more focused on numerical parity than on C++ hygiene.** The author clearly knows the ADMIXTOOLS-2 math and the GPU parity contract, and the code is careful where it counts for correctness. But it is also monolithic, over-commented, and has a few API-coupling and error-propagation rough edges that would make a senior C++ reviewer wince.

## Verdict

**B-, ship as an internal dev oracle but don't showcase it.** For a job-application piece, the 2,200-line monolith, copy-paste assembler triplet, hidden `tot_line_` state, raw-pointer surfaces, and uneven error handling would not impress a senior reviewer. If you want to show it off, split by algorithm family, unify the f4/f3 assemblers, replace raw-pointer APIs with `std::span`, and make the survivor-block rule and `Status` propagation consistent.