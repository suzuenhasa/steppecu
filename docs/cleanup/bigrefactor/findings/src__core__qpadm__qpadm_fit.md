# Review findings — src__core__qpadm__qpadm_fit

Files: /home/suzunik/steppe/src/core/qpadm/qpadm_fit.cpp, /home/suzunik/steppe/src/core/qpadm/qpadm_fit.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

Notes (why clean, not padding):
- 4.1: All arithmetic is intentional FP64 (pchisq_upper, p/chisq, weight feasibility `v < 0.0 || v > 1.0` at .cpp:101, literals `0.0`/`1.0`). FP64-by-design; no wrong narrowing — N/A.
- 4.2/4.6: This file is the host-pure orchestrator; it issues NO global indexing into the f2 tensor or genotype matrix (delegated to assemble_f4/jackknife_cov/gls_weights/etc). The only indexing is `res.rank_p[(size_t)r]` (.cpp:130), where `r` is a small rank (<= nl-1, nl bounded by P). `res.dof = (nl - r) * (nr - r)` (.cpp:43) is at most ~P*P (~6.25M) in int32 — no overflow.
- 4.3: No cudaMalloc/new/DeviceBuffer; only std::vector reserve/assign/push_back (element-count APIs, correct).
- 4.4/4.5: Only loop is the range-for `for (const double v : w)` (.cpp:98) — no countdown. `rank_p.assign((size_t)r + 1, ...)` (.cpp:128) and the compare `(size_t)r < res.rank_p.size()` (.cpp:129) are guarded by `r >= 0` (.cpp:129); both sides size_t — no signed/unsigned underflow.
- 4.7: No raw pointers; types are ComputeBackend&, F4Blocks, std::span<const int>, std::vector, and the DeviceF2Blocks vs F2BlockTensor overloads carry the host/device space distinction through the type system already.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

Notes (why clean, not padding):
- This unit is the host-pure, CUDA-FREE orchestrator (qpadm_fit.hpp:4-6 "HOST-PURE, CUDA-FREE"). All GPU work (GEMM/SVD/Cholesky) is delegated through the CUDA-free ComputeBackend seam; the includes are only <cmath>/<cstddef>/<span>/<stdexcept>/<vector> and core/device headers (.cpp:5-22). No <cuda_runtime.h>, cuBLAS, or cuSOLVER includes.
- 2.1: No sm_/compute_ arch flags or CMake arch lists in this source unit (build flags live in CMake, not here) — N/A.
- 2.2: No texture<...>/surface<...> references, no cudaBindTexture*/tex1D/2D/3D — N/A (grep clean).
- 2.3: No warp intrinsics at all (no __shfl/__ballot/__any/__all/__activemask) — host code, N/A.
- 2.4: No cudaThreadSynchronize (or any cuda*Synchronize) — host code, N/A.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

Notes (why clean, not padding):
- 3.1: No commented-out code blocks "kept just in case". Every comment is explanatory prose (the precision policy at .cpp:45-67, the M(fit-2) build-order rationale at .cpp:132-147, the carve-out notes at .cpp:222-224, .cpp:236-238). The `(void)precision` text at .cpp:222 is a *citation* of cuda_backend.cu's carve-out, not commented-out code in this file.
- 3.2: No `#if 0` / `#ifdef`-disabled regions. No code after `return`/`break`. The three early `return res` (.cpp:74, .cpp:81) and the final `return res` (.cpp:178) all terminate their paths cleanly; the `case JackknifePolicy::All: default:` (.cpp:114-117) deliberately share one `break`. The `try { ... } catch (const std::runtime_error&)` (.cpp:148-175) catch body is intentionally empty (documented non-breaking absence of a backend rank_sweep override), not dead code.
- 3.3: All includes are used — <cmath> (std::isnan .cpp:99), <cstddef> (std::size_t .cpp:128-130), <span> (throughout), <stdexcept> (std::runtime_error .cpp:172), <vector> (throughout); the core/device headers each supply a symbol that is referenced (assemble_f4, gls_weights, jackknife_cov, se_from_loo, run_rank_sweep/run_popdrop, ComputeBackend/F4Blocks/JackknifeCov/GlsWeights, DeviceF2Blocks, Resources, Precision/kDefaultMantissaBits, Status, F2BlockTensor). Both anonymous-namespace helpers are used (qpwave_from_sweep .cpp:185 → .cpp:272; run_qpwave_impl .cpp:255 → .cpp:281/288). The hpp decls (pchisq_upper, left_with_target, run_impl) are all used. No unused params (each run_qpwave/run_qpadm overload uses `resources` to select the backend). MINOR doc-comment drift (not a code defect, not flagged): the ranktest.hpp include comment (.cpp:16) lists `run_qpwave_impl`, but run_qpwave_impl is actually defined locally in this file's anonymous namespace; ranktest.hpp only provides run_rank_sweep/run_popdrop.
- 3.4: No computed-but-unread values. Every `res.*` field assigned in run_impl/run_qpwave_impl is read via the returned QpAdmResult/QpWaveResult. Locals nl/nr/r (.cpp:39-41), cov (.cpp:70), gw (.cpp:77), se (.cpp:123), rs (.cpp:149), pd (.cpp:162), the `tag` (.cpp:263), and the weights_feasible lambda are all consumed.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

Notes (why clean, not padding):
- 5.1: No unnamed magic literals. The only bare numerics are the FP64 feasibility bounds `v < 0.0 || v > 1.0` (.cpp:101) and `0.0` fill (.cpp:128) — these are the qpAdm weight-feasibility *definition* (admixture proportions each in [0,1]), a domain semantic, not a tunable constant. `nl - 1` (.cpp:41, default = full rank), `(nl - r) * (nr - r)` (.cpp:43, the AT2 dof formula), and `+ 1` (.cpp:128, rank index count = r+1) are derived from data dimensions, not magic.
- 5.2: No hardcoded sizes/bounds. All dimensions are derived at runtime from `X.nl`/`X.nr` (.cpp:39-40); ranks/dof flow from those + `opts.rank`. Nothing that should be a param is hardcoded — the orchestrator delegates all sizing to the backend.
- 5.3: No duplicated literal / drift risk. The mantissa-bits value appears in four Precision constructions (.cpp:55, .cpp:223, .cpp:239, .cpp:258) but ALL reference the single named constant `steppe::kDefaultMantissaBits` (config.hpp:44, `= 40`) — same source of truth as the f2 path, so the four sites cannot drift. This is the intended one-policy unification, not a copy-pasted literal.
- 5.4: `resources.gpus.at(0).backend` is repeated at .cpp:217, .cpp:234, .cpp:280, .cpp:287 — a hardcoded device index 0. With context this is the deliberate single-model entry-point convention (select the first backend; the SAME pattern in model_search.cpp:291), not a magic-number bug — the multi-GPU fan-out lives above this seam. LOW hygiene at most (a named `kPrimaryGpu`/helper could DRY the four wrappers), but no correctness/scale risk; not flagged.
- 5.5: No ambiguous `32` (no warp-size or block-dim literal) — this is host-pure, CUDA-FREE orchestrator code with no kernel launches or shared-mem sizing.

## Group 6 — Naming

No Group 6 issues found.

Notes (why clean, not padding):
- 6.1: The short identifiers are all tightly-scoped, AT2-domain, or standard statistical names, not opaque junk. `X` (.cpp:34,225,...) is the conventional statistical design-matrix name and the F4Blocks parameter; `nl`/`nr` (.cpp:39-40) mirror the struct field names `X.nl`/`X.nr` (number-of-left / number-of-right rows) one-to-one; `r` (.cpp:41) is the qpAdm rank, used as such throughout. The result-of-stage locals `cov`/`gw`/`se`/`rs`/`pd` (.cpp:70,77,123,149,162) are each one-line-lived and immediately consumed, and each is documented at its assignment ("S4 — covariance", "S6 — GLS weights", "S7 — SE", "M(fit-2) — the RANK SWEEP", "res$popdrop"). `be` (ComputeBackend&) is the established project-wide backend handle. No `tmp`/`data2`/`arr`/`flag` names. The lambda `v` (.cpp:98) and `row` (.cpp:163) are bona-fide loop locals.
- 6.2: No misleading names. `block_sizes` (.cpp:34) is genuinely the per-block size weights (the AT2 block_lengths, OQ-3); `left_idx`/`left_with_target` (.cpp:26-32,218) are genuinely index vectors; `compute_se` (.cpp:105) is the boolean gate it claims to be; `weights_feasible` (.cpp:96) returns the feasibility predicate. `tag`/`precision_tag` (.cpp:64,263) honestly name the reported precision. Nothing named count-but-is-index or list-but-is-map.
- 6.3: One convention per kind, consistently applied: snake_case for functions/locals/members (`run_impl`, `left_with_target`, `compute_se`, `weights_feasible`, `block_sizes`, `model_index`, `est_rank`, `rank_p`, `precision_tag`), PascalCase for types (QpAdmResult, F4Blocks, JackknifeCov, GlsWeights, RankSweep, PopDropRow), kPascalCase for the constant (kDefaultMantissaBits). No nElements-vs-num_elements-vs-n drift within the file.
- 6.4: The abbreviations used (`nl`,`nr`,`cov`,`gw`,`rs`,`pd`,`be`,`gls`,`loo`,`dof`,`se`,`p`,`chisq`) are all either AT2/qpAdm domain terms carried for parity (gls=generalized least squares, loo=leave-one-out, dof=degrees of freedom, chisq=chi-squared, se=standard error, rankdrop/popdrop/f4rank are the AT2 res$ field names) or mirror the backend struct field names — standard within this codebase, not nonstandard coinages.

## Group 7 — Duplication

- [7.1][MED] qpadm_fit.cpp:64-67 and qpadm_fit.cpp:263-266 — the "honest precision_tag" derivation `(prec.kind == Precision::Kind::EmulatedFp64 && be.capabilities().emulated_fp64_honorable) ? Precision::Kind::EmulatedFp64 : Precision::Kind::Fp64` is copy-pasted verbatim into both `run_impl` (assigned to `res.precision_tag`) and `run_qpwave_impl` (assigned to `tag`). The two must agree forever (both report what ACTUALLY ran on the SYRK per §9/§12), so the duplication is a latent drift hazard. Suggested: extract a free helper e.g. `Precision::Kind honored_tag(const Precision&, ComputeBackend&)` and call it from both sites.
- [7.1][LOW] qpadm_fit.cpp:215-230 and qpadm_fit.cpp:232-246 — the two public `run_qpadm` overloads (DeviceF2Blocks vs F2BlockTensor) are near-identical bodies differing only by the f2 source type: same `be` select, same `left_with_target`, same `prec` ctor, same `assemble_f4(be, f2, left, right, prec)`, same `run_impl(be, std::move(X), f2.block_sizes, model, opts)`. The qpWave path already solved exactly this with the `run_qpwave_impl` template (.cpp:254-273); the run_qpadm pair was left un-templated. Suggested: hoist a `template <class F2Src> QpAdmResult run_qpadm_impl(ComputeBackend&, const F2Src&, const QpAdmModel&, const QpAdmOptions&)` (mirroring run_qpwave_impl) and make the two overloads thin forwarders.
- [7.2][LOW] qpadm_fit.cpp:55, :223, :239, :258 — `const Precision prec{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits}` is constructed identically at four sites (run_impl, both run_qpadm overloads, run_qpwave_impl). All four reference the single named constant so they cannot drift in value (already noted under 5.3), but the construction boilerplate itself repeats. Suggested: a single `inline Precision default_fit_precision()` (or a constexpr Precision) folds the four ctors; collapsing the run_qpadm overloads per 7.1 also removes two of them.
- [7.2][LOW] qpadm_fit.cpp:217, :234, :280, :287 — `ComputeBackend& be = *resources.gpus.at(0).backend;` is the identical loop-invariant expression at all four public entry points (already flagged as LOW under 5.4 as a magic device-index). Suggested: a `kPrimaryGpu`-keyed accessor (e.g. `resources.primary_backend()`) folds the four copies and the magic `0` together.

## Group 8 — Comments

- [8.2][MED] qpadm_fit.hpp:32 — the `run_impl` doc-comment is stale: it describes a parameter `` `se_policy` `` that gates the S7 jackknife SE, but no such parameter (or field) exists — `run_impl`'s signature (hpp:38-40, cpp:34-35) takes only `be, X, block_sizes, model, opts`, and the gate at cpp:106 switches on `opts.jackknife` (type `JackknifePolicy`, declared qpadm.hpp:88). The parameter was renamed/moved into `QpAdmOptions::jackknife` and the doc-comment still names the old `se_policy`, so a reader looking for an `se_policy` arg finds none. Suggested: rename the comment's `se_policy` reference to `opts.jackknife` / `JackknifePolicy` so it tracks the real option field.

## Group 9 — Constants & configuration

No Group 9 issues found.

Notes (why clean, not padding):
- 9.1: No should-be-const value left mutable. The dimension/rank locals `nl`/`nr`/`r` (.cpp:39-41) and the per-stage results `prec`/`cov`/`gw`/`se`/`rs`/`pd`/`tag`/`left_idx` (.cpp:55,70,77,123,149,162,263,218) and the assembled `X` (.cpp:225,241,260) are all already `const` where not consumed by a move. `compute_se` (.cpp:105) is intentionally declared-then-assigned inside the `switch` (.cpp:106-118) so it cannot be a single const initializer — a legitimate pattern, not a missed const. `res` (.cpp:36) is the mutable result accumulator (must be non-const). The lambda local `bool any` (.cpp:97) is a running flag (mutated at .cpp:100). The four `Precision prec{...}` ctors (.cpp:55,223,239,258) are already `const`; they could fold to one `constexpr`/`inline` factory, but that is the duplication direction already captured under 7.2, not a mutability defect.
- 9.2: Config is surfaced, not tangled. Every tunable knob the orchestrator consults lives on `QpAdmOptions` (`opts.rank`, `opts.fudge`, `opts.jackknife`, `opts.se_require_p`, `opts.p_se_threshold`, `opts.rank_alpha` — declared/defaulted in include/steppe/qpadm.hpp:80-99) or on the central `steppe::kDefaultMantissaBits` (config.hpp:44, `= 40`). The precision policy is built from that one named constant at all four sites, never an inline `40`. The only buried literal is the device index `0` in `resources.gpus.at(0).backend` (.cpp:217,234,280,287) — already captured as LOW under 5.4/7.2 (the deliberate single-model "first backend" convention; the multi-GPU fan-out lives above this seam). No tunable threshold is hardcoded inside the fit logic.
- 9.3: No positional-boolean anti-pattern. None of the delegated calls (assemble_f4, jackknife_cov, gls_weights, se_from_loo, run_rank_sweep, run_popdrop, run_qpwave_impl, qpwave_from_sweep) pass a bare `true`/`false` positional flag — arguments are typed values (ComputeBackend&, F4Blocks, spans, JackknifeCov, Precision, ints, QpAdmOptions) and the precision/jackknife behavior is carried by the `Precision::Kind` and `JackknifePolicy` enums, not boolean params. The only boolean literals in the file are the local init `bool any = false` (.cpp:97) and the ternary cast `row.feasible ? char{1} : char{0}` (.cpp:170) — neither is a call-site positional flag.

## Group 10 — Initialization

No Group 10 issues found.

Notes (why clean, not padding):
- 10.1 (late/distant or uninitialized-then-assigned): The one uninitialized-then-assigned local is `bool compute_se;` (.cpp:105), assigned in the immediately-following `switch (opts.jackknife)` (.cpp:106-118). The switch is EXHAUSTIVE over the `JackknifePolicy` enum (qpadm.hpp:88) — `None`, `FeasibleOnly`, and `All`/`default:` (shared arm, .cpp:114-117) all assign `compute_se` — so every control path writes it before the only read at .cpp:122. No read-before-write path exists; the declare-then-switch-assign is the legitimate pattern (a const-initializer can't span a switch). Not flagged. Every other local is declared AT first use with an initializer: `nl`/`nr`/`r` (.cpp:39-41), `res` (.cpp:36), `cov` (.cpp:70), `gw` (.cpp:77), `prec` (.cpp:55), the `weights_feasible` lambda + its `bool any = false` (.cpp:96-97), `const SeResult se` (.cpp:123, copy-init from se_from_loo), `const RankSweep rs` (.cpp:149), `const std::vector<PopDropRow> pd` (.cpp:162); and in the namespace/entry-point code `QpWaveResult qw` (.cpp:186, .cpp:268), `left_idx` (.cpp:218, .cpp:235), `X` (.cpp:225, .cpp:241, .cpp:260), `cov`/`tag`/`rs` (.cpp:261,263,270), `be` (.cpp:217,234,280,287). None declared far from use.
- 10.2 (zero-init assumptions that don't hold): The two value-result aggregates default-constructed here — `QpAdmResult res;` (.cpp:36) and `QpWaveResult qw;` (.cpp:186, .cpp:268) — do NOT rely on implicit/garbage zero-init: every scalar member carries an explicit default member initializer in its struct (`QpAdmResult` qpadm.hpp:131-161: `p=0.0`, `chisq=0.0`, `dof=0`, `f4rank=0`, `status=Status::Ok`, `precision_tag=Fp64`, `model_index=-1`; `QpWaveResult` qpadm.hpp:217-220: `f4rank=0`, `est_rank=0`, `status=Status::Ok`, `precision_tag=Fp64`), and the `std::vector`/`std::string` members default-construct empty by their own ctors. So the early-return paths (.cpp:74, .cpp:81) and the rank_sweep `catch` (.cpp:172-175) return a `res` whose unset fields are well-defined defaults, not garbage. `res.rank_p.assign((size_t)r + 1, 0.0)` (.cpp:128) and `res.precision_tag = ...` (.cpp:64) explicitly fill rather than assume prior state. No POD/static/global here relies on implicit zero-init. N/A.

