# Group 7 — Duplication — Roll-up Summary

GROUP 7 tasks: 7.1 copy-pasted blocks differing by a constant (extract fn/template); 7.2 repeated/loop-invariant expressions (compute once); 7.3 repeated sizeof/casts (hoist/template); 7.4 collapsible boilerplate (a macro/helper would fold it). IDENTIFY-only review (no edits). §12/FP64 context applied — intentional FP64/parity/determinism choices were NOT flagged.

## 1. Coverage

- **Units in scope:** 61 (scope = all)
- **Units reviewed:** 61 (100%)
- **Clean (no Group 7 issues):** 16
- **With findings:** 45

Clean units (16): `include__steppe__config`, `include__steppe__error`, `include__steppe__fstats`, `src__core__internal__host_device`, `src__core__internal__log`, `src__core__qpadm__gls_solve`, `src__core__qpadm__jackknife`, `src__core__qpadm__model_search_core`, `src__device__backend_factory`, `src__device__cuda__decode_af_kernel`, `src__device__cuda__device_partial`, `src__device__stream_f2_blocks`, `src__io__filter__filter_plan`, `src__io__filter__include_exclude`, `src__io__genotype_tile` (plus several units whose only "duplication" was reviewed and judged intentional/idiomatic, recorded as notes not findings).

## 2. Counts

**Total findings: 134** — `1 HIGH / 23 MED / 110 LOW`

By severity:

| Severity | Count |
|---|---|
| HIGH | 1 |
| MED  | 23 |
| LOW  | 110 |

By task:

| Task | HIGH | MED | LOW | Total |
|---|---|---|---|---|
| 7.1 copy-pasted blocks   | 1 | 21 | 29 | 51 |
| 7.2 repeated expressions | 0 |  2 | 40 | 42 |
| 7.3 repeated sizeof/casts | 0 |  0 | 13 | 13 |
| 7.4 collapsible boilerplate | 0 |  0 | 28 | 28 |
| **Total** | **1** | **23** | **110** | **134** |

Most-loaded units: `src__core__fstats__f2_blocks_multigpu` (8), `src__device__cuda__block_sink` (9), `src__device__cuda__cuda_backend` (9), `src__device__cuda__qpadm_fit_kernels` (9), `src__device__cpu__cpu_backend` (6), `src__core__internal__small_linalg` (5), `src__core__qpadm__model_search` (5).

## 3. Top findings (HIGH first)

### HIGH (1)

- **[7.1][HIGH]** `src/device/cpu/cpu_backend.cpp:787-845 vs 850-904` — `opt_A` and `opt_B` are near-identical 58-line copies of the AT2 GLS-ridge solve (build `xvec`, form `W=qinv·linop`, `coeffs=linopᵀ·W` + `rhs=xvecᵀ·W`, ridge the diagonal, `core::solve`, reshape). They differ ONLY by the Kronecker operator lambda (I⊗B vs A⊗I), `t = nl*r` vs `r*nr`, and the final reshape. Largest duplication in the TU and parity-load-bearing — the ridge/solve/reshape must stay bit-identical across both halves, so a fix/precision tweak to one can silently drift from the other. Suggested: extract a shared `als_ridge_solve(linop_lambda, m, t, xvec, qinv, fudge)` returning the solved vector, leaving only the operator lambda + reshape per caller.

### MED (23) — all are copy-pasted blocks/loops with real drift risk (21×7.1, 2×7.2)

Multi-GPU f2 host/device entries (`src__core__fstats__f2_blocks_multigpu`):
- **[7.1][MED]** `f2_blocks_multigpu.cpp:161-163, 270-272` — 4-term `use_p2p` gate copy-pasted verbatim across host/device entries (one copy labelled "copied VERBATIM"); the file's own banner calls this the "SINGLE AUTHORITATIVE HOME". Suggested: one `select_p2p_combine(Resources&, G)`.
- **[7.1][MED]** `f2_blocks_multigpu.cpp:71-78, 244-251, 330-337` — the 4-line "Shared contract" STEPPE_ASSERT block triplicated, differing only by the fn-name string. Suggested: `validate_multigpu_inputs(...,const char* fn)`.
- **[7.1][MED]** `f2_blocks_multigpu.cpp:81-86, 254-259, 339-344` — `G=device_count(); if(G<1) throw` fail-fast triplicated. Fold into the same validation helper.
- **[7.1][MED]** `f2_blocks_multigpu.cpp:178-187, 282-288` — P2P resident arm near-duplicated (only the host `.to_host()` differs). Have the host entry reuse one shared resident helper.

Multi-GPU fan-out (`src__core__fstats__f2_blocks_multigpu_core`):
- **[7.1][MED]** `f2_blocks_multigpu_core.cpp:117-152, 190-229, 269-306` — the jthread fan-out + worker body triplicated across `compute_multigpu_partials`/`_resident`/`_into`, differing only by the trailing seam call; the comments admit it ("The EXACT same concurrent fan-out as ..."). Suggested: one `fan_out_shards(resources, shards, per-worker-callable)`.

qpAdm model search (`src__core__qpadm__model_search`):
- **[7.1][MED]** `model_search.cpp:221-227, 261-267` — "validate model_index then scatter into slot" copy-pasted across the G==1 fast path and the G>=2 worker (3rd near-variant at 304-307); the determinism re-sort gate is enforced in 2-3 divergent places. Suggested: `scatter_into_slots(results, batch, n)`.
- **[7.2][MED]** `model_search.cpp:175-177, 182-183, 185-193` — `replicate_f2` walks the device list THREE times recomputing the same residency predicate. Classify per-g once.

qpAdm fit (`src__core__qpadm__qpadm_fit`):
- **[7.1][MED]** `qpadm_fit.cpp:64-67 vs 263-266` — the "honest precision_tag" derivation (EmulatedFp64 honorability gate) copy-pasted into `run_impl` and `run_qpwave_impl`; both must report what ACTUALLY ran on the SYRK (§9/§12) so divergence is a latent parity hazard. Suggested: `Precision::Kind honored_tag(const Precision&, ComputeBackend&)`.

CPU oracle (`src__device__cpu__cpu_backend`):
- **[7.1][MED]** `cpu_backend.cpp:152-201 vs 271-300` — the entire per-pair f2 oracle body copy-pasted between `compute_f2` and `compute_f2_blocks` (differs only by SNP range + slab base); a formula drift in one diverges the oracle from the per-block oracle the header promises can't diverge. Suggested: `f2_pair_over_range(...)`.

Streaming sinks (`src__device__cuda__block_sink`) — 5 MED, all 7.1:
- `block_sink.cu:66-76 vs 222-232` — `acquire_slot()` byte-for-byte identical in HostRamSink/DiskSink.
- `block_sink.cu:78-87 vs 234-243` — writer-loop queue-pop preamble identical.
- `block_sink.cu:107-125 vs 268-285` — `spill_block` identical except the byte count.
- `block_sink.cu:128-135 & 138-142 vs 289-293 & 329-333` — writer stop-and-join copy-pasted FOUR times.
- `block_sink.cu:143-150 vs 334-341` — event-destroy teardown loop identical (only warn tag differs).
- Suggested across all five: a shared `StagingRing`/CRTP base owning the ring + acquire/writer/spill/stop/teardown.

Production CUDA backend (`src__device__cuda__cuda_backend`) — 3 MED, all 7.1:
- `cuda_backend.cu:474-490 vs 730-746` — `ceil_bucket` + `struct Bucket` + bucket-assign + sort BYTE-IDENTICAL across `run_f2_blocks_resident`/`stream_f2_blocks_impl` (comment: "IDENTICAL to run_f2_blocks_resident").
- `cuda_backend.cu:460-468 vs 719-727` — block_ranges→offsets/sizes derivation duplicated verbatim.
- `cuda_backend.cu:1812-1835 vs 2513-2537` — AT2 res$rankdrop nested-table build copy-pasted between `rank_sweep` and `assemble_result`. Suggested: shared bucketing/prologue helper + `fill_rankdrop(...)`.

P2P combine (`src__device__cuda__p2p_combine`):
- **[7.1][MED]** `p2p_combine.cu:121-175 vs 248-291` — the entire fixed-order g=0..G-1 placement loop copy-pasted between resident/resident_device (differs ONLY by dst base pointers); a peer-access fix to one silently skips the other. Suggested: `place_partials_into(...)`.

On-device fit kernels (`src__device__cuda__qpadm_fit_kernels`) — 5 MED (4×7.1, 1×7.2):
- `qpadm_fit_kernels.cu:212-258 vs 590-631; 263-309 vs 635-676` — each `*_large` kernel is a line-for-line copy of its templated sibling (~84 duplicated lines), differing only by VRAM `double*` scratch vs local arrays.
- `qpadm_fit_kernels.cu:314-333 vs 679-696` — `dev_chisq_of` duplicated (local vs caller residual scratch).
- `qpadm_fit_kernels.cu:388-412/419-457/462-479 vs 951-980/983-1022/1024-1047` — three model-batched kernels are the single-model body + a model axis/offset; f4/loo/xtau math duplicated verbatim.
- `qpadm_fit_kernels.cu:361-379, 557-573, 788-803, 890-917` — the constrained weight solve appears FOUR times with identical structure.
- **[7.2][MED]** `qpadm_fit_kernels.cu:354-360, 526-531, 754-759, 878-883` — the ALS opt_A→opt_B refinement loop repeated four times. Suggested: shared `__device__` cores parameterized on scratch source.

## 4. Cross-cutting patterns

1. **The "twin device/host-oracle" path is the dominant 7.1 source.** Almost every MED clusters where a device-resident path and its CPU/host-oracle (or small-vs-large, or single-model-vs-batched) sibling were written as near-identical copies that must stay bit-identical for §12 parity: cpu_backend f2/GLS, qpadm_fit run_qpadm/run_qpwave + precision_tag, qpadm_fit_kernels `*_large` and `*_models` kernels, cuda_backend resident-vs-stream bucketing/rankdrop, p2p_combine resident-vs-resident_device, f2_blocks_multigpu host-vs-device entries. The duplication is parity-load-bearing — that is exactly why it is a hazard: the property "cannot diverge on the formula" is asserted in comments but enforced by hand-copy, not by a shared definition. Many findings cite comments that literally say "IDENTICAL to ...", "copied VERBATIM", "Mirrors ... EXACTLY".

2. **RAII / move-only / teardown boilerplate (7.1+7.4) recurs across the CUDA layer.** Move-ctor/move-assign `std::exchange` pairs, warn-on-nonzero `destroy()` bodies, and the `struct G { ~G(){cudaSetDevice}}` / `DeviceGuard` device-switch guard appear across handles.hpp, stream.hpp, device_buffer.cuh, pinned_buffer.cuh, device_f2_blocks.cu, p2p_combine.cu, block_sink.cu. A shared `MoveOnlyHandle`/`UniqueCudaHandle` base + a single `ScopedDeviceSwitch` + a `warn_on_cuda_teardown(e,ctx)` helper would fold most. Several reviewers flagged the device-switch guard as duplicated at 3+ sites.

3. **Hand-inlined `n_block < 0 ? 0` clamp and `(size_t)P*(size_t)P` slab/index widening (7.2/7.3).** The non-negative-clamp idiom recurs ~10× in f2_blocks_multigpu, 6× in block_sink, 4× in f2_blocks_out/device_f2_blocks, plus tier_select/vram_budget; the floor(fraction·free)→size_t budget idiom recurs 3× (tier_select ×2, vram_budget). Common fix direction: tiny `nonneg(int)→size_t`, `clamped_n_block(int)`, `budget_bytes(frac,free)`, and a column-major `cm_index(i,j,ld)` helper. These are LOW (correct as written, §4-safe widening), pure hygiene.

4. **`Precision`/primary-backend construction repeated as a value, not a constant (7.2).** `Precision{EmulatedFp64, kDefaultMantissaBits}` is reconstructed 3-4× in model_search and qpadm_fit, and `*resources.gpus.at(0).backend` 4×. The VALUE cannot drift (single named constant) — flagged only because the construction boilerplate repeats; a `default_fit_precision()` / `resources.primary_backend()` accessor folds it (dovetails with Group 5's magic-`0` device-index note).

5. **Error-throw prefix/string-concat scaffolding (7.4) is the dominant LOW in src/io.** `"io::<fn>: " + ... + path` throw sites recur across geno_reader, ind_reader, snp_filter, block_partition_rule, f2_blocks_out, check.cuh — reviewers consistently judged most NOT worth a finding (distinct messages, a helper buys little), flagging only the genuine near-duplicate catch-pairs / identical guards. Treated as borderline keep-as-is.

## Headline

**Units: 61 (45 with findings, 16 clean). Total findings: 134 (1 HIGH, 23 MED, 110 LOW). By task: 7.1=51, 7.2=42, 7.3=13, 7.4=28.** The single HIGH and nearly all 23 MED are copy-pasted twin paths (device-vs-host-oracle / small-vs-large / single-vs-batched) that must stay bit-identical for §12 parity but are kept equal only by hand-copy.
