# Group 5 rollup — Hardcoded values / magic numbers

Tasks: 5.1 unnamed literals → named constants · 5.2 hardcoded sizes/bounds that
should be params/derived · 5.3 duplicated constants (DRIFT = correctness bug) ·
5.4 hardcoded paths/IDs/device ids · 5.5 ambiguous `32` (warp vs other) → name it.

FP64/§12 context applied: `double` literals (`0.5`/`1.0`/`0.0`), the Ozaki
mantissa-40 constants, single-stream determinism, and AT2 math coefficients are
INTENTIONAL and parity-load-bearing — they were NOT flagged.

## 1. Coverage

- Units in scope: **61** (scope = all).
- Units **with ≥1 Group-5 finding: 26**.
- Units **clean (no Group-5 issue): 35**.

Clean units are dominated by host-pure orchestration / DI / enum-header TUs with
no kernel geometry, plus the already-exemplary `include/steppe/config.hpp`
neighbours (`launch_config`, `vram_budget`, etc. mostly clean because the project
already homes most knobs as named `constexpr`).

## 2. Counts by severity

| Severity | Count |
|----------|-------|
| HIGH | **1** |
| MED  | **15** |
| LOW  | **41** |
| **Total** | **57** |

### By task (approximate — many findings tagged with a primary task; 5.3 dominates)

| Task | Theme | Rough share |
|------|-------|-------------|
| 5.3 | duplicated/drift-prone constants | the bulk of HIGH+MED (the only HIGH; ~half of MED) |
| 5.1 | unnamed literals → named constant | most LOW |
| 5.2 | hardcoded sizes/bounds | a few LOW + tier_select MED coefficients |
| 5.4 | hardcoded paths/IDs/device ids | a small handful of LOW (cache-path strings, env tokens) |
| 5.5 | ambiguous `32` | the gesvdj cap + warp-rounding (MED/LOW, see below) |

## 3. Top findings (HIGH first)

### HIGH
- [5.3][HIGH] src/core/qpadm/model_search.cpp:73 — the small-path envelope
  `nl <= 5 && nr <= 10 && r <= 4` is a bare-literal THIRD copy of the
  authoritatively-named `kQpMaxNl=5 / kQpMaxNr=10 / kQpMaxR=4`
  (qpadm_fit_kernels.cu:41-43), which SIZE the kernel's per-thread local arrays.
  A second bare copy lives in `CudaBackend::model_fits_small_path`
  (cuda_backend.cu:1490). This host gate ROUTES models onto the fixed-size kernel
  path. DRIFT is a correctness bug: widen the gate without widening the kernel
  bounds → device-side local-array overflow / UB; grow `kQpMax*` without updating
  the literal → silent mis-routing / lost throughput. Suggested: define
  `kQpMaxNl/Nr/R` once in a CUDA-free shared header and have all three sites
  (this gate, `model_fits_small_path`, the kernels) reference it.

### MED — the high-value drift / wrong-result risks
- [5.3][MED] cuda_backend.cu:1490 vs 1723-1724,2040 — same `kQpMax*` bit-parity
  envelope as the HIGH above, on the device side: if the host predicate and the
  device kernel constants drift, a model routed to the "small" path overruns the
  kernel's fixed envelope. (Same root cause as the HIGH; the two should be fixed
  together by single-sourcing `kQpMaxNl/Nr/R`.)
- [5.1][MED] cuda_backend.cu:2511 — `assemble_result` (S8 batched rotation)
  selects f4rank with a HARDCODED `if (rankp[...] > 0.05)`, while the single-model
  `rank_sweep` uses the caller's `alpha` (line 1840). The batched path therefore
  IGNORES a non-default rank alpha and silently uses 0.05 — a wrong f4rank for any
  `alpha != 0.05` and a batched-vs-single divergence. Suggested: thread the rank
  alpha into `fit_chunk`/`assemble_result` (or one named `kDefaultRankAlpha`).
- [5.3][MED] cuda_backend.cu:1739 vs 1515/1558 — `svd_path` (the REPORTED routine)
  is computed from a duplicate copy of the `(nl<=32 && nr<=32)` gesvdj predicate
  that actually selects the routine in `large_svd_V`. Drift makes `svd_path`
  mis-report the executed path, and the parity test asserting svd_path==2 for
  NRBIG silently passes on a wrong dispatch. Suggested: derive the report from the
  same `gesvdj_applicable(nl,nr)` helper the dispatch uses.
- [5.5/5.3][MED] cpu_backend.cpp:530,599 (and cuda_backend.cu:1515,1558,1739) —
  the gesvdjBatched per-dim cap `32` is an AMBIGUOUS unnamed magic number (reads
  like warp size, is NOT) AND is duplicated across 5 sites spanning the oracle and
  the GPU backend. If one drifts, the oracle's emitted `svd_path` disagrees with
  the routine the GPU runs — a parity/observability mismatch the tests key off.
  Suggested: one named `kGesvdjBatchedMaxDim = 32` in a shared header read by all
  sites.
- [5.3][MED] f2_blocks_multigpu.cpp:161-163, 270-272 (+219-220) — the four-term
  `use_p2p` combine gate is copy-pasted VERBATIM into two functions (and a third
  near-copy at 219-220 for degrade detection), despite the file's own banner
  declaring itself the "SINGLE AUTHORITATIVE HOME" of the gate. A fifth term or
  reorder in one copy → host/device entries silently disagree on transport
  selection. Suggested: extract one `select_p2p_combine(...)` helper.
- [5.3][MED] ranktest.cpp:110 — the dof fallback `(nl-r)*(nr-r)` duplicates the
  chi-square dof the backend `rank_sweep` computes internally (`rs.dof`); the
  fallback path (`ri >= rs.dof.size()`) silently drifts if the backend convention
  changes. Suggested: one shared `qpadm_dof(nl,nr,r)` helper.
- [5.3][MED] decode_af_kernel.cu:69,70 (+ cpu_backend.cpp:336) — the 2-bit packing
  radix (`/4u`, `& 3`) is re-picked as bare literals, duplicating the io single-home
  `io::kCodesPerByte = 4` that the reader used. The byte index/position MUST match
  the reader's packing; drift mis-decodes. Suggested: a named packing-radix constant
  in `decode_af.hpp` (pinned to `io::kCodesPerByte` by the existing equivalence test),
  divided by at both sites.
- [5.3][MED] decode_af.hpp:58,67-68 — the same packing convention single-homed here
  but duplicated across the core/io seam (`kMissingGenotypeCode=3` mirrors
  `io::kMissingCode`; the MSB-first bit-extract mirrors `io::code_in_byte`). Layering
  forbids the core depending on io, so the fix is keeping the cross-leaf equivalence
  test green so the two copies cannot drift.
- [5.3][MED] qpadm_fit_kernels.cu:1445-1447 — `launch_symmetrize_lower_to_full`
  repeats the tile dim `16` (and `15`=tile-1) across block(16,16) + the `/16` grid
  divisor; changing the block without the divisor under-launches and silently skips
  elements. Suggested: one `kSymTile = 16` deriving both.
- [5.1][MED] tier_select.hpp:59 / [5.3][MED] tier_select.hpp:89-91 — the resident
  feeder footprint coefficient `7u` (=3+4) and the streamed coefficients
  `3u/4u/4u/2u` re-spell the real feeder/ring allocation as policy literals; the
  `2u` is the literal value of the named `kStreamDeviceChunks`. Drift between these
  budget literals and the real malloc/ring undersizes the tier budget → tier
  mis-select / OOM. Suggested: name `kFeederRawBufsPerPop/kFeederOutBufsPerPop`, use
  `kStreamDeviceChunks`, and pull the gather/GEMM term from `per_block_chunk_bytes`.
- [5.3][MED] block_sink.cu:199 — on-disk `h.version = 1u` is a bare literal; the
  format version exists only as a comment in f2_disk_format.hpp (`// 1`), so a bump
  must be edited in two unsynchronized places (writer here, future M7 reader).
  Suggested: add `kF2DiskVersion = 1u` to f2_disk_format.hpp.

### Notable LOW (the recurring ones)
- [5.1][LOW] cuda_backend.cu:1937 — kernel status code `6` → `Status::RankDeficient`
  with no shared named constant (drift with the kernel's emitted code mis-classifies
  the fit). Suggested: a shared `kQpStatusRankDeficient`.
- [5.1][LOW] small_linalg.hpp:220 / qpadm_fit_kernels.cu:156 — the Jacobi
  off-diagonal convergence `1e-30` is the one unnamed threshold beside the named
  `kTol=1e-15`/`kMaxSweeps`; parity-frozen value, name only.
- [5.1/5.3][LOW] f2_disk_format.hpp:28/37, block_sink.cu:189-192 — the 64-byte
  header size and the version `1` live as comments/static_asserts, not as named
  constants writers+readers can share; plus the `0644` octal file mode is unnamed.
- [5.5][LOW] qpadm_fit_kernels.cu:1332-1333 — `(m*m+31)/32*32` warp-rounding uses
  bare `32`/`31`; name `kWarpSize`. And the grid clamp `65535` is duplicated across
  8 launch wrappers (:1295…:1549) — one `kMaxGridDimX`.

## 4. Cross-cutting patterns

1. **5.3 dispatch-envelope drift is the dominant risk class.** The single HIGH and
   ~half the MEDs are the same shape: a *bound that sizes a fixed kernel buffer or
   selects a routine* is re-typed as a bare literal at the *host gate that routes to
   that kernel*. Two distinct triads stand out and should be single-sourced first:
   - `kQpMaxNl/Nr/R = 5/10/4` — model_search.cpp:73, cuda_backend.cu:1490,
     qpadm_fit_kernels.cu:41-43 (the HIGH).
   - the gesvdj `32` cap — cpu_backend.cpp:530,599 + cuda_backend.cu:1515,1558,1739,
     where the *reported* `svd_path` and the *executed* routine come from separate
     copies of the same predicate (parity tests assert on the report).

2. **"Reported value vs executed value" computed from independent copies** recurs:
   svd_path (report) vs large_svd_V (dispatch); the p2p combine gate vs its degrade
   WARN; ranktest's dof fallback vs the backend dof. These are observability/parity
   hazards because a test can assert the report and still mask a wrong execution.

3. **The 2-bit EIGENSTRAT packing geometry** (`/4`, `& 3`, missing-code `3`, MSB-first)
   is re-derived in 4+ places across the core/io/CUDA/CPU seam. The project already
   has the single-home (`io::kCodesPerByte`, `decode_af.hpp` equivalence test); the
   findings are about routing the remaining bare copies through it.

4. **Budget/footprint coefficients re-spelled as policy literals** (tier_select `7u`,
   `3u+4u`, `2u`; vram_budget `2u`/`4u`; cuda_backend 4 GB / 512 MB fit fallbacks).
   These mirror real allocations elsewhere and silently undersize budgets on drift.

5. **The codebase is already strong on naming.** `include/steppe/config.hpp` is the
   promotion home and most adjacent headers (launch_config, vram_budget) are clean or
   near-clean; the FP64 math literals are correctly left alone. The residual work is
   (a) the few drift-class 5.3 dispatch bounds, and (b) routine hygiene LOWs (named
   sentinels, octal mode, env-token tables, `cudaEventWaitDefault`).

---

**Headline: 61 units in scope, 26 with findings (35 clean), 57 total findings — 1 HIGH, 15 MED, 41 LOW.**
