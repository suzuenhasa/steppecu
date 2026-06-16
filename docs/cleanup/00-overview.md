# 00 — Whole-codebase holistic review (steppe M0–M4, branch `m4-perblock-f2`)

Reviewer: holistic synthesizer over the 27 finalized per-unit reviews in this directory,
re-grounded against `docs/architecture.md` (§2/§4/§7/§8/§9/§11/§12/§13), `docs/ROADMAP.md`
(§4/§5/§6), and `docs/TODO.md` (the cleanup backlog + the re-verified CAPABILITY-TIER section,
workflow `wxz1fiiln`). This document carries only what the *individual* reviews cannot see:
cross-file layering/DRY/contract problems, the one unified capability-tier design, a deduped
prioritized master backlog (before-M4.5 vs later), and per-area + overall scores. The per-unit
files remain the authority on each unit's internals; this folds them together.

The codebase is genuinely disciplined — the per-unit floor is 6.5 and the median is ~8.5, and the
single strongest property (a shared `__host__ __device__` per-element primitive that makes the CPU
oracle and the GPU path unable to diverge on a formula) holds across the whole tree. The gap to
9.5+ is **concentrated, not pervasive**, and it clusters into a small number of cross-cutting
themes that the same M4.5/M5 milestones would otherwise calcify across G devices and N tiles.

---

## (1) CROSS-FILE issues no single-file review can see

These are real because they span ≥2 files; each per-unit review saw an edge of it but not the
whole shape.

### X-1 [HIGH] The cuBLAS workspace / `cublasSetStream` determinism void is a THREE-file bug, not one
`cuda_backend.cu` binds the §12-mandated emulated-FP64 reproducibility workspace **once** in the
ctor (`cublasSetWorkspace`, `kCublasWorkspaceBytes`), but **both** GEMM routines —
`run_f2_gemms` (`f2_block_kernel.cu:242`, M0) **and** `run_f2_gemms_group`
(`f2_blocks_kernel.cu:190`, M4) — call `cublasSetStream` on every invocation, and the cuBLAS 13.x
docs state verbatim that `cublasSetStream` *"unconditionally resets the cuBLAS library workspace
back to the default workspace pool."* So the workspace the ctor exists to install is discarded
before every GEMM batch, on both the M0 and M4 paths. The defect's root cause, fix, and prevention
are split across `handles.hpp` (the wrapper that *should* own the (stream, workspace) invariant but
exposes only a bare `get()`), `cuda_backend.cu` (binds workspace, passes `nullptr` stream), and the
two kernel TUs (re-bind the stream per call). No single-file review can mandate the fix because the
fix is architectural: `CublasHandle` must own a `set_stream()` that re-applies the workspace, and
the kernel TUs must stop calling raw `cublasSetStream`. Flagged independently by
`device-cuda-handles` (1.1/3.1), `device-cuda-cuda_backend` (1.1/3.1/5.4), and
`device-cuda-f2_block_kernel` (C-3/I-1) — all confirmed against the live cuBLAS docs.

### X-2 [HIGH] The `F2Result` M0 diagonal genuinely DIVERGES between the CPU oracle and the GPU, and every equivalence test masks it
`CpuBackend::compute_f2` walks the strict upper triangle (`j=i+1`) and leaves the diagonal `0`;
`CudaBackend::compute_f2` computes every `(i,j)` and fills the diagonal with `−2·mean_het`. So the
two reference implementations of the *same* M0 statistic produce different bytes on the diagonal —
and `test_f2_equivalence.cu` does `if (i==j) continue;` (and even uses an *inline* oracle, not the
production `CpuBackend::compute_f2`, so the production CPU method has zero diagonal coverage), so
the divergence is invisible to CI. The false "the M0 compute_f2 convention" claim is then copied
verbatim into `fstats.hpp`, `cpu_backend.cpp`, and `test_f2_blocks_equivalence.cu`. Spotted by
`include-fstats` (F1), `device-backend` (1.2), and `device-cpu-cpu_backend` (1.1/10.1). The M4 path
(`compute_f2_blocks`) is *consistent* across backends — the divergence is M0-only and never
consumed downstream (f3/f4 read off-diagonal) — so it is latent, but it is exactly the oracle≡GPU
trust seam §13 exists to protect, and it should be pinned before M4.5 replicates the M4 path.

### X-3 [HIGH] The block-range scan is re-derived in three places; the inverse of the §8 single-home rule has no home
`assign_blocks`/`block_partition_rule` is the sanctioned single-home for SNP→block, but the
**inverse** (turn the non-decreasing `block_id[]` into per-block `[begin,end)` ranges) is
hand-duplicated in `cuda_backend.cu:136-147`, `cpu_backend.cpp:212-226`, and a third near-copy in
`test_f2_blocks_equivalence.cu`. The two backend copies have *already* drifted in shape (CPU
computes `begin[]`+`end[]`; GPU computes `block_offsets[]`+derives end). None validates
`0 ≤ block_id[s] < n_block` or non-decreasing-ness, so a malformed partition is a silent OOB write
on the host vectors and an OOB device read of `block_offsets[id]` (`f2_blocks_kernel.cu:96`). M4.5
will copy it a fourth time per device. The fix is one host-pure
`block_ranges(block_id, M, n_block) → vector<BlockRange>` in `block_partition_rule.{hpp,cpp}` that
validates the contract once and is unit-testable GPU-free. Flagged by `core-domain-...` (1.4/9.1),
`device-cpu-cpu_backend` (6.2/2.1), `device-cuda-cuda_backend` (6.2/1.4), and the `f2_from_blocks`
orchestration review (F-1).

### X-4 [HIGH→MED] Phantom single-source homes the spec names but the tree does not contain
A whole family of "single home" files the architecture/ROADMAP/config comments cite **do not
exist** (verified by `find`): `core/internal/launch_config.hpp`, `core/internal/host_device.hpp`,
`internal/log.hpp`, `internal/nvtx.hpp`, `internal/expected.hpp`, `device/cuda/allocator.cu`,
`device/cuda/pinned_buffer.cuh`, `core/internal/span_view.hpp`. The consequences cascade across
files:
- `cdiv`/`grid_for` live de-facto in `f2_estimator.hpp` while `config.hpp:50`, `f2_estimator.hpp`,
  and `decode_af_kernel.cu` all point at the phantom `launch_config.hpp`; `decode_af_kernel.cu`
  re-rolls its own `cdiv_l`/`cdiv_i` "to avoid a dependency" that is one CUDA-free include away.
- `STEPPE_HD` is `#define`d identically in **both** `f2_estimator.hpp` and `decode_af.hpp` with no
  shared `host_device.hpp` and no `#undef`.
- The teardown-warning macro is triplicated (`device_buffer.cuh`, `stream.hpp`, `handles.hpp`),
  with drifted `NDEBUG` branches (one evaluates its arg, two don't) and no `#undef` (header-macro
  leak), all `fprintf(stderr,...)` placeholders for the phantom `STEPPE_LOG_WARN` — a §10
  "never printf in library code" violation acknowledged in three comments.
- `STEPPE_DEBUG_ONLY`/`STEPPE_ASSERT` (the §7 debug-assert facility) is referenced by multiple
  reviews as the fix for an unenforced precondition, but **does not exist anywhere** — so every
  "add a debug assert" fix silently depends on landing that facility first.
This is the single highest-leverage cross-cutting cleanup: create the real homes once and the DRY
gaps in ~6 files close together. Flagged by `core-internal-f2_estimator` (4.2/4.3/4.1),
`core-internal-decode_af` (D-1), `device-cuda-check` (R-4), `device-cuda-stream` (6.1), and
`device-cuda-device_buffer` (5.1).

### X-5 [HIGH] `0.80` VRAM fraction is a live magic literal; the budget gate is also subtly unsound and duplicated
`cuda_backend.cu:230` hardcodes `0.80 * free_b` — the exact ROADMAP §4 smell `config.hpp` exists to
eliminate (`kMaxVramUtilizationFraction` is missing). It also (a) does not subtract the cuBLAS
workspace reserve before applying the fraction, and (b) re-derives a *second* budget gate that
should be single-sourced with the §11.2 `build()`-time check that does not exist yet. Confirmed by
`include-config` (5.1) and `device-cuda-cuda_backend` (5.1/5.2/5.4/9.4).

### X-6 [HIGH/build-correctness] `EmulatedFp64` silently runs the REJECTED dynamic-mantissa trap when `STEPPE_HAVE_EMU_TUNING=OFF` (the in-file default)
`engage_f2_precision` always sets `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` but compiles out the
FIXED-slice pin under `#if STEPPE_HAVE_EMU_TUNING`, whose in-file default is `0`. With tuning off,
cuBLAS falls back to its documented DYNAMIC default (~60-bit) — the precise trap ROADMAP §0 / §12
forbid — while the result still reports the `EmulatedFp64{40}` tag. The remote box passes
`-DSTEPPE_HAVE_EMU_TUNING=1` so the *measured* path is safe; a stock-toolkit/CI build silently
degrades. The §9 `build()` contract ("reject a precision the backend cannot honor") already
mandates the fix but is unimplemented. The decision is split: the trap is engaged in
`f2_block_kernel.cu` (C-1/C-2), the orchestrator `cuda_backend.cu` (3.3) is where it should be
refused observably, and `config.hpp` (3.2) is where the public contract should hint at it.

### X-7 [MED] Two grid-dimension launch failures with the same root, opposite axis-orientation choices
`f2_blocks_kernel.cu` sets `grid.z = n_in_group` (the M4 grouped path) and the f2 feeder sets
`grid.y = cdiv(M, 16)` — both can exceed the hard 65,535 y/z limit (the feeder at M > ~1.05M SNPs,
which `MatView::M`'s `long` type explicitly permits). The decode launcher got the safe orientation
(M on `grid.x`, limit 2³¹−1); the f2 feeder did not. AADR passes only because both extents are
small today. The fix is the same `grid_for`-with-clamp helper X-4's `launch_config.hpp` should
own. Flagged by `device-cuda-cuda_backend` (1.2/1.7/2.6), `device-cuda-f2_block_kernel`, and
`device-cuda-decode_af_kernel` (F-COR-1).

### X-8 [MED] The packing geometry (`s/4`, `s&3`, `kCodesPerByte`, `kMissingCode`) is single-homed by halves and re-derived everywhere
`eigenstrat_format.hpp` owns the within-byte extractor (`code_in_byte`) and stride
(`packed_bytes`), but the **byte index** `s/4` is open-coded as a bare `4u`/`3` literal in 3 TUs
(`cpu_backend.cpp`, `decode_af_kernel.cu`, `mind_prepass.cpp`). Separately, `core::genotype_code`
re-implements `io::code_in_byte`'s bit order and `core::kMissingGenotypeCode` duplicates
`io::kMissingCode` (a deliberate §4 layering consequence — `core` must not depend on the `io`
leaf), but `eigenstrat_format.hpp`'s comment falsely claims it is "the SINGLE bit-order site," and
nothing asserts the two twins agree (no `static_assert`, no exhaustive cross-twin test — the
agreement is inferred only from the end-to-end real-AADR decode test). Flagged by
`io-eigenstrat_format` (5.4/9.1), `core-internal-decode_af` (M-1/L-1/A-2), and
`device-cpu-cpu_backend` (7.5).

### X-9 [MED] Backend factories declared in no header; five ad-hoc forward decls in two namespaces
`make_cpu_backend()` (defined in `steppe::core`) and `make_cuda_backend()` (defined in
`steppe::device`) are declared in **no** header; every consumer (3 tests + the M4 spike + future
M4.5 `Resources` wiring) hand-writes a prototype, in two different namespaces. `CpuBackend` itself
lives in `steppe::core` but compiles into `steppe_device` — a namespace/layer mismatch. A signature
change silently breaks ODR. Flagged by `device-backend` (9.1) and `device-cpu-cpu_backend` (9.1);
on TODO §A.

### X-10 [MED] The `GenotypeTile` ↔ `DecodeTileView` shape is mirrored across the io→device seam and has already diverged
The packing+partition layout is declared twice (`io::GenotypeTile`, `device::DecodeTileView`) and
hand-copied field-by-field at every call site; the two have drifted (`ploidy` + `int n_pop` live
only on the consumer view, `n_pop()` returns `size_t`). The §4 layering *requires* two structs (the
device layer cannot include the io header), so the fix is a bridging `as_view(tile, ploidy)` adapter
in the wiring layer (where a missed field becomes a one-TU compile error), not a merged type.
Flagged by `io-genotype_tile` (F10) and `device-backend` (9.3).

### X-11 [MED] `ploidy <= 0` produces different silent behavior in two primitives that share the Q/V/N contract
`decode_af.hpp::finalize_af` with `ploidy==0, an>0` yields `n=0`, `q=ac/0 = NaN/Inf`, and reports
`v=1.0` (a non-finite Q that is *not* masked out); `snp_filter.cpp:30` clamps `ploidy<=0` to 1.
Two different silent behaviors for the same illegal input, in the two units that are supposed to be
the single source of the Q/V/N contract. The fix (fold `ploidy>0` into the validity test in
`finalize_af`, reconcile with `snp_filter`) is cross-file. Flagged by `core-internal-decode_af`
(E-1), `device-backend` (2.3), `device-cuda-decode_af_kernel` (F-EDGE-3), `io-filter-snp_filter`
(1.5).

### X-12 [MED, cosmetic-but-pervasive] LLP64 / `long`-width assumptions stated as portability facts but only true on the LP64 target
`views.hpp` documents `M` as `long` "so a large SNP block does not overflow 32-bit" — false on
Windows/MSVC where `long` is 32-bit; the codebase is Linux-only today, so this is latent, but the
file's *stated reason* is wrong. The same `long`==`int64_t`==`size_t` LP64 assumption is relied on
(correctly, but implicitly) in `decode_af.hpp` (`ac`/`an`), `cpu_backend.cpp` (the `long double`
oracle width), and the `DeviceBuffer<long>`/kernel `const long*` seam. Either pin `std::int64_t`
explicitly or add a single `static_assert(sizeof(long)>=8)` + a documented "LP64 assumed" note.
Flagged by `core-internal-views` (7.3/6.2), `core-internal-decode_af` (E-3), `device-backend`
(1.5), `device-cpu-cpu_backend` (3.4).

### §2/§4/§8 conformance verdict
- **§4 layering (CUDA-PRIVATE, io-leaf, dependency direction): conformant and compiler-enforced.**
  Verified across every unit — CUDA never leaks into `core`/`api`/`io`; `backend.hpp` is CUDA-free;
  `io` links only `steppe::api`+`steppe::warnings`. This is the codebase's structural triumph.
- **§2 single-source: mostly conformant, with the specific cracks above** (X-3 block-ranges, X-4
  phantom homes, X-8 packing geometry) — each is the *inverse* or *neighbor* of a rule that IS
  single-homed, which is why they slipped.
- **§2 fail-fast: the weakest principle.** A recurring pattern across `f2_from_blocks`,
  `cuda_backend`, `cpu_backend`, `geno_reader`, `snp_reader`, `include_exclude`, `mind_prepass`,
  `snp_filter`, and `decode_af` is *documented-but-unenforced preconditions* that degrade to silent
  wrong answers (OOB reads, NaN-with-validity, silent SNP-axis misalignment, silently-swallowed
  prune.in). The §7 `STEPPE_DEBUG_ONLY` facility that would carry the cheap fix does not exist
  (X-4).

---

## (2) CAPABILITY-TIER COHERENCE — the ONE unified design

**Current state: incoherent by omission.** No file implements a capability probe, a tagged-degrade
log line, or a recorded "which path did this run take." The pieces that *exist* are correct and
parity-neutral (`enable_peer_access` in `DeviceConfig`, the host-staged fixed-order combine as the
baseline), but the runtime-probe + explicitly-tagged-fallback pattern the TODO `wxz1fiiln` section
mandates is absent everywhere. Several per-unit reviews independently (and correctly) concluded
"this unit is correctly tier-NEUTRAL" — and that is right for the host-pure leaves (`views`,
`decode_af`, `f2_estimator`, `block_partition_rule`, `filter_decision`, `include_exclude`) — but the
*cross-cutting machinery* those neutral units feed into has no home.

**The single unified design (to land with M4.5):**

1. **One capability probe, owned at `Resources` construction (not in any leaf).** At `build()` /
   `Resources` assembly, probe per device: compute capability + free/**total** VRAM
   (`cudaGetDeviceProperties` + `cudaMemGetInfo` — `cuda_backend.cu` already calls the latter and
   *discards* `total_b`, exactly the datum needed); `cudaDeviceCanAccessPeer` for the P2P combine;
   the effective `STEPPE_HAVE_EMU_TUNING` / emulated-FP64-honorable state; GDS availability
   (`cuFileBufRegister`/`DMA_BUF_SUPPORTED`, M5/M7); RLIMIT_MEMLOCK (M5). The probe result is a
   small `BackendCapabilities` value.

2. **Capability-tagged results recorded in `DeviceConfig`/`Resources` + the run record — never on
   the parity-critical numeric payload.** Each run records which path it took and why it degraded.
   The tag does NOT live on `F2BlockTensor` (keep it pure numeric storage) — it lives out-of-band in
   `Resources`/a result envelope; `fstats.hpp` (F20) and `f2_from_blocks` (F-13) should document
   that decision rather than carry the tag.

3. **Two knob types, cleanly separated (the recurring confusion across reviews):**
   - **Override intent → `DeviceConfig`** (append-only as levers land): `enable_peer_access`
     (exists), `prefer_p2p_combine` (M4.5), `enable_gds_ingest` (M5/M7). `config.hpp` should carry
     a banner documenting this intended growth.
   - **Discovered capability + which-path tag → `Resources`/result metadata** (runtime state, not
     intent).

4. **A non-throwing, tagged degrade path distinct from `STEPPE_CUDA_CHECK`.** This is the load-bearing
   `device-cuda-check` finding (CAP-1/CAP-2): `cudaDeviceCanAccessPeer` returning "no" and
   `cudaDeviceEnablePeerAccess` returning `cudaErrorPeerAccessAlreadyEnabled` are *expected* on the
   budget box and must `STEPPE_LOG_WARN`-and-degrade, NOT throw. Routing a probe through the fatal
   checker turns a graceful degrade into a hard failure. This needs the phantom `internal/log.hpp`
   (X-4) and a `STEPPE_CUDA_WARN`-style non-throwing variant next to the throwing checks.

5. **Parity holds on both tiers by construction** (the design is correct here): every lever is
   data-movement/observability only — the host-staged fixed-order combine and the `cudaMemcpyPeer`
   device-resident combine sum the same fixed `g=0..G-1` order; GDS vs pinned-pread only changes how
   bytes reach VRAM; the AT2-golden + native-FP64 oracle (the `cpu_backend.cpp` long-double path,
   which should carry a one-line "cross-tier parity reference" note per its review's 11.1) gate both
   boxes so a capable-path lever can never silently change a reported number.

**Files that must change for capability-tier coherence (all M4.5-scoped, none today):**
`config.hpp` (override-knob banner + `deterministic` field), `backend.hpp` (a
`BackendCapabilities probe()` + per-device-instance contract doc), `cuda_backend.cu` (the probe
itself + capture `total_b` + the `EmulatedFp64`-not-honorable tagged degrade of X-6),
`f2_from_blocks.cpp` (thread `Resources`, host the `canAccessPeer`-gated combine *policy*), the new
`internal/log.hpp` (the tagged-warn sink), and `device-cuda-check`'s non-throwing probe variant.
The leaf parsers (`snp_reader`, `geno_reader`, `mind_prepass`, `include_exclude`, `snp_filter`)
must NOT log directly (they are leaves) — they should **surface counts/provenance as return data**
so `app` emits the tagged line; this is the M5/M6 ingest-tier obligation (`geno_reader` K-1/K-2 is
the concrete `read_into(span)` + GDS-vs-pinned seam).

---

## (3) PRIORITIZED MASTER BACKLOG (deduped across all 27 units)

Severity: HIGH = correctness/determinism/silent-wrong-data or a confirmed launch crash;
MED = robustness/fail-fast/DRY/contract; LOW = polish/doc. "Files" lists the primary edit sites.

### BEFORE M4.5 — pay first; M4.5 (multi-GPU) / M5 (streaming) calcify these across G devices / N tiles

| # | Sev | Item & concrete fix | Files |
|---|-----|---------------------|-------|
| B1 | HIGH | **cuBLAS workspace reset (X-1).** Give `CublasHandle` a `set_stream()` that re-applies the owned workspace after `cublasSetStream`; bind stream+workspace once in the ctor; drop the per-call `cublasSetStream` from both GEMM routines (it passes `nullptr` today). Restores §12 emulated-FP64 reproducibility on M0 and M4. | `handles.hpp`, `cuda_backend.cu`, `f2_block_kernel.cu`, `f2_blocks_kernel.cu` |
| B2 | HIGH | **EmulatedFp64 dynamic-mantissa trap (X-6).** Route the honorability decision through ONE predicate driving both math-mode and compute-type; when `STEPPE_HAVE_EMU_TUNING=OFF` + EmulatedFp64 requested, fall back to native `Fp64` with a logged capability tag (or throw `INVALID_CONFIG`) — never silently run dynamic. | `f2_block_kernel.cu`, `cuda_backend.cu`, `config.hpp` |
| B3 | HIGH | **Block-range scan single-home + validation (X-3).** Add `block_ranges(block_id, M, n_block) → vector<BlockRange>` to `block_partition_rule.{hpp,cpp}`, validating `0≤id<n_block` + non-decreasing once; both backends call it; unit-test it. Closes the OOB write/read and the DRY triplication. | `block_partition_rule.{hpp,cpp}`, `cuda_backend.cu`, `cpu_backend.cpp`, `f2_from_blocks.cpp` |
| B4 | HIGH | **`F2Result` M0 diagonal divergence (X-2).** Fix `CpuBackend::compute_f2` to loop `j=i` (match `compute_f2_blocks` + the GPU kernel); pin the convention on `F2Result::f2`; fix the 3 parroting comments; wire the *production* `CpuBackend::compute_f2` into `test_f2_equivalence.cu` and diff the **full** matrix (diagonal included). | `cpu_backend.cpp`, `backend.hpp`, `fstats.hpp`, `test_f2_equivalence.cu`, `test_f2_blocks_equivalence.cu` |
| B5 | HIGH | **`kMaxVramUtilizationFraction` (X-5).** Add the named constant to `config.hpp`; replace the `0.80` literal; subtract `kCublasWorkspaceBytes` before applying the fraction; clamp the budget computation in `size_t` before the `int` narrowing of `max_blocks`. | `config.hpp`, `cuda_backend.cu` |
| B6 | HIGH→MED | **Grid-dimension launch failures (X-7).** Add `kMaxGridZ=65535`; clamp `nb ≤ kMaxGridZ` in the chunk sizing; re-orient the f2 feeder so the SNP count rides `grid.x` (matching the decode launcher). Fold the y/z clamp into `grid_for` once `launch_config.hpp` exists (B7). | `cuda_backend.cu`, `f2_block_kernel.cu`, `f2_blocks_kernel.cu` |
| B7 | MED | **Create the phantom single-source homes (X-4).** `core/internal/launch_config.hpp` (move `cdiv`/`grid_for` + a warp-justified decode block constant; delete `decode_af_kernel.cu`'s `cdiv_l`/`cdiv_i`); `core/internal/host_device.hpp` (one `STEPPE_HD` + `STEPPE_DEBUG_ONLY`/`STEPPE_ASSERT`); `internal/log.hpp` (the teardown-warning sink, replacing the 3 `fprintf` macros). Fixes the dangling citations in `config.hpp:50`, `f2_estimator.hpp`, `decode_af_kernel.cu`. | new `launch_config.hpp`/`host_device.hpp`/`log.hpp`, `f2_estimator.hpp`, `decode_af.hpp`, `decode_af_kernel.cu`, `check.cuh`, `stream.hpp`, `handles.hpp`, `device_buffer.cuh` |
| B8 | MED | **Backend factory header + namespace (X-9).** Declare `make_cpu_backend`/`make_cuda_backend` in one CUDA-free header; move `CpuBackend`/`make_cpu_backend` to `steppe::device`; delete the 5 forward decls. | `backend.hpp` (or new `backend_factory.hpp`), `cpu_backend.cpp`, 3 test TUs |
| B9 | MED | **`deterministic` field on `DeviceConfig`** (matches §9 spec; gates the §12 stream_count/workspace/combine rules M4.5 relies on). | `config.hpp` |
| B10 | MED | **`ploidy<=0` reconciliation (X-11).** Fold `ploidy>0` into `finalize_af`'s validity test (degrade to masked-out `{0,0,0}`); reconcile with `snp_filter.cpp`'s clamp-to-1; document and unit-test the haploid (`ploidy==1`) + illegal (`0`) cases. | `decode_af.hpp`, `snp_filter.cpp`, new `test_decode.cpp` |
| B11 | MED | **`f2_from_blocks` precondition guard.** One `validate_qvn(Q,V,N)` + `validate_partition(partition, Q.M)` (debug `assert`, CUDA-free): `Q.P==V.P==N.P`, `Q.M==V.M==N.M`, `block_id.size()==Q.M`, `n_block≤Q.M`, ids non-decreasing. Add the CUDA-free `MockBackend` unit test (the seam was designed to be GPU-free-testable and isn't). | `f2_from_blocks.cpp`, new `tests/unit/test_f2_from_blocks.cpp` |
| B12 | MED | **`compute_f2` `P<=0\|\|M<=0` guard** (sibling-consistency; `compute_f2_blocks`/`decode_af`/CPU all guard, GPU `compute_f2` doesn't → throws instead of empty result). | `cuda_backend.cu` |
| B13 | MED | **`block_size_morgans>0` guard in `assign_blocks`** (the `ConfigBuilder::build()` it would normally defer to does not exist; this is the only enforceable site today). `if (!(block_size_morgans>0.0)) return out;` rejects 0/negative/NaN; closes the float→int UB. | `block_partition_rule.cpp` |
| B14 | MED | **`snp_reader` C1+N1: silent SNP-axis misalignment.** Replace extraction-failure fall-through with token-count-based column decision; parse `genpos` with `std::from_chars` (locale-free, rejects NaN/Inf); fail-fast with line number on a malformed record. The block-rule key value must be deterministic. | `snp_reader.cpp` |
| B15 | MED | **`snp_reader` C2: uncaught `std::stoi` throw.** Replace `std::stoi` in `chrom_code` with `std::from_chars`; route overflow to the negative-sentinel path; makes the documented `runtime_error`-only contract true. | `snp_reader.cpp` |
| B16 | MED | **X→23/Y→24/MT→90 chrom codes (X-8).** Promote to `kChromCodeX/Y/MT` in `eigenstrat_format.hpp`; reference from `snp_reader`; cross-link the `config.hpp` autosome comment. The autosome filter's correctness depends on these exact codes. | `eigenstrat_format.hpp`, `snp_reader.cpp`, `config.hpp` |
| B17 | MED | **`geno_reader` C1+E8: oversized-file + out-of-range row → silent wrong genotypes.** Distinguish "partial file" from "file inconsistent with header" (throw); add `if (row >= records_present_) throw` in the gather. | `geno_reader.cpp` |
| B18 | MED | **`geno_reader`/`eigenstrat_format` integer-width + parse-overflow (E1/E2/E9, 1.1).** Guard the `streamoff` offset multiply + the `n_ind*bytes_per_rec` allocation against `size_t` overflow; harden the header `v*10+digit` decimal parse against silent wrap. | `geno_reader.cpp`, `eigenstrat_format.cpp` |
| B19 | MED | **`include_exclude` 1.1: fail-silent prune.in read.** After the getline loop, `if (in.bad() \|\| (in.fail() && !in.eof())) throw` (a directory opens but read-fails on Linux → silently empty keep-set). Add the (currently zero) prune.in file-branch tests. | `include_exclude.cpp`, `test_filters.cpp` |
| B20 | MED | **`snp_filter` bounds/contract (1.1/1.2/6.1).** Validate `pop_individuals.size()>=P` (the unguarded numerator loop can yield negative `missing_frac` that passes geno), `ploidy>=1`, non-null `q`/`n`; add the `build_snp_keep_mask(const vector<PerSnpSummary>&, ...)` overload that makes the decision cascade GPU-free-testable. | `snp_filter.{hpp,cpp}`, `test_filters.cpp` |
| B21 | MED | **`mind_prepass` 1.5 header/code contradiction** on `n_snp==0`+active (header says "fully missing/drop", code keeps all); fix the doc to match the code; kill the bare `4u`×2 (use `io::kCodesPerByte`); add the `n_snp==0`+active test. | `mind_prepass.{hpp,cpp}`, `test_filters.cpp` |
| B22 | MED | **`M ≤ INT_MAX` guard in `compute_f2`** before `static_cast<int>(M)` feeds `cublasGemmEx`'s `int k` (the M0 whole-matrix path; `MatView::M` is `long` deliberately). | `cuda_backend.cu`, `f2_block_kernel.cu` |
| B23 | MED | **`device_buffer` checked-multiply** `n*sizeof(T)` overflow guard (typed error, `<limits>`); document `bytes()` exact under the invariant for the §11.2 budget. | `device_buffer.cuh` |

### LATER — real but M5/M6/Phase-2-timed, or cosmetic

| # | Sev | Item | Files |
|---|-----|------|-------|
| L1 | MED | **`error.hpp` ABI policy (§16/§17): C `steppe_status_t` vs C++ `enum class Status`.** Decide + record (Option A: make it the C enum with all 8 §10 values; Option B: keep as internal C++ mirror, document deferral). Add `ChisqUndefined`; fix the "three (two listed)" contradiction; pin underlying type + `[[nodiscard]]` on the type. | `error.hpp` |
| L2 | MED | **`DeviceBuffer::view()` restore + span/mdspan kernel signatures** (§7 "kernels accept only views"); lands with `span_view.hpp`. | `device_buffer.cuh`, kernels |
| L3 | MED | **`Stream` non-blocking ctor + priority** — the §11.1 copy/compute overlap *cannot be built* on the current blocking-only `Stream`; fix the "non-blocking-capable" docstring; split `Event` ordering-vs-timing. M5. | `stream.hpp` |
| L4 | MED | **Pool allocator (`allocator.cu`) + pinned staging (`pinned_buffer.cuh`)** — `cuda_backend.cu`'s per-chunk `cudaMalloc`/`cudaFree` is device-wide-synchronizing; `geno_reader`'s pageable per-individual `seekg` gather is the named "clearest host liability." M5 ingest spine. | new `allocator.cu`/`pinned_buffer.cuh`, `cuda_backend.cu`, `geno_reader.cpp` |
| L5 | MED | **NVTX scopes** on the 3 backend methods (the §11.3 "Nsight first" gate needs named ranges; budget-5090 fallback is "nsys+NVTX only"). | new `nvtx.hpp`, `cuda_backend.cu` |
| L6 | MED | **`FilterPlan` is dead code** — wire the M2 factory + test, or mark as a forward stub; pin the per-tile-vs-dataset axis contract; narrow `in_tile` to a threshold sub-struct (drop the double-filter footgun). M5/M6. | `filter_plan.hpp`, `snp_filter.cpp` |
| L7 | MED | **`F2BlockTensor` schema_version + label/provenance decision** for the M7 on-disk cache; `flat_index`/`block_offset`/`well_formed` accessors so S3/M4.5/M7 stop re-deriving slab offsets. | `fstats.hpp` |
| L8 | MED | **`mind_prepass`/`geno_reader` streaming reshape** — accept tiles, accumulate across tiles, `read_into(span)`, fd-based reads for `O_DIRECT`/`fadvise`. M5. | `mind_prepass.{hpp,cpp}`, `geno_reader.{hpp,cpp}` |
| L9 | MED | **`f2_estimator` cancellation-regime test** — the load-bearing `f2_term`-vs-expanded equivalence is tested only on 3 benign SNPs at 1e-12, never in the `p_i≈p_j` cancellation regime its own comment claims to cover. | `test_f2.cpp` |
| L10 | LOW | **`MathModeScope` RAII** for the shared cuBLAS handle (M4.5 Fp64-parity-recompute coexisting with EmulatedFp64). | `handles.hpp`, `f2_block_kernel.cu` |
| L11 | LOW | **`constexpr` on the `decode_af`/`f2_estimator` numeric primitives** (enables `static_assert` tests; consistent with `cdiv`/`grid_for`/`code_in_byte`). | `decode_af.hpp`, `f2_estimator.hpp` |
| L12 | LOW | **`kPairwiseBaseCase=128` → `config.hpp`** (duplicated bare `128` in the oracle test). | `config.hpp`, `cpu_backend.cpp`, `test_f2_equivalence.cu` |
| L13 | LOW | **`kDefaultPloidy=2`** named (duplicated in `backend.hpp` + `snp_filter.hpp`). | `config.hpp` |
| L14 | LOW | **LP64/`long`-width (X-12):** `views.hpp` `M`→`int64_t` + corrected comment; `static_assert(sizeof(long)>=8)` / "LP64 assumed" notes in the affected files. | `views.hpp`, `decode_af.hpp`, `cpu_backend.cpp`, `backend.hpp` |
| L15 | LOW | **`long double` width `static_assert`** guarding the CPU oracle's "better-than-native" property. | `cpu_backend.cpp` |
| L16 | LOW | **Host unit tests for the under-tested leaves:** `test_decode.cpp`, `test_eigenstrat_format.cpp`, `test_geno_reader.cpp`, `test_snp_reader.cpp`, `test_ind_reader.cpp` (esp. the `nlargest` tie-break), `test_views.cpp`, `test_cuda_check.cpp`, `test_device_buffer.cu` — most are GPU-free, data-free, and have a sibling template (`test_f2.cpp`/`test_filters.cpp`). | `tests/unit/` |
| L17 | LOW | **`cublasGetStatusName`/`cublasGetStatusString`** replace the hand-rolled `CublasError::status_name` switch (official APIs exist in cuBLAS 11.4.4–13.3); add the human-readable description for parity with `CudaError`. | `check.cuh` |
| L18 | LOW | **`CudaError`/`CublasError` OOM-safe message** (fixed `char[N]` buffer via `snprintf`, not `std::string` — the message allocation can throw `bad_alloc` on the exact OOM path it reports, and the implicit copy ctor repeats it). | `check.cuh` |
| L19 | LOW | **Comment-accuracy sweep:** `error.hpp` "three (two)"; `kCdivBlock` "kernels never re-pick" (decode uses 32×8); the phantom `launch_config.hpp` citations; `eigenstrat_format`'s false "SINGLE bit-order site"; `f2_from_blocks`'s stale "M4 is a later milestone"; the `vector<bool>` "wrong for CUDA" framing (it's host-applied); the `geno_reader` "kept open" aspirational comment; prune spike `:line` citations when `experiments/` is deleted. | many |

---

## (4) PER-AREA SCORES + OVERALL

Per-area = the demanding-senior assessment folding the unit scores with the cross-cutting weight
(a unit can score 8.5 in isolation yet sit in an area dragged down by an X-finding that spans it).

| Area | Units (per-unit scores) | Area score | Gap to 9.5 |
|---|---|---|---|
| **Public API (`include/`)** | config 9.0, error 7.5, fstats 8.0 | **8.0** | `error.hpp` ABI policy undecided (L1); fstats diagonal-doc + missing schema/accessors; config missing `kMaxVramUtilizationFraction`/`deterministic`. |
| **Core internal/domain/fstats** | views 8.5, f2_estimator 8.5, decode_af 9.0, block_partition 8.5, f2_from_blocks 8.0 | **8.5** | The codebase's strongest area; held by X-3 (block-ranges home), X-4 (phantom `launch_config`/`host_device`), the unenforced-precondition pattern (B11/B13), and the cancellation-regime test gap (L9). |
| **Device infra (RAII/check)** | check 8.5, device_buffer 8.5, stream 8.0, handles 8.5 | **8.0** | X-1 (handles must own (stream,workspace)); X-4 (phantom `log.hpp`, triplicated teardown macro); `Stream` can't express the M5 overlap it exists for (L3); zero compile/test coverage of `Stream`. |
| **Device backends/kernels** | cpu_backend 8.5, decode_af_kernel 8.5, f2_block_kernel 7.5, f2_blocks_kernel (folded), cuda_backend 6.5 | **7.5** | The lowest area: X-1, X-2, X-5, X-6, X-7 all land here; `cuda_backend` carries the determinism void + two launch crashes + the 170-line monolith + no `device_id`/`Resources`. |
| **IO (readers + format + tile)** | eigenstrat 8.5, geno_reader 7.0, snp_reader 6.5, ind_reader 8.0, genotype_tile 8.5 | **7.5** | Parser robustness: silent SNP-axis misalignment (B14), uncaught `stoi` (B15), oversized-file/OOB (B17), integer-overflow edges (B18), unhomed chrom codes (X-8/B16); zero host unit tests on parity-bearing parsers (L16). |
| **IO filters** | filter_decision 9.3, snp_filter 8.5, mind_prepass 8.0, include_exclude 8.5, filter_plan 7.0 | **8.3** | `filter_decision` is near-reference; dragged by `filter_plan` (dead code, L6), the fail-silent prune.in (B19), and the snp_filter/mind_prepass fail-fast + bare-`4u` gaps (B20/B21). |

### OVERALL CODEBASE SCORE: **8.0 / 10**

This is a high-8 *structure* (compiler-enforced layering, textbook RAII, the shared-primitive
divergence-prevention thesis, exemplary magic-number/comment discipline, a real CPU-oracle trust
seam) carrying a low-7 *device hot-path* and *parser-robustness* tier. The weighted overall lands at
8.0 — the device backends and IO parsers (the two 7.5 areas, and the f-stat hot path that is the
project's reason to exist) pull it down from the 8.5 the core/internal layer would suggest.

### The concrete gap to 9.5+

A 9.5+ codebase closes the four things a demanding senior would not sign off on, all of which
M4.5/M5 would otherwise replicate across G devices and N tiles:

1. **The §12 determinism contract must actually hold (B1, B2).** Today it is *claimed* (the ctor
   binds a workspace, the tag says `EmulatedFp64{40}`) but *defeated* (the workspace is reset every
   GEMM; the default build runs the rejected dynamic mantissa). The whole regression-gate strategy
   rests on bit-stability that does not currently exist. This is the single most important fix.

2. **Fail-fast must replace fail-silent at the seams (B3, B11, B13, B14, B17, B19, B20).** The
   recurring pattern — documented-but-unenforced preconditions degrading to silent wrong data (OOB
   reads, NaN-with-validity, SNP-axis misalignment, swallowed prune.in) — is the antithesis of §2,
   and it is *unfixable cheaply* until the phantom `STEPPE_DEBUG_ONLY`/`log.hpp` facility exists
   (B7). Land B7, then the asserts are one-liners.

3. **The phantom single-source homes must become real (B7) and the duplicated domain rules must
   collapse into them (B3, X-8/B16).** The codebase's signature strength is single-source; its
   signature weakness is the handful of rules that are single-homed *by halves* (block-ranges, the
   byte index, the bit order, `cdiv`). Creating `launch_config.hpp`/`host_device.hpp`/`log.hpp` and
   `block_ranges()` closes ~10 findings across ~8 files at once.

4. **The capability-tier machinery must exist as the one unified design in §(2)** — a probe at
   `Resources`, tagged-and-logged degrade through a *non-throwing* path, override knobs in
   `DeviceConfig` and discovered state in `Resources`, parity-neutral on both tiers. None of it
   exists today; M4.5 is the moment it must, because that is the milestone that introduces the
   second (P2P-vs-host-staged) path it governs.

Items 1–3 are squarely the before-M4.5 set (B1–B23). Item 4 lands *with* M4.5. With those done, the
device-backend and IO areas rise to ~8.5–9, the determinism/fail-fast/single-source principles hold
end-to-end, and the codebase clears 9.5 — the structure underneath is already there.
