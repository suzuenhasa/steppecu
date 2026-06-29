# Code Conventions and Quality — First-Principles Assessment

**Scope:** C++/CUDA source and headers across `steppe`, read-only. Sample of ~28 files covering CUDA kernels, core algorithms, I/O, app commands, internal headers, device abstractions, and public headers.

**Method:** First-principles review against the dimensions requested: naming, comments, C++ idioms, duplication, cohesion, magic numbers, includes/formatting. Line references are to the versions present at review time.

---

## Executive Summary

The codebase is technically competent and well-engineered for numerical correctness, but it reads like an active excavation site. The dominant quality drag is not algorithmic error or memory safety — it is **cognitive load**: enormous comment manifestos, copy-pasted orchestration boilerplate, monolithic functions/classes, and an inconsistent surface contract (exceptions vs `Status` vs raw pointers vs `std::span`). A+ quality requires a disciplined cleanup pass focused on compression, centralization, and contract regularity rather than further feature expansion.

The most common repeated issues align with the directory’s own README observation (lines 156–160): integer-width soup, `Status`/exception schizophrenia, duplicated emit/output logic, over-commenting, and copy-paste drift.

---

## Top 10 Patterns That Must Change

### 1. Comment manifestos and internal ticket archaeology

**The problem.** File headers and inline comments routinely exceed 30–50 lines and are loaded with roadmap sections (`ROADMAP §0`), cleanup ticket IDs (`cleanup X-6/B2`, `cleanup 20.1/MED`), machine names (`box5090`), and bracketed dedup notes (`[7.1]`, `[7.2]`, `[9.1]`). This is noise for future maintainers and will become stale the moment the external tracking system changes.

**Evidence.**
- `src/device/cuda/f2_block_kernel.cu:1–54` — 54-line header essay before the first include.
- `src/device/cuda/dstat_kernel.cu:1–48` — similar manifesto including performance history and hardware notes.
- `src/device/cuda/dates_kernel.cu:1–17` — comparatively short, but still cites `dates.c:585-665` and `fftsubs.c`.
- `src/device/cuda/cuda_backend.cu:1–56` — multi-GPU scaffold history, `cleanup F19/F20`, etc.
- `src/core/stats/qpfstats.cpp:1–40` — header recaps the entire GPU pipeline and AT2 parity pins.
- Inline: `[7.1] dedup`, `OQ-12`, `B22`, `X-7/B6`, `M-FR-2` appear dozens of times across the sample.

**Recommendation.** Keep the *why* and the *non-obvious invariant*; delete the *when* and the *ticket ID*. Move historical design rationale to `docs/design/` or the per-file review notes in `docs/kimireview/`. A header should state what the file owns and its critical invariants, not the full changelog.

---

### 2. Repeated `kPrimaryGpu` / `primary_backend()` boilerplate

**The problem.** The same "device 0 is the primary backend" convention is redeclared and reimplemented in nearly every core algorithm TU and many app commands. This is a global convention that should live in one place.

**Evidence.**
- `src/core/qpadm/qpadm_fit.cpp:242` — `inline constexpr std::size_t kPrimaryGpu = 0;`
- `src/core/qpadm/f4.cpp:41` — identical constant.
- `src/core/qpadm/qpgraph_fit.cpp:39` — identical constant.
- `src/core/stats/dstat.cpp:64` — identical constant.
- `src/core/stats/qpfstats.cpp:73` — identical constant.
- Matching `primary_backend(resources)` helpers appear in the same files.

**Recommendation.** Centralize in `device/resources.hpp` or a new `device/primary.hpp` seam: `std::size_t primary_gpu_index()` and `ComputeBackend& primary_backend(Resources&)`. Delete the per-TU duplicates.

---

### 3. Duplicated genotype decode front-end

**The problem.** `run_dstat` and `run_qpfstats` share almost the entire read→partition→tile→decode→autosome-keep pipeline, yet it is copied and pasted with minor local renames. Any filter or ploidy change risks diverging the two paths.

**Evidence.**
- `src/core/stats/dstat.cpp:100–190` — `GenoReader`, `read_ind_partition`, `read_snp_table`, `read_canonical_tile`, `decode_af_compact_autosome`, and the host fallback loop.
- `src/core/stats/qpfstats.cpp:233–309` — the same sequence, including the identical `resident ? be.decode_af_compact_autosome(...) : ...` branch and the host fallback loop.

**Recommendation.** Extract a single function such as `core::decode_genotype_path(geno, snp, ind, pops, resources) -> DecodedPops` that returns the compacted device result or host Q/V plus `chrom_kept`/`genpos_kept`. Both `dstat` and `qpfstats` should consume it.

---

### 4. Monolithic functions and classes

**The problem.** Several public entry points and the CUDA backend class do too much. They mix I/O, validation, device dispatch, host fallback, and emit logic in one scope, making testing and review hard.

**Evidence.**
- `src/core/stats/dstat.cpp:77–236` — `run_dstat` is ~160 lines covering I/O, decode, autosome filtering, block assignment, and kernel dispatch.
- `src/core/stats/qpfstats.cpp:214–411` — `run_qpfstats` is ~200 lines; the popcomb/design builder alone is 90 lines.
- `src/app/cmd_extract_f2.cpp:133–423` — `run_extract_f2_command` is ~290 lines with dry-run, hash thread, library call, and meta writing.
- `src/device/cuda/cuda_backend.cu` — the `CudaBackend` class spans the entire 5,679-line file and mixes f2 blocks, decode, D-stat, qpfstats, qpGraph, sweep, rank tests, and streaming tiers.

**Recommendation.** Decompose into smaller private helpers with explicit in/out contracts. For `CudaBackend`, split by workflow into `cuda_backend_f2.cpp`, `cuda_backend_decode.cpp`, `cuda_backend_qpadm.cpp`, etc., or at least into internal `friend` namespaces. Each helper should be unit-testable in isolation.

---

### 5. Magic numbers in numerical kernels

**The problem.** Several numerical tunables are written as bare literals. They may be correct today, but they are not discoverable and are easy to drift.

**Evidence.**
- `src/device/cuda/dates_kernel.cu:278` — `const int coarse = 4000;` (exp-fit grid search).
- `src/device/cuda/dates_kernel.cu:305` — `const int ternary_iters = 200;`.
- `src/device/cuda/dates_kernel.cu:303–304` — literal `1e-9` tolerance bounds with no named constant.
- `src/device/cuda/dstat_kernel.cu:248` — `constexpr int kThreads = 256;` is named, but `256` also appears as `kBlock` in `dates_kernel.cu:31`; the two names suggest the value is not globally standardized.
- `src/device/cuda/f2_block_kernel.cu:354` — uses `kCdivBlock`, which is good, but the kernel body still contains literal `16` in the comment and the block math could be clearer.

**Recommendation.** Promote every algorithmic/tuning literal to a named `constexpr` in a single home (e.g., `dates_kernel.cu` internal constants, `core/internal/launch_config.hpp` for launch values). Include a one-line justification for the chosen value.

---

### 6. Abbreviated names in non-hot orchestration code

**The problem.** Single-letter and two-letter identifiers are used for variables with broad lifetimes, especially in the fit engine and app commands. This is fine inside a 5-line hot loop, but not across 100-line functions.

**Evidence.**
- `src/core/qpadm/qpadm_fit.cpp:53` — `ComputeBackend& be`.
- `src/core/qpadm/qpadm_fit.cpp:74` — `const Precision prec`.
- `src/core/qpadm/qpadm_fit.cpp:88` — `GlsWeights gw`.
- `src/core/qpadm/model_search.cpp:116` — `const Precision prec`.
- `src/core/qpadm/f4.cpp:54` — `F4Result run_f4_impl(ComputeBackend& be, ...)`.
- `src/app/cmd_f4.cpp` / `cmd_qpadm.cpp` / `cmd_qpgraph.cpp` — `dir`, `opts`, `fmt`, `wr` are pervasive.

**Recommendation.** In orchestration code with non-trivial scope, use full words: `backend`, `precision`, `gls_weights`, `options`, `format`, `write_result`. Reserve abbreviations for local lambda captures and tight numerical loops.

---

### 7. Schizophrenic error contract (exceptions vs `Status` vs catch-as-control-flow)

**The problem.** The architecture claims domain outcomes are `Status` values, never exceptions, yet several paths use `try/catch` for normal control flow or swallow `std::runtime_error` to paper over unimplemented backend methods.

**Evidence.**
- `src/core/qpadm/qpadm_fit.cpp:163–190` — catches `std::runtime_error` from `run_rank_sweep` because some backends lack the override, treating it as a non-fatal absence. This is control flow by exception.
- `src/core/qpadm/model_search.cpp:134–140` — catches `std::runtime_error` from `fit_models_batched` to fall back to per-model fitting.
- `src/app/cmd_extract_f2.cpp:340–347` — catches `std::invalid_argument` separately from `std::exception`.
- `src/app/cmd_f4.cpp:176–182` / `cmd_qpadm.cpp:118–124` / `cmd_qpgraph.cpp:176–179` — identical `try { build_resources; upload; run; } catch (const std::exception&) { ... }` blocks.

**Recommendation.** Decide the contract and enforce it:
- Domain outcomes → `Status` return value.
- Programming/config errors → exception with a specific type or `std::runtime_error`.
- Optional backend capabilities → return an empty/optional result or a capability flag, not an exception. Replace the `try/catch` fallback in `qpadm_fit.cpp` and `model_search.cpp` with a capability query.

---

### 8. Header bloat and tangled seams

**The problem.** Key seam headers have grown into monoliths that pull in many concrete types and mix interface with large POD definitions. This increases compile times and coupling.

**Evidence.**
- `src/device/backend.hpp` — 1,857 lines; includes `steppe/config.hpp`, `steppe/error.hpp`, `steppe/fstats.hpp`, `steppe/qpadm.hpp`, `core/internal/views.hpp`, `device/device_partial.hpp`, `device/device_f2_blocks.hpp`, etc.; defines `F2Result`, `F4Blocks`, `JackknifeCov`, `QpfstatsSmooth`, `RatioBlockJackknife`, `QpGraphTopoArena`, `RankSweep`, `PopDropRow`, `DecodeTileView`, `SweepConfig`, and the full `ComputeBackend` interface.
- `src/device/cuda/cuda_backend.cu` — 30+ includes (lines 57–123), many of which are only needed for specific methods.
- `include/steppe/qpadm.hpp` — reasonably sized, but `backend.hpp` including `qpadm.hpp` creates a dependency where the internal device seam depends on a public header.

**Recommendation.** Split `backend.hpp` into:
- `device/compute_backend.hpp` — the pure abstract interface.
- `device/value_types.hpp` or per-workflow value headers — the POD structs.
- Keep public headers (`include/steppe/*.hpp`) free of internal device dependencies; forward-declare where possible.

---

### 9. Copy-paste drift in SNP-major tile readers

**The problem.** `GenoReader` implements four format-specific SNP-major tile readers with nearly identical skeletons: format check, `snp_begin==0` guard, bounds check, empty-partition guard, selection build, checked-multiply allocation, and read/pack loop. The differences are small (PLINK bit-flip, EIGENSTRAT ASCII parse, ANCESTRYMAP triple parse), yet each function is ~100–150 lines.

**Evidence.**
- `src/io/geno_reader.cpp:512–635` — `read_snp_major_tile`.
- `src/io/geno_reader.cpp:637–783` — `read_eigenstrat_snp_major_tile`.
- `src/io/geno_reader.cpp:785–930` — `read_plink_snp_major_tile`.
- `src/io/geno_reader.cpp:932–1132` — `read_ancestrymap_snp_major_tile`.

Common duplicated blocks: selection+pop-offsets build (lines ~563–583, ~689–705, ~837–858, ~976–999), checked multiply + allocation (lines ~586–610, ~707–731, ~860–884, ~1000+), and the format check/bounds/empty partition prologue.

**Recommendation.** Factor the common prologue/selection/allocation into `read_snp_major_tile_common(...)`, and inject only the format-specific record parser via a small strategy lambda or virtual interface.

---

### 10. Integer-width and span/pointer API inconsistency

**The problem.** The codebase mixes `int`, `long`, `std::size_t`, and `std::ptrdiff_t` for counts and indices, and it mixes `std::span<const T>` with raw `const T* + size` APIs. This is exactly the README’s first flagged issue (line 157) and it creates real narrowing risk.

**Evidence.**
- `src/core/internal/views.hpp:60` — `long M = 0;` for SNP count.
- `src/device/cuda/f2_block_kernel.cu:123` — `int P, long M`.
- `src/device/cuda/dstat_kernel.cu:74` — `int P, long M`.
- `src/device/backend.hpp:433–438` — `compute_f2_blocks(..., const int* block_id, int n_block, ...)` uses raw pointer + size.
- `src/core/qpadm/qpadm_fit.cpp:271` — `std::span<const int>(X.block_sizes)` wraps the same data.
- `src/core/stats/dstat.cpp:222–229` — passes `.data()` pointers to `dstat_blocks_jackknife`.
- `src/core/qpadm/model_search.cpp` — mixes `std::size_t G`, `std::size_t n`, `int nl`, `int nr`, `int r`.

**Recommendation.** Adopt a project-wide index type alias (`steppe::Index` → `std::ptrdiff_t` or `std::size_t`) and use it consistently. Convert backend virtuals that take raw pointers to `std::span`. Audit all `static_cast<int>(...)` and `static_cast<long>(...)` sites for narrowing; replace with checked conversions or wider types.

---

## Additional Observations (Honorable Mentions)

- **`[[nodiscard]]` is used well** — e.g., `launch_config.hpp`, `small_linalg.hpp`. Keep this.
- **`constexpr` constants are mostly good** — `kRidge`, `kPloidyDiploid`, `kInvSqrt2`, `kMaxGridX/Y/Z`. Extend this discipline to the remaining literals.
- **Formatting is consistent** — 4-space indent, K&R braces, aligned continuation. The main issue is overly long lines and dense comment blocks.
- **Public headers are mostly CUDA-free** — good. Only `backend.hpp` violates the spirit by pulling in public `qpadm.hpp`.
- **RAII for CUDA resources** — `DeviceBuffer`, `Stream`, `CublasHandle` are present and correct. This is a strength, not a defect.

---

## Priority Order for Cleanup

If only one cleanup sprint is possible, do it in this order:

1. Centralize `kPrimaryGpu` / `primary_backend()` (Pattern 2) — low risk, high dedup.
2. Extract the shared decode front-end (Pattern 3) — reduces parity risk.
3. Trim comment manifestos (Pattern 1) — immediate readability win.
4. Split `backend.hpp` (Pattern 8) — improves compile times and layering.
5. Fix the error contract (Pattern 7) — removes hidden control flow.
6. Refactor SNP-major readers (Pattern 9) — reduces copy-paste surface.
7. Name remaining magic numbers (Pattern 5) and expand abbreviations (Pattern 6).
8. Decompose monolithic functions/classes (Pattern 4) — longest task, do last.
9. Standardize index types and `std::span` APIs (Pattern 10) — requires careful audit.

