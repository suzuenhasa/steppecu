# Discussion Summary — Path to 10/10 & f2 Architecture

This file compiles the key points from the recent discussion about the `steppe` project's overall engineering quality, flow, and the role of f2 precompute.

---

## 1. Overall Project Verdict

- `steppe` is a **credible, well-architected GPU population-genetics compute engine**.
- Current grade: **B+/A-**.
- It is **not slop** and **not a mess**.
- It is **not yet 10/10 or fully shippable** as a polished production product.
- The issues are **finishing work**, not fundamental design failures — no rewrite required.

### Genuine strengths

- Compiler-enforced CUDA-free seam: CUDA headers stay inside `src/device/`.
- Strong RAII ownership of buffers, handles, streams.
- Two-oracle correctness strategy: steppe CPU vs. steppe GPU vs. ADMIXTOOLS 2 goldens.
- Real-AADR performance discipline with documented numbers.
- Broad CLI/Python surface: 14 subcommands.
- Numerically careful; parity with AT2 is a real signal.

### Biggest blockers before senior review

- **No CI** — `.clang-format`, `.clang-tidy`, and warning flags exist but are not enforced.
- **`ComputeBackend` is a 1,857-line god interface** with ~40 virtuals spanning every feature.
- **`core` depends on `io`** despite `architecture.md` §4 claiming otherwise.
- **Output layer is half-centralized**: `result_emit.cpp` handles some commands; `dates`, `qpgraph`, `fstat-sweep`, `extract-f2`, `qpfstats` hand-build CSV/JSON.
- **Error taxonomy is incomplete**: no `Status::IoError`, no `Status::CudaRuntime`; disk-full and CUDA OOM both collapse to `kExitRuntimeError`.
- **Public API does not deliver the C ABI** promised in `architecture.md` §16.
- **Project docs are stale**: `NEXT-STEPS.md`, `ROADMAP.md`, etc. describe features that already ship.
- **Root clutter**: `.agents/`, `.claude/`, `.codex/`, `agentscripts/`, `aadr/` data in project root.

---

## 2. Flow & Cognitive Coupling

- Verdict: **C+**.
- Structurally layered, but control flow **bounces between app/core/io/device**.
- A reader must hold **8–15 files** in mind to follow a single command.

### Main flow problems

- Every fit command (`f4`, `f3`, `f4-ratio`, `qpadm`, `qpwave`, `qpgraph`, `rotate`) duplicates the same 5-step dance:
  - `read_f2_dir()`
  - `build_resources()`
  - `upload_f2_blocks_to_device()`
  - `run_*()`
  - emit output
- Each command repeats `--f2-dir` validation, no-GPU checks, `--out`/`stdout` opening, and format parsing.
- `src/core/stats/read_canonical_tile.cpp` does genotype I/O and calls `backend.transpose_to_canonical()` — this belongs in `app`/`extract`, not `core`.
- `src/core/CMakeLists.txt:153` links `steppe::io`, contradicting the documented layering.
- `src/device/backend.hpp:624-651` declares `core::qpadm` symbols — a device header pointing upward into core.
- `cuda_backend.cu` is ~5,600 lines and owns decode, f2 GEMMs, qpadm fit, qpgraph, dates, qpfstats — a monolithic seam.
- `cmd_extract_f2.cpp` mixes CLI validation, dry-run sizing, thread management, SHA-256, and metadata assembly at multiple abstraction levels.
- `model_search.cpp:215-248` uses `replicate_f2()` — a host bounce because precompute targets one GPU and rotation rebroadcasts.

### What organic flow would look like

- One `app/command_runner.hpp` template owns the fit-command orchestration.
- `core` is pure math on `F2BlockTensor`/`MatView`; never touches genotype files.
- `io` is only entered from `app`/`extract`.
- `ComputeBackend` is split into focused interfaces (`DecodeBackend`, `PrecomputeBackend`, `FitBackend`, etc.).
- Library calls return one `Result<T>` status type that maps to one exit code.
- Precompute is explicit and centralized.

---

## 3. f2 Precompute Architecture

### Current state

- `f2` is a **manual precompute artifact**, not a precompute engine.
- `extract-f2` is the **only** command that creates `F2BlockTensor` from genotypes.
- All other commands require `--f2-dir` to already exist; they fail if it is missing.
- The user must manually run `extract-f2` before `f4`/`qpadm`/`qpgraph`/etc.

### Difference between `f2` and `extract-f2`

- `f2` = the statistic / data structure (allele-sharing counts per SNP block).
- `extract-f2` = the CLI command that builds an f2 directory from `.geno`/`.ind`/`.snp` or PLINK files.
- `extract-f2` would be better named `build-f2`, `precompute f2`, or `f2 build`.

### f2 is the centerpiece

- The architecture is essentially:
  ```
  raw genotypes → f2 blocks → f4/f3/qpadm/qpgraph/dates/sweeps
       (expensive)      (cheaper)
  ```
- `F2BlockTensor` / the f2 directory is the canonical precomputed intermediate.
- Everything else is a transformation on top of f2.

### Levels of ambition for f2

1. **Manual precompute with better UX** (current, defensible)
   - Cache by genotype hash + population list + block params.
   - Clear "run `steppe extract-f2 ...` first" message.
   - Document f2 as the expensive, reusable artifact.

2. **Lazy/on-demand precompute**
   - Commands like `steppe f4 --geno ... --ind ... --snp ...` automatically build f2 if missing.
   - Requires cache key, cache invalidation, default cache location, force-rebuild flag.

3. **Managed, tiered f2 engine**
   - Resident GPU / host RAM / disk / cloud tiers.
   - Reference counting or LRU eviction.
   - Block-level granularity and background prefetch.

### Recommendation

- **Stick with Level 1 for now**, but clean it into a real subsystem.
- The trap is not precompute vs. recompute — it is **unclear ownership of expensive intermediate state**.
- Proposed subsystem:
  ```
  src/f2/
    f2_cache.hpp      // cache key, lookup, invalidation
    f2_engine.hpp     // create_or_load_f2(...) → F2DirHandle
    f2_dir_format.hpp // STPF2BK1 reading/writing
    f2_dir_writer.hpp
    f2_dir_io.hpp
  ```
- Move the scattered responsibility out of `src/app/extract_f2_core.cpp`, `src/core/stats/read_canonical_tile.cpp`, `src/core/fstats/f2_blocks_multigpu.cpp`, `src/app/f2_dir_writer.cpp`, and `src/app/f2_dir_io.cpp`.

### Other precompute/caching opportunities (secondary)

- Decode-compact results (raw genotype → compact allele-frequency arrays).
- SNP filter masks.
- `qpgraph` topology enumeration.
- `qpadm` model-search setup work.
- `dates` covariance structure.

These should only be addressed **after** the f2 engine is clean.

---

## 4. Shippability

| Definition | Verdict |
|---|---|
| Internal research tool | **Yes, basically shippable** |
| Open-source repo you publish | **Shippable with caveats** (tests/README good; docs stale; no CI) |
| Production pipeline others depend on | **Not yet** (error handling, output consistency, lifecycle missing) |
| Job-application showcase | **Needs polish first** (warts will dominate the conversation) |

- None of the issues require a rewrite.
- Estimated finishing work:
  - 1–2 weeks hygiene (CI, docs, root cleanup, status taxonomy).
  - 2–4 weeks structural cleanup (split `ComputeBackend`, decouple `core`/`io`, centralize output).
  - 1–2 months polish (C ABI, result structs, perf regression, wheel pipeline).

---

## 5. CI (Continuous Integration)

- CI = automated system that builds/tests code on every commit or pull request.
- It catches "works on my machine" problems.
- For `steppe`, CI would run:
  - Host-only build.
  - CUDA build.
  - Python wheel build.
  - Unit/reference/Python tests.
  - `clang-format` and `clang-tidy` checks.
  - Sanitizers / CUDA compute-sanitizer.
- Easiest path: GitHub Actions via `.github/workflows/ci.yml`.
- Free for public repos; CUDA runners available if needed.

---

## 6. Highest-Impact Fixes ( ranked )

1. Add CI that enforces format/lint/build/test on every commit.
2. Slim `ComputeBackend` into role-specific interfaces.
3. Decouple `core` from `io`; move genotype-path code into `src/extract/`.
4. Build one shared output layer (`OutputSink`, `CsvRowWriter`, `JsonWriter`, `DoubleFormatter`).
5. Fix the status/exit-code taxonomy (`IoError`, `CudaRuntime`, `DeviceOom`).
6. Wire `cusolverDnSetDeterministicMode` and add a second compute stream.
7. Refactor public result structs and remove polymorphic Python returns.
8. Refresh stale project-status docs and clean the repo root.
9. Implement the C ABI layer or stop advertising it.
10. Finish multi-GPU rotation: per-device precompute, dynamic model queue, `PeerTopology`.

---

## 7. What to Highlight vs. What to Fix First

### Highlight with confidence

- Compiler-enforced CUDA-free seam.
- Strong RAII ownership.
- Two-oracle correctness strategy (CPU/GPU/AT2 goldens).
- Real-AADR performance discipline.
- Broad CLI/Python surface.

### Fix before high-stakes showcase

- Missing CI.
- Stale project-status docs.
- `core` → `io` layering lie.
- 1,857-line `ComputeBackend` god interface.
- Half-centralized output layer.
- Silent file overwrites.
- Unimplemented C ABI.
- Root-level clutter.
