# steppe — RUN GUIDE (how to run everything)

The one-stop "how do I actually run this" doc. Every command here is **verified at
HEAD** (`phase2-fit-engine`, the docs-refresh tip; the code is `main @ 25c882a`) — no
invented variants. For *standing up / verifying a fresh box* see `docs/BOX-RUNBOOK.md`;
for *what is built / what's next* see `docs/RESUME.md` + `docs/ROADMAP.md`; the fit
internals live in `docs/design/fit-engine.md`.

> **SCOPE (read this first).** steppe ships a **14-subcommand CLI** (`steppe extract-f2 /
> qpadm / qpadm-rotate / qpwave / f4 / f3 / f4-ratio / f4-sweep / f3-sweep / qpdstat /
> qpfstats / qpgraph / qpgraph-search / dates`) **and** a **Python facade** (the
> `steppe` package: `read_f2` / `extract_f2` / `qpadm` / `qpwave` / `qpgraph` /
> `qpgraph_search` / `f4` / `f3` / `f4ratio` / `qpdstat` / `dstat` / `dates` /
> `qpfstats` / `qpadm_search`). For the user-facing front door — install, the CLI command
> table, the Python quickstart — see the top-level **`README.md`** and the canonical
> **`docs/RUN-SHEET.md`**; the full feature/golden/wall-clock matrix is
> `docs/feature-matrix.md`. **This guide is the lower-level / developer view**: the
> build, the GPU-vs-AT2-golden ctest, the **test-harness binaries**
> (`test_qpadm_parity` / `test_qpadm_rotation`), and the **C++ library API**
> (`steppe::run_qpadm` / `run_qpadm_search`, `include/steppe/qpadm.hpp`) for embedding the
> fit directly.

Nothing builds locally — **author locally → rsync → build/test on the box.**

---

## 0. The box + the nvcc-PATH prefix + rsync-from-local

**The box:** `box5090` — 2× RTX 5090 (Blackwell, `sm_120`), CUDA 13. SSH alias
`ssh box5090` (the alias/IP is ephemeral; set it up per `docs/BOX-RUNBOOK.md` §0). The
RTX 5090s are **consumer GeForce: no GPU↔GPU P2P** — which is why the rotation runs
single-GPU (see §6).

**⚠ The nvcc-PATH prefix.** `nvcc` is usually **not on PATH** on the box (without this you
get a bogus "Failed to detect a default CUDA architecture"). Prefix every build/test
command with:

```bash
export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0
```

(`ulimit -c 0` — the fail-fast/assert tests call `abort()`, and the kernel can dump a
~404 MB core per abort and fill the disk. See §6 for the cleanup.)

**Sync the repo from local:**

```bash
rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh /home/suzunik/steppe/ box5090:/workspace/steppe/
```

---

## 1. BUILD (Release, `build-rel`)

**PERF/bench work MUST be a Release build.** A debug build inserts a per-kernel
`cudaDeviceSynchronize` that voids all timing. Use `build-rel`:

```bash
# on the box, AFTER the nvcc-PATH prefix from §0, from /workspace/steppe:
cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build-rel
```

This builds the core library, the device (CUDA) backend, and the test/harness binaries
into `build-rel/bin/`.

---

## 2. TEST — the GPU-vs-AT2-golden ctest

The validation is **real-AADR AT2 goldens** (admixtools 2.0.10 / R 4.3.3 / v66.p1_HO).
The GPU path matches them to the recorded tolerance tier; the `CpuBackend` is the native
oracle, run only under `STEPPE_THOROUGH`.

### Default lane (GPU vs golden, fast ~1 min)

```bash
ctest --test-dir build-rel --output-on-failure
```

The two qpAdm tests in this lane:

| ctest name | binary | runtime | what it covers |
|---|---|---|---|
| `qpadm_parity` | `test_qpadm_parity` | ~0.86 s | the **single-model fit + rank test** on the GPU vs `golden_fit0.json` (9-pop, `nr≤32` small path) **and** `golden_fit1_NRBIG.json` (`nr=39`, the cuSOLVER `gesvd` large path). |
| `qpadm_rotation` | `test_qpadm_rotation` | ~3.42 s | the **S8 model-space rotation** on the GPU vs `golden_rot.json` (84-model rotation), genuinely batched, G=1 vs G=2 bit-identical. |

(The lane also runs the Phase-1 precompute equivalence/determinism tests — those need
`STEPPE_AADR_ROOT` pointed at the real AADR data; the qpAdm tests use the bundled
fixtures and do not.)

### Thorough lane (adds the CpuBackend oracle; the CI-without-GPU localizer)

```bash
STEPPE_THOROUGH=1 ctest --test-dir build-rel -R qpadm
```

`STEPPE_THOROUGH=1` is **~56 s** and adds: the `CpuBackend` native oracle diff (the
bit-exact reference the device path is diffed against) and the NRBIG full-SE path. This
is the lane to run when you've touched the fit math and want the oracle to localize a
divergence — and it's the CI lane for a host without a GPU.

---

## 3. RUN A qpAdm FIT / RANK TEST / ROTATION

### 3a. Via the test-harness binaries (loads the real f2 fixtures, runs the GPU fit)

The harness binaries take the goldens dir as their one argument and load the real f2
fixtures themselves:

```bash
# from /workspace/steppe, AFTER the §0 prefix + a §1 build:
build-rel/bin/test_qpadm_parity   tests/reference/goldens/at2
build-rel/bin/test_qpadm_rotation tests/reference/goldens/at2
```

What they load + assert:

- **`test_qpadm_parity`** — uploads `fixtures/f2_fit0_9pop.bin` (the 9-pop f2 tensor) to
  VRAM as a `DeviceF2Blocks` and runs the production CUDA-backend fit + rank test,
  asserting the GPU result matches `golden_fit0.json` (`f4rank`, `dof`, `chisq`, the
  `rankdrop`/`popdrop` tables); then `fixtures/f2_fit1_NRBIG.bin` (real AADR,
  target=England_BellBeaker, `nr=39`) for the cuSOLVER `gesvd` large path vs
  `golden_fit1_NRBIG.json`. Under `STEPPE_THOROUGH=1` it also runs the `CpuBackend`
  oracle and the NRBIG full SE.
- **`test_qpadm_rotation`** — uploads `fixtures/f2_rot.bin` (real AADR, an 8-pop source
  pool, `nr≤32` right set ⇒ all 28 two-source + 56 three-source = 84 models) and runs
  `run_qpadm_search` batched on the GPU, asserting per-model weights/`f4rank`/feasibility
  vs `golden_rot.json`, that the run is genuinely batched (`batched_dispatch_count` ≪ the
  84 models — 2 buckets), and that G=1 and G=2 are bit-identical + identically ordered.

These are the same commands `ctest` invokes (§2), just run directly.

### 3b. Via the C++ library API (your own translation unit)

The public, CUDA-free entry points are in **`include/steppe/qpadm.hpp`**:

```cpp
#include "steppe/qpadm.hpp"   // run_qpadm, run_qpadm_search, run_qpwave + value types
// device::DeviceF2Blocks + device::Resources come from the device headers.

using namespace steppe;

QpAdmModel model;
model.target = tgt_idx;              // population INDICES into the f2_blocks P axis
model.left   = { src0_idx, src1_idx };
model.right  = { r0_idx, r1_idx, /* ... outgroups ... */ };
model.model_index = 0;              // stable identity (echoed back)

QpAdmOptions opts;                   // AT2 parity constants are NAMED here, not literals
// opts.fudge          = 1e-4;       // AT2 ridge constant
// opts.als_iterations = 20;         // opt_A/opt_B ALS iterations (NOT a single Cholesky)
// opts.rank           = -1;         // -1 ⇒ default nl-1 (best 2-way rank)
// opts.rank_alpha     = 0.05;       // rank-decision significance (f4rank)
// opts.jackknife      = JackknifePolicy::All;  // SE policy — rotation only (see below)

// SINGLE model, device-resident f2 (the GPU-first primary entry; zero D2H):
QpAdmResult r = run_qpadm(f2 /*DeviceF2Blocks*/, model, opts, resources);
//   r.weight, r.se, r.z, r.p, r.chisq, r.dof, r.f4rank,
//   r.rankdrop_*, r.popdrop_*, r.status (Ok / RankDeficient / NonSpdCovariance)
```

Key API facts (all from `include/steppe/qpadm.hpp`):

- **A model references populations by INDEX** into the device-resident f2_blocks P axis —
  no strings at the compute seam (name→index resolution is an app/binding concern, owned
  by the built `app/` CLI layer and the Python facade).
- **`run_qpadm`** has two overloads: the `DeviceF2Blocks` form (the GPU-first primary, zero
  D2H) and a host `F2BlockTensor&` form (the parity/oracle door used by the test).
- **`run_qpwave`** — the qpWave rank-sweep form (no target prepend), same two overloads,
  returns `QpWaveResult`.
- **`run_qpadm_search`** — the **S8 rotation**: fit a *pool* of `std::span<const QpAdmModel>`
  against the same resident f2, batched on the GPU and sharded across `Resources::gpus`,
  returning a per-model `std::vector<QpAdmResult>` **in input order** (deterministic
  regardless of GPU count). Domain outcomes are per-model `status`, never exceptions — a
  search of thousands of models records-and-continues.
- **`JackknifePolicy { None=0, FeasibleOnly=1, All=2 }`** (`QpAdmOptions::jackknife`,
  default `All`) governs **only** the rotation's expensive LOO jackknife SE — the point
  estimate (weights/χ²/p/f4rank/feasible/popdrop) is identical across all three. The
  single-model `run_qpadm` / `run_qpwave` **ignore** it (they always compute SE). `None` =
  no SE (fastest screen); `FeasibleOnly` = SE only for survivors; `All` = SE for every
  model (today's behavior, the goldens).

The harness `.cu` files (`tests/reference/test_qpadm_parity.cu`,
`tests/reference/test_qpadm_rotation.cu`) are the **worked, compiling examples** of
calling this API end-to-end (fixture load → `upload_f2_blocks_to_device` → `run_qpadm` /
`run_qpadm_search` → assert).

---

## 4. THE PRECOMPUTE (how `f2_blocks` is produced, device-resident)

The fit reads a **`DeviceF2Blocks`** — the Phase-1 precompute output (S0–S2), kept
**resident in VRAM** so the fit does zero D2H on the CUDA path. Two ways it gets there:

1. **From a precompute run (production):** the device backend computes the per-block f2
   tensor straight into VRAM via `compute_f2_blocks_device` /
   `compute_f2_blocks_streamed` (`src/device/backend.hpp`). M5 streaming auto-selects a
   tier (VRAM → host RAM → disk) so the GPU footprint stays bounded
   (O(P·tile + P²)) — a **full-autosome P=2500** precompute completes on a single 32 GB
   5090 in **~51.5 s**, parity bit-identical (`docs/RESUME.md`). The on-disk Disk-tier
   cache path is `QpAdmConfig`-driven (`include/steppe/config.hpp`: `disk_cache_path` /
   `STEPPE_F2_CACHE_PATH`, default `./steppe_f2_blocks.cache`) — the precompute-once /
   fit-many artifact.

2. **From a host tensor (tests / staging a fixture):**
   `steppe::device::upload_f2_blocks_to_device(const F2BlockTensor& host, int device_id)`
   (`src/device/device_f2_blocks.hpp`) copies a host `F2BlockTensor` to VRAM as a
   `DeviceF2Blocks`. This is exactly what the harness binaries do with the bundled
   `fixtures/*.bin` real-AADR f2 tensors before calling the fit.

The Phase-1 precompute equivalence/determinism is validated by its own ctest lane
(`f2_blocks_equivalence`, `f2_determinism`, etc.) against real AADR — those need
`STEPPE_AADR_ROOT` pointed at the data (see `docs/BOX-RUNBOOK.md` §5).

---

## 5. The dev loop (edit local → rsync → build → ctest)

```bash
# 1) edit locally in /home/suzunik/steppe
# 2) sync to the box:
rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh /home/suzunik/steppe/ box5090:/workspace/steppe/
# 3) on the box (after the §0 nvcc-PATH prefix), from /workspace/steppe:
cmake --build build-rel                                 # incremental
ctest --test-dir build-rel --output-on-failure         # fast GPU-vs-golden (~1 min)
# touched the fit math? localize with the oracle:
STEPPE_THOROUGH=1 ctest --test-dir build-rel -R qpadm  # ~56 s
```

The box's network is flaky — for long jobs (a full build, a bench), run **detached on the
box and poll a log** (`docs/BOX-RUNBOOK.md` §7) rather than holding an ssh session open.

---

## 6. Gotchas

- **RELEASE only for any timing.** A debug build's per-kernel
  `cudaDeviceSynchronize` voids all perf numbers. Always `build-rel` (§1) for bench/perf.
- **Clear core dumps.** Fail-fast/assert tests `abort()`; the box can accumulate ~404 MB
  cores. Run with `ulimit -c 0` (§0) and reclaim a flooded box:
  ```bash
  rm -f /var/lib/vastai_kaalia/data/core-*
  ```
- **Run the rotation SINGLE-GPU on the 5090s.** The consumer 5090s have **no GPU↔GPU
  P2P**, so multi-GPU rotation pays a one-time f2 replication as a **host bounce**
  (~8.72 GB / ~3.8 s) — only **~1.21× at 9086 real models**, no crossover. Multi-GPU
  rotation is **deferred** (`TODO(multigpu-host-bounce)`; needs P2P hardware like the
  RTX PRO 6000, or per-device precompute). **Do not expect a multi-GPU rotation
  speedup on box5090 — run single-GPU.** Parity stands (84/84 real models == AT2) and
  G=1 == G=2 bit-identical regardless of GPU count.
- **All validation is REAL AADR.** Every parity number above is on real AADR goldens
  (admixtools 2.0.10 / R 4.3.3 / v66.p1_HO); there are no synthetic-data results.

---

## The CLI / Python surface (built — wraps this same API)

The `app/` layer **is built**: the `steppe` CLI (`src/app/`) owns name→index resolution
(users pass population *names*, not f2 indices), argument parsing, and CSV/JSON emit; the
Python facade (`bindings/steppe`, nanobind) exposes the same surface — both wrap the same
`run_qpadm` / `run_qpadm_search` / `run_qpwave` library API this guide documents, plus the
standalone f-stats (f4 / f3 / f4-ratio / qpDstat), qpfstats, qpGraph (fit + topology
search), and DATES. For the user-facing install + command table + quickstart, see
`README.md` and `docs/RUN-SHEET.md`; the per-feature golden/wall-clock matrix is
`docs/feature-matrix.md`. Still pending (not yet built): older `.GENO`/EIGENSTRAT/PLINK
readers (steppe is TGENO-only) and the precompute M6 multi-dataset merge / M7 on-disk
cache (`docs/research/desirable-features-survey.md`).
