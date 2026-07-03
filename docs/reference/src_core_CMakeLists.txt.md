# `src/core/CMakeLists.txt` reference

## 1. Purpose

This build file defines the **core layer** of steppe: the host-side orchestration
that decides *what* GPU work to do and in *what order*, without ever touching CUDA
directly. It declares two library targets and pins the rules that keep the core
layer physically separate from the GPU code.

The two targets are:

- **`steppe_core_internal`** — a header-only ("INTERFACE") library of small,
  shared, host-pure helpers that both the core layer and the GPU layer include. It
  contains no compiled code of its own; it just makes a set of headers available
  under a common include root.
- **`steppe_core`** — a compiled static library holding all the host orchestration:
  the f2 assembly driver, the qpAdm fit engine, the standalone f-statistics
  (f2/f3/f4 and their ratio and sweep variants), qpGraph, the genotype-path tools
  (qpDstat, qpfstats, DATES), the shared genotype read front-end, and the
  command-line/bindings configuration layer.

The single most important rule this file enforces is that the core layer reaches the
GPU **only** through a CUDA-free seam called `ComputeBackend`. CUDA itself stays
private to a separate `steppe_device` library. Because of how the link dependencies
are set up (described below), a file in the core layer physically *cannot* include a
CUDA header — the compiler would not find it. That guarantees the core layer stays
host-pure and independently testable without a GPU.

---

## 2. `steppe_core_internal` — the shared host-pure helpers

This is an INTERFACE (header-only) library. It compiles nothing; it exists to expose
a small set of headers that are shared by *both* the core layer and the GPU layer, so
those two layers can never drift apart on the pieces they must agree on.

The headers it exposes are the "kernel" of the codebase:

- The **Q / V / N views** (`internal/views.hpp`) — lightweight typed views over the
  genotype, variance, and count data.
- The **shared f2 estimator primitive** (`internal/f2_estimator.hpp`) — the single
  source of the f2 formula. Because both the CPU reference oracle and the GPU feeder
  include this one header, they cannot compute f2 differently. The header is written
  to compile both under the CUDA compiler (as a `__host__ __device__` function) and
  under a plain host compiler (as an ordinary inline function), using a macro that
  expands to the right annotation for each.
- The **SNP-to-block domain rule** (`domain/block_partition_rule.hpp`) — the shared
  rule for assigning SNPs to jackknife blocks.

The reason this library is header-only and CUDA-free is exactly *because* it is
consumed by both layers. If it pulled in CUDA, the host-only core layer could not use
it. Keeping it as pure, dual-compilable headers lets the same code serve the CPU
oracle and the GPU path from one place.

**Include root.** Headers under `src/core/` are referenced with paths like
`core/internal/views.hpp` and `core/domain/...`, and headers elsewhere (for example
the backend header) include `core/internal/views.hpp`. So the include root this
library advertises is `src/` — one directory *above* `src/core/`. It also links the
public API library (for the shared config and error headers) and requires C++20.

---

## 3. `steppe_core` — the host orchestration

This is a compiled **static** library. It contains all of steppe's host-side
orchestration: the code that assembles f2 statistics, runs the qpAdm fit, computes
the standalone statistics, and wires up configuration — all without issuing a single
GPU instruction itself. Every GPU operation is routed through the `ComputeBackend`
seam.

It links the internal helpers and public API publicly, and the GPU layer, the
warnings preset, the I/O layer, and the threads library privately (see section 6).

The list of source files that make up this library is grouped by role in section 5.

---

## 4. The CUDA-free layering rule

The core layer links the GPU library (`steppe::device`) with **private** visibility.
"Private" here means: the core library uses the GPU library internally, but does not
pass its headers or include paths along to anything that links the core library. CUDA
stays private to the GPU library the same way.

The practical effect is a hard wall:

- The core layer can *call into* the GPU only through `ComputeBackend`, which is
  itself a CUDA-free header.
- Because the CUDA include paths never reach the core layer, a core-layer source file
  literally cannot `#include` a CUDA header — it would fail to compile.

This is what makes the core layer host-pure and testable without a GPU. Several of
the units below (the multi-GPU shard planner, the config builder, the fit engine) are
exercised by plain host tests that link a fake backend and never touch a real device.

---

## 5. The source files and what they do

### f2 assembly and multi-GPU orchestration

- **`fstats/f2_from_blocks.cpp`** — the single-device f2 assembly driver. It walks
  the blocks and drives the backend through the `ComputeBackend` seam to produce the
  f2 block tensor. It never issues a matrix multiply itself; it only orchestrates.

- **`fstats/f2_combine.cpp`** — the host-staged combine step. When results are
  computed on several GPUs, this gathers each device's partial result to the host and
  sums them in a fixed device order. It is the portable baseline that always works,
  and it is itself CUDA-free host code.

- **`fstats/f2_blocks_multigpu.cpp`** — the multi-GPU entry point. It shards whole
  jackknife blocks across the available devices, drives each device's backend through
  the CUDA-free seam, and combines the per-device partial results in a **fixed device
  order** (device 0, then 1, and so on). Fixing that order is what makes a multi-GPU
  run produce bit-for-bit the same result as a single-GPU run. This file chooses
  between two combine strategies at one decision point (see "the combine gate"
  below): the host-staged combine, or a faster direct device-to-device (peer-to-peer)
  combine. It stays CUDA-free because it names only host types and *CUDA-free
  declarations* of the combine routines — the actual CUDA implementation of the
  peer-to-peer combine lives in the GPU library and is reached through a plain header
  declaration.

- **`fstats/f2_blocks_multigpu_core.cpp`** — the host-pure, peer-to-peer-free *core*
  of that orchestrator, factored out on purpose. It holds the block-aligned shard
  **plan** and the per-device concurrent **fan-out**, and it references only the
  CUDA-free backend seam plus the block-planning helpers — never the device-side
  peer-to-peer combine. Because it names nothing that requires a device link, a
  completely GPU-free host test can drive it against a fake backend. The entry-point
  file above composes this core with the combine choice.

  **The combine gate.** Whether the run uses the fast peer-to-peer combine or the
  host-staged combine is decided in exactly one place — the peer-access capability
  check inside the multi-GPU entry point. Both combine paths sum partials in the same
  fixed device order, so they produce identical results; the choice only changes how
  bytes move between GPUs. If peer access turns out to be unavailable, the code falls
  back safely to the host-staged path.

- **`domain/block_partition_rule.cpp`** — the SNP-to-block assignment pass. This is
  the single home of the jackknife block id that both the I/O layer and the GPU
  kernels consume, so they can never disagree about which SNP belongs to which block.
  Only the *stateful* whole-ordering pass compiles here; the small per-SNP "which
  block is this?" primitive and the centimorgan-to-Morgan unit conversion stay inline
  in the header so they can be used from either layer.

### The qpAdm fit engine

- **`qpadm/qpadm_fit.cpp`** — the qpAdm fit orchestrator, plus the standard-error
  computation and the chi-squared p-value. Like everything here it is host-pure; the
  device operations route through the backend seam (a CPU reference backend for
  parity testing, a CUDA backend in production). The lower-level pieces (building the
  f4 matrix, the jackknife, the least-squares solve) are thin header-only drivers;
  this file holds the orchestration and the final statistics.

- **`qpadm/ranktest.cpp`** — the rank test that qpAdm uses to judge how many source
  populations a target needs.

- **`qpadm/nested_models.cpp`** — comparison of nested qpAdm models.

### The standalone f-statistics

- **`qpadm/f4.cpp`** — the standalone f4 statistic. It is the sibling of the qpWave
  runner and reuses the same quartet assembly and jackknife-covariance machinery; it
  introduces no new math.

- **`qpadm/f3.cpp`** — the standalone f3 statistic, the three-population sibling of
  f4. It reuses the same jackknife covariance; its one genuinely new piece is the
  three-slab combine.

- **`qpadm/f4ratio.cpp`** — the standalone f4-ratio. It reuses the shared quartet
  assembly (one assembly serves both the numerator and the denominator). Its one new
  piece is the per-block *ratio* weighted block-jackknife.

- **`qpadm/fstat_sweep.cpp`** — the GPU-only f-statistic **sweep** (over f4 or f3).
  This is the production-scale version of the standalone statistics: instead of
  computing one statistic, it enumerates *every* combination of populations, on the
  device, using a combinatorial-unranking kernel. It reuses the same quartet assembly
  and a per-item jackknife, filters survivors by significance and compacts them **on
  the GPU**, and returns only the survivors. There is no host-side enumeration, no
  host-side filtering, and no per-item host loop. A safety cap limits how many
  combinations a single sweep may enumerate.

### qpGraph

- **`qpadm/qpgraph_model.cpp`** — the qpGraph topology data model. It parses an edge
  list into a path-table model of population weights. This is the general version of
  the fixed weight-fill used by an earlier optimizer prototype.

- **`qpadm/qpgraph_fit.cpp`** — the single-graph qpGraph fit driver. It builds the f3
  basis resident on the device, runs the jackknife covariance, drives the on-device
  fleet optimizer, and returns the fitted result. (The optimizer's objective function
  is a separate header-only oracle shared by the CPU backend and the parity test.)

- **`qpadm/qpgraph_enumerate.cpp`** — the topology-search enumerator. It reproduces
  the reference tool's tree- and graph-generation exactly (both the count and the set
  of canonical graph hashes), so its enumeration can be checked one-to-one against the
  reference.

- **`qpadm/qpgraph_search.cpp`** — the qpGraph topology **search** driver. It fits an
  exhaustive bounded space of candidate topologies in one launch (the optimizer can
  fit many different topologies at once), picks the global best, and then runs a
  mutation / hill-climb step that is checked to recover the exhaustive best. Host-pure
  and CUDA-free.

### Model-space search

- **`qpadm/model_search.cpp`** — the model-space search orchestrator that runs qpAdm
  over many candidate models, plus the default batched-fit path.

- **`qpadm/model_search_core.cpp`** — the GPU-free planner that maps a set of models
  onto per-device shards. Factored out so it can be host-tested without a device.

### The genotype-path tools

These three tools read genotypes directly and compute their result from the raw
decoded data. None of them read the precomputed f2 cache.

- **`stats/dstat.cpp`** — qpDstat's normalized-D (the "Part B", genotype path). It
  reuses the shared genotype decode front-end (open reader, decode allele
  frequencies, assign blocks), then diverges into a per-SNP D kernel on the backend
  and a numerator/denominator block-jackknife on the host.

- **`stats/qpfstats.cpp`** — the joint f2 **smoother**. It reuses the same genotype
  front-end, drives the f4-numerator engine over the full set of f2/f3/f4 population
  combinations, then runs an on-device shared-factor smoothing solve and scatters the
  result back into a smoothed f2 block tensor.

- **`stats/dates.cpp`** — the DATES admixture-dating tool. It reuses the genotype
  front-end (including a per-SNP genetic-position step), then diverges into an FFT-
  based autocorrelation engine that builds the weighted linkage-disequilibrium decay
  curve, fits an exponential-plus-offset model to it, and runs a leave-one-chromosome
  weighted block jackknife. The FFT reformulation deliberately avoids the naive
  quadratic-in-SNPs pairwise loop.

### The shared genotype read front-end

Two files exist so the four genotype-path tools (extract-f2, qpDstat, qpfstats,
DATES) all read data through one code path and cannot drift apart:

- **`stats/read_canonical_tile.cpp`** — the format dispatch. It is the one place that
  turns an open genotype reader plus an individual partition into the canonical
  individual-major genotype tile, regardless of how the data is laid out on disk. For
  the individual-major on-disk format it is a straight read; for a SNP-major format it
  does a SNP-major gather and transposes to canonical layout on the backend. This is
  an orchestration point (the CUDA-free reader leaf meets the CUDA-free backend seam),
  not a CUDA source file. It is reused by both this library's tools and the standalone
  extract-f2 tool.

- **`stats/genotype_front_end.cpp`** — the shared decode front-end that wraps the
  tile reader above. It turns the genotype file triple into the canonical tile plus
  the parsed SNP table and individual partition, and stops at that hand-off — each
  caller then diverges into its own decode. Having a single front-end means any read-
  time change is made in one place instead of in four lockstep copies (which would be
  a parity hazard).

### The configuration / access layer

- **`config/config_builder.cpp`** — the CUDA-free, host-pure configuration layer used
  by the command-line tool and the language bindings. It implements the settings
  precedence merge — compiled-in defaults, then a TOML file, then `STEPPE_*`
  environment variables, then command-line flags, in that order — and a validating
  build step that maps the raw flags onto the real config structs and fails fast on
  bad input. It lives in the core library (not the app) so the bindings can reuse it
  and so it can be host-tested without a GPU.

---

## 6. Link dependencies

The core library declares its dependencies with deliberate visibility:

**Public** (passed along to anything that links the core library):

- **`steppe::api`** — the public API, for the shared config and error types.
- **`steppe::core_internal`** — the shared host-pure helpers from section 2.

**Private** (used internally, not passed along):

- **`steppe::device`** — the GPU library. Private on purpose: the core layer reaches
  the GPU *only* through the `ComputeBackend` seam, and keeping this private is what
  prevents CUDA headers from leaking into the core layer.
- **`steppe::warnings`** — the shared compiler-warning preset.
- **`steppe::io`** — the I/O layer, needed by the genotype-path tools for the reader
  and the individual/SNP table parsing.
- **`Threads::Threads`** — the threads library, needed for the per-device concurrent
  fan-out in the multi-GPU orchestration.

The library requires C++20.

---

## 7. The device-symbol resolution setting

The core library sets `CUDA_RESOLVE_DEVICE_SYMBOLS` to **OFF**, and this is a
deliberate, load-bearing choice worth understanding.

The core library has no CUDA sources of its own — it is pure host orchestration. It
links the GPU library privately only to reach the GPU through the backend seam.
Turning device-symbol resolution *off* tells the build **not** to have the core
library produce and archive its own device-link object for the GPU library's
separately-compiled device symbols.

Why this matters: if the core library *did* archive its own device-link object, then
a CUDA executable that links **both** the core library and the GPU library directly
(for example an end-to-end equivalence test) would end up resolving the CUDA
registration symbols **twice** — once from each library — and the final host link
would fail to find them cleanly.

With the setting off, the consuming executable device-links the GPU library directly,
which is the same path the existing CUDA tests already use successfully. In short:
the core library is a host archive that happens to depend on GPU code, so it must not
try to own the device-link step — the final executable owns it.
