# `extract_f2_core.cpp` reference

## 1. Purpose

`src/app/extract_f2_core.cpp` holds the one real implementation of
`steppe::run_extract_f2` — the library call that turns a genotype triple
(`.geno`/`.snp`/`.ind`, or the equivalent PLINK `.bed`/`.bim`/`.fam`) into f2
statistics. It runs the full chain: read the files, decode genotypes into
allele frequencies, filter SNPs, partition the surviving SNPs into jackknife
blocks, compute the per-block f2 tensor on the GPU, and hand back a plain result
value.

Both the command-line tool and the Python bindings call this same function, so
there is exactly one copy of the computation. The command-line wrapper is thin:
it parses flags and forwards them here. Because there is one shared body, the
numbers are identical no matter how the run was started, and the reference
outputs (the "goldens") that this code is checked against never depend on which
front end invoked it.

Two things shape how the function reports problems and where it can live:

- **It throws, it never prints.** Where the command-line tool would print an
  error to the terminal and exit with a status code, this function raises a
  `std::runtime_error` or `std::invalid_argument` and returns a value on
  success (an `F2ExtractResult`). That is what lets the Python bindings surface
  a real exception instead of a process exit.
- **It contains no CUDA code.** The GPU is reached only indirectly, through the
  resource and backend objects passed in. This keeps the file on the CUDA-free
  side of the build so it can be compiled into the library, the CLI, and the
  bindings without any of them pulling in the GPU toolchain.

---

## 2. The pipeline at a glance

The function is a pipeline of five stages. Most stages feed the next in a
straight line; the branches are the GPU-versus-CPU split in the filter stage
and, on the GPU path, the output-tier choice in the compute stage — the
resident dense path versus the streamed re-decode path (section 2a) — and the
move-versus-copy of the finished result.

1. **Open and read.** Open the genotype file (which pins the on-disk format),
   read the individual/population table for the requested selection, read the
   SNP table, and read one canonical individual-major tile of packed genotypes.
2. **Decode and filter.** Decode packed genotypes into per-population
   allele-frequency quantities and decide which SNPs to keep. This stage has two
   implementations that agree on a bit-identical kept set — a GPU path and a CPU
   reference path. On the GPU path this is a *metadata-only* pass: it records the
   kept SNPs' identity, order, and count but **discards** the decoded
   `Q`/`V`/`N` per tile (see section 2a). The CPU reference path still compacts
   the kept columns into dense host arrays.
3. **Assign blocks.** Partition the kept SNPs into jackknife blocks along the
   genome.
4. **Compute f2 blocks.** Run the GPU f2 computation over the kept, blocked
   data — either from a dense host buffer (Resident tier) or by re-decoding each
   chunk on-device (streamed tiers). See section 2a.
5. **Materialize the result.** Move or copy the f2 tensor to host memory
   (whichever avoids doubling host RAM for the chosen tier) and fill in the
   result struct (labels, SNP counts, ploidy counts, precision tag, memory
   tier).

The three per-population quantities the f2 stage consumes are named `Q`, `V`,
and `N` throughout: a per-population allele-frequency-style value, its paired
variance term, and a per-population sample-count. They must stay aligned on the
SNP axis. On the GPU path for large models they are never carried as a dense
`P × M_kept` host buffer at all — they are re-decoded on demand (section 2a).

---

## 2a. The two-phase decode and how f2 is fed (GPU path)

The GPU path does not build the full dense `P × M_kept` host `Q`/`V`/`N` input
before computing f2. Building it was the memory wall: three `P × M_kept` double
arrays are roughly 96 GB at `--auto-top-k 2000` on the ~2.14M-SNP 2M panel, and
materializing them got the process OOM-killed (exit 137) before any f2 work ran.
The current flow is two phases with a tier decision between them.

### Phase A — metadata pass

Phase A decodes **every** SNP-tile in file order, but only to learn the kept
set: for each tile it takes that tile's kept-SNP `chromosome`, `genetic
position`, and `physical position`, appends them in file order, and adds the
tile's kept count to `M_kept`. Each tile's decoded `Q`/`V`/`N` is **discarded**
when the tile's device buffers are released at the end of the loop body. Host
RAM after Phase A is therefore `O(M_kept)` metadata — three short per-kept-SNP
arrays — not the old `O(P × M_kept)` dense wall. If nothing survived, the
function throws. The tiling mechanics of this pass (tile width, ploidy hoisting,
what runs on-device) are described in section 7.

### The tier decision (now host-input aware)

With `M_kept` known, `resolve_output_tier` picks the output tier — the f2 result
kept all-in-GPU-memory (Resident), spilled to host RAM (HostRam), or spilled to
disk (Disk). This decision is now aware of the *dense-input* host cost: the
Resident output engine needs the full dense `P × M_kept` host `Q`/`V`/`N`, so the
tier selector reserves that cost (three stacks of `P × M_kept × sizeof(double)`)
before it will choose Resident. A model whose dense `Q`/`V`/`N` would not fit in
host RAM is routed to the streamed HostRam/Disk path instead of blowing up. The
free-host-RAM figure this budget is measured against is container-aware: it
clamps the whole-host free figure to the cgroup memory headroom, so a
memory-capped box is not over-budgeted.

### Phase B, Resident tier — re-decode the dense buffer

For small and mid-size models that fit, Phase B re-runs the Phase-A tile loop,
but this time keeps each tile's `Q`/`V`/`N` and appends them into the dense
`P × M_kept` host buffer, then hands that buffer to the **unchanged** resident f2
engine. The decode is deterministic, so this rebuilds exactly the `Q`/`V`/`N` a
single non-tiled pass would have produced; this tier is byte-identical to the
pre-fix behavior.

### Phase B, HostRam/Disk tiers — re-decode per chunk on-device

For large models, Phase B feeds the **unchanged** streamed f2 engine without ever
building the dense host buffer. It does two things:

- **Build `kept_to_raw`.** This maps each kept column back to its raw `.snp` row.
  Because the kept set is a monotone subset of the file-order raw rows, a single
  forward two-pointer scan matches each kept SNP (by chromosome and physical
  position) to its raw row in order; a mismatch on the final count throws.
- **Construct a `RedecodeSource`.** This descriptor bundles the base decode view,
  the SNP-table columns, the filter, the per-population individual counts, the
  `maxmiss` value, and `kept_to_raw`. It is passed to the streamed engine, which
  supplies each per-chunk tile by **re-decoding** those kept columns on-device
  instead of copying them out of a dense host buffer that is never built.

Only the per-chunk tile *source* changed here — a re-decode in place of a
dense-buffer copy. The f2 GEMM, feeder, gather, assemble, ring, and spill are
byte-for-byte unchanged, so the streamed re-decode output is bit-identical to the
resident/dense-fed path[^at2] (verified: same `f2.bin` sha256, and the CordedWare
qpAdm weights matched the pre-fix run to all 17 digits).

### Moving, not copying, the HostRam result

For the HostRam tier the streamed f2 result already lives in a host tensor inside
the engine's sink, so the extract function **moves** it out rather than calling
`to_host()`. A `to_host()` copy would allocate a second full `P² × n_block × 2`
host tensor (f2 plus its paired variance term) — roughly 46 GB for a top-2000 2M
model — doubling host RAM and re-triggering the OOM. The Resident and Disk tiers
still materialize via `to_host()` (from VRAM and the on-disk cache respectively).

### Payoff

`--auto-top-k 2000` on the 2M panel, which was OOM-killed at exit 137 before the
change, now completes via `auto` tier=host in about 88 seconds with peak host RAM
around 54 GiB, under the box's ~90 GiB cap.

---

## 3. Parity pins

Four decisions are made deliberately so that the results stay parity-exact[^at2]. They are called out here because each one is easy to get subtly wrong
in a way that would still "run" but produce different numbers.

1. **Population-axis order.** The populations are ordered by reading the
   individual file for the selection and sorting ascending by population label.
   That sorted order is the P-axis index order used everywhere downstream, and
   it is the parity-fixed order.
2. **Per-sample pseudo-haploid detection.** Unless a ploidy is forced, each
   sample's ploidy is detected individually (some ancient-DNA samples are
   pseudo-haploid, some are diploid). This mirrors the parity
   pseudo-haploid adjustment. See section 5.
3. **The `maxmiss` coverage test is on the population axis.** The
   `maxmiss` filter is a per-SNP test over *populations*, not over individuals, and it
   is applied as a separate step. The individual-axis missingness filter is
   forced off during extract so it cannot double-filter. See section 6.
4. **Autosomes-only is a filter flag.** Restricting to chromosomes 1–22 is a
   filter setting carried in the filter configuration, not a hardcoded step.

Because these pins fix the exact SNP set and its exact order, the block
partition and every golden output are reproduced bit-for-bit.

---

## 4. Inputs, validation, and the population axis

The function first rejects impossible inputs before touching any data: an empty
path in the genotype triple, or a run with no CUDA device available (steppe is a
GPU product and requires one).

It then opens the files through a shared genotype front-end helper, which opens
the reader, reads the population partition for the requested selection, reads
the SNP table, and reads the packed genotype tile — all with the parser chosen
automatically from the on-disk format.

### Validating an explicit population list

When the caller passes an explicit list of population labels, every label must
be present in the resolved partition. This check lives in this function rather
than in the shared helper because it needs the resolved partition to compare
against, and because it is part of the library's contract to fail clearly.

The reason it is a separate, deliberate check: the underlying individual reader
silently drops an unknown label and only complains if the *entire* selection
comes back empty. So a request for pops `A, B, Zzz` where `Zzz` does not exist
would otherwise quietly proceed with just `A, B`. This validation throws an
`std::invalid_argument` naming the missing label instead.

### Deriving the axes

After reading, the function establishes two counts and a name map:

- `P` — the number of populations (the P axis).
- `M` — the number of SNPs (the SNP axis).
- `pop_labels` — the P labels in P-axis index order, i.e. the map from a
  population index back to its name.

It guards against an empty selection (`P <= 0`) or no SNPs (`M <= 0`), and
against the `.snp` table and the genotype tile disagreeing on how many SNPs
exist (the two SNP axes must line up).

---

## 5. Per-sample ploidy detection

Each sample is either pseudo-haploid (effectively one allele observed, encoded
as ploidy `1`) or diploid (ploidy `2`). These two values mirror the parity
pseudo-haploid adjustment. The caller chooses one of three modes:

| Mode | What happens |
|---|---|
| `Auto` | Ploidy is detected per sample. The detection runs on the GPU: the backend scans the uploaded packed genotypes itself and derives each sample's ploidy. The CPU reference backend runs the equivalent host scan. No ploidy vector is filled on the host in this mode. |
| `PseudoHaploid` | Every sample is forced to ploidy `1` (a uniform vector). No detection runs. |
| `Diploid` | Every sample is forced to ploidy `2` (a uniform vector). No detection runs. |

For `Auto`, detection is driven from inside the decode by a flag on the decode
view, so the ploidy is derived on the same device that will use it rather than
being computed eagerly on the host.

### The reporting counts

The result reports how many samples ended up pseudo-haploid versus diploid. For
the explicit modes the vector is already in hand. For `Auto`, these counts come
from the same on-device detection pass (one cheap pass over the samples), not
from a separate host scan. This count vector is kept in its own local variable
so it can never disturb the decode view's ploidy pointer, which stays null in
`Auto` mode precisely so the in-decode detection flag is what drives the work.

---

## 6. The `maxmiss` filter (population-axis coverage)

The `maxmiss` value is a per-SNP coverage test measured over populations. For a
given SNP, it looks at the fraction of selected populations that have *no data*
at that SNP (a sample count of zero), and drops the SNP if that fraction exceeds
`maxmiss`.

- `maxmiss >= 1` keeps every SNP (the coverage test never fires).
- `maxmiss == 0` keeps only SNPs where *every* selected population has data —
  the global intersection.

This is separate from, and must not be confused with, the individual-axis
missingness filter. To honor the parity semantics[^at2], the individual-axis
missingness threshold is forced to its no-op value (`1.0`) during extract, so
only the population-axis `maxmiss` decides coverage. Letting both run would
double-filter and change the kept set.

Internally the same numeric knob (`geno_max_missing`) carries the `maxmiss`
value here, but it is applied with the population-axis meaning, and the
individual-axis predicate is disabled.

---

## 7. The SNP-tiled decode and filter (Phase A)

This is the load-bearing algorithm in the file. When a CUDA device is present,
the decode-and-filter stage runs on the GPU, and it processes the SNP axis in
tiles rather than all at once. On the GPU path this is the Phase-A metadata pass
of section 2a: it establishes the kept set and its metadata, and discards each
tile's decoded `Q`/`V`/`N`.

### Why tile

The decode allocates on the order of six full `P × M` double-precision buffers
before the f2 computation even begins (the decoded quantities plus their
compacted counterparts). At large `P × M` that is enough to exhaust GPU memory.
Tiling caps the peak footprint at `P × tile_M` — one tile's worth — instead of
`P × M`. The tile width is chosen from the free VRAM, the population count, and
the individual count; it is always a multiple of four so every tile boundary
stays aligned to the 2-bit genotype packing, and the final tile may be shorter.

### Hoisting ploidy to an explicit vector before the loop

Right before the tile loop, the per-sample ploidy is frozen into a single
explicit vector that *every* tile uses, and the in-decode auto-detection flag is
turned off.

This is required for correctness under tiling, not an optimization. Auto
detection scans a fixed prefix of the leading SNPs. If detection re-ran per
tile, a later tile would scan *its own* leading SNPs — a different set of SNPs —
and could assign a sample a different ploidy than the first tile did, producing
inconsistent allele-frequency quantities. Freezing one vector for all tiles
avoids that.

The frozen vector is exactly the one already computed for the reporting counts
in section 5, which was detected over the full-data prefix. So for `Auto` this
is bit-identical to running a single non-tiled on-device detection, and for the
explicit modes it is just the uniform vector that was already in place — nothing
actually moves.

### Why the tiled kept set is bit-identical to a single pass

For each tile the backend returns that tile's compacted `Q`/`V`/`N` plus the
kept-SNP metadata (chromosome, genetic position, physical position). In Phase A
the host keeps only the metadata: it appends each tile's `chromosome`, `genetic
position`, and `physical position` arrays in file order, adds the tile's kept
count to `M_kept`, and then frees that tile's device buffers — the tile's
`Q`/`V`/`N` is discarded with them. A full-size `P × M_kept` `Q`/`V`/`N` tensor
is never held resident; that was the OOM wall the two-phase design removed
(section 2a).

Appending each tile's kept metadata in tile order reconstructs exactly the same
kept set, in the same order, that a single non-tiled pass would have produced.
Three properties make this hold:

- each SNP's keep decision is independent of the others;
- the per-tile scan visits SNPs in monotone order;
- tiles are appended in file order.

Because the kept set, its file order, and its metadata are reproduced exactly,
the block partition and every downstream f2 value are reproduced bit-for-bit —
whether the f2 stage later re-decodes into a dense buffer (Resident) or per chunk
(streamed). Tiles that keep no SNPs are simply skipped. If, after all tiles,
nothing survived, the function throws — every SNP was filtered out and the
filters need relaxing.

### What runs on the device

The GPU path computes the keep decision and the compaction on-device: the
pooled minor-allele-frequency accumulation across populations, the shared
keep decision, and the separate population-axis `maxmiss` coverage test. The old
approach of copying the whole decoded tile back to the host and running a
per-SNP filter loop there is gone. In Phase A only the small kept-SNP metadata
crosses back to the host; the compacted `Q`/`V`/`N` stays on the device and is
released with the tile. The Phase-B feeds that actually consume `Q`/`V`/`N`
re-run this same decode (section 2a).

---

## 8. The CPU parity-oracle path

When no CUDA device is present, the same stage runs entirely on the host. This
path exists as the reference oracle that the GPU path is validated against; it
is intentionally the straightforward, un-tiled version.

It decodes the full tile, builds the keep mask (with the individual-axis
threshold forced to its no-op so only `maxmiss` governs coverage), then applies
the population-axis `maxmiss` loop by hand: for each still-kept SNP it counts the
populations with a zero sample count and drops the SNP if that fraction exceeds
`maxmiss`.

It then compacts `Q`, `V`, and `N` together with the parallel chromosome,
genetic-position, and physical-position arrays in lockstep, so the SNP axis of
the compacted data stays aligned with its metadata (which the block assignment
in the next stage depends on). Like the GPU path, it throws if nothing survived.

Both paths are contractually required to produce the identical kept set, the
identical kept order, the identical kept metadata, and the identical compacted
`Q`/`V`/`N` — the equivalence is enforced by comparing against the goldens.

---

## 9. Block assignment, f2 computation, and the result

### Assigning jackknife blocks

The kept SNPs are partitioned into blocks by `assign_blocks`, working over the
kept chromosome and genetic-position arrays. The kept physical-position array is
passed alongside for one specific fallback: when a dataset ships with no genetic
map (the genetic-position column is all zeros, common for data derived from VCF
or PLINK), blocks are formed by a fixed span of physical position instead. When a
real genetic map is present (as in the AADR data[^aadr]), the physical positions are
ignored and the partition is bit-identical to a genetic-map partition. The
function throws if the partition comes out empty.

### Computing the f2 blocks

How the f2 stage is fed depends on the path and, on the GPU path, on the tier
chosen by `resolve_output_tier` (section 2a):

- **GPU, Resident tier.** The dense `P × M_kept` host `Q`/`V`/`N` (rebuilt by the
  Phase-B re-decode) are wrapped as matrix views and handed to the f2 entry
  point along with the block partition and the precision policy.
- **GPU, HostRam/Disk tiers.** No dense buffer exists. The f2 entry point is
  called with null `Q`/`V`/`N` views plus a `RedecodeSource`, and the streamed
  engine re-decodes each per-chunk tile on-device. The f2 math is byte-for-byte
  the same as the Resident path (section 2a).
- **CPU reference path.** The compacted `Q`/`V`/`N` built by the host oracle
  (section 8) are wrapped as views and handed to the f2 entry point.

In every case the entry point returns the per-block f2 output along with the tier
it actually used.

### Building the result

Finally the f2 output is materialized to a host tensor and packaged into the
result. For the **HostRam** tier the streamed result already lives in a host
tensor inside the engine's sink, so it is **moved** out rather than copied via
`to_host()` — a copy would allocate a second full host tensor and, for a large
model, double host RAM and re-trigger the OOM (section 2a). The **Resident** and
**Disk** tiers still materialize via `to_host()` (from VRAM and the on-disk cache
respectively). The packaged result carries:

- the f2 tensor and the population labels;
- `n_snp_total` (the SNPs read, `M`) and `n_snp_kept` (the SNPs that survived
  filtering, `M_kept`);
- the pseudo-haploid and diploid sample counts from section 5;
- the precision tag actually used;
- the memory tier that was chosen, translated from the internal tier enum into
  the CUDA-free public mirror so a caller (or the CLI summary) can see whether a
  forced-tier override was honored;
- a status of `Ok`.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
[^aadr]: **AADR** — the Allen Ancient DNA Resource, the ancient-genome dataset steppe validates against. Mallick S, Micco A, Mah M, et al. *The Allen Ancient DNA Resource (AADR): a curated compendium of ancient human genomes.* Scientific Data 2024;11:182.
