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

The function is a straight-line pipeline of five stages. Each stage feeds the
next; there are no branches other than the GPU-versus-CPU split inside the
filter stage.

1. **Open and read.** Open the genotype file (which pins the on-disk format),
   read the individual/population table for the requested selection, read the
   SNP table, and read one canonical individual-major tile of packed genotypes.
2. **Decode, filter, compact.** Decode packed genotypes into per-population
   allele-frequency quantities, decide which SNPs to keep, and compact the kept
   columns into dense arrays. This stage has two implementations that produce
   bit-identical output — a GPU path and a CPU reference path.
3. **Assign blocks.** Partition the kept SNPs into jackknife blocks along the
   genome.
4. **Compute f2 blocks.** Run the GPU f2 computation over the kept, blocked
   data.
5. **Materialize the result.** Copy the f2 tensor to host memory and fill in the
   result struct (labels, SNP counts, ploidy counts, precision tag, memory
   tier).

The three quantities carried between the filter stage and the f2 stage are
named `Q`, `V`, and `N` throughout: a per-population allele-frequency-style
value, its paired variance term, and a per-population sample-count. They travel
together and must stay aligned on the SNP axis.

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

## 7. The SNP-tiled decode and filter

This is the load-bearing algorithm in the file. When a CUDA device is present,
the decode-filter-compact stage runs on the GPU, and it processes the SNP axis
in tiles rather than all at once.

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

### Why the tiled output is bit-identical to a single pass

For each tile the backend returns that tile's compacted `Q`/`V`/`N` plus the
kept SNP metadata (chromosome, genetic position, physical position). The host
appends each tile's arrays in file order and then frees that tile's device
buffers before the next iteration, so a full-size `P × M_kept` tensor is never
held resident — that would defeat the whole point and reintroduce the `O(P × M)`
footprint.

The compacted host layout is column-major `P × mk`: the value for population `i`
at kept SNP `d` sits at index `i + P·d`, so the P values for one SNP are
contiguous and consecutive SNPs are P apart. Appending each tile's arrays in
tile order therefore reconstructs exactly the same contiguous
`P × M_kept` layout a single non-tiled pass would have produced — byte for byte.
Three properties make this hold:

- each SNP's keep decision is independent of the others;
- the per-tile scan visits SNPs in monotone order;
- tiles are appended in file order.

Tiles that keep no SNPs are simply skipped. If, after all tiles, nothing
survived, the function throws — every SNP was filtered out and the filters need
relaxing.

### What runs on the device

The GPU path computes the keep decision and the compaction on-device: the
pooled minor-allele-frequency accumulation across populations, the shared
keep decision, and the separate population-axis `maxmiss` coverage test. The old
approach of copying the whole decoded tile back to the host and running a
per-SNP filter loop there is gone; only the small compacted `Q`/`V`/`N` and the
kept-SNP metadata cross back to the host.

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

The compacted `Q`, `V`, and `N` are wrapped as matrix views over the
`P × M_kept` data and handed to the unified, adaptive f2 entry point along with
the block partition and the precision policy. That call chooses the memory tier
(all-in-GPU-memory, spill to host RAM, or spill to disk) on its own and returns
the per-block f2 output.

### Building the result

Finally the f2 output is materialized to a host tensor and packaged into the
result:

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
