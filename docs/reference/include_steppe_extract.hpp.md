# `extract.hpp` reference

## 1. Purpose

`include/steppe/extract.hpp` declares the public, library-level entry point for
turning a genotype dataset into f2 statistics: the function `run_extract_f2`,
plus the small result type and enums it needs. It is the callable counterpart of
the `steppe extract-f2` command-line tool. Where the command-line tool prints
progress and errors to the terminal and returns a process exit code, this
function takes plain arguments, returns a value, and reports any failure by
throwing an exception. That difference is the whole reason the function exists.

The command-line command cannot be called from another program or from Python:
it writes to standard output and standard error and communicates success or
failure only through an exit code. Language bindings need the opposite — a
function that returns a real object on success and throws on failure, and that
never prints anything on its own. `run_extract_f2` is that function. The
command-line command is now just a thin wrapper around it, so both paths run the
exact same computation.

The header is deliberately free of any CUDA (GPU) code. It uses only the C++
standard library and forward-declares the two GPU-adjacent types it references
(`device::Resources` and `io::PopSelection`) rather than including their real
headers. That keeps the header lightweight enough to include from the public API
surface and the language bindings without dragging the GPU code along with it.
The bindings reach the GPU *only* through this function and the resources handle,
which is what lets the binding layer stay CUDA-free.

Under the hood the function runs the same pipeline the command-line tool does:
decode the genotypes, apply the quality-control filters, partition SNPs into
jackknife blocks, compute the per-block f2 values on the GPU, and copy the result
back to host memory. That math was lifted verbatim out of the command-line
command — it is byte-identical, and the reference result files (the "goldens")
were left untouched — so a plain extract through this function reproduces the
same numbers ADMIXTOOLS 2 produces.

---

## 2. The parity pins

Four behaviors are carried over exactly from the command-line command so that a
bare extract reproduces the ADMIXTOOLS 2 reference result. These are fixed
conventions, not tunable options.

1. **Population-axis order.** The populations are read from the individual
   (`.ind`) file using the explicit selection you pass in, then sorted in
   ascending order by label. That sorted order is the order of the population
   axis in the result, and it matches the order ADMIXTOOLS 2 uses (the order of
   a `pops.txt` file).

2. **Per-sample ploidy auto-detection.** Unless you force a ploidy, each sample's
   ploidy is detected individually — this is the same pseudo-haploid adjustment
   ADMIXTOOLS 2 performs. It is the default for extract.

3. **Coverage test on the population axis.** ADMIXTOOLS 2's `maxmiss` missingness
   test is a *population*-axis coverage test, applied as its own per-SNP check,
   not the *sample*-axis predicate. To reproduce that, the sample-axis
   missing-data filter is forced to its do-nothing setting while the
   population-axis coverage test does the real work.

4. **Autosomes only.** Keeping only chromosomes 1 through 22 is controlled by a
   flag on the filter configuration. ADMIXTOOLS 2's `extract_f2` defaults to
   autosomes-only, so that is the matching setting.

---

## 3. ExtractPloidy

`ExtractPloidy` is the ploidy convention used when decoding the genotypes. It is
named here — rather than reusing the command-line tool's internal ploidy enum —
so the public extract surface does not have to pull in the command-line tool's
run-configuration types.

| Value | Meaning |
|---|---|
| `Auto` | **The default.** Detect each sample's ploidy individually, using the same pseudo-haploid adjustment ADMIXTOOLS 2 uses. This is the `extract-f2` default. |
| `PseudoHaploid` | Force every sample to pseudo-haploid — a single uniform per-sample setting instead of per-sample detection. |
| `Diploid` | Force every sample to diploid — again a single uniform per-sample setting. |

---

## 4. ExtractTier

`ExtractTier` reports which memory tier the f2 computation actually used. steppe
can keep the f2 result entirely in GPU memory, spill it to host RAM, or spill it
to disk, depending on how large the problem is. This enum is a plain mirror of
the internal tier type, named here so the public surface does not have to include
the internal tier-selection header.

| Value | Meaning |
|---|---|
| `Resident` | The result stayed in GPU memory. This is the fast path, and the one chosen when the number of populations is small enough to fit. |
| `HostRam` | The result was spilled to host RAM, using the input-streaming path for larger problems. |
| `Disk` | The result was spilled to disk, again via the input-streaming path. |

This value is for observability only. Which tier runs changes *where* the result
lands, never its numbers — the bits are identical across all three tiers. It is
echoed back in the command-line summary and the run metadata so that a forced
tier choice (a `--tier` flag or the `STEPPE_FORCE_TIER` environment variable) is
visibly honored.

---

## 5. F2ExtractResult

`F2ExtractResult` is the value `run_extract_f2` returns on success. It bundles the
computed f2 tensor together with the labels and provenance a caller needs to make
sense of it. It is returned by value; the binding layer wraps it in a handle or
hands it to the directory writer.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `f2` | `F2BlockTensor` | — | The f2 tensor in host memory: the per-block f2 values and their paired variances. These are the real, double-precision values, and they are double-precision in every precision mode. |
| `pop_labels` | `vector<string>` | — | The population labels in population-axis index order (the same ascending-sorted order as a `pops.txt` file). The compute engine works purely with indices and has no names of its own, so this is the name-to-index map that goes with the tensor. |
| `n_snp_total` | `long` | `0` | How many SNPs were read from the `.snp` file, before any filtering. |
| `n_snp_kept` | `long` | `0` | How many SNPs survived the filters and were actually used. |
| `n_pseudo_haploid` | `size_t` | `0` | How many samples were decoded as pseudo-haploid. Observability only. |
| `n_diploid` | `size_t` | `0` | How many samples were decoded as diploid. Observability only. |
| `precision_tag` | `Precision::Kind` | `EmulatedFp64` | Which precision mode the computation actually engaged. |
| `tier` | `ExtractTier` | `Resident` | Which memory tier the computation used (see section 4). Observability only. |
| `status` | `Status` | `Ok` | Always `Ok` on a returned result. A genuine failure throws instead of returning, so the status never carries a fault. |

A note on `status`: extract has no per-row domain outcome that would ride on this
field. Anything that goes wrong — no usable device, an unknown population name, a
missing or unreadable file, or every SNP being filtered out — is a hard fault, and
a hard fault is thrown, never returned as a partial result with a non-`Ok`
status. So a value you receive back is always a complete, successful result.

---

## 6. run_extract_f2

`run_extract_f2` is the single public function this header exists to declare. It
performs the genotype-to-f2 extraction over the genotype triple — the `.geno`,
`.snp`, and `.ind` files (in EIGENSTRAT or TGENO format) — and returns an
`F2ExtractResult`. It routes the decode and the f2 computation through the first
GPU in the resources handle (GPU-first and device-resident; multi-GPU support is
parked, so a single device is used).

### Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `geno` | `const string&` | Path to the genotype (`.geno`) file. |
| `snp` | `const string&` | Path to the SNP (`.snp`) file. |
| `ind` | `const string&` | Path to the individual (`.ind`) file. |
| `pops` | `const io::PopSelection&` | Which populations to use. The named-subset form is what the bindings use, and it is what the population axis is partitioned from; the top-K and minimum-count selection forms are also supported, identical to the command-line tool. The population axis is this selection sorted in ascending order by label. |
| `filter` | `const FilterConfig&` | The on-the-fly quality-control configuration: the population-axis coverage test (through the missing-data field), autosomes-only, minor-allele-frequency, drop-monomorphic, transversions-only, and the SNP include/exclude lists. |
| `precision` | `const Precision&` | Which arithmetic the f2 matrix multiplications run in. The default is emulated double precision at 40 mantissa bits. |
| `blgsize_morgans` | `double` | The jackknife block size, in Morgans. The ADMIXTOOLS 2 default is `0.05`. |
| `ploidy` | `ExtractPloidy` | The ploidy convention (see section 3). `Auto` by default, which is the ADMIXTOOLS 2 per-sample pseudo-haploid detection. |
| `resources` | `device::Resources&` | The GPU resources handle. The decode and the f2 computation run on this handle's first GPU. |

### Failure behavior

The function is marked `[[nodiscard]]`, so its return value must not be ignored.

On any fault it throws a standard exception — `std::runtime_error` or
`std::invalid_argument` — rather than returning. The faults that throw are: no
usable device, an unknown population name in an explicit selection, a missing or
unreadable input file, an empty selection, and every SNP being filtered out. It
never returns a partial or half-built result; a returned `F2ExtractResult` is
always complete and successful.
