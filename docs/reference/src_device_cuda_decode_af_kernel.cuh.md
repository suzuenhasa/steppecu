# `decode_af_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/decode_af_kernel.cuh` declares one function:
`launch_decode_af`. That function is the entry point for the first GPU stage of
the pipeline — turning packed genotype data straight from a file into per-population
allele frequencies.

The header is deliberately thin. It declares the launch wrapper and nothing else.
The actual GPU kernel — the code that runs on the device, and the `<<<...>>>`
launch syntax that starts it — lives entirely in the matching `.cu` file
(`decode_af_kernel.cu`). Host-side orchestration code (the CUDA backend) calls
`launch_decode_af` as an ordinary function and never sees the kernel body. This
split keeps the GPU-launch machinery in one place and lets the rest of the code
treat the decode step as a plain function call.

The header does name one CUDA type, `cudaStream_t`, which is why it is private to
the device layer rather than part of the public, CUDA-free interface. Section 6
explains that boundary.

---

## 2. What the kernel computes

`launch_decode_af` decodes a tile of packed genotype records and reduces it into
allele-frequency statistics, in a single GPU pass.

The input is a packed genotype tile laid out **individual-major** and
**population-contiguous**: each individual is one fixed-size record of bytes, the
records for one population sit next to each other, and the populations follow one
after another. Inside each record, one SNP's genotype is stored in 2 bits (a packed
raw-value encoding), so several SNPs share a byte.

The work is split so that **one GPU thread owns exactly one `(population p, SNP s)`
output entry**. That thread walks over just the individuals belonging to population
`p` — a contiguous segment of the record axis — and for each individual it unpacks
SNP `s` out of that record's byte using the shared decode primitive. As it goes it
accumulates two integer running totals:

- **AC** — the number of reference-allele copies seen (the allele count).
- **AN** — the number of individuals that had a non-missing genotype at this SNP.

After the segment is fully scanned, a single double-precision divide converts those
integer totals into the three reported outputs (see section 3). Everything up to
that final divide is integer arithmetic; the divide and the outputs are native
double precision.

Adjacent threads (which handle adjacent SNPs) read adjacent bytes of the same
record, so the memory reads are **coalesced** along the SNP axis — the access
pattern the GPU is fastest at. This stage is limited by memory bandwidth, not by
arithmetic, which is why the layout is chosen to keep reads contiguous.

---

## 3. The Q/V/N output contract

The kernel produces three arrays, one value per `(population, SNP)` cell. They are
often referred to together as the **Q / V / N** contract:

| Output | Meaning |
|---|---|
| `Q` | The reference-allele frequency for this population at this SNP: `AC / (ploidy · AN)` — allele copies observed divided by allele copies possible. |
| `N` | The number of observed allele copies backing that frequency: `ploidy · AN`. This is the effective sample size / weight for the cell, used later when frequencies are combined. |
| `V` | A validity flag: `1` when at least one individual was non-missing (`AN > 0`), `0` otherwise. It marks cells where `Q` is meaningful versus cells that had no data. |

All three arrays are `P × M` and stored **column-major**: `P` is the number of
populations (the row axis), `M` is the number of SNPs (the column axis), and the
cell for population `i`, SNP `s` lives at flat index `i + P·s`. `M` is passed as a
64-bit value because a full panel can have far more SNPs than fit comfortably in a
32-bit count.

---

## 4. The `launch_decode_af` parameters

```cpp
void launch_decode_af(const std::uint8_t* d_packed,
                      std::size_t bytes_per_record,
                      const std::size_t* d_pop_offsets,
                      int P, long M, int ploidy,
                      const int* d_sample_ploidy,
                      double* d_Q, double* d_V, double* d_N,
                      cudaStream_t stream);
```

The `d_` prefix on a pointer means it points into GPU (device) memory, not host
memory.

| Parameter | What it is |
|---|---|
| `d_packed` | The packed genotype tile on the GPU: `n_individuals × bytes_per_record` bytes, laid out individual-major and population-contiguous (see section 2). |
| `bytes_per_record` | How many bytes each individual's record occupies. Used to step from one individual to the next inside `d_packed`. |
| `d_pop_offsets` | An array of `P + 1` boundaries marking where each population starts and ends along the individual axis. Population `p` owns the individuals from `d_pop_offsets[p]` up to `d_pop_offsets[p+1]`. This is what tells each thread which segment of individuals to reduce over. |
| `P` | The number of populations — the row axis of the outputs. |
| `M` | The number of SNPs — the column axis of the outputs. Passed as a 64-bit `long` because SNP counts can be large. |
| `ploidy` | The uniform, per-sample fallback ploidy used when `d_sample_ploidy` is null (typically 2 for diploid). See section 5. |
| `d_sample_ploidy` | Optional per-sample ploidy vector, or null. When provided it overrides the uniform `ploidy` sample by sample. See section 5. |
| `d_Q`, `d_V`, `d_N` | The three output arrays described in section 3 — `P × M`, column-major, on the GPU. |
| `stream` | The CUDA stream the launch runs on, so the caller controls ordering and overlap with other GPU work. |

---

## 5. Per-sample ploidy and pseudo-haploid handling

Some samples in a dataset are genuine diploids (ploidy 2) while others are treated
as **pseudo-haploid** (ploidy 1) — a common situation with low-coverage ancient
DNA, where only a single allele can be confidently called. The two kinds must
contribute to an allele frequency differently, and this kernel supports both.

There are two modes, chosen by whether `d_sample_ploidy` is null:

- **Per-sample mode (`d_sample_ploidy` is non-null).** This is a device array of
  length `n_individuals`, running in lockstep with the sample axis, holding each
  sample's ploidy (2 for diploid, 1 for pseudo-haploid). It is auto-detected from
  the data upstream[^at2]. Each non-missing sample
  is folded in using the `adjust_pseudohaploid` accumulation:
  `AC += code / (3 − ploidy)` and `N += ploidy`. For a diploid sample `3 − ploidy`
  is `1`, so its raw allele code adds in unchanged; for a pseudo-haploid sample
  `3 − ploidy` is `2`, so its contribution is halved. This reproduces the
  handling of mixed-ploidy panels exactly.

- **Uniform fallback mode (`d_sample_ploidy` is null).** Every sample is treated as
  having the single scalar `ploidy` passed in the call — the older all-diploid
  path. This path is **bit-identical** to what the code produced before per-sample
  ploidy existed, so turning the feature off changes nothing for datasets that were
  always uniform.

---

## 6. Why this header is private to the device layer

steppe has two kinds of internal boundaries. The public one — the compute backend
interface — is deliberately free of any CUDA types, so that code above it (the core
library, the command-line tool, the language bindings) can be built without a CUDA
toolchain in reach.

This header is on the other side of that line. It mentions `cudaStream_t`, a CUDA
type, so anything that includes it must be compiled as CUDA. That is why the file
belongs to the private device seam: it is the internal handoff between the CUDA
backend and the decode kernel's translation unit, not part of the CUDA-free public
surface. Keeping the declaration here — rather than in a public header — is what
lets the rest of the project stay CUDA-free while the device layer alone deals with
streams and kernel launches.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
