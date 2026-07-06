# `decode_keep_autosomes.hpp` reference

## 1. Purpose

`src/core/stats/decode_keep_autosomes.hpp` declares the two small front-end helpers
that the genotype-path tools call right after they have loaded a slab of genotypes
into memory, and right before they start computing statistics. Between those two
moments every one of those tools has to do the same two boring-but-fiddly jobs:

1. **Describe** the loaded slab to the decoder — package up the raw bytes, the
   sample layout, and the ploidy into the little struct the decode kernels read.
2. **Decode and trim** — turn the packed genotype codes into per-population allele
   frequencies, and throw away the SNPs that don't sit on an autosome.

"Decode" here means converting the compact on-disk genotype codes into allele
frequencies: for each population at each SNP, what fraction of that population's
copies carry the reference allele. "Autosome" means an ordinary numbered chromosome
(1 through 22) as opposed to the sex chromosomes (X, Y) or mitochondrial DNA;
f-statistics are conventionally computed on the autosomes only[^at2], so the sex
chromosomes and everything else get dropped before any statistic is formed.

The reason these two jobs live in one shared header — instead of being retyped
inside each tool — is drift. steppe's D-statistics tool, its qpfstats tool, and its
DATES tool all need this exact front-end, and if each kept its own copy the copies
would slowly diverge: one would fix an off-by-one in the keep loop, another would
change how ploidy is set, and quietly the three tools would stop agreeing about
which SNPs they used. Pulling the code into one place makes that impossible — there
is a single source of the decode-and-keep behavior, so the tools cannot disagree.
The sharing is deliberately layered:

- **D-statistics and qpfstats** share the *whole* thing — they both call
  `decode_and_keep_autosomes` and get an identical decode plus autosome-keep.
- **DATES** shares only the *first* job — it calls `make_decode_tile_view` to build
  the same tile description, then runs its own decode path, because dating needs the
  data organized differently downstream.

This header only *declares* the helpers; the actual code lives in the matching
`decode_keep_autosomes.cpp`, whose `// Reference:` pointer points back here so the
implementation and this document stay together rather than each drifting on its own.

This file sits at what the codebase calls the **CUDA-free seam**: it is plain C++
that any host compiler can build, with no CUDA in sight. It talks to the GPU only
through the abstract `ComputeBackend` interface, so the tools above it never have to
be CUDA-aware to get GPU-decoded data.

---

## 2. Describing the tile to the decoder — `make_decode_tile_view`

```
make_decode_tile_view(const io::GenotypeTile& tile,
                      const std::vector<int>& sample_ploidy, int P) -> DecodeTileView
```

A `GenotypeTile` is steppe's in-memory slab of loaded genotypes — the packed bytes,
how many SNPs and individuals it covers, and where each population's individuals
start. A `DecodeTileView` is the lighter, pointer-only struct the decode kernels
actually read. This function is the wiring between the two: it copies the tile's
fields and pointers across into a fresh view and hands it back.

Concretely it fills the view with:

| View field | Where it comes from | What it is |
|---|---|---|
| `packed` | `tile.packed.data()` | pointer to the raw packed genotype bytes |
| `bytes_per_record` | `tile.bytes_per_record` | stride of one SNP's record |
| `n_snp` | `tile.n_snp` | number of SNPs in the slab |
| `n_individuals` | `tile.n_individuals` | number of individuals in the slab |
| `pop_offsets` | `tile.pop_offsets.data()` | where each population begins |
| `n_pop` | `P` | number of populations |
| `sample_ploidy` | `sample_ploidy.data()` | per-sample ploidy vector |
| `ploidy` | `core::kPloidyDiploid` (= 2) | the default whole-tile ploidy |

Two details are worth calling out. First, the view does **not** own any of this
data — it holds bare pointers into the caller's `tile` and into the caller's
`sample_ploidy` vector. That is why the header carries an explicit lifetime note:
the view aliases `sample_ploidy`, so the caller must keep that vector alive for as
long as the view is used. Hand the view to a decoder after the ploidy vector has
gone out of scope and you are reading freed memory.

Second, `ploidy` is set to the diploid default of 2, meaning "assume two copies per
individual unless told otherwise." Ancient-DNA samples are often *pseudo-haploid*
(effectively one observed copy), and the actual per-sample ploidy travels in the
`sample_ploidy` vector; the decoder reconciles the two. This function's job is only
to set the sensible default and point at the vector — it does not itself decide any
sample's ploidy.

---

## 3. The decode-and-keep result — `DecodeKeepResult`

```
struct DecodeKeepResult {
    bool resident;
    device::DeviceDecodeResult ddr;
    std::vector<double> Qk, Vk;
    std::vector<int>    chrom_kept;
    std::vector<double> genpos_kept, physpos_kept;
};
```

This is the bundle `decode_and_keep_autosomes` returns, and it is shaped to carry
*either* of two outcomes depending on whether a GPU was present:

- **`resident`** is the switch. When true, the decoded allele frequencies were left
  living on the GPU, and they are reachable through **`ddr`** (a
  `DeviceDecodeResult`, essentially a handle to device-side buffers). In this case
  `Qk` and `Vk` are left empty — the numbers never came back to the host, which is
  exactly the point of keeping them resident: no needless copy off the card.
- When `resident` is false (no GPU — the CPU test path), the decoded frequencies
  come back on the host in **`Qk`** and **`Vk`**: the kept allele frequencies and
  their matching validity/weight values, laid out one population-block per kept SNP.

Either way the three small **position vectors** — `chrom_kept`, `genpos_kept`,
`physpos_kept` — always come back on the host. These are the chromosome number,
genetic-map position, and physical base-pair position of each SNP that survived the
autosome filter. They are cheap host-side metadata that downstream code (jackknife
block assignment, dating) needs regardless of where the big frequency arrays ended
up, so they are always materialized.

---

## 4. Decode plus autosome keep — `decode_and_keep_autosomes`

```
decode_and_keep_autosomes(ComputeBackend& be, const io::GenotypeTile& tile,
                          const io::SnpTable& snptab, int P, long M) -> DecodeKeepResult
```

This is the shared workhorse. It takes the loaded `tile`, the SNP metadata table
`snptab` (chromosome, genetic and physical positions for each of the `M` SNPs), the
population count `P`, and the compute backend `be`, and returns a filled
`DecodeKeepResult`.

It starts by building an all-diploid `sample_ploidy` vector (one entry of
`kPloidyDiploid` per individual) and calling `make_decode_tile_view` (section 2) to
describe the tile. Then it branches on whether the backend actually has a GPU:

**Resident path (a GPU is present).** It calls the backend's
`decode_af_compact_autosome`, handing it the view plus spans over the SNP table's
chromosome, genetic-position, and physical-position columns, and the autosome
bounds `kAutosomeChromMin` (1) and `kAutosomeChromMax` (22). The GPU does the whole
job in one shot: it decodes the allele frequencies *and* applies the autosome
filter on the device, so the frequencies never leave the card. The result comes
back as a `DeviceDecodeResult` stored in `out.ddr`, `resident` is set true, and the
three host-side position vectors are copied out of the device result for downstream
use.

**Host path (no GPU — the CPU parity oracle).** With no device to lean on, it calls
`be.decode_af(view)` to decode every SNP on the host, then does the autosome keep
itself in a plain loop. It walks all `M` SNPs and, for each one whose chromosome
falls in `[1, 22]`, copies that SNP's `P` allele frequencies from the decoder's `q`
array into `Qk` and the matching `P` validity values from `v` into `Vk`, and appends
the SNP's chromosome and two positions to the kept-position vectors. SNPs on any
other chromosome are skipped with a `continue`, so they contribute nothing — exactly
matching what the device path keeps. The per-SNP source offset is `P * s` because the
decoder lays its output out SNP-major (all `P` populations for one SNP sit
contiguously).

Both branches use the `idx(...)` helper when computing array indices; that is
steppe's checked cast from signed `long`/`int` counts to the unsigned `size_t` that
vector indexing wants, so a negative or overflowing count is caught rather than
silently wrapping into a wild offset.

The important invariant is that the two branches produce the **same kept set** —
same SNPs, same order, same per-population frequencies — just in different places
(device buffers versus host vectors). That is what lets the CPU path serve as a
trusted parity oracle for the GPU path: run the same tile through both and the kept
data must agree. Keeping the branch logic in this one shared function is what makes
that guarantee hold for D-statistics and qpfstats alike, rather than depending on
two hand-copied loops staying in sync.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity, and the source of the autosomes-only convention followed here. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
