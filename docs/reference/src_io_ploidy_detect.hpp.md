# `ploidy_detect.hpp` reference

## 1. Purpose

`src/io/ploidy_detect.hpp` declares one small function that classifies each
sample in a genotype tile as either **diploid** (ploidy 2) or **pseudo-haploid**
(ploidy 1). This is the step that reproduces ADMIXTOOLS 2's automatic
pseudo-haploid handling (its `adjust_pseudohaploid = TRUE` behavior).

Why this matters: pseudo-haploid samples — common in ancient DNA, where a single
read is used to represent a genotype — use a different per-sample counting
convention than true diploid samples in the downstream f2 statistic. Getting the
ploidy label right per sample is what makes steppe's f2 estimates match
ADMIXTOOLS 2 on datasets that mix the two.

The whole header is a single function declaration plus the reasoning behind it.
It sits at the front of the decode path: after the reader has gathered the
selected samples into a tile of packed genotype bytes, but before those bytes are
turned into allele frequencies. The result — a ploidy label per sample — is
carried alongside the tile so the decode uses the correct per-sample convention.

---

## 2. The detection rule

The rule is deliberately simple and is copied exactly from ADMIXTOOLS 2 (verified
against admixtools version 2.0.10, its `cpp_readgeno.cpp` ploidy routines):

1. Start every sample at ploidy **1** (pseudo-haploid). That is the default
   assumption.
2. Scan the first several hundred SNPs of that sample (see section 4 for the
   exact window).
3. The moment a **heterozygous** call is seen — a genotype that carries one copy
   of each allele — bump that sample to ploidy **2** (diploid) and stop scanning
   it.
4. If no heterozygous call turns up anywhere in the window, the sample stays at
   ploidy 1 (pseudo-haploid).

The logic behind it: a genome that is genuinely haploid can never be
heterozygous, because it only has one allele at each position. So a single
heterozygous call is proof that the sample is diploid. The absence of any
heterozygous call across a large window is taken as evidence that the sample is
pseudo-haploid.

### What counts as "heterozygous"

Genotypes arrive as small 2-bit codes. The code that means heterozygous is the
value **1** (one copy of the reference allele). This value has a name,
`kHetCode`, defined in the format header (`eigenstrat_format.hpp`) so the same
definition is shared everywhere it is needed. The other codes are: 0 and 2 for
the two homozygous states, and 3 for a missing genotype.

### Edge cases that stay pseudo-haploid

A sample whose window is entirely missing, or entirely homozygous, never triggers
the heterozygous test, so it stays at the default ploidy 1. That is the intended
behavior — with no positive evidence of diploidy, the pseudo-haploid default
holds.

---

## 3. Per-sample classification

The classification is done **one sample at a time**, and the result is one ploidy
label per sample.

This is the key improvement over an older approach that tagged ploidy per
*population*. Real ancient-DNA populations can be mixed: the same population label
can cover both a genuinely diploid sample and a pseudo-haploid one. Named examples
where this actually happens include Turkey_N, Serbia, Yamnaya, and Karitiana. A
per-population tag cannot represent that mixture and would mislabel some samples;
the per-sample rule classifies each individual correctly regardless of what
population it belongs to.

The returned labels are laid out **parallel to the tile's gathered sample axis**:
entry `g` in the returned vector is the ploidy of the `g`-th gathered sample in
the tile, in the same order the tile stores them. The vector's length equals the
tile's sample count.

---

## 4. The detection window

Detection scans a bounded prefix of each sample's SNPs rather than the whole
genome.

- **The window length** is `min(kPloidyDetectSnps, tile.n_snp)` SNPs.
- `kPloidyDetectSnps` is **1000** (defined in `eigenstrat_format.hpp`). This
  matches ADMIXTOOLS 2's `ntest` default of 1000 SNPs.
- When the tile happens to cover fewer than 1000 SNPs, the window is simply the
  whole record. ADMIXTOOLS 2 behaves the same way — its window caps at however
  many SNPs are actually available.

### It scans the tile, not the disk

Detection reads the **exact same packed bytes** that the decode will later read.
The tile already holds each gathered sample's leading packed SNP bytes in memory,
so scanning for a heterozygous call costs **no extra input/output** — there is no
second pass over the file. It walks the first `window` SNPs of each sample by
unpacking the 2-bit code at byte `s / 4`, position `s % 4` (the standard
most-significant-bit-first code layout), using the shared `code_in_byte`
primitive from the format header.

---

## 5. `detect_sample_ploidy` — the contract

```cpp
[[nodiscard]] std::vector<int> detect_sample_ploidy(const GenotypeTile& tile);
```

- **Input:** a `GenotypeTile` — the packed, gathered genotype bytes plus the
  layout describing them.
- **Output:** a `std::vector<int>` of length `tile.n_individuals`, parallel to
  the gathered sample axis. Each entry is `2` (diploid) if that sample had a
  heterozygous call in its window, or `1` (pseudo-haploid) otherwise.
- **Does not mutate the tile.** It reads the tile's bytes and returns a fresh
  vector; the tile is passed by const reference and is left untouched.
- **Pure function of the tile bytes.** The same tile always yields the same
  labels, with no hidden state, no randomness, and no side effects. It is safe to
  call once per tile.
- **`[[nodiscard]]`** marks the return value as the whole point of the call —
  ignoring it is almost certainly a mistake, and the compiler will warn.

The two ploidy values, 1 (pseudo-haploid) and 2 (diploid), are the only values
ADMIXTOOLS 2 uses, so they are the only values this function returns.

---

## 6. Layering and dependencies

This is a leaf header in the input/output (`io`) layer. It is deliberately kept
lightweight:

- **Pure host C++20.** No CUDA, and no dependency on the core or device layers.
- It uses only the `io` format primitives — the `code_in_byte` bit extractor and
  the `kHetCode` constant from `eigenstrat_format.hpp` — plus the `GenotypeTile`
  struct it takes as input.

Keeping detection here, on plain host bytes with no GPU dependency, means the
reader front-end can classify ploidy without pulling in any of the GPU code. The
same rule is mirrored on the device side (there is an equivalent GPU kernel and a
CPU-backend version), and all of them scan the same window with the same
heterozygous test so they agree by construction.
