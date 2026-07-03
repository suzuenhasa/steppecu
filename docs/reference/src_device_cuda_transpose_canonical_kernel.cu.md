# `transpose_canonical_kernel.cu` reference

## 1. Purpose

`src/device/cuda/transpose_canonical_kernel.cu` is the GPU primitive that turns a
genotype source stored *SNP-major* into the *individual-major* layout the rest of
the pipeline expects. It does three things in a single pass: it **transposes** the
two axes (SNP-by-individual becomes individual-by-SNP), it **gathers** only the
selected individuals and reorders them, and it **re-encodes** each genotype value
into the canonical convention.

Some genotype file formats (PACKEDANCESTRYMAP and the older GENO layout) store one
record per SNP, with every individual's genotype for that SNP interleaved inside the
SNP's bytes. Everything downstream in steppe — the allele-frequency decode and the
ploidy detection — instead reads one record per individual, with that individual's
SNPs laid out across its bytes. This file is the bridge. After it runs, the gathered
data sits in exactly the individual-major packing the downstream stages already
consume, so nothing after this point has to change: only the read-time layout is
transposed.

The file owns two symbols:

- **`transpose_to_canonical_kernel`** — the GPU kernel that does the work, one thread
  per output byte.
- **`launch_transpose_to_canonical`** — a thin launch wrapper so that no `<<<>>>`
  launch syntax appears in ordinary host code, following the same pattern as the
  other device kernels in this directory.

This is a CUDA translation unit and is private to the device layer of steppe. It is
not part of the public, CUDA-free interface.

---

## 2. The two byte layouts

Both the source and the output pack genotypes as **2-bit codes, four to a byte,
most-significant-bits first**. A code is `0`, `1`, or `2` for the number of reference
alleles, and `3` for missing. The same bit-extraction helper (`core::genotype_code`)
that the allele-frequency decode uses is reused here, so the transpose can never
disagree with the decoder about how a byte is unpacked.

**Source (SNP-major).** Record `s` holds SNP `s`. Within that record, individual `i`
lives at byte `i/4`, position `i%4`. The distance in bytes from one SNP record to the
next is `src_bytes_per_record` (see section 7 — this stride can be larger than the
number of individuals strictly requires).

**Output (individual-major, "canonical").** Record `g` holds gathered individual `g`
and occupies the byte range `[g * out_bytes_per_record, (g+1) * out_bytes_per_record)`.
Within that record, SNP `s` lives at byte `s/4`, position `s%4`. `out_bytes_per_record`
is `ceil(n_snp / 4)` — just enough bytes to hold all the SNPs the tile covers.

Because both layouts share the same code convention and the same
most-significant-bits-first bit order, the re-encoding step for these formats is the
identity (section 6): the only real work is moving each code from its source position
to its transposed output position.

---

## 3. The gather: selecting and reordering individuals

The transpose is not a straight axis swap of every individual — it simultaneously
picks out a chosen subset of individuals and puts them in a specific order. That
mapping is supplied in `sel_rows`: output column `g` reads from source individual row
`sel_rows[g]`. The array holds the selected individuals grouped and ordered by
population, so the output tile comes out already selected and population-contiguous.

Doing the selection here is natural because the source is SNP-major: every SNP record
interleaves *all* individuals, so choosing "just these individuals, in this order"
has to happen at the moment the data is read out. The transpose is **output-driven** —
the kernel walks over output bytes and pulls each one's data from wherever it lives in
the source — which is exactly what makes the gather fall out for free. It mirrors the
per-individual gather the individual-major reader already does, just with the two axes
swapped.

`sel_rows[g]` is read once per output byte, because the four SNPs packed into one
output byte all belong to the same individual and therefore share the same source row.

---

## 4. Inside the kernel: one thread per output byte

The kernel assigns **one thread to each output byte**. A flat thread id `tid` is split
into an output individual `g = tid / out_bytes_per_record` and an output byte index
`b = tid % out_bytes_per_record`. That byte holds up to four SNPs — SNP numbers
`4b, 4b+1, 4b+2, 4b+3`.

For each of those SNPs `s` the thread:

1. Finds the source row for this individual, `src_row = sel_rows[g]`, and from it the
   source byte `src_row / 4` and in-byte position `src_row % 4`.
2. Reads the source byte at SNP record `s`: `snp_major[s * src_bytes_per_record + src_row/4]`.
3. Unpacks the 2-bit code at that position with `core::genotype_code`.
4. Applies the encoding map (section 6) to get the canonical code.
5. Packs that code into the output byte at the right position, most-significant-bits
   first.

**The packing math.** SNP `4b` goes into bits 7–6, `4b+1` into bits 5–4, `4b+2` into
bits 3–2, `4b+3` into bits 1–0. For the `k`-th SNP in the byte (`k` from 0 to 3) the
left shift is `(4 - 1 - k) * 2`, giving the sequence 6, 4, 2, 0. The codes are OR-ed
together into one byte and written with a single store.

**The partial last byte.** When the tile's SNP count isn't a multiple of four, the
final output byte is only partly used. Any SNP position `s >= n_snp` contributes zero
bits — the loop simply breaks — leaving an all-zero tail. That is precisely the tail
the host-side packer produces for the same data, so the two agree bit-for-bit.

**Race-free by construction.** Every output byte has exactly one writing thread, and
each thread writes exactly one byte with a plain store. No two threads ever touch the
same byte, so there are no atomics and no partial-bit OR-ing across threads. The
distinct pairs `(g, b)` own distinct output bytes.

---

## 5. Grid-stride coverage and the grid-size clamp

Ideally the launch would use exactly one thread per output byte. But the total number
of output bytes (`n_individuals * out_bytes_per_record`) can exceed the largest grid
dimension a launch is allowed to request. To stay correct at any size, two mechanisms
work together:

- **Grid-stride loop.** Inside the kernel, each thread doesn't stop after one byte — it
  strides forward by the total number of threads (`gridDim.x * blockDim.x`) and keeps
  going until it has covered every output byte. Coverage is therefore independent of how
  large the launch grid actually is.
- **Clamped grid extent.** The launch wrapper computes the ideal grid size, then caps it
  at the maximum allowed grid dimension (`core::kMaxGridX`). Whatever the cap forces the
  grid down to, the grid-stride loop still visits every byte.

This clamp is a deliberate safety net, and it replaced a weaker guard. The kernel
previously relied on a debug assertion that the grid fit in one dimension — but that
assertion compiles out under the release build, which is the build that actually runs
the pipeline. Without the clamp, an over-large byte count would silently truncate when
cast to the unsigned grid type and the kernel would quietly under-cover its output,
leaving part of the tile unwritten. The clamp makes correctness independent of whether
assertions are enabled.

The block size is fixed by the named constant `kTransposeBlock = 256` (threads per
block). The wrapper also returns immediately if the tile is empty — zero individuals or
zero SNPs means there is nothing to pack.

---

## 6. The encoding map

`apply_encoding` converts a source 2-bit code into the canonical code before it is
packed. For the formats this file handles today (PACKEDANCESTRYMAP and GENO), the
source already uses the canonical raw-value convention (`0`/`1`/`2` reference-allele
copies, `3` missing) and the same most-significant-bits-first bit order, so the map is
the **identity** — the native code *is* the canonical code. This is the only value of
the `TransposeEncoding` enum at present (`Identity`).

The map is written as a device-side `switch` rather than as a lookup table uploaded
from the host. That choice matters for extension: when later formats need a real
remap (for example ASCII EIGENSTRAT or PLINK), a new case is added at this one site
and the transpose body does not change. The encoding is the single knob that lets one
kernel serve multiple source formats.

---

## 7. Padding safety (the record-length floor)

The GENO layout floors each SNP record's byte length at `max(48, ceil(n_ind/4))`.
For a dataset with few individuals, that floor means each SNP record carries **padding
bytes** beyond the bytes the real individuals occupy. The stride passed in as
`src_bytes_per_record` reflects this floored length, so it can be larger than
`ceil(n_ind/4)`.

Those padding bytes are never mistaken for genotype data. The kernel only ever reads
byte `src_row / 4`, and every `src_row` comes from `sel_rows`, where each value is a
genuine individual index strictly less than the individual count. The kernel never
walks the source by full record strides looking for individuals, so a padding byte is
never read as a phantom individual. The record stride is used only to jump from one
SNP record to the next.

---

## 8. Why the result is bit-for-bit correct

This transpose is made of integer and bit operations only: the 2-bit unpack, the
identity re-encoding, and the most-significant-bits-first re-pack. There is no
floating-point arithmetic, no summation whose order could matter, and no reduction.
As a result the device output is identical, bit for bit, to two independent
references:

- the CPU reference implementation's host-loop version of the same transpose, and
- for PACKEDANCESTRYMAP, the individual-major tile produced by reading the same data
  through the individual-major (TGENO) path.

Because there is no floating-point math here, the emulated-double-precision policy that
governs the matrix-multiply stages elsewhere in steppe simply does not apply — there is
no precision knob to set, and nothing to make approximate. The correctness of this pass
rests entirely on the layout and bit math being exact, which the single-writer-per-byte
design and the shared unpack helper guarantee.
