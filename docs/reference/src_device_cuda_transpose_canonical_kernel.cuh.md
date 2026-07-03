# `transpose_canonical_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/transpose_canonical_kernel.cuh` declares a single GPU launch
helper, `launch_transpose_to_canonical`, plus the small enum that configures it.
It is the entry point for one focused job in the genotype-file readers: take
genotype bytes that are laid out **one record per SNP** and rewrite them into the
**one record per individual** layout the rest of steppe expects, while at the same
time picking out and reordering the individuals actually wanted and translating
each genotype value into the internal ("canonical") convention.

The header is deliberately narrow. It contains only the enum, one function
declaration, and their documentation. The actual kernel — the GPU code and its
`<<<>>>` launch — lives in the matching `.cu` file. Everything a caller needs to
invoke the primitive correctly is captured here in the declaration and its contract.

---

## 2. The two byte layouts

The whole primitive exists to convert between two ways of packing the same
genotype matrix. In both layouts a single genotype is a **2-bit code**: values
`0`, `1`, `2` count how many copies of the reference allele a sample carries, and
`3` means the genotype is missing. Four codes pack into one byte, and the bit
order is **MSB-first** — the code at logical position 0 sits in the most
significant two bits of the byte, position 3 in the least significant.

**Source layout — SNP-major.** This is how the on-disk PACKEDANCESTRYMAP / GENO
family stores data. There is one record per SNP. Inside a SNP's record the
*individuals* are the thing packed four-to-a-byte: individual `i` lives at byte
`i / 4`, bit position `i % 4` within that byte.

**Target layout — canonical individual-major.** This is what the downstream
decode and ploidy-detection stages read. There is one record per individual.
Inside an individual's record the *SNPs* are packed four-to-a-byte: SNP `s` lives
at byte `s / 4`, position `s % 4`, again MSB-first.

Converting between the two is a transpose: the axis that was packed inside a byte
(individuals in the source) becomes the axis that indexes across records in the
output, and vice versa.

---

## 3. What one launch does

A single call to `launch_transpose_to_canonical` performs three things at once, in
one pass over the data:

1. **Transpose.** Swap the SNP and individual axes, as described above, so the
   result is individual-major.
2. **Gather and reorder.** The output does not necessarily include every
   individual in the source, and not in source order. The caller supplies an
   explicit list of source rows, one per output individual, in the desired order
   (the population-contiguous selection). Output column `g` is filled from source
   individual `d_sel_rows[g]`. This is how a chosen, regrouped subset of samples
   is produced without a separate copy step.
3. **Encode.** Each 2-bit source code is passed through an encoding map before it
   is written, so a format whose raw codes differ from the canonical convention
   can be normalized in the same pass (see section 4).

---

## 4. TransposeEncoding

`TransposeEncoding` selects the map applied to every source 2-bit code before it is
packed into the output.

| Value | Numeric | Meaning |
|---|---|---|
| `Identity` | `0` | The source code already *is* the canonical code — copy it through unchanged. |

The PACKEDANCESTRYMAP / GENO format shares steppe's canonical conventions exactly:
the same `0/1/2` reference-allele-count meaning, the same `3`-means-missing
sentinel, and the same MSB-first bit order. So for that format the map is the
identity and no translation is needed.

The enum exists so that other formats can be added later without touching the
transpose kernel itself. A format whose raw codes use a different numbering (for
example an ASCII-based or PLINK-style source) would introduce a new enumerator
here and a matching branch in the kernel, and everything else — the transpose, the
gather, the packing — stays the same.

---

## 5. `launch_transpose_to_canonical` parameters

```
void launch_transpose_to_canonical(const std::uint8_t* d_snp_major,
                                   std::size_t src_bytes_per_record,
                                   const std::size_t* d_sel_rows,
                                   std::size_t n_individuals, std::size_t n_snp,
                                   std::size_t out_bytes_per_record,
                                   TransposeEncoding encoding,
                                   std::uint8_t* d_out, cudaStream_t stream);
```

| Parameter | Meaning |
|---|---|
| `d_snp_major` | Device pointer to the SNP-major source bytes: a grid of `n_snp_records × src_bytes_per_record`. Record `s` is SNP `s`; individual `i` within it is at byte `i / 4`, position `i % 4`. |
| `src_bytes_per_record` | The stride, in bytes, from one SNP record to the next in the source. This is the format's record length, which may be **larger** than the `ceil(n_ind / 4)` bytes actually needed to hold the individuals (see section 7). The kernel only ever reads within the meaningful part of each record. |
| `d_sel_rows` | Device array of length `n_individuals`. Entry `g` is the source individual row that fills output column `g`. This is where selection and reordering are expressed — it is the population-contiguous list of chosen samples. |
| `n_individuals` | Number of gathered output individuals. This is the number of output records produced. |
| `n_snp` | Number of SNPs the output tile covers. This is the output column axis and it bounds how many codes are written. |
| `out_bytes_per_record` | The output record stride, `ceil(n_snp / 4)` bytes — how many bytes each individual's canonical record occupies. |
| `encoding` | Which code map to apply (section 4); `Identity` for the PACKEDANCESTRYMAP / GENO path. |
| `d_out` | Device pointer to the output buffer, sized `n_individuals × out_bytes_per_record`. Receives the canonical individual-major tile. |
| `stream` | The CUDA stream the kernel launches on. |

---

## 6. The one-thread-per-output-byte design

The kernel assigns **one thread to each output byte**, indexed by the pair
(gathered individual `g`, byte `b` within that individual's record). That thread is
solely responsible for producing that one byte and nothing else.

Each output byte holds up to four SNPs — `4b`, `4b+1`, `4b+2`, `4b+3` — for
individual `g`. The thread does this for each of those SNPs `s`:

1. Look up the source row: `src_row = d_sel_rows[g]`.
2. Read the source code from the SNP-major grid: byte `s * src_bytes_per_record +
   src_row / 4`, at bit position `src_row % 4`.
3. Apply the encoding map to get the canonical code.
4. Place that canonical code into the output byte at position `s % 4`, MSB-first.

Because every output byte has exactly one owning thread, **no two threads ever
write the same byte.** That makes the whole kernel race-free without needing atomic
operations or any read-modify-write "OR in my bits" step — each thread simply
assembles its four codes locally and stores the finished byte once. This is the key
design property of the primitive: the output partition, not synchronization, is
what guarantees correctness.

---

## 7. Edge-case invariants

Two boundary conditions are handled deliberately so the GPU result matches the
byte-for-byte behavior of the host (CPU) packer.

### The partial last byte

When `n_snp` is not a multiple of four, the final output byte of each record covers
SNP indices that run past the real data — some of `4b … 4b+3` are `>= n_snp`. Those
out-of-range SNP slots contribute **zero bits** to the output byte. The host packer
does exactly the same, so the padded tail bits agree on both paths.

### The record-length floor (padding never becomes a phantom individual)

The GENO family stores each SNP record with a minimum length — `max(48,
ceil(n_ind / 4))` bytes — so a file with few individuals still has records padded
out to at least 48 bytes. Those padding bytes are meaningless, and if the kernel
walked the record by its full byte length it could mistake padding for extra
individuals.

That never happens here because individuals are addressed only through the explicit
`d_sel_rows` entries, and every such row is a real individual index (`< n_ind`).
The kernel reads the source byte at `src_row / 4`, which therefore always lands
inside the genuine `ceil(n_ind / 4)` bytes and never in the padding region.
`src_bytes_per_record` is used only as the stride to step from one SNP to the next,
not as a count of individuals to scan.

---

## 8. Why this header is device-private

This header names a CUDA type (`cudaStream_t`) in its function signature and
includes `<cuda_runtime.h>`. Because of that it is **internal to the GPU
(`steppe_device`) layer** and must not be pulled into steppe's CUDA-free public
interface.

It sits at the seam between the compute backend and the transpose kernel's own
translation unit — a private, device-side boundary. The public compute-backend
interface, by contrast, is kept entirely free of CUDA types so that the core
library, the command-line tool, and the language bindings can include it without
depending on the GPU toolkit. Keeping the kernel's declaration in a device-private
header preserves that separation: the `<<<>>>` launch and the kernel body live only
in the paired `.cu` file, and no CUDA type leaks past this seam.
