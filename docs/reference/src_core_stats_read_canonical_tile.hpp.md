# `read_canonical_tile.hpp` reference

## 1. Purpose

`src/core/stats/read_canonical_tile.hpp` declares a single function,
`read_canonical_tile`, which is the one place the genotype-path tools turn an open
genotype file into the in-memory tile shape the rest of the pipeline expects,
regardless of how the data is laid out on disk.

Four tools take this genotype path — the f2-statistic extractor (`extract_f2`), the
genotype-based D/f4 engine (`qpDstat-B`), the multi-statistic estimator
(`qpfstats`), and the admixture-dating tool (`DATES`). Rather than each of them
learning how to read every on-disk format, they all call this one function. It reads
a window of SNPs for the selected populations and hands back a tile in a fixed,
canonical layout. Everything downstream is written against that one layout, so the
tools never have to branch on file format themselves.

"Canonical" here means **individual-major**: the tile stores each individual's
genotypes together. Some on-disk formats already store data that way; others store
it the opposite way (each SNP's genotypes across all individuals together). This
function absorbs that difference so that what comes out is always individual-major.

---

## 2. The three on-disk formats it handles

The function inspects the format of the open reader and picks one of three paths.
Two of the three store data SNP-first and need a layout flip; the first already
matches the canonical layout and is passed straight through.

| On-disk format | What it is | What the function does |
|---|---|---|
| **TGENO** (transposed geno) | Already individual-major on disk. | Calls the reader's `read_tile` directly. This is the original, unchanged path — no transpose is needed because the disk layout already matches the canonical layout. The compute backend is never consulted. |
| **GENO** (packed binary ancestrymap) | SNP-major: each SNP's genotypes across individuals are packed together. | Calls the reader's `read_snp_major_tile` to gather the raw SNP-major data, then calls the backend's `transpose_to_canonical` to flip it into an individual-major tile. |
| **EIGENSTRAT** (ASCII SNP-major) | SNP-major, stored as text rather than packed bits. | Calls the reader's `read_eigenstrat_snp_major_tile` to parse the ASCII and pack it into the same intermediate SNP-major buffer, then runs the **same** transpose as the GENO path. |

The two SNP-major formats deliberately share the transpose step. Once EIGENSTRAT has
been parsed and packed into the same intermediate SNP-major shape that GENO produces,
there is only one flip-to-canonical operation, not two.

### The transpose runs on the GPU

The flip from SNP-major to individual-major is `transpose_to_canonical`, a method on
the compute backend. On the real (GPU) backend it runs as a device kernel. There is
also a plain host-loop version used only as a correctness oracle for testing — the
two must produce the same result. The reader itself never does the transpose; it only
produces raw data, and the backend does the layout change.

---

## 3. The "transpose-on-read" idea

The whole point of this function is that the layout difference between formats is
resolved **once, at read time**, and nothing after it has to know or care which
format the data came from.

After `read_canonical_tile` returns, the tile is in exactly the same canonical
packing — same byte layout, same population offsets, same population labels — no
matter which of the three formats it started as. Every later stage (ploidy
detection, allele-frequency decoding, the quality-control filter, and the f2 / D /
DATES / qpfstats math) already expects that one canonical packing. So adding support
for a new SNP-major format only requires teaching this function how to read and flip
it; nothing downstream changes.

---

## 4. The API contract

```cpp
[[nodiscard]] io::GenotypeTile read_canonical_tile(io::GenoReader& reader,
                                                   const io::IndPartition& part,
                                                   ComputeBackend& backend,
                                                   std::size_t snp_begin,
                                                   std::size_t snp_end);
```

It reads the half-open SNP window `[snp_begin, snp_end)` for the populations named in
`part` and returns a canonical individual-major `io::GenotypeTile`.

| Parameter | Meaning |
|---|---|
| `reader` | An already-open genotype file reader. Its format determines which of the three paths runs. |
| `part` | The individual partition — which populations (and their individuals) to read, and in what order. |
| `backend` | The compute backend used only to run the transpose. See the note below on when it is actually touched. |
| `snp_begin`, `snp_end` | The half-open range of SNP indices to read into this tile. |

### The backend is consulted only on the SNP-major paths

On the TGENO path the `backend` argument is never used, because no transpose is
needed. It is consulted only on the GENO and EIGENSTRAT paths, where the SNP-major
data has to be flipped to canonical. Callers must still pass a valid backend (the
signature requires it), but they should not assume it is exercised on every call.

### The returned tile has no ploidy filled in

The tile that comes back has an **empty** `sample_ploidy`. This function does not
detect or set ploidy. Ploidy is worked out later, on the already-canonical tile,
either by a separate on-device pre-pass or from a ploidy vector the caller supplies
explicitly. Keeping ploidy out of this function keeps its single responsibility —
format dispatch and layout — clean.

### Failure behavior

Any reader or transpose failure is surfaced as a `std::runtime_error`. This is the
same exception contract the underlying reader calls (`read_tile`,
`read_snp_major_tile`) already carry, so callers do not face a new error model. An
unrecognized on-disk format is also thrown as a `std::runtime_error`.

The `[[nodiscard]]` marking means the returned tile must be used — ignoring the
result is almost certainly a mistake, since reading a tile with no consumer does
work for nothing.

---

## 5. Where this code lives in the layering

This function sits at a deliberate seam between two parts of the codebase that are
both kept free of any GPU (CUDA) headers:

- the **io** layer, which knows how to read files but is not allowed to call GPU
  kernels, and
- the **compute backend** seam, whose interface is also CUDA-free even though its
  real implementation runs on the GPU.

`read_canonical_tile` is the wiring point where those two meet: it takes raw data
from the io layer and asks the backend to transpose it. It is intentionally **not**
a GPU translation unit and is intentionally **not** placed inside the io layer,
because the io layer may not reach into GPU code. No CUDA header appears here; the
transpose is reached only through the backend's abstract interface.

Because of this placement it can be compiled once into the core library (which links
both the io layer and the device backend) and reused in two places: by the core
library's own genotype tools, and by the standalone f2-extraction executable, which
links against the core library. One dispatch function, one compiled copy, shared by
every genotype-path caller.
