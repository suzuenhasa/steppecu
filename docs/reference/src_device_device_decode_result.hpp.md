# `device_decode_result.hpp` reference

## 1. Purpose

`src/device/device_decode_result.hpp` declares one class, `DeviceDecodeResult`.
It is a small, opaque handle that owns the output of the genotype-decode step and
keeps that output sitting in GPU memory instead of copying it back to the host.

The decode step turns raw genotype data into three per-population, per-SNP arrays —
called Q, V, and N here — and then throws away every SNP that isn't wanted (for
example, the non-autosome chromosomes) so the surviving columns are packed together
with no gaps. `DeviceDecodeResult` is the object that carries the packed, kept
arrays forward, together with just enough plain host-side bookkeeping (how many
populations, how many surviving SNPs, and which chromosome and position each
surviving SNP came from) for the rest of the pipeline to use them.

The central design idea is that the heavy arrays never leave the GPU. Older versions
of the pipeline decoded on the GPU, copied roughly a gigabyte of Q/V/N back to the
host, filtered SNPs in a host loop, then re-uploaded the survivors. All of that
host round-tripping is gone: the decode, the keep/drop decision, and the packing
now all happen on the GPU, and only a small amount of metadata crosses to the host.
This handle is the seam where the still-resident GPU arrays are handed off.

Two vocabulary notes used throughout:

- **Q, V, N** are the three decoded arrays. Q holds the per-population allele
  frequencies, V holds a paired variance-like quantity, and N holds the count of
  non-missing samples behind each frequency. All three are laid out the same way
  (see section 7).
- **"Compacted" / "kept axis"** means the columns have already been filtered down
  to the SNPs that survive, packed contiguously. `M_kept` is how many survived.

---

## 2. What the decode step produces on-device

The producer of this handle is the GPU backend's decode routine. Before this handle
existed, several distinct pieces of work each involved a host round-trip; the point
of the handle is that all of them now finish on the GPU:

1. **Decode stays resident.** The raw genotype tile is decoded into Q/V/N directly
   in GPU memory. There is no copy of the full Q/V/N arrays back to the host — the
   large (~1.1 GB) transfer that used to happen here is eliminated.
2. **The keep/drop decision runs on the GPU.** A per-SNP mask kernel decides which
   SNPs survive (for the autosome path, which SNPs are on chromosomes 1–22). The old
   host filter loop that made this decision one SNP at a time is gone.
3. **The survivors are packed on the GPU.** A scan-and-gather compaction (using the
   CUDA library's stream-compaction primitive) rewrites Q/V into a dense block of
   `M_kept` columns with the dropped columns removed. The old lockstep host subset
   plus re-upload is gone.

After all three steps, the packed Q/V (and, in one regime, N) remain in GPU memory
and escape only through this handle. The only data that crosses to the host is the
small per-surviving-SNP metadata — the chromosome, genetic position, and physical
position of each kept SNP — which a later, deliberately CUDA-free step uses to assign
each SNP to a jackknife block. Because that block-assignment step is unchanged and
reads the same metadata the old host loop produced, the block assignments come out
identical.

---

## 3. Regime A versus regime B — the N-buffer rule

There are two callers of the decode routine, and they need different amounts of data
back. This is the single most important invariant in the file, because it decides
whether the N array exists at all.

- **Regime A (autosome-only).** The consumers here read only Q and V; they never
  need N. So the regime-A decode routine leaves the N buffer empty and compacts only
  Q and V. On a regime-A result, `n_device()` returns null.
- **Regime B (filtered `extract_f2`).** This path applies the more delicate,
  floating-point-sensitive filters (minor-allele frequency, max-missing, SNP class)
  and then feeds a matrix multiply that genuinely needs the sample counts. So the
  regime-B decode routine compacts Q, V, **and** N together in lockstep onto the same
  kept axis. On a regime-B result, `n_device()` is non-null.

The practical rule to remember: **`n_device()` being null tells you which regime
produced the handle.** Null means regime A (no N was ever compacted); non-null means
regime B (N is present and packed to match Q and V). Any code that needs N must
either be a regime-B consumer or must check for null first. The `to_host_qvn` method
(section 8) enforces this rule rather than trusting the caller.

---

## 4. Ownership: move-only, and CUDA-free by construction

`DeviceDecodeResult` is **move-only**. It has a move constructor and move assignment,
but the copy constructor and copy assignment are deleted. This is because it owns GPU
memory: copying it would either duplicate a large device allocation implicitly or
create two owners of the same buffer, both of which are mistakes. Passing ownership
along by moving is the only supported way to hand the result to the next stage. A
default-constructed or moved-from instance is the empty state — it owns no GPU
buffers — and the destructor frees whatever resident Q/V (and N) the handle still
holds.

The header is deliberately **CUDA-free**: it names no CUDA type and includes no CUDA
headers, only three standard-library headers. It achieves this with the
pointer-to-implementation (PIMPL) pattern. The actual owners of the GPU memory are
`DeviceBuffer<double>` objects that live inside a nested `Impl` struct, and `Impl` is
only forward-declared here (`struct Impl;`) and defined in a separate CUDA header. The
handle stores it behind a `std::unique_ptr<Impl>`.

The reason for going to this trouble is that the code that orchestrates the pipeline
(the core library) is itself CUDA-free and must be compilable by an ordinary C++
compiler. Because the shape fields and the kept-axis metadata are plain host data,
that CUDA-free orchestrator can hold the handle, run its CUDA-free block-assignment
step over the kept axis, and forward the still-resident Q/V into the next stage — all
without ever touching a CUDA type. This mirrors the same pattern used by the
device-side f2-blocks handle.

---

## 5. Shape fields

Three plain, host-side scalars describe the size and location of the resident arrays.
They are ordinary public members with in-class defaults.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `P` | `int` | `0` | The population count. This is the number of rows in Q/V/N and the leading dimension of the column-major layout. |
| `M_kept` | `long` | `0` | The number of SNPs that survived filtering and were packed — the column count of the compacted arrays. This is a `long` because a large panel can have more surviving SNPs than a 32-bit count comfortably holds. |
| `device_id` | `int` | `-1` | The CUDA device ordinal that the Q/V (and N) buffers are resident on. `-1` means no device — the empty or moved-from state. |

---

## 6. The kept-axis metadata and its file-order guarantee

Three vectors carry the per-surviving-SNP metadata to the host. They are parallel to
the resident Q/V columns: element *k* of each vector describes the SNP in column *k*
of the compacted arrays. The later block-assignment step reads them to decide which
jackknife block each kept SNP belongs to.

| Field | Type | Length | Meaning |
|---|---|---|---|
| `chrom_kept` | `vector<int>` | `M_kept` | The chromosome number of each kept SNP. |
| `genpos_kept` | `vector<double>` | `M_kept` | The genetic position of each kept SNP, in Morgans. |
| `physpos_kept` | `vector<double>` | `M_kept`, or empty | The physical position of each kept SNP, in base pairs. Empty when the producer was given no physical-position axis; otherwise length `M_kept`. |

Two properties matter:

- **File order is guaranteed.** All three vectors are in the original file order of
  the SNPs, and that same ordering is preserved by the GPU compaction primitive, which
  keeps flagged elements in their original relative order. Because the metadata and the
  resident Q/V columns are both packed by the same order-preserving pass, column *k*
  of Q/V and element *k* of every metadata vector always refer to the same SNP. The
  result is bit-identical to what the old host autosome loop produced for the same
  fields, which is why block assignments still match.
- **Physical position is the fallback for block assignment.** The block-assignment
  step normally splits blocks by genetic position (`genpos_kept`). When a dataset ships
  with no genetic map — the genetic-position column is all zeros, common for data
  derived from VCF or PLINK — it falls back to splitting by physical position instead.
  `physpos_kept` exists to feed that fallback and is compacted in lockstep with the
  other two by the same compaction pass. It is left empty when the producer had no
  physical positions to supply.

---

## 7. Device pointers to the resident arrays

Three accessor methods hand back borrowed (non-owning) pointers into the GPU memory
the handle owns. "Borrowed" means the caller must not free them; the handle still owns
the memory and frees it in its destructor. The pointers are only valid for as long as
the handle is alive.

| Method | Returns | Notes |
|---|---|---|
| `q_device()` | `const double*` | Pointer to the compacted Q array. Null when the handle is empty. |
| `v_device()` | `const double*` | Pointer to the compacted V array. Null when the handle is empty. |
| `n_device()` | `const double*` | Pointer to the compacted N array. Non-null **only** for a regime-B (filtered `extract_f2`) result; null for a regime-A (autosome-only) result, which never compacts N. See section 3. |

**Memory layout.** All three arrays are column-major with shape `[P × M_kept]`. The
element for population *i* and SNP *s* is at index `i + P·s`. In other words, each
SNP's `P` population values are contiguous, and consecutive SNPs are `P` apart. Both
the shape fields and this layout are what the next stage relies on to read the arrays
correctly.

These accessors are `noexcept` and declared `[[nodiscard]]`, and their definitions
live in a CUDA source file because they dereference the CUDA-typed `Impl`.

---

## 8. Reading results back to the host: `to_host_qvn`

```cpp
void to_host_qvn(std::vector<double>& q_host,
                 std::vector<double>& v_host,
                 std::vector<double>& n_host) const;
```

This method exists for the regime-B path, whose downstream consumer takes host-side
matrix views rather than GPU pointers. It synchronously copies the compacted Q, V, and
N arrays from the GPU down into the three supplied host vectors, resizing each to
`P·M_kept`.

This is still a large improvement over the old design even though it does copy to the
host: the host per-SNP filter loop and the copy of the *full, unfiltered* tile are both
gone, because the keep-decision and the compaction already ran on the GPU. Only the
*already-shrunken* compacted arrays cross to the host. In spirit it is the same kind of
transfer as the small metadata copy on the regime-A path, just with the bigger packed
arrays.

Two behaviors to know:

- **It requires a regime-B result and enforces that.** Because it reads N, it needs
  `n_device()` to be non-null. If it is called on a non-empty regime-A result — one
  whose `Impl` is set but whose N is empty — it throws `std::invalid_argument` rather
  than dereferencing a null N source. This turns the section-3 invariant into a
  checked precondition instead of a silent crash.
- **On an empty result it is a safe no-op.** If the handle is empty, the method simply
  clears the three output vectors and returns.

Its definition also lives in a CUDA source file.

---

## 9. `empty()` and the opaque payload

`empty()` is an inline, `noexcept`, `[[nodiscard]]` predicate that returns true for a
degenerate result that owns no resident buffers. It reports empty when `M_kept <= 0`
or `P <= 0` — that is, whenever there is nothing meaningful to point at. Default-
constructed and moved-from handles are empty by this test.

The final member is the PIMPL payload itself:

```cpp
struct Impl;                  // defined in a separate CUDA header
std::unique_ptr<Impl> impl;   // null => no resident buffers
```

`Impl` is only forward-declared in this header; its real definition (holding the
`DeviceBuffer<double>` owners of Q/V and, for regime B, N) lives in the accompanying
CUDA header. The `unique_ptr` being null is another way of saying the handle owns no
resident buffers. Keeping the CUDA-typed owners hidden behind this forward-declared
`Impl` is exactly what lets the header stay CUDA-free (section 4).
