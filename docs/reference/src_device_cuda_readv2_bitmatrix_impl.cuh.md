# `readv2_bitmatrix_impl.cuh` reference

## 1. Purpose

`src/device/cuda/readv2_bitmatrix_impl.cuh` defines the one small struct that holds
the actual GPU memory behind a `Readv2Bitmatrix`. It is the CUDA half of a deliberate
two-file split, the exact same pattern used by `device_decode_result_impl.cuh`:

- The public handle `Readv2Bitmatrix` (in `device/readv2_bitmatrix.hpp`) is written to
  be free of any CUDA code. It only forward-declares a nested type named `Impl` and
  keeps a `std::unique_ptr<Impl>` to it, alongside a handful of plain host scalars that
  describe the window geometry (`n_samples`, `window_snps`, `wpw`, `n_win`,
  `words_per_sample`, and so on). The CUDA-free orchestrator in the core library — the
  READv2 driver that streams genotype chunks in and asks for the mismatch sweep back —
  can carry that handle around without ever pulling in GPU code.
- This file is where `Impl` is actually spelled out. It is the only place that names the
  real GPU buffer type (`DeviceBuffer<Readv2Word>`), so it is the only file in the pair
  the CUDA toolchain has to compile.

That split is the whole reason the file exists. It lets the bit-matrix be a
device-resident thing — its data lives in VRAM and stays there across the entire
all-pairs sweep — while still being passable through code that knows nothing about
CUDA. The host never touches the packed bits.

---

## 2. The single device buffer

`Readv2Bitmatrix::Impl` is just one GPU buffer:

| Field | Type | Shape | What it holds |
|---|---|---|---|
| `words` | `DeviceBuffer<Readv2Word>` | `n_samples × words_per_sample` (flattened) | The packed `[sample × SNP-window]` bit-matrix — every sample's genotypes for the whole genome, encoded 64 SNPs to a word. |

The buffer is a flat, sample-major array of `n_samples * words_per_sample` cells.
Sample `s` owns the contiguous run `words[s * words_per_sample ..
(s+1) * words_per_sample)`. Within that run the words are laid out window by window:
`words_per_sample = n_win * wpw`, where `n_win` is the number of SNP-count windows
tiling the genome and `wpw` is the words per window (`ceil(window_snps / 64)`). Window
`g` of a sample occupies `wpw` consecutive words. All of that geometry lives as host
scalars on the public handle; this struct just holds the memory.

Each cell is a `Readv2Word` — a 16-byte, 128-bit AoS pair of `allele` and `valid`
bit-planes (defined in `readv2_layout.cuh`) that the mismatch inner loop fetches in one
coalesced `ulonglong2` transaction. Bits are LSB-first: local SNP index `l` in `[0,64)`
maps to bit `l`. The `allele` bit is the single pseudo-haploid allele (0 for ref-copy
code 0, 1 for ref-copy code 2) and is meaningful only where `valid == 1`; `valid` is 1
exactly at a genuine 0/2 hardcall, and 0 at a missing call, a het, or a padding SNP.

---

## 3. The zero-at-allocation contract

The single most load-bearing invariant on this buffer is that it is **fully zeroed at
allocation**. `CudaBackend::readv2_alloc_bitmatrix` (in `cuda_backend_readv2.cu`)
`cudaMemsetAsync`s the whole thing to zero right after allocating it, before any pack
kernel runs.

This is not tidiness — it is correctness. The pack step only ever writes the words for
SNPs that actually exist. Everything it never touches must already read as `valid == 0`:

- **Window tails.** The last word of each window carries `window_snps % 64` real SNPs;
  its unused high bits are never written and must stay `valid = 0`.
- **The genome tail.** The final window can be partial (windows tile with
  `ceil(m0 / window_snps)`, keeping a short last window), so the words past the last real
  SNP are never written either.
- **Any missing / het / dropped site.** These are simply left as the zeroed default.

Because a padding or never-written cell reads `valid = 0`, the mismatch kernel's
both-valid `AND` doubles as the padding mask automatically — it never counts a comparison
at a bit that isn't a real, shared hardcall, and it never has to special-case window
edges. The zero-fill is what makes that free. Skip it and the sweep would read garbage
bits as genotypes.

---

## 4. Privacy and who may include this file

This header is private to the `steppe_device` build unit. It is not part of the public
API, and code outside the GPU layer must not include it — that is exactly what the
CUDA-free public handle protects against.

Exactly two translation units include it:

- **`readv2_bitmatrix.cu`** — defines the handle's special members (default constructor,
  destructor, and the two move operations) out of line. They have to be here, not on the
  header, because `std::unique_ptr<Impl>` needs a *complete* `Impl` to destroy it; putting
  these definitions in a CUDA TU that has seen the full struct is what lets the public
  header stay CUDA-free while still owning a `unique_ptr`.
- **`cuda_backend_readv2.cu`** — the producer and consumer side. It reaches through the
  `unique_ptr` into `words` to (a) allocate and zero the matrix, (b) pack streamed 2-bit
  genotype chunks into it with the pack kernel, and (c) run the all-pairs windowed
  `__popc` mismatch reduction over it.

Because `Impl` is only ever handled through a `std::unique_ptr`, a `Readv2Bitmatrix`
whose pointer is null represents an empty or moved-from matrix with no resident buffer at
all. The handle's `empty()` predicate reflects the same idea from the host-scalar side
(no samples or zero words per sample), and every backend method guards on `bits.impl`
before dereferencing, so a null or empty matrix is a clean no-op rather than a crash.
