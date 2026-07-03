# `detect_ploidy_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/detect_ploidy_kernel.cuh` declares a single GPU launch
wrapper, `launch_detect_ploidy`. That function runs a small prepass on the GPU
that decides, for each sample in the data, whether the sample should be treated
as **diploid** or as **pseudo-haploid**, and writes the answer into an output
array.

This is a prepass in the sense that it runs before the main statistics work and
produces a small per-sample table (one integer per sample) that later stages
read. It exists because the same decision used to be made on the host CPU, one
sample at a time; moving it onto the GPU keeps the data on the device and avoids
a round trip.

The header itself is deliberately narrow. It contains only the one function
declaration plus its documentation. The actual GPU kernel body — the code that
runs on each thread — and the `<<<...>>>` launch live in the matching
`detect_ploidy_kernel.cu` file, not here.

---

## 2. The ploidy rule it implements

Some ancient-DNA samples are stored in a **pseudo-haploid** form: even though the
organism is diploid, only a single allele is recorded at each site, so a
heterozygous call (one copy of each allele) can never appear. A genuinely diploid
sample, by contrast, will show heterozygous calls. The rule keys off exactly that
difference.

For one sample, the kernel scans that sample's genotype calls from the start.
As soon as it finds a **heterozygous** call, it concludes the sample is diploid
and stops. If it scans the whole detection window without finding a single
heterozygous call, it concludes the sample is pseudo-haploid.

The output uses a simple integer encoding:

| Written value | Meaning |
|---|---|
| `2` | Diploid — at least one heterozygous call was seen |
| `1` | Pseudo-haploid — no heterozygous call was seen in the window |

A call counts as heterozygous when its decoded genotype code equals the shared
heterozygous-genotype code constant (`core::genotype_code == kHeterozygousGenotypeCode`).

### The detection window

The kernel does not scan every SNP. It scans only the first
`min(kPloidyDetectSnps, n_snp)` SNPs of the sample's record, where
`kPloidyDetectSnps` is a fixed cap defined elsewhere and `n_snp` is however many
SNPs the current data tile actually covers. Whichever of the two is smaller
bounds the scan. A handful of SNPs is more than enough to catch a heterozygous
call in a truly diploid sample, so scanning the whole genome would be wasted
work.

This matches how ADMIXTOOLS 2 detects pseudo-haploid samples, and it is a literal
port of the equivalent host-side loop, so it produces the same diploid/pseudo-haploid
verdict as that reference. See section 5 for the exact sense in which the two agree.

---

## 3. The `launch_detect_ploidy` contract

```cpp
void launch_detect_ploidy(const std::uint8_t* d_packed,
                          std::size_t bytes_per_record,
                          std::size_t n_individuals, std::size_t n_snp,
                          int* d_ploidy, cudaStream_t stream);
```

The launch assigns **one GPU thread to one sample**: thread `g` owns the sample
at index `g`, scans that sample's window, and writes that sample's single result.
The threads do not interact.

| Parameter | Meaning |
|---|---|
| `d_packed` | Device pointer to the packed genotype bytes for all gathered samples, laid out one sample record after another (`n_individuals × bytes_per_record` bytes total, contiguous by population). This is the same byte buffer the main decode reads — see section 4. |
| `bytes_per_record` | The stride, in bytes, from the start of one sample's record to the start of the next. Multiplying it by `g` gives the byte offset of sample `g`'s record. |
| `n_individuals` | The number of samples to process — the size of the prepass. This is also the number of threads and the length of the output array. |
| `n_snp` | How many SNPs the current data tile covers. It caps the detection window (section 2); the scan never runs past it even if the fixed cap is larger. |
| `d_ploidy` | Device pointer to the output array, one `int` per sample (`n_individuals` entries). Each entry is `2` for diploid or `1` for pseudo-haploid. |
| `stream` | The CUDA stream the launch is queued on. |

---

## 4. How the packed genotypes are laid out

The kernel reads the very same packed byte layout that the main genotype decode
reads, using the same shared decode primitives. Reusing one decode path is what
makes the two agree bit-for-bit (section 5).

The layout is **individual-major**: each sample owns one contiguous record of
`bytes_per_record` bytes, and the records sit back to back. Within a sample's
record, four SNP calls are packed into each byte, two bits per call. For SNP
index `s` inside a record:

- the byte holding it is at offset `s / 4` within that sample's record, and
- its two bits sit at position `s % 4` inside that byte, read **most-significant
  bits first**.

Decoding those two bits into a genotype code is done through the shared
`core::genotype_code` helper, so the kernel interprets a packed byte identically
to every other part of the codebase that reads this format.

---

## 5. Invariants and where this file fits

### Bit-identical to the host detector by construction

The kernel is a literal port of the host-side ploidy loop, and it decodes through
the *same* shared primitives the host loop uses. Because the whole computation is
made of integer and bit operations only — there is no floating-point math and no
precision mode involved — the GPU result vector is bit-for-bit identical to what
the host detector would produce for the same input. There is no tolerance and no
rounding to reason about; the two paths agree exactly, by construction.

This is why the prepass can safely move to the GPU without changing any downstream
result: the diploid/pseudo-haploid table it produces is indistinguishable from the
one the host used to produce.

### Private to the device library

This header names a CUDA type (`cudaStream_t`) in its signature, so it is a
**private, device-internal** header. It is the seam between the GPU backend and
the prepass kernel's own translation unit — not a public, CUDA-free interface.
Code outside the GPU library must not include it. The public compute interface
that outside code does use is defined separately and pulls in no CUDA types.

The header includes only `<cstddef>`, `<cstdint>`, and `<cuda_runtime.h>`, and is
guarded by `STEPPE_DEVICE_CUDA_DETECT_PLOIDY_KERNEL_CUH`. Everything else about
the kernel — its body and its launch configuration — lives in the paired `.cu`
file.
