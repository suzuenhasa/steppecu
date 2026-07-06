# `detect_ploidy_kernel.cu` reference

## 1. Purpose

`src/device/cuda/detect_ploidy_kernel.cu` decides, on the GPU, whether each
gathered individual should be treated as **diploid** or **pseudo-haploid**. It looks
at the individual's genotype data and writes one small integer per individual: `2`
for diploid, `1` for pseudo-haploid.

That per-individual number matters downstream because it changes how an allele
frequency is computed. A diploid genotype call ranges over 0, 1, or 2 copies of an
allele and is divided by 2; a pseudo-haploid call is treated as 0 or 1 copy and is
divided by 1. Getting the ploidy wrong would silently halve or double a frequency, so
this prepass has to run before any frequency is formed.

The file contains exactly two things:

1. `detect_ploidy_kernel` — the GPU kernel that does the scan, one thread per
   individual.
2. `launch_detect_ploidy` — a thin host-side wrapper that picks the launch geometry
   and starts the kernel, so that no `<<<...>>>` launch syntax ever appears in
   ordinary host code.

This is the on-device twin of a host (CPU) detector that lives in
`src/io/ploidy_detect.cpp`. The two are written to produce **bit-for-bit identical**
output on the same input, and the reasons they cannot drift apart are spelled out in
section 5.

---

## 2. What ploidy detection is, and why the scan works

A diploid genome carries two copies of each chromosome, so at a given site it can be
**heterozygous** — carrying one of each of the two alleles. A haploid or
pseudo-haploid genome carries only a single call per site and, by construction,
*cannot* be heterozygous.

That asymmetry is the whole trick. To decide an individual's ploidy, scan its first
several genotype calls: the moment you see a single heterozygous call, the individual
must be diploid and you can stop. If you scan the whole window and never see one, you
treat the individual as pseudo-haploid.

The default when in doubt is **pseudo-haploid** (ploidy `1`). Every individual starts
out labeled pseudo-haploid and is only ever *promoted* to diploid by the discovery of
a heterozygous call. This "default pseudo-haploid, promote on first het" rule is the
parity behavior steppe reproduces[^at2].

---

## 3. The detection algorithm (the kernel)

`detect_ploidy_kernel` assigns **one thread to one gathered individual**. Thread `g`
handles individual `g`, for `g` in `[0, n_individuals)`. Threads whose index runs past
the number of individuals simply return and do nothing.

Each thread does this:

1. Start with the label pseudo-haploid (`kPloidyPseudoHaploid`, value `1`).
2. Point at the start of that individual's packed record: `packed + g * bytes_per_record`.
3. Walk the first `window` SNP positions of the record (see section 6 for how `window`
   is chosen).
4. At each SNP position, unpack the 2-bit genotype code and compare it to the
   heterozygous code. On the **first** heterozygous call, set the label to diploid
   (`kPloidyDiploid`, value `2`) and break out of the loop immediately.
5. Write the final label to `ploidy[g]`.

Because a scan stops at the first heterozygous call, a diploid individual usually
exits after only a few SNPs; only genuinely pseudo-haploid individuals (or ones whose
first calls are all missing/homozygous) walk the full window.

The entire scan is **integer and bit operations only** — a 2-bit unpack, an equality
check, and a loop bound. There is no floating-point math anywhere in it.

---

## 4. Byte layout and the shared unpack primitive

Genotypes arrive packed four calls to a byte (two bits each). To find SNP `s` inside
an individual's record, the kernel computes:

- **which byte**: `s / kCodesPerByte` (i.e. `s / 4`), and
- **which slot in that byte**: `s % kCodesPerByte` (i.e. `s % 4`).

The 2-bit code is then extracted by `core::genotype_code`, which reads the slots
**most-significant bits first** within each byte. The specific 2-bit values it can
return are shared constants: `kHeterozygousGenotypeCode` is `1` (a het call), and the
missing-data code is `3`.

The crucial point is that this kernel does **not** roll its own unpacking. It calls
the same shared `core::genotype_code` primitive — from `core/internal/decode_af.hpp`
— that the main genotype decoder uses, and it reads the exact same
individual-major packed bytes (byte `s/4`, slot `s%4`, most-significant-bits-first)
that the decoder reads. So this prepass sees precisely the calls the rest of the
pipeline will later see for the same tile; it is the on-device counterpart of the same
scan over the same bytes.

---

## 5. Why it is bit-identical to the host detector

There is a host (CPU) version of this detector in `src/io/ploidy_detect.cpp`. steppe
guarantees the GPU kernel and the host loop produce **the same integer for every
individual**, and that guarantee is structural rather than something checked after the
fact:

- Both use the **same shared core primitives** — the same `genotype_code` unpack, the
  same `kHeterozygousGenotypeCode == 1` comparison, and the same
  `min(kPloidyDetectSnps, n_snp)` window cap.
- Both read the **same packed bytes** in the **same layout** (byte `s/4`, slot `s%4`,
  most-significant-bits-first).
- The work is **integer/bit only**. There is no floating-point arithmetic, so there is
  no rounding to differ; there is no reduction, so there is no summation order to
  differ. The output is a plain label, not an accumulated number.

Because every operation that produces the label is an exact integer operation drawn
from the same shared code, the two detectors cannot diverge by construction — there is
no "precision" difference for them to have. The host detector serves as the reference
that the device path is expected to match exactly.

---

## 6. The detection window (and the empty-window case)

The window is how many leading SNPs each thread will scan before giving up and calling
the individual pseudo-haploid:

```
window = min(kPloidyDetectSnps, n_snp)
```

`kPloidyDetectSnps` is `1000` — the parity number of leading SNPs tested[^at2]. If the
tile actually carries fewer SNPs than that (`n_snp < 1000`), the window shrinks to
`n_snp`, so a short record scans its whole available prefix rather than reading past
its end. This is exactly what the host loop does in the same situation.

The `window == 0` case (a tile with no SNPs) is handled without any special branch.
The launch wrapper still launches the kernel unconditionally; each thread's scan loop
simply never executes its body, so every individual keeps its initial pseudo-haploid
label and the output vector is still fully written. That matches the host loop, which
leaves every individual at the pseudo-haploid default when there is nothing to scan.
The one thing the wrapper *does* short-circuit is `n_individuals == 0`, where it
returns early because there is no output to write at all.

---

## 7. The launch wrapper and grid geometry

`launch_detect_ploidy` is the only host-visible entry point. It:

1. Returns immediately if `n_individuals == 0` (nothing to do).
2. Computes the scan `window` as described in section 6.
3. Computes the grid width in one call to the shared `core::grid_for_x` helper (from
   `core/internal/launch_config.hpp`): it divides the number of individuals by the
   block size rounding up, and in the same step asserts that the resulting grid width
   does not exceed the maximum grid dimension the hardware allows (`kMaxGridX`). If it
   ever did, the fix is to tile the individual axis into multiple launches — the
   message passed into `grid_for_x` says so. In practice this is a guard against an
   impossibly large individual count, not something that fires on real data.
4. Launches the kernel on the caller-supplied stream and checks for a launch error.

The block size is a fixed compile-time constant, `kPloidyBlock = 256` threads per
block, defined privately in this file. Kernels here never choose their own block size
ad hoc; the geometry is computed in this one wrapper.

This wrapper's header names a CUDA type (`cudaStream_t`), which makes it part of the
device-internal interface between the backend and this kernel — it is not part of the
CUDA-free public interface. The kernel body and the `<<<...>>>` launch live only in
this `.cu` file.

---

## 8. Constants and values

Most of these values live in the shared decode header (`core/internal/decode_af.hpp`)
and are reused here so the host and device detectors reference the same numbers; only
`kPloidyBlock` is defined in this file.

| Name | Value | What it means |
|---|---|---|
| `kPloidyBlock` | `256` | Threads per block for the kernel launch. Defined in this file. |
| `kPloidyDetectSnps` | `1000` | How many leading SNPs to scan before defaulting to pseudo-haploid. The parity tested count. Caps the scan window (section 6). |
| `kPloidyPseudoHaploid` | `1` | The label written for a pseudo-haploid individual — and the default every individual starts with. |
| `kPloidyDiploid` | `2` | The label written once a heterozygous call is found. |
| `kHeterozygousGenotypeCode` | `1` | The unpacked 2-bit code that means "heterozygous." Finding one promotes the individual to diploid. |
| `kCodesPerByte` | `4` | How many 2-bit genotype calls are packed into each byte; drives the `s/4` byte index and `s%4` slot index. |

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
