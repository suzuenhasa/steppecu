I read through this carefully. This is **not slop** — it's clearly written by someone who understands both the population-genetics domain and the project's backend seam, but a senior developer would flag some maintainability and convention issues.

## What's genuinely good

- **The domain parity narrative is impressive.** The comments explicitly pin three behaviors to the AT2 `qpdstat_geno` golden (allele frequency calculation, block partitioning, SNP mask) and explain *why* this path diverges from the f2 cache (`run_f4`/`run_f4ratio`). That level of cross-reference-aware design shows real competence, not just code slinging.
- **Clean use of modern C++ at the API boundary.** `std::span<const std::string>`, `std::span<const std::array<int, 4>>`, `std::vector<int> flat; flat.reserve(...)`, and `std::nan("")` all indicate the author is comfortable with contemporary C++ idioms.
- **Good architectural seam discipline.** The file routes everything through `ComputeBackend`, with a single `resident` branch deciding whether the decode/jackknife stays on the device or falls back to the CPU oracle. The result struct `DstatResult` is populated consistently from `RatioBlockJackknife`, and domain degeneracy is returned as per-row NaN rather than thrown.
- **The forced-diploid ploidy decision is well-explained.** Lines 58–61 and 137 make it clear why this path ignores the extract-f2 auto-detection and pins AT2's plain `ref/an/2` frequency calculation. A domain-expert reader can immediately see the parity argument.

## What a senior developer would flag

**Dense, cross-referential comments that risk going stale:**

```cpp
// run_dstat is the genotype-reading SIBLING of run_f4 / run_f4ratio: it does NOT read the f2
// cache. It REUSES the extract-f2 decode FRONT-END (the io reader + decode_af [per-SNP Q/V/N]
// + assign_blocks [from genpos]) and DIVERGES at S2 into the per-SNP D kernel
```

This is good context, but comments like `mirrors cmd_extract_f2.cpp:157-356` (line 100) and `mirrors f4.cpp/f4ratio.cpp's kPrimaryGpu` (line 64) are line-number anchors in other files. Those will rot the first time someone refactors `cmd_extract_f2.cpp`. A senior reviewer would ask for stable identifiers (function names, section tags) rather than brittle line numbers.

**The `resident` heuristic is too simple:**

```cpp
const bool resident = (be.capabilities().device_count > 0);
```

At line 161 this decides whether to call the device-resident compact path or the host decode path. But `device_count > 0` does not necessarily mean this backend instance is the CUDA backend or that the current tile is resident. If the caller ever runs a `CpuBackend` in a machine with GPUs, this branch will try to call `decode_af_compact_autosome` on a CPU backend. The backend itself should probably expose this capability (`supports_device_resident_decode()`), or the branch should be typed on the backend kind.

**Repeated NaN-filling boilerplate:**

Lines 128–133, 192–198, and 206–213 are nearly identical blocks:

```cpp
res.est.assign(static_cast<std::size_t>(N), std::nan(""));
res.se.assign(static_cast<std::size_t>(N), std::nan(""));
res.z.assign(static_cast<std::size_t>(N), std::nan(""));
res.p.assign(static_cast<std::size_t>(N), std::nan(""));
res.status = Status::Ok;
return res;
```

A small local helper (`fill_nan_result(res, N)`) would remove the copy-paste surface area and make the early-return guard clauses read like guard clauses rather than wall-of-text.

**No validation that quadruple indices are in `[0, P)`:**

```cpp
for (const std::array<int, 4>& q : quadruples) {
    res.p1.push_back(q[0]); res.p2.push_back(q[1]);
    res.p3.push_back(q[2]); res.p4.push_back(q[3]);
    flat.push_back(q[0]); flat.push_back(q[1]); flat.push_back(q[2]); flat.push_back(q[3]);
}
```

The comment at lines 101–109 says the caller resolved indices against the same `PopResolver`, which is fine if the contract is airtight. But a senior reviewer would want at least an `assert` or `if (q[i] < 0 || q[i] >= P) return DomainError;` barrier. Silent out-of-bounds indices handed to a CUDA kernel are a bad day.

**Two copies from the device decode result:**

```cpp
chrom_kept = ddr.chrom_kept;
genpos_kept = ddr.genpos_kept;
```

At lines 171–172, the host path copies two modest vectors out of the device decode result. This is probably negligible compared to the Q/V arrays, but it's a small reminder that the "resident" path still does some host-side bookkeeping. A comment acknowledging the cost/benefit would help.

**CPU-path SNP compaction is not cache-friendly:**

```cpp
for (long s = 0; s < M; ++s) {
    const int chr = snptab.chrom[static_cast<std::size_t>(s)];
    if (chr < kAutosomeChromMin || chr > kAutosomeChromMax) continue;
    const std::size_t src = static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
    for (int p = 0; p < P; ++p) {
        Qk.push_back(dec.q[src + static_cast<std::size_t>(p)]);
        Vk.push_back(dec.v[src + static_cast<std::size_t>(p)]);
    }
    ...
}
```

Lines 179–189 push `P` doubles into `Qk` and `Vk` per SNP. Since the decode layout is SNP-major (`src = P * s`), this is the correct memory order, but the per-element `push_back` with two separate vectors is a lot of branchy insertion. Precomputing the kept count or using a single `std::copy` per population would be cleaner and likely faster.

**Mixed integer-width types:**

The file uses `int N`, `long M`, `long M_kept`, `std::size_t P`, and `int n_block` somewhat interchangeably. `long` is 32-bit on Windows and 64-bit on LP64 Unix; for SNP counts that can run into the millions, `std::ptrdiff_t` or an explicit project type (`int64_t`) would be safer. The casts are everywhere and add visual noise.

## The "slop" test

**Not slop.** Slop is:
- Magic numbers without explanation
- Copy-pasted code with stale comments
- No error checking
- Obviously wrong algorithms that happen to pass tests

This file has none of those. The magic numbers (`kPloidyDiploid = 2`, `kAutosomeChromMin/Max`, `kPrimaryGpu = 0`) are named and motivated. The comments are dense but explain *why*, not just *what*. The only slop-adjacent smell is the repeated NaN-filling and the line-number cross-references, which are maintainability issues rather than correctness issues.

## What it actually looks like

This looks like **solid research-engineering code written by a domain expert who is comfortable with the project's backend seam but still thinks in terms of function-level orchestration rather than library-level abstraction.** The author clearly knows the admixture-statistics pipeline, knows where the CUDA path must diverge from the CPU oracle, and documents those decisions well.

A senior C++ reviewer would say: "Competent, well-reasoned, but let's DRY the NaN fills and replace the brittle line-number comments with stable tags." A CUDA reviewer would say: "The host-side orchestration is fine; the actual device work is hidden behind `dstat_blocks_jackknife`, so I can't judge occupancy or register pressure here, but the seam is clean."

The biggest risk is **comment rot**. There are entire paragraphs in the header comment and above `kPrimaryGpu` that describe implementation details in other files. When those files change, this file will silently become misleading. That is the most senior-reviewer-visible flaw in an otherwise respectable file.

**Verdict:** B+ — strong domain work, clean seam, but let down by brittle cross-file references, repeated boilerplate, and a simplistic device-residence heuristic.
