# The shared genotype-decode + allele-frequency primitive (decode_af)

> Source: `src/core/internal/decode_af.hpp` — the single source of the decode math so the
> CPU reference oracle and the GPU kernel CANNOT diverge. Cross-refs: architecture.md §5
> (S0/S1), §7, §8, §13; ROADMAP §2 (Q/V/N contract, M1).

THE shared, per-element genotype-decode + allele-frequency primitive — the single source
of the decode math so the CPU reference oracle and the GPU kernel CANNOT diverge
(architecture.md §5 S0/S1, §7, §8, §13; ROADMAP §2 Q/V/N contract, M1).

Mirrors `core/internal/f2_estimator.hpp`'s role: thin `__host__ __device__` scalar
functions that are (a) the per-element numerics the GPU kernel calls inside its segmented
reduction and (b) the exact same numerics the CPU reference calls in its scalar loop — so
the 2-bit unpack, the raw-value→genotype mapping, the missing-handling, and the AC/AN→Q/V/N
finalize are identical on both paths.

## THE DECODE CONVENTION (verified bit-for-bit against the on-box oracle `build_tgeno_matrix.py`; the corrected M1 facts)

- **2-bit code per genotype, RAW VALUE:** `0→0` ref-allele copies, `1→1`, `2→2`,
  `3→MISSING`. (NOT the binary mapping `00→0,10→1,11→2,01→missing`, which mis-decodes — the
  raw-value mapping reproduces the oracle, `max|Δ|=0`.)
- **Per (population, SNP):** over the individuals of that population,
  ```
      AC += code            for each NON-missing individual (code != 3)
      AN += 1               counting NON-missing INDIVIDUALS (not alleles)
  ```
  Missing individuals are excluded from BOTH AC and AN.
- Then, with a PER-SAMPLE PLOIDY (2 diploid / 1 pseudo-haploid — AUTO-detected per sample
  from the genotypes the AT2 way, see below). AT2's `cpp_*_to_afs` convention
  (`src/cpp_readgeno.cpp`:
  ```
      altalleles(pop)     += val / (3.0 - ploidy(i));
      observedalleles(pop) += ploidy(i);
  ```
  ) accumulates a per-sample-WEIGHTED allele count and a per-sample-summed haploid count:
  ```
    AC = Σ_i code_i / (3 - ploidy_i)   over NON-missing individuals
    N  = Σ_i ploidy_i                  over NON-missing individuals
    Q  = AC / N                        (ref-allele frequency in [0,1]; 0 where N==0)
    V  = (N > 0) ? 1 : 0               (validity mask)
  ```
  For a diploid sample (ploidy 2): `code/(3-2) = code`, contributes 2 to N — so an
  ALL-DIPLOID population reduces EXACTLY to the legacy `AC = Σcode`, `N = 2·non-missing`,
  `Q = AC/(2·AN)` (the modern-data path is bit-identical). For a pseudo-haploid sample
  (ploidy 1): `code ∈ {0,2}`, `code/(3-1) = code/2 ∈ {0,1}`, contributes 1 to N.
  MIXED-PLOIDY populations (a diploid sample and a pseudo-haploid sample in the same pop —
  real for aDNA: Turkey_N, Serbia, Yamnaya, Karitiana) are therefore handled correctly
  because the weighting is PER SAMPLE, not per population.

- **AT2 PSEUDO-HAPLOID AUTO-DETECTION** (the per-sample ploidy SOURCE, matching
  `adjust_pseudohaploid=TRUE`; `src/cpp_readgeno.cpp cpp_*_ploidy` verified against
  admixtools 2.0.10): scan the FIRST `kPloidyDetectSnps` (= 1000) SNPs of each sample; the
  sample is DIPLOID (ploidy 2) iff ANY of those SNPs is a HETEROZYGOUS call (`code == 1`),
  else PSEUDO-HAPLOID (ploidy 1). A haploid genome cannot be heterozygous, so observing a
  het ⇒ diploid; observing none ⇒ pseudo-haploid. (AT2 initializes `ploidy = 1` and bumps
  to 2 on the first het; an all-missing / all-homozygous prefix stays ploidy 1.) The
  detection is done UPSTREAM in the io leaf (`geno_reader::detect_sample_ploidy`) and
  carried as a per-sample vector on `DecodeTileView`, never re-derived in the primitive.

## `__host__ __device__` portability

Like `f2_estimator.hpp`, `STEPPE_HD` (from the single home `core/internal/host_device.hpp`)
expands to the CUDA qualifiers under nvcc and to nothing otherwise, so the SAME functions
compile and unit-test on the CPU and run on the GPU (the point of one shared primitive).
CUDA-free-compilable (it includes no CUDA header).
