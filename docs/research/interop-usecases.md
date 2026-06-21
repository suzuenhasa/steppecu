# steppe array-interop — GENOTYPE INPUT lens (feed `extract_f2` from an in-memory array, not files)

Research analyst note: HONEST + adversarial. The default popgen workflow today is *file IO and it already works*. The bar for a new array-input path is not "zero-copy is cool" — it is "a real user has data in RAM that does not exist as a file, and the file round-trip is a genuine friction or a correctness hazard." Where that bar is not met, I say YAGNI.

---

## 0. The one fact that reframes this whole lens

steppe does **not** actually need a *genotype* array to feed the f2 precompute. Its real upstream contract is **Q/V/N**, and that contract already has a documented, byte-stable binary format that an array could fill directly.

From the steppe source (read, not inferred):

- `src/core/internal/views.hpp` — **THE Q/V/N CONTRACT**: three column-major `[P × M]` FP64 arrays per SNP-block. `Q` = reference-allele frequency in [0,1] (zero-filled where invalid), `V` = validity mask (1.0/0.0), `N` = **non-missing haploid count** (2× non-missing diploids, or 1× pseudo-haploids). Invariant `V != 0 ⟺ N > 0`. The header explicitly says this is "byte-compatible with the validated inputs" written by the oracle `build_tgeno_matrix.py` as `Q.f64 / V.f64 / N.f64`, and there is already a `--load` path that consumes them.
- `src/core/internal/decode_af.hpp` — the file path's job is *only* to get from packed 2-bit genotype bytes to Q/V/N: for each (pop, SNP) it accumulates `AC += code` (0/1/2), `AN += 1` over non-missing individuals, then `N = ploidy·AN`, `Q = AC/N`, `V = (AN>0 && ploidy>0)`. **ploidy is a metadata parameter, never inferred from genotypes.**
- `src/io/genotype_tile.hpp` — the file format produces TGENO individual-major 2-bit packed bytes + a population partition (`pop_offsets`) over the sample axis; the GPU kernel reduces individuals → per-pop Q/V/N.

So there are genuinely **two candidate array seams on the input side**, at different altitudes:

| Seam | Array shape | What it is | steppe analog | AT2 analog |
|---|---|---|---|---|
| **(A) Genotype seam** | `(n_variants, n_samples, ploidy)` int, + a population→sample map | raw per-individual calls; steppe must reduce to Q/V/N | replaces the `io` decode front-end | `f2_from_geno` |
| **(B) Allele-frequency seam (Q/V/N)** | three `[P × M]` FP64 `Q,V,N` (+ block partition) | per-population frequencies already reduced | feeds the f2 kernel directly, bypasses decode | `afs_to_f2` / `afs_to_f2_blocks` |

This split is the crux of the honest assessment, and AT2 itself ships **both** ([`f2_from_geno`](https://uqrmaie1.github.io/admixtools/reference/f2_from_geno.html) from genotypes, [`afs_to_f2`](https://rdrr.io/github/uqrmaie1/admixtools/man/afs_to_f2.html) from precomputed frequencies). AT2's per-SNP f2 is literally `(p1-p2)^2 - p1(1-p1)/(n1-1) - p2(1-p2)/(n2-1)` — i.e. exactly steppe's Q/V/N → het-corrected f2. **Seam B is not a steppe invention; it is the AT2 `afs_to_f2` contract expressed as an array.**

---

## 1. The popgen array ecosystem — what a "genotype array" actually carries (cited)

All three major Python popgen representations converge on the SAME shape, which matters for the mapping question:

- **scikit-allel `GenotypeArray`**: 3-D `(n_variants, n_samples, ploidy)`, dtype `int8`; allele encoding `0`=ref, `1..`=alt, `-1`=missing. Has `count_alleles()` → `AlleleCountsArray` `(n_variants, n_alleles)` int32, and `to_n_ref()` → reference-allele count per call. ([scikit-allel data structures](https://scikit-allel.readthedocs.io/en/stable/model/ndarray.html))
- **tskit `TreeSequence.genotype_matrix()`**: `(n_sites, n_samples)` numpy array, `int8`, where **samples are haploid sample nodes (not diploid individuals)**; missing data is the negative sentinel. Diploid grouping is via the Individuals table / sample sets. ([tskit Python API](https://tskit.dev/tskit/docs/stable/python-api.html), [tskit getting started](https://tskit.dev/tutorials/getting_started.html))
- **sgkit / xarray**: `call_genotype` is 3-D `(variants, samples, ploidy)` int; the Dataset also carries `variant_allele`, sample IDs, and supports mixed ploidy. ([sgkit user guide](https://sgkit-dev.github.io/sgkit/latest/getting_started.html))

The interop transport story is solid and dependency-free for steppe:

- **nanobind `nb::ndarray`** already exchanges with NumPy, PyTorch, TensorFlow, JAX, CuPy, MLX over the **buffer protocol** (CPU) and **DLPack** (GPU-capable), with `nb::device::cpu` / `nb::device::cuda` device tags and `c_contig`/`f_contig` constraints. Zero new deps — exactly the ADR-0002 conclusion. ([nanobind ndarray](https://nanobind.readthedocs.io/en/latest/ndarray.html))
- **GPU zero-copy**: CuPy, PyTorch, Numba all implement `__cuda_array_interface__` and DLPack; CuPy↔PyTorch GPU exchange is zero-cost. ([CuPy interoperability](https://docs.cupy.dev/en/stable/user_guide/interoperability.html), [Numba CUDA Array Interface v3](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html))

---

## 2. The mapping problem — does a raw genotype array carry enough? (the adversarial core)

**steppe needs Q, V, N with a known ploidy and a population partition. A raw `(n_variants, n_samples, ploidy)` array does NOT carry all of that.** Concretely:

1. **Population partition is NOT in the array.** steppe reduces *over the individuals of each population* (`pop_offsets` in `genotype_tile.hpp`). A bare genotype matrix has a sample axis with no pop labels. The user must supply a sample→population map (the `.ind`-file's 3rd column). Every workflow below must carry this alongside the array. This is a real, unavoidable second input.
2. **Ploidy is metadata, never inferred.** `decode_af.hpp` is explicit: ploidy is a parameter `∈{1,2}`. scikit-allel's ploidy axis tells you the *storage* ploidy, not the *biological* ploidy you want for N. **Pseudo-haploid ancient DNA is the trap**: AADR pseudo-haploids are coded 0/2 in a diploid slot but only one allele is observed, so N must use ploidy=1. AT2 handles this with `adjust_pseudohaploid` (auto-detects samples with no heterozygous calls in the first 1000 SNPs). ([f2_from_geno](https://uqrmaie1.github.io/admixtools/reference/f2_from_geno.html)) **A naive array→Q/V/N path that assumes ploidy=2 will silently miscompute N for exactly the ancient-DNA data steppe targets** — a correctness hazard, not a convenience. This is the single strongest argument that the *genotype* seam (A) is more dangerous than it looks.
3. **Reference-allele orientation.** steppe's Q is frequency of the *fixed reference allele* (`.snp` col 5). scikit-allel/tskit allele indices are relative to each site's own allele table; "allele 0" is ref by convention but tskit's "ancestral state" / msprime's `0` is not guaranteed to be the same physical allele as a target .snp file's ref. For an **end-to-end simulated** pipeline (msprime → steppe → qpAdm) this is internally consistent and fine. For **mixing** a simulated/QC'd array against a file-based reference panel, allele polarity must be reconciled — out of scope for M(py-1), flag loudly.
4. **Missing data.** Maps cleanly: array sentinel (`-1` / negative) → steppe's "exclude from AC and AN". Low risk.

**Verdict on the mapping:** A genotype array (seam A) carries the genotypes and (with the storage-ploidy axis) *most* of what's needed, but **not** the population map, **not** reliably the biological ploidy/pseudo-haploid status, and **not** guaranteed ref polarity. The frequency seam (B, Q/V/N) sidesteps ploidy and decode entirely because the producer already did the reduction — but then the producer owns the het-correction N convention and the precondition `V!=0 ⟺ N>0`.

---

## 3. Concrete use cases, each tied to a real workflow

### UC-1 — msprime/tskit simulation → steppe f2 → qpAdm (methods-paper / power-analysis loop)
**Workflow (real):** simulate a demography in msprime, get `ts.genotype_matrix()` `(n_sites, n_samples)` int8 in RAM, group haploid sample nodes into diploid individuals and into populations, run steppe f2 + qpAdm; repeat over thousands of parameter draws to characterize qpAdm power / false-positive rate. This is *the* canonical reason a methods person wants array input — they generate millions of replicate datasets that **never should touch disk**.
- **Value: HIGH.** This is the one workflow where files are pure friction: writing EIGENSTRAT per replicate dominates runtime and pollutes the filesystem; the data is *born* in numpy. It also exercises steppe's precompute-once/fit-many at its best (one sim → many qpAdm models). Aligns with the project's own "design for the S8 rotation / power analysis" memory.
- **Effort: M.** The transport is trivial (numpy int8 over the buffer protocol). The work is the **reduction kernel from a genotype array → Q/V/N** plus accepting the sample→pop map and a per-sample ploidy/haploid-flag vector. msprime data is clean (no missingness, known polarity, true ploidy), so the dangerous parts of §2 don't bite *here* — but the code path must still exist.
- **FP64 caveat:** msprime genotypes are integer; no fp32 issue at input. steppe stays FP64 internally. Fine.
- **What M(py-1) exposes:** see §5, "should."

### UC-2 — Allele-frequency / Q-V-N array → steppe f2 (the AT2 `afs_to_f2` analog) — seam B
**Workflow (real):** a user (or an upstream tool) has already computed per-population allele frequencies + sample counts (e.g. from a huge biobank where individual genotypes never fit in one array, or from a tool that emits AFs), and wants f2/qpAdm without re-deriving frequencies. Also: someone who wrote out steppe's own `Q.f64/V.f64/N.f64` and wants to re-feed them.
- **Value: MED-HIGH.** Direct AT2 precedent ([`afs_to_f2`](https://rdrr.io/github/uqrmaie1/admixtools/man/afs_to_f2.html), [`afs_to_f2_blocks`](https://rdrr.io/github/uqrmaie1/admixtools/man/afs_to_f2_blocks.html)); matches steppe's *actual* internal contract (`views.hpp`) one-to-one; sidesteps every ploidy/polarity/pop-map hazard in §2 by pushing them to the producer. It is the cleanest, most honest seam steppe can offer.
- **Effort: S-M.** The Q/V/N → f2 kernel already exists; this is "accept three `[P×M]` arrays + a block partition and call the existing precompute." Smallest new surface of any input option.
- **Caveat:** the contract burden moves to the user — they must honor `V!=0 ⟺ N>0` and the haploid-count N convention (alleles, not individuals). Needs a validating wrapper, not a raw pointer grab.
- **What M(py-1) exposes:** this is the **MUST**, see §5.

### UC-3 — scikit-allel QC pipeline → steppe
**Workflow (real):** a user filters/LD-prunes/QCs real genotypes in scikit-allel (`GenotypeArray`), then wants f2/qpAdm on the survivors without writing a new EIGENSTRAT.
- **Value: MED.** Real, but the honest counter is that scikit-allel users typically already round-trip through VCF/Zarr and a one-time EIGENSTRAT export is minutes against a precompute that is the expensive step anyway. The win is convenience + avoiding a lossy/again-polarized re-export, not throughput. `GenotypeArray.count_alleles()` / `to_n_ref()` ([scikit-allel](https://scikit-allel.readthedocs.io/en/stable/model/ndarray.html)) make it easy for the *user* to produce a Q/V/N triple themselves — which argues for routing this through seam B (UC-2) rather than a bespoke `GenotypeArray` adaptor in C++.
- **Effort: M if native genotype-array path; S if documented as "compute counts in allel, hand steppe Q/V/N."**
- **Caveat:** real data ⇒ all of §2's hazards (pseudo-haploid, missingness, polarity, pop map) are live. This is where assuming ploidy=2 would corrupt AADR.
- **What M(py-1) exposes:** **document the allel→Q/V/N recipe** (a few lines using `count_alleles`), feed via UC-2. No bespoke C++ `GenotypeArray` reader.

### UC-4 — CuPy/PyTorch/JAX **GPU-resident** genotype or AF array → steppe (zero-copy, same device)
**Workflow (claimed):** a user did GPU QC/imputation/simulation (e.g. a CuPy AF matrix, or a torch genotype tensor) and wants steppe to consume it in-place via DLPack/CAI, no D2H.
- **Value: LOW-MED, and mostly aspirational.** Adversarial read: the f2 precompute is the expensive, throughput-dominant step; saving one H2D of a genotype/AF array in front of it is in the noise relative to the precompute GEMMs. The genuine zero-copy win in steppe is on the **f2_blocks interchange** (the precompute-once artifact) and **results**, not the input. Also: keeping the input on the *same* device steppe runs on is a real constraint (DLPack carries device id; mismatched device ⇒ copy or error anyway).
- **Effort: S incrementally** *if* the host array path (UC-2) already exists — nanobind's `nb::device::cuda` + DLPack gives it nearly for free. So it's cheap to add **as an annotation on the same entry point**, not a separate feature.
- **FP64 CAVEAT (load-bearing here):** PyTorch and JAX **default to float32** (`torch.float32`; JAX needs `jax_enable_x64=True`). ([PyTorch get_default_dtype](https://docs.pytorch.org/docs/stable/generated/torch.get_default_dtype.html), [JAX default dtypes / X64 flag](https://docs.jax.dev/en/latest/default_dtypes.html)) steppe's Q/V/N is **FP64** by contract. A torch/JAX AF array handed to steppe will silently be fp32 unless the user opted into x64 — steppe must **reject fp32 with a clear error (or copy-and-promote with a warning), never silently downcast its precision contract.** This caveat does not apply to integer genotype arrays (msprime/allel int8) — only to float AF arrays from the DL frameworks.
- **What M(py-1) exposes:** accept DLPack/CAI on the *same* entry point as UC-2/UC-1 via `nb::device::cuda`, but **enforce fp64** and **same-device**. Do not build a separate "GPU input" feature.

### UC-5 — pandas / cuDF / anndata DataFrame genotypes
- **Value: LOW.** Genotype/AF data is fundamentally a dense numeric matrix; the DataFrame layer adds nothing steppe needs and anndata/scanpy is single-cell-shaped, not popgen-shaped. A user with a DataFrame does `.to_numpy()` and lands in UC-1/UC-2.
- **Effort: n/a.** **YAGNI.** Document `.to_numpy()`/`.values` and stop.

---

## 4. Ranking (value × effort)

| Rank | Use case | Seam | Value | Effort | Net |
|---|---|---|---|---|---|
| 1 | **UC-2** AF / Q-V-N array → f2 (`afs_to_f2` analog) | B | MED-HIGH | S-M | **Best ratio.** Matches steppe's real contract; smallest surface; AT2 precedent. |
| 2 | **UC-1** msprime/tskit sim → f2 → qpAdm | A | HIGH | M | The flagship workflow; clean data dodges the §2 hazards. |
| 3 | **UC-4** GPU-resident DLPack/CAI input | A/B | LOW-MED | S (incremental) | Cheap *as an annotation* on #1/#2; not its own feature. fp64 caveat. |
| 4 | **UC-3** scikit-allel QC → steppe | A→B | MED | S (as doc) / M (native) | Route through #1 via a documented recipe, not bespoke C++. |
| 5 | **UC-5** pandas/cuDF/anndata | — | LOW | — | YAGNI; `.to_numpy()`. |

---

## 5. What M(py-1) should ship

**MUST**
- **A Q/V/N array entry to the f2 precompute (seam B, UC-2).** Accept three host `[P×M]` FP64 contiguous `nb::ndarray` (Q, V, N) + a block-partition vector (SNP counts per block) + pop labels; validate `V!=0 ⟺ N>0`, FP64 dtype, contiguity, shape agreement; call the existing precompute. This is the smallest, safest, highest-fidelity seam, it mirrors AT2's `afs_to_f2`, and every other input use case can be expressed on top of it. ([afs_to_f2](https://rdrr.io/github/uqrmaie1/admixtools/man/afs_to_f2.html))

**SHOULD**
- **A genotype-array → Q/V/N reduction entry (seam A, UC-1).** Accept an int `(n_variants, n_samples)` or `(n_variants, n_samples, ploidy)` `nb::ndarray` + a **sample→population map** + a **per-sample ploidy/pseudo-haploid flag** (NOT inferred — match `decode_af.hpp`'s "ploidy is metadata" rule), reduce to Q/V/N, then reuse the MUST path. This unlocks the msprime power-analysis loop. Ship the pseudo-haploid handling (AT2's `adjust_pseudohaploid` behavior) or at minimum require an explicit per-sample ploidy and document that ploidy=2-by-default would corrupt aDNA.
- **DLPack/CAI device acceptance on those same two entries** via `nb::device::cuda`, enforcing **same-device** and **fp64-for-float-inputs**, with a hard error on fp32 (UC-4). Reuse, don't fork.
- **Docs: the scikit-allel and tskit recipes** — three or four lines each showing `GenotypeArray.count_alleles()`/`to_n_ref()` and `ts.genotype_matrix()` → the MUST/SHOULD entry. This satisfies UC-3 with near-zero C++.

**NICE-TO-HAVE**
- A thin convenience wrapper that takes a scikit-allel `GenotypeArray`/`AlleleCountsArray` or a tskit `TreeSequence` directly and does the count→Q/V/N reduction in Python (pure-Python sugar over the SHOULD entry). Optional, no C++.

**YAGNI**
- A bespoke C++ scikit-allel/sgkit/anndata adaptor; pandas/cuDF DataFrame input; GPU-resident input as a *separate* feature; auto-detecting ploidy from genotypes (explicitly forbidden by steppe's own contract); polarity reconciliation against external reference panels (push to the user; flag the mixing hazard in docs).

---

## 6. Caveats to carry forward (the honest list)

1. **Pseudo-haploid / ploidy is the correctness landmine.** Never assume ploidy=2 on real aDNA. Require explicit per-sample ploidy or replicate AT2's `adjust_pseudohaploid`. A silent ploidy bug corrupts N → het correction → every f2. ([f2_from_geno](https://uqrmaie1.github.io/admixtools/reference/f2_from_geno.html))
2. **Population map is a mandatory second input** — a genotype array alone is insufficient (steppe reduces per-population; `genotype_tile.hpp` `pop_offsets`).
3. **Reference-allele polarity** is internally consistent within one simulated/QC'd dataset but must be reconciled when mixing array data with file-based panels; out of scope for M(py-1), flag.
4. **FP64 vs fp32:** PyTorch/JAX default to fp32; steppe's Q/V/N is fp64. Reject/promote, never silently downcast. ([PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.get_default_dtype.html), [JAX X64](https://docs.jax.dev/en/latest/default_dtypes.html))
5. **The input seam is the low-value zero-copy site.** The precompute dominates; saving an input H2D is noise. The real zero-copy prizes are the f2_blocks interchange and results — keep input zero-copy *opportunistic* (free via DLPack), not a headline feature.

---

## Sources
- steppe headers (read): `src/core/internal/views.hpp` (Q/V/N contract), `src/core/internal/decode_af.hpp` (decode → Q/V/N, ploidy-is-metadata, pseudo-haploid), `src/io/genotype_tile.hpp` (packed bytes + pop partition), `include/steppe/fstats.hpp` (F2BlockTensor), `src/device/device_f2_blocks.hpp`, `include/steppe/qpadm.hpp` (results).
- scikit-allel data structures: https://scikit-allel.readthedocs.io/en/stable/model/ndarray.html
- tskit Python API (genotype_matrix / variants): https://tskit.dev/tskit/docs/stable/python-api.html ; getting started: https://tskit.dev/tutorials/getting_started.html
- sgkit getting started / user guide: https://sgkit-dev.github.io/sgkit/latest/getting_started.html
- nanobind ndarray: https://nanobind.readthedocs.io/en/latest/ndarray.html
- CuPy interoperability (DLPack + `__cuda_array_interface__`): https://docs.cupy.dev/en/stable/user_guide/interoperability.html
- Numba CUDA Array Interface v3: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
- ADMIXTOOLS 2 tutorial (extract_f2 / f2_from_geno / precompute-once): https://uqrmaie1.github.io/admixtools/articles/admixtools.html
- ADMIXTOOLS 2 `f2_from_geno` (pseudo-haploid, per-SNP f2 formula): https://uqrmaie1.github.io/admixtools/reference/f2_from_geno.html
- ADMIXTOOLS 2 `afs_to_f2` / `afs_to_f2_blocks` (allele-frequency → f2): https://rdrr.io/github/uqrmaie1/admixtools/man/afs_to_f2.html , https://rdrr.io/github/uqrmaie1/admixtools/man/afs_to_f2_blocks.html
- admixr (EIGENSTRAT ind/snp/geno file workflow, the baseline being replaced): https://uqrmaie1.github.io/admixtools/ , https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6736366/
- JAX default dtypes / X64 flag: https://docs.jax.dev/en/latest/default_dtypes.html
- PyTorch default dtype: https://docs.pytorch.org/docs/stable/generated/torch.get_default_dtype.html
