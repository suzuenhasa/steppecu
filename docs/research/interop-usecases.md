# steppe Python interop — USE-CASE + recommendation report

**Synthesis of 4 lenses** (results-out · f2-interchange · genotype-in · protocols-ecosystem)
**Scope:** what is the nanobind DLPack/`__cuda_array_interface__` (CAI) interop seam *practically for*, across the Python array ecosystem, and what M(py-1) should expose.
**Decision frame (ADR-0002 / the PyCUDA research):** nanobind-only binding; the interop seam is realized **on the nanobind module** via `nb::ndarray` / `nb::device::cuda` (zero new deps). **No PyCUDA.** DLPack is the primary GPU protocol; CAI is a compatibility fallback.

---

## TL;DR — the headline

The single most valuable Python deliverable is **`results → pandas DataFrame`**, and it is **NOT an array-interop seam at all** — it is host-side numpy/pandas, zero DLPack/CAI/GPU machinery. The canonical workflow steppe exists to serve ("precompute one f2, screen thousands of models, then **sort/filter/rank a table**") terminates in a DataFrame, not a GPU array. Conflating "we need DLPack" with "we need DataFrames" is the central trap; all four lenses independently flagged it.

The DLPack/CAI **array seam earns its keep at exactly one object: the f2_blocks interchange.** Even there, the *must* is the boring host-numpy export (`to_host()` → `(P,P,n_block)` float64), and the GPU zero-copy export is a **guarded SHOULD** — genuinely valuable but only for methods-developers at large P, and the one feature most at risk of being "zero-copy because we can."

**The must-haves (ship in M(py-1)):**
1. Bound typed result objects + `results → pandas DataFrame` with AT2/admixr column names verbatim. *(value HIGH × effort S — host numpy/pandas, not an array protocol.)*
2. `f2.to_host()` → numpy `(P, P, n_block)` float64 view (+ `vpair`, `block_sizes`). *(HIGH × S — the one sanctioned D2H, fp64-clean, AT2-parity-exact.)*
3. Q/V/N allele-frequency array → f2 precompute (seam B; AT2 `afs_to_f2_blocks` analog). *(MED–HIGH × S–M — matches steppe's real upstream contract.)*

**The one-or-two real GPU-interop use cases (SHOULD, guarded):**
- **Read-only DLPack/CAI zero-copy export of the device-resident f2 tensor** → CuPy/PyTorch for a methods-developer who wants to *touch* the multi-GB resident f2 in place (PCA over f2, custom block-jackknife, a novel statistic) without an 8 GB D2H round-trip. Real value appears only when the resident f2 is multi-GB (≈8 GB at P≈600). Gate behind lifetime keep-alive + stream-sync + read-only + a loud FP64 warning.
- **DLPack/CAI device acceptance of a GPU-resident Q/V/N array** → feed an msprime/tskit simulation or a GPU QC pipeline straight into the precompute without a host bounce. Cheaper as an *annotation on the existing import entry* than as its own feature; same-device + fp64-enforced.

Everything else (DLPack-on-results, GPU genotype zero-copy as a headline, cuDF/anndata result frames, autograd-over-fit, JAX-as-flagship, TF/MLX targets, internal zarr writer) is **YAGNI for M(py-1).**

---

## 1. The mental model — interop at steppe's 3 data objects × the ecosystem

steppe touches the Python world at exactly three data objects. They sit at **different altitudes** and have **completely different interop profiles** — the central insight is that they are not symmetric, and the GPU-array hype attaches to only one of them.

```
   GENOTYPE / Q-V-N IN            f2_blocks INTERCHANGE              RESULTS OUT
   ───────────────────           ─────────────────────             ────────────
   files (EIGENSTRAT/            DeviceF2Blocks (VRAM,              QpAdmResult{weight,se,z,
   PACKEDANCESTRYMAP/            P×P×n_block FP64,                  p,chisq,dof,f4rank,
   PLINK) → io leaf →            precompute-once)                  feasible,rankdrop_*,
   per-pop Q/V/N  →              ──to_host()──▶                    popdrop_*,status}
   ──extract_f2──▶               F2BlockTensor (host mirror)       + vector<QpAdmResult>
                                 + STPF2BK1 disk cache             (the rotation)
                                 + AT2-compat f2 dir
   ─────────────────────────────────────────────────────────────────────────────
   altitude: TALL input          altitude: the INTERCHANGE         altitude: SHORT output
   (must reduce to Q/V/N)        (the precompute↔fit artifact)     (tabular, host-side)
   interop: HOST import          interop: HOST numpy (must) +      interop: HOST numpy/
   (numpy/allel/tskit);          GPU DLPack/CAI (the ONLY real     pandas ONLY — NOT an
   GPU accept is incremental     GPU-array use case)               array-protocol case
```

**Why the asymmetry matters (the load-bearing facts, verified against the headers):**

- **`F2BlockTensor` is bit-for-bit AT2's `f2_blocks`.** Header `include/steppe/fstats.hpp`: `f2[i + P·j + P·P·b]`, length `P·P·n_block`, FP64 in every precision mode, plus `vpair` and `block_sizes`. ADMIXTOOLS 2 already hands this object to users **as a 3-D array they index directly** — `dim(f2_blocks) == c(7,7,708)`, `f2_blocks[,,1]`, `apply(f2_blocks, 1:2, mean)`, `qpadm(f2_blocks, ...)`. This is the strongest, *already-exercised* interop match in the whole project — not speculative. (Source: AT2 tutorial / `extract_f2` docs.)
- **`DeviceF2Blocks` is the resident, multi-GB GPU artifact** (header `src/device/device_f2_blocks.hpp`): column-major `i + P·j + P·P·b`, `f2_device()`/`vpair_device()` borrowed pointers, **move-only, frees in its destructor**, and `to_host()` is documented as **"THE ONLY D2H + host alloc in the whole pipeline."** This is what the GPU zero-copy seam would export — and its move-only self-freeing lifetime is the #1 spike risk.
- **`QpAdmResult` maps ~1:1 onto AT2/admixr column conventions** (header `include/steppe/qpadm.hpp`): `weight/se/z`; `f4rank/dof/chisq/p` + `rankdrop_{dofdiff,chisqdiff,p_nested}`; `popdrop_{pat,wt,feasible}`. R users porting notebooks will read these columns 1:1. This is a DataFrame, not a GPU tensor.
- **steppe's real upstream contract is Q/V/N, not genotypes.** Per `src/core/internal/views.hpp`, `extract_f2` consumes three column-major `[P×M]` FP64 arrays (ref-allele freq, validity mask, non-missing haploid count) — there is already a byte-stable `Q.f64/V.f64/N.f64` `--load` format. So the genotype seam has *two* altitudes: the tall genotype array (must reduce; AT2 `f2_from_geno` analog) and the lower allele-frequency/Q-V-N seam (AT2 `afs_to_f2_blocks` analog, which already exists in AT2). The lower seam is the better interop target.

**Ecosystem map (which libraries actually attach where):**

| Object | numpy | pandas | CuPy | PyTorch | JAX | numba | cuDF | popgen stack |
|---|---|---|---|---|---|---|---|---|
| genotype/Q-V-N IN | host import (must-able) | — | GPU accept (incr.) | GPU accept (fp32 trap) | GPU accept (fp32 trap) | — | — | **scikit-allel `count_alleles()`, tskit/msprime AFS, sgkit** |
| f2 INTERCHANGE | **`to_host()` export (must)** | label helper (nice) | **DLPack export (the real GPU case)** | DLPack export (fp32 trap) | DLPack export (fp32 trap) | CAI export (legacy) | DLPack (rare) | AT2 `f2_blocks` 3-D array convention |
| results OUT | **vector fields (must)** | **`to_dataframe()` (THE must)** | — | — | — | — | YAGNI | **admixr/AT2 `qpadm_rotate` nested-frame convention** |

---

## 2. The practical use cases (ranked: value × effort)

| # | Avenue | Data object | Concrete popgen workflow | Library | Value | Effort | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | `results → DataFrame` (rotation → ranked feasibility table) | results OUT | precompute once → screen 1000s of models → **sort/filter by feasible/p/chisq, rank candidates** — the canonical AT2/admixr loop | pandas (soft dep) | **H** | **S** | **MUST** |
| 2 | typed result objects + `weights_df`/`rankdrop_df`/`popdrop_df` per-type | results OUT | inspect one model's weights±SE, nested rank table, popdrop table — 1:1 with AT2 `res$weights/$rankdrop/$popdrop` | numpy/pandas | **H** | **S** | **MUST** |
| 3 | `f2.to_host()` → numpy `(P,P,n_block)` float64 (+ vpair, block_sizes) | f2 INTERCHANGE | export the f2 tensor for AT2-parity debugging, custom block stats, `f2[,,b]` slicing — the AT2 power-user pattern | numpy | **H** | **S** | **MUST** |
| 4 | Q/V/N (allele-freq) array → f2 precompute | genotype IN | feed allele freqs computed elsewhere straight to the f2 kernel — AT2 `afs_to_f2_blocks` analog, steppe's real contract | numpy/CuPy | **M–H** | **S–M** | **MUST/SHOULD** |
| 5 | genotype array `(n_var, n_samp, ploidy)` → Q/V/N reduction → f2 | genotype IN | **msprime/tskit sim → qpAdm power analysis** (the flagship sim loop); clean sim data dodges ploidy/polarity hazards | tskit, scikit-allel | **H** | **M** | **SHOULD** |
| 6 | numpy `(P,P,n_block)` → `F2BlockTensor` import → `upload_f2_blocks_to_device` | f2 INTERCHANGE | load an f2 computed by AT2 (or a cached/edited one) and fit it in steppe | numpy | **M** | **S–M** | **SHOULD** |
| 7 | **read-only DLPack/CAI zero-copy export of `DeviceF2Blocks`** | f2 INTERCHANGE | **methods-dev touches the resident multi-GB f2 in place** (PCA over f2, custom jackknife, novel stat) without 8 GB D2H | **CuPy** (fp64-clean), PyTorch | **M** (H at large P, methods-dev) | **M** | **SHOULD (guarded)** |
| 8 | DLPack/CAI device acceptance of GPU-resident Q/V/N | genotype IN | GPU sim/QC pipeline → precompute with no host bounce (annotation on #4's entry) | CuPy | **L–M** | **S** (incremental) | **SHOULD (guarded)** |
| 9 | scikit-allel QC (`GenotypeArray.count_alleles`) → steppe | genotype IN | filter/QC in allel, then fit in steppe — route through #4/#5 via a **doc recipe**, not bespoke C++ | scikit-allel | **M** | **S** (doc only) | **SHOULD (doc)** |
| 10 | pure-Python `to_xarray()`/`to_pandas()` label helper on the f2 tensor | f2 INTERCHANGE | attach pop names/coords to the 3-D f2 array for labelled slicing | xarray/pandas | **L–M** | **S** | **NICE** |
| 11 | DLPack export **on results** (weights as GPU tensors) | results OUT | — (no real consumer; results are tiny + tabular) | torch/JAX | **L** | **M** | **YAGNI** |
| 12 | GPU-resident **genotype** zero-copy as a headline feature | genotype IN | — (semantics minefield: pop-partition, ploidy, polarity all lost) | cuDF/torch | **L** | **M–L** | **YAGNI** |
| 13 | cuDF / anndata **result** frames | results OUT | — (`.to_numpy()`/pandas covers it; results are small + host) | cuDF/anndata | **L** | **M** | **YAGNI** |
| 14 | autograd-over-fit / differentiable qpAdm | results OUT | — (fit is a fixed GLS/SVD pipeline, not a layer) | JAX/torch | **L** | **L** | **YAGNI** |
| 15 | internal zarr/xarray **writer** for the f2 cache | f2 INTERCHANGE | — (free on the exported numpy handle; STPF2BK1 already exists) | zarr | **L** | **M** | **YAGNI** |
| 16 | per-framework hand-written paths (TF/MLX/JAX-specific) | all | — (DLPack covers the whole ecosystem; tags are docstring sugar) | TF/MLX | **L** | **M–L** | **YAGNI** |

---

## 3. The honest split — genuine value vs. "zero-copy because we can"

### Genuinely valuable (real workflows, real users)

- **results → pandas DataFrame (the #1 must).** This is the deliverable the whole precompute-once/fit-many design (ADR-0005) exists to produce: one expensive f2, thousands of cheap fits, then a **table you sort and filter**. AT2's `qpadm_rotate`/`qpadm_multi` set the expectation of a nested results frame; admixr is built around `data.frame` results. **It is not an array-protocol case at all** — bind the structs, expose numpy float64 vector fields, layer `.to_dataframe()` with AT2/admixr column names verbatim. **Column-name fidelity is the single highest-leverage decision** (R notebooks port mentally 1:1). FP64 is native-correct here (numpy/pandas default float64). pandas a **soft/lazy** dep (the scikit-allel pattern), not the core return.

- **f2 host-numpy export (must).** `to_host()` already exists (the one sanctioned D2H). A real minority of the exact users steppe wants to win — AT2 power-users — already work at the array level (`f2_blocks[,,1]`, `apply(f2_blocks, 1:2, mean)`). fp64 is clean (numpy native). It backstops scikit-allel/pandas/zarr/xarray and is the parity/debug trust anchor. Cheap, high value, no GPU machinery.

- **Q/V/N array → f2 (must/should), and genotype-array → Q/V/N (should).** steppe's real contract is Q/V/N, and AT2 already exposes both altitudes (`afs_to_f2` / `f2_from_geno`). The **flagship is the msprime/tskit simulation → qpAdm power-analysis loop** — clean simulated data sidesteps the worst hazards. Route scikit-allel through a documented `count_alleles()` recipe rather than bespoke C++.

- **The ONE real GPU-array use case: read-only DLPack/CAI export of the resident f2 (should, guarded).** This is the only place the hyped GPU zero-copy seam is genuinely earned, and only for a methods-developer at large P. When the resident f2 is multi-GB (≈8 GB at P≈600 for `f2`+`vpair`), forcing an 8 GB D2H just to compute a statistic CuPy could compute in place is real waste. **CuPy is the fp64-clean consumer.** But it serves a minority, and steppe's own fit engine already owns the VRAM-consumer role — so **build the mechanism, don't build analytics on it, and don't advertise it as the headline.**

### Speculative / "because we can" (be adversarial)

- **GPU zero-copy as a JAX/PyTorch headline** — actively an **anti-feature** for the f2 path. JAX defaults to float32 and silently truncates unless `jax_enable_x64=True`; PyTorch defaults to float32 and users habitually `.float()`. Advertising a JAX path on an fp64 artifact invites a silent precision catastrophe (see §4). DLPack/CAI preserve dtype on handoff — **the consumer's default is the footgun, not the protocol.**
- **DLPack on results** — results are tiny and tabular; no consumer wants weights as a GPU tensor.
- **GPU-resident genotype zero-copy** — a raw genotype array does not carry what steppe needs: the **population partition** is a mandatory second input (`pop_offsets`), **ploidy is metadata never inferred** (assuming ploidy=2 silently corrupts N for pseudo-haploid AADR — the single biggest correctness landmine), and **ref-allele polarity** must be reconciled. Pushing the reduction to the producer (Q/V/N seam) sidesteps all three.
- **cuDF/anndata result frames, autograd-over-fit, internal zarr writer, per-framework paths** — all covered by `.to_numpy()`/pandas, a fixed (non-differentiable) pipeline, the existing STPF2BK1 cache, or DLPack's universal coverage respectively.

---

## 4. Protocol recommendation + must-spike risks

### Protocol: one DLPack export + a CAI `@property` fallback; consume via DLPack

DLPack covers the whole GPU ecosystem (numpy/CuPy/PyTorch/JAX/cuDF). CAI (`__cuda_array_interface__`) is needed only for **numba** and **legacy RAPIDS**. **Per-library code is unnecessary** — nanobind's `nb::device::cuda` framework tags are docstring sugar; one DLPack path + one CAI property serves everyone. Host export is plain `nb::ndarray` over numpy buffers. (Sources: nanobind ndarray docs, DLPack Python spec, Array API, numba CAI spec, CuPy interop.)

### The four must-spike risks (all SILENT-correctness failures — each needs a round-trip parity test)

1. **VRAM ownership / lifetime (the spike the PyCUDA report flagged).** `DeviceF2Blocks` is **move-only and frees in its destructor** (header confirmed). A CuPy/torch view that outlives it is a **use-after-free on VRAM**. The exported view's lifetime MUST be tied to the owner via the `nb::capsule` deleter / keep-alive. Highest-priority spike.
2. **Stream sync.** DLPack/CAI v3 put the **producer** on the hook to make data valid on the consumer's stream; nanobind "knows neither the stream nor the runtime." Safest mitigation: export only **after the precompute stream is synced** (the f2 is a finished artifact, so this is free) and honor the DLPack `__dlpack__(stream=...)` handshake.
3. **FP64 (the brief's flagged caveat).** steppe is FP64 end-to-end; the emulated-FP64 fit policy protects ~1e-12 precision. **Host path is safe** (numpy/pandas/CuPy default float64). **GPU export is the danger: JAX silently truncates to float32 unless `jax_enable_x64=True`; PyTorch defaults to float32 and down-casts on any mixed op.** A bad import silently destroys the precision steppe exists to provide. Export must carry float64 and the docstring must warn loudly; **CuPy is the only fp64-clean default consumer.**
4. **Layout.** The f2 tensor is **column-major** (`i + P·j + P·P·b`, F-contiguous per slab). Must export F-contiguous or numpy/CuPy users silently get a per-slab transpose — and the f2 slab's **symmetry hides the bug** until an asymmetric op exposes it. (Sources: DLPack Python spec, JAX x64 docs, PyTorch dtype docs, numba CAI v3, CuPy interop.)

---

## 5. Concrete M(py-1) interop scope

**MUST (ship):**
- Bound `QpAdmResult` / `QpWaveResult` typed objects with numpy float64 vector fields.
- `rotation → ranked-feasibility DataFrame` (the headline) + per-type `weights_df` / `rankdrop_df` / `popdrop_df`, **AT2/admixr column names verbatim**; pandas a soft/lazy dep.
- `f2.to_host()` → numpy `(P, P, n_block)` float64 (+ `vpair`, `block_sizes`), with a loud "this is the one D2H, GBs at large P" docstring.
- Q/V/N (allele-frequency) array → f2 precompute (seam B), with validation (`V≠0 ⟺ N>0`, fp64, contiguity).

**SHOULD (ship if cheap, guarded):**
- numpy `(P,P,n_block)` → `F2BlockTensor` import → `upload_f2_blocks_to_device` (load an external/cached f2 and fit it).
- genotype array → Q/V/N reduction (seam A), requiring an **explicit sample→pop map** and a **per-sample ploidy / pseudo-haploid flag** (never auto-detect ploidy).
- read-only DLPack/CAI **export** of the resident f2, ONLY with: synced-artifact export + lifetime keep-alive + read-only flag + honest `device_id` + a prominent FP64 warning.
- DLPack/CAI device **acceptance** of a same-device, fp64-enforced Q/V/N array (annotation on the import entry).
- scikit-allel / tskit / msprime **doc recipes** (no bespoke C++ adaptors).
- The documented FP64 caveat in every GPU-export docstring.

**NICE (defer unless free):**
- pure-Python `to_xarray()` / `to_pandas()` label helper on the f2 numpy handle.

**YAGNI (do not build in M(py-1)):**
- DLPack on results; GPU-resident genotype zero-copy; cuDF/anndata result frames; autograd-over-fit; JAX-as-headline; TF/MLX targets; internal zarr writer; per-framework hand-written paths; writable GPU views; auto-ploidy detection; polarity reconciliation; zero-copy validated only on the tiny golden (validate at scale).

---

## Sources (primary)

- ADMIXTOOLS 2 — `extract_f2` / `f2_from_precomp` tutorial; `f2_blocks` 3-D array convention (`dim==c(7,7,708)`, `f2_blocks[,,1]`, `qpadm(f2_blocks,...)`); `afs_to_f2`/`f2_from_geno`; `qpadm`/`qpadm_rotate`/`qpadm_multi` result frames (`uqrad.github.io/admixtools`, `comppopgenworkshop2022` docs).
- admixr — `data.frame` qpAdm results convention (bodkan.net/admixr).
- nanobind — `nb::ndarray`, `nb::device::cuda`, DLPack/CAI framework tags (nanobind.readthedocs.io ndarray docs).
- DLPack — Python array-interchange spec, `__dlpack__(stream=...)` handshake, capsule lifetime (dmlc.github.io/dlpack; data-apis.org array-API `__dlpack__`).
- numba — `__cuda_array_interface__` v3 spec (numba.readthedocs.io CUDA CAI).
- CuPy — DLPack/CAI interoperability (`cupy.from_dlpack`, `toDlpack`), float64 default (docs.cupy.dev interoperability).
- JAX — `jax_enable_x64` / float32-default truncation (jax.readthedocs.io "Double (64 bit) precision").
- PyTorch — default float32 dtype, DLPack (`torch.utils.dlpack`) (pytorch.org docs).
- scikit-allel — `GenotypeArray`, `count_alleles`, `AlleleCountsArray` shapes (scikit-allel.readthedocs.io).
- tskit / msprime — `TreeSequence`, `allele_frequency_spectrum`, genotype matrix shapes (tskit.dev).
- sgkit — xarray/zarr genotype call conventions (sgkit-dev.github.io).
- steppe headers (read directly): `include/steppe/qpadm.hpp` (QpAdmResult fields), `include/steppe/fstats.hpp` (F2BlockTensor `f2[i+P·j+P·P·b]`, length `P·P·n_block`, fp64), `src/device/device_f2_blocks.hpp` (DeviceF2Blocks: move-only, `to_host()` = the only D2H, column-major borrowed device ptrs, `upload_f2_blocks_to_device`).
