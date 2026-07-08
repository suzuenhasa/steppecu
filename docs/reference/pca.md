# `steppe pca` reference

## 1. What it is

`steppe pca` runs a genotype **principal component analysis** on the GPU. You
hand it a genotype triple (`.geno`/`.snp`/`.ind`, in EIGENSTRAT / PLINK /
ANCESTRYMAP / packed form), it standardizes every SNP the classic Patterson way,
builds the sample-by-sample covariance matrix, eigendecomposes it, and hands back
the top-K principal-component coordinates for each sample. Those coordinates are
the scatter plot everyone in population genetics reaches for first: samples that
share ancestry land near each other, and the leading axes trace the big structure
in the data — continental splits on PC1/PC2, finer clines further down.

It answers a plain question: *given a pile of individuals, how do they group, and
along what axes of variation?* No model, no reference populations, no admixture
proportions — just the shape of the genotype cloud. It is a standalone,
engine-independent stat: it does not touch the f2 cache or the qpAdm fit engine,
it just reads genotypes and does linear algebra.

On top of the numbers, `--emit-html` writes a **self-contained interactive
scatter** you can double-click open — pan, zoom, hover for sample labels, swap
which PCs are on the axes, toggle populations in the legend. No server, no
internet, no dependencies (details in §6).

---

## 2. The method, and the tool it matches

### The standardization (Patterson 2006)

For each SNP, let `p` be the estimated alternate-allele frequency. steppe:

- **centers** each genotype dosage by subtracting `2p` (the expected diploid
  dosage),
- **scales** by `1 / sqrt(p(1-p))` — the Patterson normalization that gives every
  SNP a comparable variance contribution regardless of its frequency,
- **mean-imputes missing calls to 0** *after* centering (a missing genotype
  contributes nothing to that sample's projection at that SNP),
- **drops monomorphic SNPs** — a SNP with no variation has `p(1-p) = 0` and would
  divide by zero, so it is excluded and counted separately.

This is exactly what `scikit-allel`'s `allel.pca(scaler='patterson')` does, and
the per-SNP allele-frequency fold is the same one the FST and SFS paths already
use — PCA adds no new decode of its own.

### The linear algebra

With the standardized `N x M` matrix `Z` (N samples, M usable SNPs), steppe forms
the `N x N` covariance `C = Z Zᵀ` with a SNP-tiled cuBLAS **SYRK**, accumulated
tile by tile so the whole genotype block never has to sit in VRAM at once. It then
eigendecomposes `C` with cuSOLVER's symmetric solver. The PC coordinate is

```
coord(sample i, PC k) = eigenvector_k(i) * sqrt(eigenvalue_k)
```

which is exactly scikit-allel / sklearn's `U * S` (the singular-value scaling), so
the coordinates line up with the reference tool axis for axis. The reported
`var_explained[k]` is `eigenvalue_k / Σ eigenvalues`. Signs of individual PCs are
arbitrary (an eigenvector and its negation are both valid), the same as in every
PCA tool — a comparison sign-aligns per PC before checking.

### The precision split

This follows steppe's standard fit-precision policy. The covariance SYRK is
matmul-heavy, so it runs the **emulated-FP64 default** (the Ozaki matmul
accelerator). The eigendecomposition is cancellation-sensitive, so it is the
**native-FP64 carve-out** — cuSOLVER in true FP64. You can override the covariance
precision with `--precision`; the eigen solve is always native FP64.

### What it is gated against

Concordance was checked on box5090 (Release, `sm_120`, real data) against a
`scikit-allel 1.3.13` + `sklearn` PCA oracle — a numpy<2 venv fed through a
`plink2 --recode A` bridge, using the identical Patterson standardization, with no
steppe number fed back into the oracle. On **AADR v66 1240K, N = 430** across 9
real ancestries (Corded Ware, Bell Beaker, Han, Iran_N, Natufian, Karitiana,
Mbuti, Papuan, Turkey_N), **522,483 polymorphic SNPs**:

- per-PC sign-aligned **|Pearson r| = 1.000000 for PC1..PC12**,
- subspace Gram relative difference **6.7e-12**,
- variance-explained identical to 10 significant figures.

(Raw eigenvalues differ from scikit-allel's only by the documented `1/N`
convention factor — the variance-explained *ratios*, which is what you read, are
identical.) `nsys` confirmed the covariance and eigen steps are device-resident.
This matches **scikit-allel / sklearn PCA, not ADMIXTOOLS2** — ADMIXTOOLS2 has no
PCA, so there is nothing there to match.

---

## 3. Inputs

- **`--prefix PREFIX`** (required) — the genotype triple `PREFIX.{geno,snp,ind}`.
  Read through steppe's shared genotype front-end, so EIGENSTRAT, PLINK,
  ANCESTRYMAP, and packed formats all work.
- **`--pops A B C ...`** (optional) — restrict to these population labels (they
  become the color groups in the HTML). Omit it to keep **all** populations.

The samples are processed in tile (file) order, and that order is the row order of
every output — it is the join key the oracle gate used.

---

## 4. Outputs

By default steppe writes the **per-sample coordinate table**: one row per sample,
columns `sample`, `pop`, then `PC1 … PCk`. With `--eigenvalues` it instead writes
the **scree table**: one row per PC, columns `pc_index`, `eigenvalue`,
`var_explained`. Either goes to stdout, or to a file with `--out`, in `csv`
(default), `tsv`, or `json` (`--format`). The JSON coordinate form nests the
eigenvalue and var-explained arrays alongside the per-sample coords.

Independently, `--emit-html PATH` writes the interactive scatter to `PATH`. This
is a separate write path — you can produce the table **and** the HTML in one run.

To stderr (kept out of the parseable stream) steppe prints a one-line diagnostic:
how many samples and usable SNPs were used, how many monomorphic SNPs were
dropped, and the PC1 / PC2 variance-explained percentages.

---

## 5. The CLI

```
steppe pca --prefix PREFIX [--pops A B C ...] [-k K]
           [--eigenvalues] [--emit-html PATH]
           [--out FILE] [--format csv|tsv|json]
           [--precision ...] [--device N]
```

| flag | meaning |
| --- | --- |
| `--prefix PREFIX` | genotype triple `PREFIX.{geno,snp,ind}` (required) |
| `--pops A B C ...` | populations to include / color; omit = all |
| `-k`, `--k K` | number of principal components (default 10; must be ≥ 1) |
| `--eigenvalues` | emit the scree table instead of the coordinate table |
| `--emit-html PATH` | also write the self-contained interactive scatter here |
| `--out FILE` | write the table to a file (stdout if omitted) |
| `--format` | `csv` (default), `tsv`, or `json` |
| `--precision` | precision for the covariance SYRK (emulated-FP64 default) |
| `--device N` | CUDA device ordinal |

The exact invocation used for the timing below (top-12 PCs, TSV out):

```
steppe pca --prefix /workspace/data/aadr/fixtures_eig/v66_fit9_ped \
           -k 12 --device 0 --out t_pca.tsv --format tsv
```

**Measured wall-clock:** on that real AADR fixture (430 samples × 522,483
polymorphic SNPs, 61,648 dropped monomorphic, top-12 PCs), steppe runs in **2.53 s**
(median of 3; 674 MB peak RSS) on a single RTX 5090. The scikit-allel + sklearn
oracle was not separately wall-clocked at the gate, so there is no head-to-head
speedup number to quote here.

---

## 6. The interactive HTML artifact

`--emit-html` writes **one strictly self-contained file**: the CSS is inline in a
`<style>`, the JS is inline in a `<script>`, and every PC coordinate is embedded as
a JSON data literal in the page. There is **no external reference at all** — no
CDN, no `<link>`, no remote font, no network `fetch` — so the file opens offline by
double-click and was grep-proven to have zero external references. The renderer is
a dependency-free canvas-2D scatter with population coloring, a scree strip, and
PC-axis selectors, plus pan / zoom / hover-label / click-to-toggle-legend
interaction.

The data schema (generic `axisNames` + `coords`) is deliberately UMAP-ready: a
future nonlinear embedding slots in as two more axis entries with no template
change (see caveats).

---

## 7. Honest caveats

- **This is exact dense PCA, sized for AADR fixtures.** It forms the full `N x N`
  covariance and does a full symmetric eigensolve — right for hundreds to a few
  thousand samples. It is **not** a randomized-SVD biobank solver; at very large N
  the dense eigensolve is the wall. Randomized SVD for biobank-scale N is a
  documented follow-up, not built.
- **No UMAP (yet).** A nonlinear `--embed umap` axis is a documented follow-up and
  is deliberately **not** built — steppe takes no RAPIDS / cuML dependency for it.
  The coord output and HTML schema are shaped so it can be added later without a
  break, but today `steppe pca` is linear PCA only.
- **No EMU imputation and no projection of new samples.** Missing calls are
  Patterson mean-imputed to 0, not model-imputed (no EMU), and there is no
  shrinkage projection of unseen samples onto an existing PCA. Both are documented
  follow-ups.
- **Monomorphic SNPs are dropped, not an error.** A SNP with no variation is
  excluded and counted in the `dropped monomorphic` line — expected behavior, not a
  warning you need to act on.
- **PC signs are arbitrary.** As in every PCA tool, the sign of each individual PC
  is not meaningful; only the geometry (relative positions, the subspace) is. A
  comparison against another tool must sign-align per PC first.
- **For genotype likelihoods, use `steppe pcangsd` instead.** `steppe pca` reads
  hard genotype calls. If your data is low-coverage genotype likelihoods (a Beagle
  GL file), the PCAngsd-style GL-weighted PCA is a separate command.
