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
pulls the **top-K eigenpairs** out of `C` with a **Halko randomized eigensolver** —
rather than solving the whole `N`-wide spectrum and throwing away everything past
PC-K, it sketches the leading K-dimensional subspace with a couple of matmul passes
and solves only the tiny K-sized problem inside it. That is exactly the PCA you
asked for (you plot PC1..PCk, never the tail), and it is what lets the full AADR
cohort fit in memory — see §7. The PC coordinate is

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
           [--project-pops X Y ...] [--project-samples FILE] [--project-mode lsq|scaled]
           [--eigenvalues] [--emit-html PATH]
           [--out FILE] [--format csv|tsv|json]
           [--precision ...] [--device N]
```

| flag | meaning |
| --- | --- |
| `--prefix PREFIX` | genotype triple `PREFIX.{geno,snp,ind}` (required) |
| `--pops A B C ...` | populations to include / color; omit = all |
| `-k`, `--k K` | number of principal components (default 10; must be ≥ 1) |
| `--project-pops X Y ...` | populations placed by **projection only** (see §5a) |
| `--project-samples FILE` | file of Genetic IDs (one per line), each projected-only |
| `--project-mode` | `lsq` (default, full least-squares) or `scaled` (diagonal ratio) |
| `--eigenvalues` | emit the scree table instead of the coordinate table |
| `--emit-html PATH` | also write the self-contained interactive scatter here |
| `--out FILE` | write the table to a file (stdout if omitted) |
| `--format` | `csv` (default), `tsv`, or `json` |
| `--precision` | precision for the covariance SYRK (emulated-FP64 default) |
| `--device N` | CUDA device ordinal |

### 5a. Projecting samples onto a reference (smartpca `lsqproject`)

`--project-pops` and `--project-samples` mark samples to be **placed by projection
only**. They are excluded from the PCA covariance **and** from the per-SNP allele
frequencies — the eigenbasis is built on the reference (everything not projected),
exactly as smartpca does with `lsqproject: YES`. Each projected sample is then
coordinated by a per-sample **least-squares fit over its non-missing sites**, so a
low-coverage sample is placed at its true position instead of being shrunk toward
the origin. The two flags combine (union); a projected population that is not in
`--pops` is auto-added to the decode set. The reference must be non-empty, `K` must
be `≤` the reference sample count, and a label cannot be both selected (`--pops`)
and projected.

`--project-mode`:
- **`lsq`** (default) solves the full `K × K` least-squares system per sample
  (`a = (WₒᵀWₒ)⁻¹ Wₒᵀ zₒ`), correcting both the shrink-to-origin **and** the
  coverage-induced correlation between PCs — identical to `lsqproject: YES`.
- **`scaled`** uses the diagonal ratio only (`a = (M_used / m_obs) · Wₒᵀ zₒ`), a
  faster / more robust path for extreme low coverage. A sample whose normal matrix
  is rank-deficient (fewer usable sites than `K`) automatically falls back to this
  ratio.

The **reference** samples keep their ordinary `U·S` coordinates (identical numbers
to a plain run), the **eigenvalues / var_explained are the reference spectrum**
(projected samples never perturb it), and every output row gains a projection flag:

- **CSV / TSV**: a trailing `is_projected` column (`0`/`1`) after `pop`.
- **JSON**: a `"projected": true|false` field per sample.
- **HTML**: projected points render as **hollow diamonds** to set them apart.

The flag/column appears **only** when a projection flag is given — with no
`--project-*` the output is byte-identical to a plain run. Example (build the
eigenbasis on five present-day panels, project a sixth):

```
steppe pca --prefix PREFIX_v66_1240K \
           --pops TSI,CHB,JPT,GWD,CHS --project-pops ITU -k 4
# stderr: 527 reference + 103 projected (lsqproject lsq) samples
```

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

**Full-cohort measured wall-clock:** the same command, pointed at the entire
AADR v66 1240K panel instead of the 9-pop fixture —

```
steppe pca --prefix PREFIX_v66_1240K -k 10 --device 0 --out pca_full.tsv --format tsv
```

— runs the whole **23,089-sample × 1.23M-SNP** cohort in **6:24** at **18 GB peak
VRAM** on a single RTX 5090. This used to be impossible: the old full-spectrum
eigensolve wanted an 8.5 GB workspace and OOM'd the full cohort outright. The Halko
top-K solver needs only ~190 MB of workspace, so the complete cohort now goes
through in one pass. The top-10 PCs it returns match the old full-spectrum solver
to **|r| = 1.0 on PC1–4** and **≥ 0.99998 on PC5–10** — the truncation costs you
nothing you would ever plot.

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

- **It scales to the full AADR cohort, and returns the top-K PCs.** steppe forms
  the full `N x N` covariance, then pulls the leading K eigenpairs out of it with a
  Halko randomized eigensolver instead of solving the whole `N`-wide spectrum. That
  swap is what lifted the old ~22k-sample cap: the full-spectrum solve needed an
  8.5 GB workspace and OOM'd the whole cohort, while the Halko path needs ~190 MB,
  so the complete **23,089-sample × 1.23M-SNP** panel now runs (**6:24, 18 GB peak
  VRAM**, one RTX 5090). What you get back is exactly the K PCs you asked for with
  `-k` — there is no full spectrum to inspect past that — and those top PCs match
  the old full solver (|r| = 1.0 on PC1–4, ≥ 0.99998 on PC5–10). The one remaining
  wall is the covariance itself: it is still a dense `N x N` matrix, so truly
  biobank-scale N (hundreds of thousands of samples and up) is still out of reach —
  but the whole present AADR cohort sits comfortably inside the envelope.
- **No UMAP (yet).** A nonlinear `--embed umap` axis is a documented follow-up and
  is deliberately **not** built — steppe takes no RAPIDS / cuML dependency for it.
  The coord output and HTML schema are shaped so it can be added later without a
  break, but today `steppe pca` is linear PCA only.
- **No EMU imputation.** Missing calls are Patterson mean-imputed to 0, not
  model-imputed (no EMU) — a documented follow-up. Least-squares **projection** of
  samples onto a reference eigenbasis (smartpca `lsqproject`) **is** now supported —
  see §5a.
- **Monomorphic SNPs are dropped, not an error.** A SNP with no variation is
  excluded and counted in the `dropped monomorphic` line — expected behavior, not a
  warning you need to act on.
- **PC signs are arbitrary.** As in every PCA tool, the sign of each individual PC
  is not meaningful; only the geometry (relative positions, the subspace) is. A
  comparison against another tool must sign-align per PC first.
- **For genotype likelihoods, use `steppe pcangsd` instead.** `steppe pca` reads
  hard genotype calls. If your data is low-coverage genotype likelihoods (a Beagle
  GL file), the PCAngsd-style GL-weighted PCA is a separate command.
