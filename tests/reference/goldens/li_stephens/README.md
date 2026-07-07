# Li-Stephens `paint` forward-backward golden (kalis oracle)

`golden.txt` freezes a **kalis** ([Aslett & Christ 2024, *BMC Bioinformatics* 25:1](https://doi.org/10.1186/s12859-024-05688-8))
exact Li & Stephens forward-backward posterior on a **tiny real phased panel**, as the
near-bit numerics oracle for steppe's CpuBackend reference forward-backward
(`ComputeBackend::ls_forward_backward`). Gated by `tests/reference/test_li_stephens_parity.cu`.

## Panel (real data — a dev parity oracle, not a user result)

- **12 real 1000 Genomes phase3 samples** (HG00096..HG00109), chr22:20,000,000–20,300,000,
  biallelic SNPs, phased → **24 phased haplotypes**.
- **256** polymorphic SNPs kept.
- A **real cM map**: HapMap-interpolated genetic positions
  ([joepickrell/1000-genomes-genetic-maps](https://github.com/joepickrell/1000-genomes-genetic-maps),
  `chr22.interpolated_genetic_map`).

kalis copies each recipient from **all other** haplotypes (Pi uniform, self excluded =
leave-one-out). The golden freezes kalis's **own** FB inputs — the donor allele matrix,
`rho` (in steppe's per-column convention: `rho[0]=1` unused, `rho[l]` = recomb entering
column `l`), a scalar `mu` emission rate (0.002), and each recipient's `Pi` — **and** its
full per-SNP posterior `gamma` (K donors × M SNPs) for 3 recipients. The gate feeds those
identical inputs to steppe's FB and reports `max |gamma_steppe − gamma_kalis|`.

Freezing kalis's own inputs isolates the check to the FB **recursion + rescaling** math;
`gamma = alpha·beta` normalized per column is invariant to the per-column rescaling
constants, so a correct recursion matches kalis regardless of rescale bookkeeping.

## Regeneration (`gen_golden.R`, on box5090 with `kalis` installed)

```
# real phased slice + real map (see the scp'd gen_golden.R header)
bcftools view -r 22:20000000-20300000 -s <12 samples> -m2 -M2 -v snps <1000G chr22 phased VCF> \
  | bcftools norm -d all | bcftools view -Oz -o slice.vcf.gz
bcftools query -f '%POS[\t%GT]\n' slice.vcf.gz > gt.tsv
wget https://raw.githubusercontent.com/joepickrell/1000-genomes-genetic-maps/master/interpolated_from_hapmap/chr22.interpolated_genetic_map.gz -O map.gz
Rscript gen_golden.R   # kalis: CacheHaplotypes -> CalcRho -> Parameters -> Forward/Backward/PostProbs
```

## Format (`golden.txt`, whitespace-tokenized, one record per line)

```
K <n_donors>            # == the 24 cache haplotypes
M <n_snps>
R <n_recipients>
mu_scalar <double>      # informational
D <M ints 0/1>          # x K, donor-major donor allele rows
RHO <M doubles>         # steppe convention (rho[0]=1 unused)
MU <M doubles>          # per-site emission rate
REC <self_donor_index0> # then, per recipient:
A <M ints 0/1>          #   recipient alleles
PI <K doubles>          #   copying prior (self entry 0)
G <K*M doubles>         #   kalis posterior gamma, donor-major [k*M+l]
```
