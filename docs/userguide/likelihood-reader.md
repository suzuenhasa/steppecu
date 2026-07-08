# The genotype-likelihood reader (`steppe ingest --likelihoods`)

## 1. What it is, and what it's for

Most of steppe works on **hard calls** — every genotype is one of `0`, `1`, `2`
copies of an allele, no hedging. That's fine for high-coverage data, but for a
low-coverage or ancient sample the caller often isn't sure: it might think a site
is 70% likely to be a het, 25% homozygous-reference, 5% homozygous-alt. Throwing
that away and forcing a single hard call loses real information.

`steppe ingest --likelihoods` keeps the hedge. It reads a **genotype-likelihood**
field out of a VCF and turns it into a normalized `[n_site × n_sample × 3]`
likelihood tensor — three numbers per (site, sample), one for each possible
genotype, summing to 1. That tensor is the shared input for the two
low-coverage tools in this family:

- **`steppe pcangsd`** — PCA that reads genotype likelihoods instead of hard
  calls (the PCAngsd method), so it stays honest about uncertain sites.
- **`steppe ibd`** — the ancIBD 5-state forward-backward pass, which runs over
  genotype posteriors.

This command is the **reader only**. It parses, normalizes, uploads the tensor to
the GPU, and (optionally) writes it to disk. The two tools above are the "faces"
that consume it.

---

## 2. The three fields it can read

A VCF can store per-genotype uncertainty in one of a few `FORMAT` fields, and
they're scaled differently. You pick which one with `--gl-field`; there is **no
auto-detect** (a file can carry both PL and GL, and guessing wrong would silently
corrupt every site), so the choice is always explicit. `PL` is the default.

| `--gl-field` | What it is | How it's normalized (`src/io/gl_normalize.hpp`) |
|---|---|---|
| `PL` (default) | Phred-scaled likelihoods, non-negative integers | subtract the min, `10^(-(PL−min)/10)`, then divide by the sum |
| `GL` | log10-scaled likelihoods, may be negative | subtract the max, `10^(GL−max)`, then divide by the sum |
| `GP` | linear genotype **posteriors** (e.g. from GLIMPSE/Beagle) | just renormalize so it sums to exactly 1 |

The PL and GL paths subtract the min/max **before** exponentiating on purpose:
that keeps every exponent ≤ 0, so nothing overflows and the arithmetic stays
stable. All of this runs in native double precision — this is the
genotype-likelihood compute path that feeds PCAngsd and ancIBD, not the f2 cache,
so the emulated-FP64 matmul policy doesn't apply here.

One honesty note the reader carries through: GP is a *posterior*, not a
likelihood. The field it came from is recorded in the tile and in the on-disk
artifact header so a downstream consumer never mistakes a posterior for a
likelihood.

---

## 3. Missing data, and the present-mask

Real files have gaps: a site with no field, a `./.` call, the wrong number of
values, a multiallelic record, a gVCF reference block, something that won't
parse. The reader never emits a NaN for these (NaN sentinels are barred — they'd
poison a GPU reduction). Instead a missing genotype becomes the **uninformative**
triplet `(1/3, 1/3, 1/3)` — numerically safe for every consumer — and its slot in
a parallel **present-mask** is set to `0`.

That mask matters. It lets a consumer tell apart two very different things: a site
that was **observed but uninformative** (a real read gave a flat likelihood) from
a site that was simply **absent**. Both look like `(1/3, 1/3, 1/3)` in the tensor;
only the mask distinguishes them.

---

## 4. Polarity: the same axis as the hard calls

The third dimension of the tensor is indexed by `g` = **copies of the panel's A1
allele** — the exact same axis the 2-bit hard-call dosage uses. The reader reaches
this by reusing the hard-call reader's `reconcile()` step unchanged: when the
VCF's REF equals the panel A1, the RR and AA entries are swapped (and RA stays
put); otherwise the native order is kept.

The practical payoff: a genotype likelihood can't end up mis-polarized relative to
the hard-call tile. An `argmax` over `g` in the likelihood tensor points at the
same genotype the hard-call path would have reported — which is exactly what the
gate below checks.

---

## 5. What goes in, what comes out

**Inputs**

- `--vcf` — the input VCF (`.vcf.gz` BGZF/gzip, or a plain `.vcf`). Required in GL
  mode.
- A **target-site source**, same as the hard-call ingest path — either a pre-built
  `--targets` table, or the native `--panel` (+ `--fasta`, `--lift`) build. The
  tensor rows are joined to this panel, in panel order, and keyed by both rsID and
  position so a call only lands on the row it was built from.
- `--gl-field PL|GL|GP` — which field to read (default `PL`).
- `--sample` — which sample to read (default: the sole sample in the VCF).

**Outputs** (you must ask for at least one)

- `--emit-likelihoods PATH` — write the **STPGL1** tensor artifact. This is the
  path that touches the GPU: the tensor is uploaded device-resident, and a
  residency checksum (a real on-device reduction compared against the host sum,
  tolerance `1e-6`) proves it actually landed on the GPU rather than being a
  host-only parse. Needs a CUDA device.
- `--emit-pl-raw PATH` — a DEBUG dump of the raw parsed triplets in VCF-native
  order, self-keyed by rsID/chrom/pos38/sample. This is host-only (no device) and
  exists so the values can be diffed bit-for-bit against a `bcftools query` dump on
  the same positions. It's a validation aid, not a production output.

### The STPGL1 artifact

`--emit-likelihoods` writes a self-describing, seekable, little-endian binary file
(`src/io/likelihood_tensor_writer.hpp`): a 64-byte header (magic `STPGL1`,
version, site-major layout flag, which field it came from, FP64 dtype, site/sample
counts, and a section-offset table), then the sample-id table, the per-site
metadata table (rsID, chrom, pos37, pos38, A1, A2 — panel order), then the
`n_site × n_sample × 3` FP64 payload and the `n_site × n_sample` present-mask. The
layout is **site-major** — `l[(site·n_sample + sample)·3 + g]` — which gives
coalesced per-site tiles for the PCAngsd covariance and a fixed per-sample stride
for the ancIBD forward-backward.

---

## 6. The CLI

Minimal single-sample PL run against a pre-built target table (the shape used at
the gate):

```
steppe ingest --likelihoods \
  --vcf sample.vcf.gz \
  --targets target_sites.tsv \
  --gl-field PL \
  --emit-likelihoods sample.stpgl1 \
  --device 0
```

The real flags (read from `src/app/cli_parse.cpp` and `src/app/cmd_ingest.cpp`):

| Flag | Meaning |
|---|---|
| `--likelihoods` | Engage GL mode. Without it, `--gl-field`/`--emit-likelihoods`/`--emit-pl-raw` are rejected. |
| `--vcf PATH` | The input VCF. Required in GL mode. |
| `--targets PATH` | Pre-built GRCh38 target-site table (mutually exclusive with the native `--panel` build). |
| `--panel` / `--fasta` / `--lift` / `--assembly` | The native target-site build (same as hard-call ingest), if you're not passing `--targets`. |
| `--gl-field PL\|GL\|GP` | Which FORMAT field to read. Default `PL`. Case-insensitive. |
| `--sample ID` | Sample to read. Default: the only sample in the VCF. |
| `--emit-likelihoods PATH` | Write the STPGL1 tensor artifact (uploads device-resident + residency checksum; needs a CUDA device). |
| `--emit-pl-raw PATH` | DEBUG raw-triplet TSV in VCF-native order (host-only). |
| `--device N` | CUDA device ordinal for the tensor upload (default auto). |

GL mode deliberately does **not** apply the `--min-dp` / `--min-gq` /
FILTER floors that the hard-call path uses — a genotype likelihood is soft
information, and clipping it on a coverage threshold would defeat the point. Those
flags are accepted (they're shared machinery) but not applied to the tensor.

If you pass `--likelihoods` but don't ask for any output, the command tells you so
and exits rather than doing invisible work.

---

## 7. How it was validated

Gated on box5090 (Release, sm_120) against **real** data — the nikki 30× WGS gVCF,
which carries a `PL` field — at commit `8b97ee3`:

- **Decode, bit-exact.** Every parsed PL triplet matched a `bcftools` PL
  extraction on the same positions: **475,265 / 475,265 = 100%**.
- **ML-call concordance.** Taking `argmax` over the likelihood at each site and
  comparing to the already-validated hard-call GT path: **474,211 / 474,211 =
  100.0000%**. The likelihood tensor and the hard calls agree on the called
  genotype at every site — which is what the shared A1-copy axis is supposed to
  guarantee.
- **Device residency proven** by a real on-GPU kernel round-trip (the checksum
  above), not a host-side stand-in.
- Full `ctest` green (95/95), including the new `gl_normalize` unit test, the VCF
  GL-parse unit test, and the CLI likelihoods test.

**Measured wall-clock** (Release, `--device 0`, warmup + median-of-3): **12.27 s**
(runs 12.30 / 12.27 / 12.27; max RSS 719 MB) on the nikki 30× gVCF — 13,948,059
VCF records joined to the 1,103,526-site 1240K target table, 475,265 sites with PL
present. It's the slowest command in this family purely because of VCF I/O volume:
the time is dominated by scanning a 13.9M-record 30× gVCF, and the tensor upload
itself is device-resident and cheap. `bcftools` was used as a correctness oracle
at the gate and was not separately wall-clocked, so there's no honest speedup
number to quote here — the two weren't timed head-to-head.

---

## 8. Honest caveats

- **GP is unit-validated only.** The PL path is gated bit-exact on real data. The
  GP (posterior) path is exercised by unit tests but has **no** end-to-end real-data
  gate yet, because no imputed GP data was staged at release — and no synthetic GP
  was fabricated to fake one. Treat GP as tested-in-the-small until a real GP file
  is run through it.
- **Reader v1 scope.** The gate covered **biallelic, autosomal SNPs, single
  sample**. The reader accepts multi-sample input, but the multi-sample and GP
  paths get their real hardening from the ancIBD and PCAngsd face workflows that
  exercise them, not from this reader's own gate.
- **This is a reader, not an analysis.** It produces a tensor; it doesn't compute
  anything about the sample on its own. The science lives in `steppe pcangsd` and
  `steppe ibd`.
