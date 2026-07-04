# extract-f2 (build the f2 cache)

Precompute the f2 statistics for a set of populations and write them to an f2-dir that the other tools read.

## What it does

`extract-f2` is the foundational step. It takes a genotype dataset (an EIGENSTRAT/PACKEDANCESTRYMAP triple, or a PLINK `.bed`/`.bim`/`.fam`), decodes it into per-population allele frequencies, filters the SNPs, splits the surviving SNPs into jackknife blocks, and computes the per-block f2 tensor on the GPU. The output is a directory (`f2.bin` + `pops.txt` + `meta.json`) — the **f2-dir**.

Most of steppe's f2-path tools (`qpadm`, `qpwave`, `qpgraph`, `qpfstats`, the f-statistics, `dates`) consume an f2-dir rather than raw genotypes. So you almost always run `extract-f2` once for a chosen set of populations, then point the analysis commands at the directory it produced. Building it once and reusing it is much faster than re-reading genotypes for every analysis.

The numbers match ADMIXTOOLS 2 (the reference implementation steppe reproduces), so the defaults below are chosen to reproduce its SNP set bit-for-bit: autosomes only, monomorphic SNPs dropped, per-sample pseudo-haploid detection, and a population-axis `maxmiss` coverage test.

## Flags

You pick the populations with **exactly one** of `--pops`, `--auto-top-k`, or `--min-n` (mutually exclusive). If you pass none, the tool uses every population in the dataset.

| flag | what it does | default |
|---|---|---|
| `--prefix TEXT` | Sets `--geno`/`--snp`/`--ind` from one prefix (`PREFIX.geno`, `PREFIX.snp`, `PREFIX.ind`). The usual way to name the input. | — |
| `--geno TEXT` | Genotype file, overrides the `--prefix`-derived path. | from `--prefix` |
| `--snp TEXT` | SNP file, overrides the `--prefix`-derived path. | from `--prefix` |
| `--ind TEXT` | Individual file, overrides the `--prefix`-derived path. | from `--prefix` |
| `--out-dir`, `--out TEXT` | Output f2-dir to write (`f2.bin` + `pops.txt` + `meta.json`). Omit it with `--dry-run` to just report sizes. | — |
| `--pops TEXT ...` | Explicit population list. Every label must exist in the dataset or the run errors naming the missing one. *(pop-selection: mutually exclusive)* | — |
| `--auto-top-k INT` | Keep the K populations with the most individuals. *(pop-selection: mutually exclusive)* | — |
| `--min-n INT` | Keep every population that has at least N individuals. *(pop-selection: mutually exclusive)* | — |
| `--blgsize FLOAT` | Jackknife block size **in Morgans** (AT2 convention). `0.05` = 5 cM. | `0.05` |
| `--maf FLOAT` | Minimum minor-allele frequency; SNPs below it are dropped. | — |
| `--geno-max-miss FLOAT` | Max per-SNP missing fraction across populations (the coverage test). | — |
| `--maxmiss FLOAT` | AT2 alias for `--geno-max-miss` (same knob). | — |
| `--mind-max-miss FLOAT` | Max per-sample missing fraction. | — |
| `--auto-only` / `--no-auto-only` | Keep only autosomes (chr 1–22). `--no-auto-only` keeps sex/other chromosomes too. | on |
| `--drop-mono` / `--no-drop-mono` | Drop monomorphic SNPs (AT2 `poly_only` parity). `--no-drop-mono` keeps them. | on |
| `--transversions` | Keep only transversion SNPs (drop transitions). | off |
| `--strand-mode TEXT` | Strand-ambiguous (A/T, C/G) SNP policy: `drop` (merge-safe) \| `keep` (retain, AT2 default) \| `flip` (not yet implemented, behaves as `keep`). | `drop` |
| `--ploidy TEXT` | Ploidy policy: `auto` (per-sample pseudo-haploid detection, AT2 `adjust_pseudohaploid`) \| `1` (force pseudo-haploid) \| `2` (force diploid). | `auto` |
| `--tier TEXT` | Output memory tier: `auto` \| `resident` (all in GPU memory) \| `host` (spill to host RAM) \| `disk` (spill to disk). `auto` picks the fastest that fits; `host`/`disk` stream the input so very large runs that would OOM in `resident` still complete. | `auto` |
| `--dry-run` | Report SNPs/blocks/tier/precision/VRAM and exit without computing. | off |
| `--hash` / `--no-hash` | Compute a SHA-256 provenance hash of the source dataset (overlapped on a background thread). | off |
| `--device TEXT` | CUDA device(s): `auto` \| an ordinal (e.g. `0`) \| two ordinals (e.g. `0,1`). GPU only — there is no `cpu`. | `auto` |
| `--precision TEXT` | Matmul precision: `emu40` \| `emu32` \| `fp64` \| `tf32`. | `emu40` |
| `--config TEXT` | **Reserved — not yet supported** (passing one currently errors). | — |

## Examples

Build a 700-population f2-dir from the 1240K panel, keeping SNPs covered in at least half the populations, on GPU 0:

```
steppe extract-f2 --prefix v66.p1_1240K.aadr.patch.PUB \
  --auto-top-k 700 --maxmiss 0.5 --device 0 --out-dir f2_700
```

This is a streamed-tier build (700 pops is past what fits resident on one card), and it writes `f2_700/{f2.bin,pops.txt,meta.json}`.

Check what a build would do before committing to it — report the kept-SNP count, block count, chosen tier, and VRAM, with no compute and no output written:

```
steppe extract-f2 --prefix v66.p1_1240K.aadr.patch.PUB \
  --auto-top-k 700 --maxmiss 0.5 --device 0 --dry-run
```

Build a smaller f2-dir for a named handful of populations:

```
steppe extract-f2 --prefix v66.p1_1240K.aadr.patch.PUB \
  --pops Mbuti.DG French.DG Han.DG Sardinian.DG \
  --maxmiss 0.5 --device 0 --out-dir f2_dir
```

## Gotchas

- **`--blgsize` is in Morgans, not centimorgans.** The default `0.05` means 5 cM. Passing `5` would ask for 5-Morgan (500 cM) blocks, which is almost never what you want.
- **The population-selection flags are mutually exclusive.** Use exactly one of `--pops`, `--auto-top-k`, `--min-n`. Passing none keeps every population in the dataset.
- **`--pops` labels must exist.** An unknown label is a hard error naming the missing population — the run does not silently drop it and proceed with a smaller set.
- **`--maxmiss` is a population-axis coverage test, not per-individual.** For each SNP it looks at the fraction of *selected populations* that have no data there. `--maxmiss 0` keeps only SNPs where every population has data (the intersection); `--maxmiss 1` keeps everything. It is a different knob from `--mind-max-miss` (per-sample). The two aliases `--maxmiss` and `--geno-max-miss` are the same value.
- **The memory tier is automatic — you normally leave `--tier auto`.** steppe measures free VRAM and free host RAM (container/cgroup-aware) at runtime and picks the fastest home for the result. The tier only changes *where* the numbers live, never the numbers themselves. Force `host` or `disk` only to reproduce a specific run or work around a constrained box. See [precision-and-tiers.md](./precision-and-tiers.md) for how the tier and `--precision` are chosen.
- **If every SNP gets filtered out, the run errors** rather than writing an empty f2-dir — relax `--maf`/`--maxmiss`/`--transversions` if that happens.
- **The population order is fixed** (labels sorted ascending) and is the axis order used by every downstream tool, so the f2-dir is self-describing via its `pops.txt`.

## See also

- [precision-and-tiers.md](./precision-and-tiers.md) — how `--tier` and `--precision` are chosen, and what each value means.
- [qpadm.md](./qpadm.md), [qpwave.md](./qpwave.md), [qpgraph.md](./qpgraph.md) — tools that consume the f2-dir this command builds.
- `docs/reference/src_app_extract_f2_core.cpp.md` — the extract pipeline internals (decode, filter, blocks, two-phase tiering).
- `docs/reference/src_device_tier_select.hpp.md` — the tier-selection policy internals.
