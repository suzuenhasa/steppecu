# qpfstats (smoothed f2 producer)

Reads a genotype dataset and a list of populations and writes a **smoothed** f2 directory that qpAdm, f4, and qpGraph can read exactly like a normal f2 cache.

## What it does

Most of steppe's f-statistic tools (qpAdm, f4/D, qpGraph) do not read genotypes directly — they read a precomputed **f2 directory** (the per-block f2 cache). Normally you build that cache with [`extract-f2`](./extract-f2.md), which computes each f2 in isolation. `qpfstats` is an **alternative producer** of that same kind of directory. Instead of computing each f2 on its own, it enumerates every f2, f3, and f4 you can form from your population set, then fits them all jointly (a ridge regression over a shared pairwise-f2 basis) so the numbers are mutually consistent. The result is a *smoothed* per-block f2 tensor, written out in steppe's ordinary f2-dir format.

Reach for `qpfstats` when you want the joint-smoothing behavior of ADMIXTOOLS 2's `qpfstats` before running downstream models — the smoothed cache can make qpAdm/qpGraph fits more stable. The workflow is two steps: run `qpfstats` to produce the smoothed dir, then point qpAdm / f4 / qpGraph at that dir with their `--f2-dir` flag, just as you would with an `extract-f2` output. Unlike an f4 or D report, `qpfstats` does **not** print estimates / SE / z / p — its output is a cache on disk, not a table.

## Flags

| flag | what it does | default |
|---|---|---|
| `--prefix TEXT` | Genotype triple to read: `PREFIX.geno`, `PREFIX.snp`, `PREFIX.ind` (EIGENSTRAT / TGENO). Required. | — |
| `--pops TEXT ...` | The population set to smooth over (space-separated after the flag, or comma-separated). Only these pops are read from the file, not the whole dataset. Sorted ascending internally (the ADMIXTOOLS 2 dimnames order). Required. | — |
| `--out-dir TEXT` | Where to write the smoothed f2 directory: `f2.bin` + `pops.txt` + `meta.json`. This is the dir you feed to downstream tools. | — |
| `--blgsize FLOAT` | Jackknife block size, in **Morgans** (the AT2 convention). `0.05` means 5 cM. | `0.05` |
| `--device TEXT` | CUDA device: `auto`, a single ordinal like `0`, or `0,1`. GPU-only — there is no `cpu` option. | `auto` |
| `--precision TEXT` | Arithmetic for the **matmul sub-steps** of the smoothing solve: `emu40`, `emu32`, `fp64`, or `tf32`. The delicate parts (the Cholesky factor and triangular solve) stay in native double regardless of this flag. | `emu40` |
| `--config TEXT` | **Reserved — not yet supported** (passing one currently errors). | — |
| `-h`, `--help` | Print help and exit. | — |

## Examples

Smooth a 9-population set and write the cache to `/tmp/smoothed_f2`:

```
steppe qpfstats \
  --prefix /workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB \
  --pops Czechia_EBA_CordedWare,Russia_Samara_EBA_Yamnaya,Turkey_N,Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --out-dir /tmp/smoothed_f2 \
  --device 0
```

Expect an `f2.bin` + `pops.txt` + `meta.json` under `/tmp/smoothed_f2`. That directory is now a drop-in f2-dir.

Then run a downstream tool against the smoothed dir — e.g. an f4/D report:

```
steppe qpdstat --f2-dir /tmp/smoothed_f2 \
  --pops Mbuti,Han,Czechia_EBA_CordedWare,Turkey_N \
  --device 0
```

The populations you query downstream must be a subset of the `--pops` you smoothed over — the cache only contains those.

## Gotchas

- **`--blgsize` is in Morgans, not centimorgans.** `0.05` = 5 cM. Passing `5` means 5 Morgans (500 cM), which is almost certainly not what you want. Leave it at the default unless you have a reason.
- **`--pops` defines the whole cache.** The output f2-dir contains only the populations you listed here. Every downstream qpAdm / f4 / qpGraph pop (target, sources, right/outgroups) must be in this set. Plan the set up front.
- **Pop order does not matter to you.** `qpfstats` sorts the pops ascending internally (matching AT2's dimname order), so however you type `--pops`, the on-disk order is the sorted one.
- **Producer, not a report.** `qpfstats` writes a cache; it prints no estimates or p-values. If you wanted a statistic table, you want the f4/D/qpAdm tools reading a dir — not this.
- **Fixed parity behaviors.** Ploidy is forced diploid, autosomes only, and the "all SNPs" mask is used (a SNP counts per combination only where all four pops have data). There is no `--maxmiss` / MAF / drop-monomorphic filter here, unlike `extract-f2`. All of f2/f3/f4 are always included.
- **GPU only.** There is no `--device cpu`; steppe is a GPU tool.

## See also

- [extract-f2](./extract-f2.md) — the other f2-dir producer (per-f2, no joint smoothing); the more common way to build the cache.
- [qpadm](./qpadm.md), [qpwave](./qpwave.md), [qpgraph](./qpgraph.md) — downstream consumers that read the f2-dir this produces.
- `docs/reference/include_steppe_qpfstats.hpp.md` — internals: the enumeration counts, the design-matrix basis, the ridge solve, and the output tensor layout.
