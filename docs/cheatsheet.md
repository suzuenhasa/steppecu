# steppe — usage cheatsheet

Every command as you'd actually run it with steppe **installed** (the `steppe` binary on your
PATH, or `./steppe`). For installation see the [README](../README.md); for the exact
box/dev invocations (ssh + `/workspace` paths) see [commands.md](commands.md).

Real AADR population labels are used below — swap in ones present in **your** f2 dir
(`cat f2_dir/pops.txt`).

---

## 0. Setup (once per shell)
```bash
# CUDA 13 must be on the loader path (only if `steppe` can't find libcudart.so.13):
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
steppe --version          # 0.1.0
steppe --help             # list all subcommands
```

## 1. Get data — the AADR
```bash
bash download-aadr.sh 1240K ./aadr        # or: HO | 2M   (see docs/download-aadr.sh)
# -> ./aadr/v66.p1_1240K.aadr.patch.PUB.{geno,snp,ind}
```

## 2. Build the f2 cache (once per population set)
```bash
steppe extract-f2 --prefix aadr/v66.p1_1240K.aadr.patch.PUB \
  --auto-top-k 200 --maxmiss 0.5 --device 0 --out-dir f2_dir
#  --pops A,B,C   explicit set        --dry-run   preview SNPs/blocks/tier/VRAM (no compute)
#  --tier disk    stream very high pop×SNP builds     cat f2_dir/pops.txt  # available labels
```

## 3. qpAdm — model a target as a mixture of sources
```bash
steppe qpadm --f2-dir f2_dir --target England_BellBeaker \
  --left  Czechia_EBA_CordedWare,Turkey_N \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --jackknife 2 --format json
```

## 4. qpAdm rotation — enumerate every source subset of a pool
```bash
steppe qpadm-rotate --f2-dir f2_dir --target England_BellBeaker \
  --pool  Russia_Samara_EBA_Yamnaya,Turkey_N,Serbia_IronGates_Mesolithic,Czechia_EBA_CordedWare \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --min-sources 1 --max-sources -1 --format csv
```

## 5. qpWave — rank sweep (no target; left[0] is the reference)
```bash
steppe qpwave --f2-dir f2_dir \
  --left  Czechia_EBA_CordedWare,Turkey_N,Russia_Samara_EBA_Yamnaya \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N --format csv
```

## 6. f-statistics (single)
```bash
steppe f4 --f2-dir f2_dir --pop1 England_BellBeaker --pop2 Czechia_EBA_CordedWare --pop3 Han --pop4 Iran_GanjDareh_N --format csv
steppe f3 --f2-dir f2_dir --pop1 Czechia_EBA_CordedWare --pop2 Russia_Samara_EBA_Yamnaya --pop3 Turkey_N     # outgroup-f3 / admixture-f3
steppe f4-ratio --f2-dir f2_dir --pops Czechia_EBA_CordedWare,Mbuti,Russia_Samara_EBA_Yamnaya,Han,Turkey_N
steppe qpdstat --f2-dir f2_dir --pops Mbuti,Han,Czechia_EBA_CordedWare,Turkey_N                             # D = f4 + Z/p
```

## 7. Sweeps — every quartet/triple on the GPU (top-K by |z|)
```bash
steppe f4 --all-quartets --f2-dir f2_dir --top-k 1000000 --sure --shard-dir sweep_out --device 0   # C(P,4)
steppe f3 --all-triples  --f2-dir f2_dir --top-k 100000  --sure --shard-dir f3_sweep  --device 0   # C(P,3)
# read the survivors back:   python -c "import steppe; print(steppe.read_fstats('sweep_out'))"
```

## 8. qpGraph — fit / search admixture graphs
```bash
steppe qpgraph --f2-dir f2_dir --graph mygraph.txt --format json       # mygraph.txt: 2 cols per line, "parent child"
steppe qpgraph-search --f2-dir f2_dir --pops England_BellBeaker,Czechia_EBA_CordedWare,Turkey_N,Mbuti,Han,Karitiana \
  --max-nadmix 1 --numstart 10 --format json
```

## 9. Genotype-path tools (take the `--prefix`, not an f2 dir)
```bash
# admixture dating — needs a real cM genetic map in the .snp
steppe dates --prefix aadr/v66.p1_1240K.aadr.patch.PUB --target England_BellBeaker \
  --left Russia_Samara_EBA_Yamnaya,Turkey_N --device 0
# joint f2 smoother -> a smoothed f2 dir you can then feed to qpadm/f4/qpgraph
steppe qpfstats --prefix aadr/v66.p1_1240K.aadr.patch.PUB \
  --pops Czechia_EBA_CordedWare,Russia_Samara_EBA_Yamnaya,Turkey_N,Mbuti,Han,Papuan,Karitiana \
  --out-dir smoothed_f2 --device 0
# genotype-path normalized D (vs the f2-path f4 above)
steppe qpdstat --prefix aadr/v66.p1_1240K.aadr.patch.PUB --pops Mbuti,Han,Czechia_EBA_CordedWare,Turkey_N
```

---

## Common flags (most subcommands)
| flag | meaning |
|---|---|
| `--device 0` | GPU ordinal (single-GPU; no `cpu`) |
| `--format csv\|tsv\|json` | output format; `--out FILE` writes to a file (else stdout) |
| `--precision emu40\|emu32\|fp64\|tf32` | matmul precision (default `emu40`; `fp64` = native; `tf32` = screening only) |
| `--jackknife 0\|1\|2` | SE policy: none · feasible-only · all |
| `--f2-dir DIR` | run over a prebuilt f2 cache (fit/stat subcommands) |
| `--prefix PREFIX` | run over genotypes `PREFIX.{geno,snp,ind}` (extract-f2, dates, qpfstats, qpdstat-D) |

*Keep this file live — it's the user-facing usage sheet (companion to the dev runbook `commands.md`).*
