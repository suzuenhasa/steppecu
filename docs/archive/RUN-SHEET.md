# steppe — command sheet (runnable, from the real agent runs)

Every command below is a real invocation we've run on box5090 (`main @ 433b71e`), single-GPU, Release build. Copy-paste directly.

## 0. Setup (run once per shell, on the box)
```bash
ssh box5090            # or run these on the box directly
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
S=/workspace/steppe/build-rel/bin/steppe                              # the binary
HO=/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB                  # Human Origins (~600K SNPs)
K1240=/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB         # 1240K panel (~1.23M SNPs, 23k ind)
RIGHT="Mbuti,Russia_UstIshim_IUP,Russia_Kostenki_UP,Russia_Malta_UP,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian"
```
Notes: always `--device 0` (single-GPU; multi-GPU is parked). Default precision is `emu40` (emulated FP64). `--blgsize` is in **Morgans** (0.05 = 5 cM, AT2 convention). `$S <subcmd> --help` lists every flag.

---

## 1. extract-f2 — genotypes → f2 dir

### no-hash (the default — fast)
```bash
$S extract-f2 --prefix $HO --out-dir /tmp/f2_haak \
  --pops Czechia_EBA_CordedWare,Czechia_BellBeaker,Sardinian,Russia_Samara_EBA_Yamnaya,Turkey_N,Serbia_IronGates_Mesolithic,Mbuti,Russia_UstIshim_IUP,Russia_Kostenki_UP,Russia_Malta_UP,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --device 0 --blgsize 0.05 --maxmiss 0 --auto-only
```

### with the provenance hash (`--hash`; overlapped on a background thread)
```bash
$S extract-f2 --prefix $HO --out-dir /tmp/f2_haak_hashed --pops <...> \
  --device 0 --blgsize 0.05 --maxmiss 0 --auto-only --hash
```
`--hash` adds the source-`.geno` SHA-256 to `meta.json` (`geno_sha256`); without it, `meta.json` records `source_hash_computed: false`. (Default OFF — the hash was ~37 s of the old ~40 s wall.)

### dry-run (report SNPs / blocks / tier / VRAM, no compute)
```bash
$S extract-f2 --prefix $K1240 --auto-top-k 700 --maxmiss 0.5 --device 0 --dry-run
```

### big-P streaming (700 pops on 1240K) — auto-selects a streamed tier, ~58 s
```bash
$S extract-f2 --prefix $K1240 --auto-top-k 700 --maxmiss 0.5 --device 0 --out-dir /workspace/f2_700
```
`--auto-top-k 700` keeps the 700 largest pops. Default `--tier auto` streams to `host`/`disk` when the result/feeder won't fit resident (this is what lets P>~250 complete on one 32 GB card). Force it explicitly:
```bash
$S extract-f2 --prefix $K1240 --auto-top-k 700 --maxmiss 0.5 --device 0 --tier disk --out-dir /workspace/f2_700_disk
```
`--tier auto|resident|host|disk` (default auto). `host`/`disk` use the SNP-tile input streaming; `resident` is the all-in-VRAM path (OOMs past ~250 pops on a full SNP set).

### pop-selection variants
```bash
--pops a,b,c          # explicit list (comma OR space separated)
--auto-top-k 700      # keep the 700 largest pops
--min-n 5             # keep pops with >= 5 individuals
```

### timing a run
```bash
/usr/bin/time -v $S extract-f2 --prefix $K1240 --pops <...> --out-dir /tmp/x --device 0 --blgsize 0.05 --maxmiss 0 --auto-only
```

---

## 2. qpadm — fit a model over an f2 dir
```bash
$S qpadm --f2-dir /tmp/f2_haak \
  --target Czechia_EBA_CordedWare \
  --left Russia_Samara_EBA_Yamnaya,Turkey_N \
  --right $RIGHT --format csv
```
Add full block-jackknife standard errors with `--jackknife 2` (`0`=none [default], `1`=feasible-only, `2`=all). `--format csv|tsv|json`. `--out FILE` to write (else stdout).
```bash
$S qpadm --f2-dir /tmp/f2_haak --target Sardinian \
  --left Serbia_IronGates_Mesolithic,Turkey_N,Russia_Samara_EBA_Yamnaya \
  --right $RIGHT --jackknife 2 --format json
```

---

## 3. qpadm-rotate — enumerate source-pool subsets (the batched rotation)
```bash
$S qpadm-rotate --f2-dir /tmp/f2_haak \
  --target Czechia_EBA_CordedWare \
  --pool Russia_Samara_EBA_Yamnaya,Turkey_N,Serbia_IronGates_Mesolithic,Iran_GanjDareh_N \
  --right $RIGHT --min-sources 1 --max-sources -1 --format csv
```
`--max-sources -1` = use the whole pool. (Engine ~2,866 models/sec single-GPU; the CLI subcommand is being wired — check `$S qpadm-rotate --help`.)

## 4. qpwave — rank sweep (no target; `left[0]` is the reference)
> The `qpwave` CLI is wired to the GPU `run_qpwave` engine (M(cli-2)) and golden-gated through the CLI (`tests/cli/test_cli_qpwave` vs `golden_qpwave`). qpWave takes NO `--target`: `--left[0]` is the reference row.
```bash
$S qpwave --f2-dir /tmp/f2_haak \
  --left Czechia_EBA_CordedWare,Turkey_N,Russia_Samara_EBA_Yamnaya \
  --right $RIGHT --format csv
```

---

## 5. f-stats — f4 / f3 / f4-ratio / qpDstat (explicit) + the all-quartets SWEEP
> Standalone f-statistics over an f2 dir, GPU-batched (reuse `assemble_f4_quartets` + the diagonal block-jackknife); `est/se/z/p` per item. All golden-gated; EmulatedFp64{40}, single-GPU.

### explicit — a quartet/triple, or a list
```bash
# f4(p1,p2;p3,p4)
$S f4 --f2-dir /tmp/f2_haak --pop1 England_BellBeaker --pop2 Czechia_EBA_CordedWare \
   --pop3 Han --pop4 Iran_GanjDareh_N --format csv
$S f4 --f2-dir DIR --pops A,B,C,D,E,F,G,H        # a list, names in groups of 4
$S f3 --f2-dir DIR --pop1 Target --pop2 SrcA --pop3 SrcB     # f3(Target; SrcA, SrcB)
$S f4-ratio --f2-dir DIR --pops ...              # admixture proportion alpha
$S qpdstat --f2-dir DIR --pops A,B,C,D           # f2-path = f4 + Z/p (== AT2 f2-path qpdstat)
$S qpdstat --prefix <geno-prefix> --pops A,B,C,D # genotype-path = the NORMALIZED D magnitude
```

### the all-quartets / all-triples SWEEP (GPU-only, memory-bounded)
> Enumerates EVERY C(P,4)/C(P,3) **on the GPU** (unrank → f4 → diagonal jackknife → `|z|` filter → CUB compact), keeping only survivors. `--top-k` (default **1,000,000**) bounds the output to the K most-significant via a **device-side reservoir** — host RAM stays a few GB no matter how many billions are computed. `--min-z` is the alternative filter (but on real AADR ~70% pass `|z|>6`, so `--top-k` is usually what you want). `--sure` lifts the enumeration cap. `--shard-dir` writes the survivor CSV.
```bash
# the FULL C(500,4) = 2,573,031,125-quartet sweep, top-1M most-extreme |z|:
$S f4 --all-quartets --f2-dir /workspace/data/f2_500 --top-k 1000000 --sure \
   --shard-dir /tmp/sweep_out
#  → one RTX 5090: ~177 s (2:57), ~14.5M quartets/sec, GPU ~100%, peak host RSS ~3.1 GB
#    (bounded), VRAM ~14.6 GB. (top-K cut landed at |z|=140 — |z|>6 keeps ~1.8B, so top-K is the real bound)
$S f3 --all-triples --f2-dir DIR --top-k 100000 --sure --shard-dir /tmp/f3_sweep
$S f4 --all-quartets --f2-dir DIR --pops A,B,C,...  --min-z 8     # sweep a SUBSET, |z|>=8
```

### read the sweep output back (Python)
```python
import steppe
df = steppe.read_fstats("/tmp/sweep_out")   # survivor table -> pandas (the filtered/top-K set fits in RAM)
```

---

## 5b. The Python wheel — build, pip install, extract_f2 (M(py-2))
```bash
# Build ONE GPU wheel (Release, sm_120) — scikit-build-core, no CLI:
cd /workspace/steppe && export PATH=/usr/local/cuda/bin:$PATH \
  && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
  && python3.12 -m build --wheel        # -> dist/steppe-0.1.0-cp312-cp312-linux_x86_64.whl

# Install into a CLEAN venv (numpy is the ONLY hard dep; CUDA 13 is a SYSTEM requirement
# resolved at load — _core.so DT_NEEDED libcudart.so.13/libcublas.so.13/libcusolver.so.12;
# the toolkit is NOT bundled, so the box's CUDA 13 must be on the loader path):
python3.12 -m venv /tmp/v && /tmp/v/bin/pip install dist/steppe-0.1.0-cp312-cp312-linux_x86_64.whl
# optional: pandas for the .to_dataframe accessors -> pip install 'steppe[pandas]'
```
```python
# extract_f2: genotypes -> f2 ON THE GPU (the same chain as the CLI extract-f2)
import steppe
# (A) in-memory handle (no disk round-trip) -> feed straight to qpadm:
f2 = steppe.extract_f2("/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB",
                       pops=["England_BellBeaker", "Turkey_N", "Mbuti", "Han"],
                       device=0, blgsize=0.05, maxmiss=0.0)   # ploidy="auto" (AT2 per-sample)
res = steppe.qpadm(f2, target="England_BellBeaker", left=["Turkey_N"], right=["Mbuti", "Han"])
# (B) or write an STPF2BK1 dir + reload it (out=DIR returns the path string):
path = steppe.extract_f2("/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB",
                         pops=["England_BellBeaker", "Turkey_N", "Mbuti"], out="/tmp/ex_f2")
f2b = steppe.read_f2(path)        # out=None vs read_f2(DIR) are bit-identical (the parity law)
```
NOTE (B3): steppe's decode is **TGENO-only**, so extract reads the raw HO TGENO prefix
(`raw/v66.p1_HO.aadr.patch.PUB`) — the convertf-PA `v66_HO_pa` is SNP-major PACKEDANCESTRYMAP
(`GENO` magic), unreadable by the decode path (same ind/snp axes — a lossless transcode).

---

## 6. Full study end-to-end (Haak 2015, the exact run)
```bash
# 1) build the f2 dir for the 15-pop union (no-hash, autosomes, maxmiss 0)
$S extract-f2 --prefix $HO --out-dir /tmp/haak \
  --pops Czechia_EBA_CordedWare,Czechia_BellBeaker,Sardinian,Russia_Samara_EBA_Yamnaya,Turkey_N,Serbia_IronGates_Mesolithic,Mbuti,Russia_UstIshim_IUP,Russia_Kostenki_UP,Russia_Malta_UP,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --device 0 --blgsize 0.05 --maxmiss 0 --auto-only

# 2) fit the three models (add --jackknife 2 for SEs)
$S qpadm --f2-dir /tmp/haak --target Czechia_EBA_CordedWare --left Russia_Samara_EBA_Yamnaya,Turkey_N --right $RIGHT --jackknife 2 --format json
$S qpadm --f2-dir /tmp/haak --target Czechia_BellBeaker  --left Russia_Samara_EBA_Yamnaya,Turkey_N --right $RIGHT --jackknife 2 --format json
$S qpadm --f2-dir /tmp/haak --target Sardinian           --left Serbia_IronGates_Mesolithic,Turkey_N,Russia_Samara_EBA_Yamnaya --right $RIGHT --jackknife 2 --format json
```
Expect: Corded Ware ≈ 0.74 Yamnaya / 0.26 Anatolia_N; Bell Beaker ≈ 0.53/0.47; Sardinian ≈ 0.17 WHG / 0.72 EEF / 0.11 steppe.

---

## 7. Tests / gates (validation, not studies)
```bash
# from /workspace/steppe, after a Release build:
STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure        # full suite (45 tests)
./build-rel/bin/test_cli_extract_qpadm ./build-rel/bin/steppe /workspace/data/aadr ./tests/reference/goldens/at2
```

## Build (on the box)
```bash
cd /workspace/steppe && export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON && cmake --build build-rel
```
