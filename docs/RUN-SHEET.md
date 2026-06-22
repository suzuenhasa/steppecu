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
$S extract-f2 --prefix $HO --out /tmp/f2_haak \
  --pops Czechia_EBA_CordedWare,Czechia_BellBeaker,Sardinian,Russia_Samara_EBA_Yamnaya,Turkey_N,Serbia_IronGates_Mesolithic,Mbuti,Russia_UstIshim_IUP,Russia_Kostenki_UP,Russia_Malta_UP,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --device 0 --blgsize 0.05 --maxmiss 0 --auto-only
```

### with the provenance hash (`--hash`; overlapped on a background thread)
```bash
$S extract-f2 --prefix $HO --out /tmp/f2_haak_hashed --pops <...> \
  --device 0 --blgsize 0.05 --maxmiss 0 --auto-only --hash
```
`--hash` adds the source-`.geno` SHA-256 to `meta.json` (`geno_sha256`); without it, `meta.json` records `source_hash_computed: false`. (Default OFF — the hash was ~37 s of the old ~40 s wall.)

### dry-run (report SNPs / blocks / tier / VRAM, no compute)
```bash
$S extract-f2 --prefix $K1240 --auto-top-k 700 --maxmiss 0.5 --device 0 --dry-run
```

### big-P streaming (700 pops on 1240K) — auto-selects a streamed tier, ~58 s
```bash
$S extract-f2 --prefix $K1240 --auto-top-k 700 --maxmiss 0.5 --device 0 --out /workspace/f2_700
```
`--auto-top-k 700` keeps the 700 largest pops. Default `--tier auto` streams to `host`/`disk` when the result/feeder won't fit resident (this is what lets P>~250 complete on one 32 GB card). Force it explicitly:
```bash
$S extract-f2 --prefix $K1240 --auto-top-k 700 --maxmiss 0.5 --device 0 --tier disk --out /workspace/f2_700_disk
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
/usr/bin/time -v $S extract-f2 --prefix $K1240 --pops <...> --out /tmp/x --device 0 --blgsize 0.05 --maxmiss 0 --auto-only
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
> ⚠️ **The `qpwave` CLI is still a scaffold** (prints "not yet implemented") — M(cli-2) pending. The *engine* (`run_qpwave`) works and is golden-gated; only the CLI subcommand isn't wired yet. The command below is the intended form.
```bash
$S qpwave --f2-dir /tmp/f2_haak \
  --left Czechia_EBA_CordedWare,Turkey_N,Russia_Samara_EBA_Yamnaya \
  --right $RIGHT --format csv
```

---

## 5. Full study end-to-end (Haak 2015, the exact run)
```bash
# 1) build the f2 dir for the 15-pop union (no-hash, autosomes, maxmiss 0)
$S extract-f2 --prefix $HO --out /tmp/haak \
  --pops Czechia_EBA_CordedWare,Czechia_BellBeaker,Sardinian,Russia_Samara_EBA_Yamnaya,Turkey_N,Serbia_IronGates_Mesolithic,Mbuti,Russia_UstIshim_IUP,Russia_Kostenki_UP,Russia_Malta_UP,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --device 0 --blgsize 0.05 --maxmiss 0 --auto-only

# 2) fit the three models (add --jackknife 2 for SEs)
$S qpadm --f2-dir /tmp/haak --target Czechia_EBA_CordedWare --left Russia_Samara_EBA_Yamnaya,Turkey_N --right $RIGHT --jackknife 2 --format json
$S qpadm --f2-dir /tmp/haak --target Czechia_BellBeaker  --left Russia_Samara_EBA_Yamnaya,Turkey_N --right $RIGHT --jackknife 2 --format json
$S qpadm --f2-dir /tmp/haak --target Sardinian           --left Serbia_IronGates_Mesolithic,Turkey_N,Russia_Samara_EBA_Yamnaya --right $RIGHT --jackknife 2 --format json
```
Expect: Corded Ware ≈ 0.74 Yamnaya / 0.26 Anatolia_N; Bell Beaker ≈ 0.53/0.47; Sardinian ≈ 0.17 WHG / 0.72 EEF / 0.11 steppe.

---

## 6. Tests / gates (validation, not studies)
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
