Every command as a standalone ssh box5090 '...' one-liner, all paths expanded — nothing to set up first:

# 🚀 sweep every quartet over 500 pops — C(500,4) = 2,573,031,125 quartets, ~177s
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe f4 --all-quartets --f2-dir /workspace/data/f2_500 --top-k 1000000 --sure --shard-dir /tmp/sweep500 --device 0'

# 🚀 sweep every triple over 500 pops — C(500,3) = 20,708,500 triples
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe f3 --all-triples --f2-dir /workspace/data/f2_500 --top-k 100000 --sure --shard-dir /tmp/f3sweep500 --device 0'

# dedicated sweep subcommands (same engine)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe f4-sweep --f2-dir /workspace/data/f2_500 --top-k 1000000 --sure --out /tmp/f4sweep.csv --device 0'
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe f3-sweep --f2-dir /workspace/data/f2_500 --min-z 8 --out /tmp/f3sweep.csv --device 0'

# qpAdm — fit a model (15-pop dir, full outgroups), with jackknife SEs, JSON
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe qpadm --f2-dir /workspace/data/haak/steppe_f2 --target Czechia_EBA_CordedWare --left Russia_Samara_EBA_Yamnaya,Turkey_N --right Mbuti,Russia_UstIshim_IUP,Russia_Kostenki_UP,Russia_Malta_UP,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian --jackknife 2 --format json --device 0'

# qpAdm-rotate — 2^5-1 = 31 competing models in one batched GPU run
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe qpadm-rotate --f2-dir /workspace/data/1240k_sweep/f2_30 --target Sweden_Viking --pool Russia_Samara_EBA_Yamnaya,Turkey_N,Serbia_IronGates_Mesolithic,France_Yonne_N,Czechia_EBA_CordedWare --right Mbuti,Han,Papuan,Karitiana,Israel_Natufian,Iran_GanjDareh_N --min-sources 1 --max-sources -1 --format csv --device 0'

# qpWave — rank sweep (no target; left[0] = reference)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe qpwave --f2-dir /workspace/data/haak/steppe_f2 --left Czechia_EBA_CordedWare,Turkey_N,Russia_Samara_EBA_Yamnaya --right Mbuti,Russia_UstIshim_IUP,Russia_Kostenki_UP,Russia_Malta_UP,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian --format csv --device 0'

# qpGraph — fit the 9-pop golden admixture graph
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe qpgraph --f2-dir /workspace/data/qpgraph_9pop_stpf2bk1 --graph /workspace/steppe/tests/reference/goldens/at2/golden_qpgraph_edges.csv --format json --device 0'

# qpGraph-search — search topologies over a bounded leaf set (fleet, global-best)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe qpgraph-search --f2-dir /workspace/data/qpgraph_9pop_stpf2bk1 --pops England_BellBeaker,Czechia_EBA_CordedWare,Turkey_N,Mbuti,Han,Karitiana --max-nadmix 1 --numstart 10 --format json --device 0'

# f4 — a single quartet f4(p1,p2;p3,p4)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe f4 --f2-dir /workspace/data/haak/steppe_f2 --pop1 England_BellBeaker --pop2 Czechia_EBA_CordedWare --pop3 Han --pop4 Iran_GanjDareh_N --format csv --device 0'

# f3 — f3(C; A, B)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe f3 --f2-dir /workspace/data/haak/steppe_f2 --pop1 Czechia_EBA_CordedWare --pop2 Russia_Samara_EBA_Yamnaya --pop3 Turkey_N --device 0'

# f4-ratio — admixture proportion alpha
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe f4-ratio --f2-dir /workspace/data/haak/steppe_f2 --pops Czechia_EBA_CordedWare,Mbuti,Russia_Samara_EBA_Yamnaya,Han,Turkey_N --device 0'

# qpDstat — f2-path D (= f4 + Z/p)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe qpdstat --f2-dir /workspace/data/haak/steppe_f2 --pops Mbuti,Han,Czechia_EBA_CordedWare,Turkey_N --device 0'

# dates — admixture dating (needs a real cM map in the .snp): date + jackknife SE
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe dates --prefix /workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB --target Czechia_EBA_CordedWare --left Russia_Samara_EBA_Yamnaya,Turkey_N --device 0'

# qpfstats — genotype -> a smoothed f2 dir (feed it to qpadm/f4/qpgraph after)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe qpfstats --prefix /workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB --pops Czechia_EBA_CordedWare,Russia_Samara_EBA_Yamnaya,Turkey_N,Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian --out-dir /tmp/smoothed_f2 --device 0'

# extract-f2 — dry-run: report SNPs/blocks/tier/VRAM for a 700-pop build (no compute)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe extract-f2 --prefix /workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB --auto-top-k 700 --maxmiss 0.5 --device 0 --dry-run'

# extract-f2 — build a 700-pop f2 dir (streamed tier, ~58s)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe extract-f2 --prefix /workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB --auto-top-k 700 --maxmiss 0.5 --device 0 --out-dir /workspace/f2_700'

# 🚀 then sweep YOUR fresh 700-pop dir — C(700,4) = 9,895,415,325 quartets
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe f4-sweep --f2-dir /workspace/f2_700 --top-k 1000000 --sure --shard-dir /tmp/sweep700 --device 0'

That's every subcommand as a self-contained one-liner. The only two that need a /workspace/f2_700 to exist first are the last sweep (run the extract-f2 build line above it first); everything else runs against data already on the box.

---

## 2M panel — v66.p1_2M (2,142,271 SNPs, 23,104 individuals; the big one)

Newer/larger AADR capture panel (~2.14M SNPs vs 1240K's 1.23M). The 700-pop extract-f2 relies on
the SNP-tiled decode (landed on main, commit 1099aee) — before that it OOM'd on a 32 GB card.
Files: /workspace/data/aadr/2m/v66.p1_2M.aadr.patch.PUB.{geno,snp,ind}

# extract-f2 on the full 2M panel — 700 pops, ~75s (streamed host tier; ~1.73M of 2.14M SNPs kept)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe extract-f2 --prefix /workspace/data/aadr/2m/v66.p1_2M.aadr.patch.PUB --auto-top-k 700 --maxmiss 0.5 --device 0 --out-dir /workspace/data/2m_f2_700'

# qpAdm over the 2M f2 dir (confirm labels first: cat /workspace/data/2m_f2_700/pops.txt)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe qpadm --f2-dir /workspace/data/2m_f2_700 --target Czechia_EBA_CordedWare --left Russia_Samara_EBA_Yamnaya,Turkey_N --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian --jackknife 2 --format json --device 0'

# 🚀 f4-sweep over the 2M 700-pop dir — C(700,4) = 9,918,641,075 quartets, top-1M  (~729s / 12.1 min, ~13.6M quartets/s, 1 RTX 5090)
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe f4 --all-quartets --f2-dir /workspace/data/2m_f2_700 --top-k 1000000 --sure --shard-dir /tmp/sweep_2m_700 --device 0'

# f3-sweep over the 2M 700-pop dir — C(700,3) = 56,921,900 triples
ssh box5090 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64; /workspace/steppe/build-rel/bin/steppe f3 --all-triples --f2-dir /workspace/data/2m_f2_700 --top-k 100000 --sure --shard-dir /tmp/f3sweep_2m_700 --device 0'

---

## AADR data — download from Harvard Dataverse (doi:10.7910/DVN/FFIDCW)

Robust script (resolves fileIds from the dataset API at runtime → survives new AADR versions; resumable; MD5-verified):
```
bash docs/download-aadr.sh 1240K ./aadr_1240K      # or:  HO   |   2M
```

Or raw curl one-liners (v66 version-14.0 fileIds; `-C -` resumes a partial download; the .geno is the big one):
```
# --- 1240K panel (~7.1 GB) ---
curl -L -C - -o v66.p1_1240K.aadr.patch.PUB.geno https://dataverse.harvard.edu/api/access/datafile/13994829
curl -L -C - -o v66.p1_1240K.aadr.patch.PUB.snp  https://dataverse.harvard.edu/api/access/datafile/13994514
curl -L -C - -o v66.p1_1240K.aadr.patch.PUB.ind  https://dataverse.harvard.edu/api/access/datafile/13994513
curl -L -C - -o v66.p1_1240K.aadr.PUB.anno       https://dataverse.harvard.edu/api/access/datafile/13994515
# --- HO panel (~4.0 GB) ---
curl -L -C - -o v66.p1_HO.aadr.patch.PUB.geno https://dataverse.harvard.edu/api/access/datafile/13994808
curl -L -C - -o v66.p1_HO.aadr.patch.PUB.snp  https://dataverse.harvard.edu/api/access/datafile/13994527
curl -L -C - -o v66.p1_HO.aadr.patch.PUB.ind  https://dataverse.harvard.edu/api/access/datafile/13994526
curl -L -C - -o v66.p1_HO.aadr.PUB.anno       https://dataverse.harvard.edu/api/access/datafile/13994528
# --- 2M panel (~12 GB) ---
curl -L -C - -o v66.p1_2M.aadr.patch.PUB.geno https://dataverse.harvard.edu/api/access/datafile/13994830
curl -L -C - -o v66.p1_2M.aadr.patch.PUB.snp  https://dataverse.harvard.edu/api/access/datafile/13994517
curl -L -C - -o v66.p1_2M.aadr.patch.PUB.ind  https://dataverse.harvard.edu/api/access/datafile/13994516
curl -L -C - -o v66.p1_2M.aadr.PUB.anno       https://dataverse.harvard.edu/api/access/datafile/13994518
# then feed a prefix to steppe:
#   steppe extract-f2 --prefix v66.p1_1240K.aadr.patch.PUB --auto-top-k 200 --maxmiss 0.5 --device 0 --out-dir f2_dir
```

---

Note: keep this file (docs/commands.md) LIVE — excluded from the docs/archive sweep; it's the working cheatsheet.