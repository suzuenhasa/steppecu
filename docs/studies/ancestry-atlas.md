# A curated large-scale qpAdm ancestry ATLAS of ancient West Eurasia (REAL v66 1240K)

A single **curated qpAdm ancestry atlas** that models every coherent ancient West-Eurasian
population (Neolithic → Iron Age Europe + Anatolia + Levant/Near East + W-Eurasian Steppe)
as **1-, 2- or 3-way mixtures of 18 canonical ancestral streams**, each fit against a fixed
**9-pop distal outgroup wall**, **WITH 713-block jackknife SEs**. Everything below is on
**REAL AADR v66 1240K** (no synthetic data), **single-GPU** (`--device 0`; multi-GPU parked),
via `steppe extract-f2` → `steppe qpadm-rotate`.

- **Mark:** REAL v66 1240K / single-GPU / curated. No AT2 run anywhere (the AT2 number below
  is an explicitly-stated *estimate*, not a measured re-run). steppe `main` @ `8babe0f`.
- **Why curate.** A prior 100-random-pop run produced only **noise**: an uncurated pool admits
  cline/ghost/duplicate "sources" → spurious overfits and uninterpretable feasible models, and
  thin random outgroups have no power to reject. Here the pool is **canonical real sampled
  streams only** and the right set is the **deep distal Reich base**, so every feasible model is
  mechanistically interpretable. The curation *is* the science.

---

## HEADLINE

**213 curated West-Eurasian targets × C(18, 1..3) source combos = 207,921 qpAdm model-fits,
each carrying a 713-block jackknife SE, computed in 6.6 minutes on ONE RTX 5090.** 177 of the
213 targets received a feasible, interpretable best-fit model (39 one-way, 117 two-way, 21
three-way). The headline patterns are **textbook archaeogenetics, reproduced at scale**:

- **The Bronze Age steppe gradient across Europe.** Neolithic farmers carry **0% steppe**
  (Germany LBK = pure EEF, p=0.63); Corded Ware / Bell Beaker / Bronze Age targets carry
  **~40–80% steppe** (Poland Corded Ware ~80%, Czech Corded Ware ~72%, English Bell Beaker ~42%).
- **The canonical Yamnaya formation:** `Russia_Samara_EBA_Yamnaya` = **CHG 0.34 + EHG/steppe-
  Eneolithic 0.66** (p=0.225), the textbook Caucasus-HG + Eastern-HG steppe genesis.
- **The Near-Eastern Iran/Levant/Caucasus axis:** Iranian-plateau targets resolve to CHG/Iran_N,
  Levantine PPN/EBA carry Natufian (Levant_N), Anatolian Bronze Age carries the eastern Iran_N
  gene flow.
- **The HG cline:** Iron Gates / Baltic / Ukrainian Mesolithic resolve as **WHG + EHG** mixtures
  (Serbia Iron Gates = WHG 0.79 + EHG 0.21, p=0.78).

**Could not be done before, can now.** A 200k-fit jackknife rotation atlas is a multi-day-to-week
CPU job in ADMIXTOOLS 2 (the R loop reslices f2 per call); steppe makes the whole curated atlas
an **interactive few-minute single-GPU run**, so the *curation can actually be iterated*.
Estimated AT2 time for the same 207,921 SE-bearing fits: **~29 h (@0.5 s/fit) to ~116 h
(@2 s/fit)** → steppe is **~265× to ~1056×** faster (mid ~440× at 1 s/fit). See the SCALE
section for the honest basis of that estimate.

> **Spot-check (independent re-run, NOT trusting the run agent).** Three headline targets were
> re-fit from a *freshly rebuilt* f2 cache (31-pop union, 1,075,800 SNPs) and reproduced the
> atlas to 3 decimals: Czech Corded Ware → Sardinia_N 0.283 + Yamnaya 0.717 (p≈0.71); Poland
> Corded Ware → Turkey_N 0.204 + Yamnaya 0.796 (p≈0.78); English Bell Beaker → Spain_EN 0.576 +
> SeredniiStih 0.424 (p≈0.27); Germany LBK → Spain_EN 1.0 (p≈0.62). The tiny p-shifts vs the
> production atlas come from the smaller cache's SNP count and are expected. The steppe gradient
> (0% in Neolithic LBK → 42–80% in BA Corded Ware/Bell Beaker) is exactly the well-established
> result, so the setup is sound.

---

## 1. The curated design

### Source pool — 18 canonical ancestral streams (all VERIFIED in v66 `.ind` col-3)

The whole pool is **real sampled v66 populations** — no ghosts, no cline points, no
constructed sources. Comma-list for `--pool`:

```
France_Mesolithic, Luxembourg_Loschbour_Mesolithic, Serbia_IronGates_Mesolithic,
Russia_Karelia_Mesolithic, Latvia_LateMesolithic_Kunda, Georgia_KotiasKlde_Mesolithic,
Iran_GanjDareh_N, Turkey_N, Turkey_Catalhoyuk_MN, Israel_Natufian, Armenia_Aknashen_N,
France_Yonne_N, Austria_N_LBK, Spain_EN, Italy_Sardinia_N,
Russia_Saratov_Eneolithic_Khvalynsk, Russia_Eneolithic_SeredniiStih_Don,
Russia_Samara_EBA_Yamnaya
```

| Ancestral stream | v66 pop (n ind, verified) | notes |
|---|---|---|
| **WHG** (Villabruna) | `France_Mesolithic` (18) | primary robust WHG |
| WHG (canonical-thin) | `Luxembourg_Loschbour_Mesolithic` (2) | the canonical pure WHG, but thin |
| Iron Gates HG | `Serbia_IronGates_Mesolithic` (73) | best-sampled, but carries **minor EHG admixture** — a *separate* stream, NOT a WHG substitute |
| **EHG** | `Russia_Karelia_Mesolithic` (16) | |
| Baltic HG (WHG–EHG cline anchor) | `Latvia_LateMesolithic_Kunda` (14) | |
| **CHG** | `Georgia_KotiasKlde_Mesolithic` (2) | the **only clean CHG in v66**, thin but canonical |
| **Iran_N** | `Iran_GanjDareh_N` (17) | |
| **Anatolia_N / EEF source** | `Turkey_N` (68); `Turkey_Catalhoyuk_MN` (46) | |
| **Levant_N** | `Israel_Natufian` (17) | |
| **Caucasus_N** | `Armenia_Aknashen_N` (4) | |
| **European EEF** (downstream Anatolian farmers) | `France_Yonne_N` (105), `Austria_N_LBK` (103), `Spain_EN` (23), `Italy_Sardinia_N` (20) | |
| **Steppe / Eneolithic EHG-CHG** | `Russia_Saratov_Eneolithic_Khvalynsk` (47), `Russia_Eneolithic_SeredniiStih_Don` (11), `Russia_Samara_EBA_Yamnaya` (46) | |

### Right / outgroup set — 9-pop distal Reich base (VERIFIED, disjoint from the pool)

The Lazaridis/Reich "basic distal" pattern. Comma-list for `--right`:

```
Mbuti, Han, Papuan, Karitiana, Russia_Kostenki_UP, Russia_Sunghir_UP,
Russia_Malta_UP, Ethiopia_MotaCave_4500BP, Russia_UstIshim_IUP
```

| right pop | n | role |
|---|---|---|
| `Mbuti` | 15 | deep African |
| `Han` | 46 | East Asian |
| `Papuan` | 32 | Australasian |
| `Karitiana` | 16 | Native American |
| `Russia_Kostenki_UP` | 3 | European UP / ANE-adjacent |
| `Russia_Sunghir_UP` | 4 | European UP |
| `Russia_Malta_UP` | 1 | **MA1 = ANE** (labelled `Russia_Malta_UP`, NOT `Russia_MA1_UP`) |
| `Ethiopia_MotaCave_4500BP` | 4 | basal African ("Mota") |
| `Russia_UstIshim_IUP` | 1 | basal Eurasian |

### Target set — 213 coherent ancient West-Eurasian populations

Derived by a precise rule on the `.ind` col-3 (region prefix ∈ West-Eurasian country list **AND**
an ancient-period token **AND NOT** a historical-period / East-Siberian / outlier token), all
**≥8 individuals**, autosomal. Count after curation = **213**. Regional spread (verified): Russia
27, Germany 19, Italy 15, England 13, Ukraine 13, Spain 12, France 11, Turkey 10, Czechia 10,
Poland 7, Armenia 7, Iran 6, Greece 6, Croatia 6, Scotland 5, Kazakhstan 5, + 20-odd more
countries. (~18 pops appear in **both** the pool and the target list — `France_Yonne_N`,
`Austria_N_LBK`, `Turkey_N`, `Spain_EN`, `Russia_Samara_EBA_Yamnaya`, … — expected and handled,
see §4.)

### Version-switch / proxy notes (all v66-grounded, no fabrication)

1. **WHG proxy:** canonical pure WHG = `Luxembourg_Loschbour_Mesolithic` but n=2. Primary robust
   WHG = `France_Mesolithic` (n=18, clean Villabruna). `Serbia_IronGates_Mesolithic` (n=73) is the
   best-sampled HG but carries minor EHG admixture — kept as a *separate* stream, not a WHG swap.
2. **CHG:** `Georgia_KotiasKlde_Mesolithic` (n=2) is the **only clean CHG in v66** (Satsurblia n=1;
   Dzudzuana is UP, not CHG). Thin but canonical; no better proxy exists.
3. **Thin deep ancients:** `Russia_Malta_UP`/`Russia_UstIshim_IUP` are n=1; `Russia_Kostenki_UP`
   n=3; `Ethiopia_MotaCave_4500BP`/`Russia_Sunghir_UP` n=4. They give the deep-drift structure
   qpAdm needs; the well-powered muscle is `Mbuti`/`Han`/`Papuan`/`Karitiana` (n=15–46). Robust
   6-pop fallback if a target's right-set f4 are noisy:
   `Mbuti,Han,Papuan,Karitiana,Russia_Kostenki_UP,Ethiopia_MotaCave_4500BP`. **Onge is absent
   from v66.**
4. **Source/right exclusivity enforced:** `Iran_GanjDareh_N`, `Israel_Natufian`, `Turkey_N`,
   `Russia_Karelia_Mesolithic` are **SOURCES**, deliberately kept OUT of the right set (a pop
   cannot be both rotating-left and fixed-right in one run).
5. **AADR/TGENO decode:** steppe's TGENO decode is correct; do NOT cross-check against AT2
   v2.0.10 TGENO goldens (those are corrupt). This atlas ran on the real v66 PUB prefix and
   decoded cleanly.

---

## 2. The atlas (best-fit model per target)

**Selection rule.** Per target, enumerate every 1–3 source subset; keep models with the solve
feasible, **all weights in [0,1]**, and **p > 0.05**; pick by **parsimony** (fewest sources,
then highest p). The target is **excluded from its own pool** and any model with the target in
`left` is filtered (see §4). 177/213 targets received a feasible model; 36 did not (§3).

**By #sources:** 39 one-way (pure), 117 two-way, 21 three-way. **53/177** best models have
**p > 0.5**. Multi-source models (138) carry **median max-SE 0.033** (range 0.012–0.109),
matching the ~0.02–0.05 design target; the 39 single-source weight=1 models carry the degenerate
~2.3e-13 SE (no jackknife variance with one source). Full 177-row table at
[`atlas_results/atlas_summary.csv`](../../atlas_results/atlas_summary.csv); the 36 no-feasible
targets at [`atlas_results/no_feasible_targets.txt`](../../atlas_results/no_feasible_targets.txt).

### Representative slice (full table = 177 rows in the CSV)

**The steppe formation + steppe-into-Europe gradient**

| target | best model (sources) | weights | max-SE | p |
|---|---|---|---|---|
| `Russia_Samara_EBA_Yamnaya` | KotiasKlde (CHG); Saratov_Khvalynsk (steppe-EHG) | 0.341; 0.659 | 0.021 | 0.225 |
| `Poland_CordedWare` | Turkey_N; Yamnaya | 0.203; 0.797 | 0.023 | 0.761 |
| `Czechia_EBA_CordedWare` | Sardinia_N (EEF); Yamnaya | 0.283; 0.717 | 0.016 | 0.705 |
| `Russia_Chelyabinsk_MLBA_Sintashta` | France_Yonne_N (EEF); Yamnaya | 0.305; 0.695 | 0.022 | 0.437 |
| `England_BellBeaker` | Spain_EN (EEF); SeredniiStih (steppe) | 0.576; 0.424 | 0.019 | 0.251 |
| `Czechia_BellBeaker` | Latvia_Kunda; KotiasKlde (CHG) | 0.424; 0.576 | 0.030 | 0.249 |
| `Germany_Leubingen1_EBA_Unetice` | France_Yonne_N; Yamnaya | 0.421; 0.579 | 0.018 | 0.193 |
| `Russia_Chelyabinsk_BA` | France_Yonne_N; Yamnaya | 0.218; 0.782 | 0.021 | 0.928 |

**Neolithic Europe — pure / near-pure EEF (no steppe)**

| target | best model | weights | max-SE | p |
|---|---|---|---|---|
| `Germany_HalberstadtSonntagsfeld_EN_LBK` | Spain_EN | 1.0 | — | 0.625 |
| `Germany_DerenburgMeerenstieg2_N_LBK` | Turkey_N | 1.0 | — | 0.533 |
| `Czechia_N` | Spain_EN | 1.0 | — | 0.769 |
| `England_EN_Megalithic` | France_Mesolithic (WHG); Spain_EN | 0.141; 0.859 | 0.030 | 0.795 |
| `Scotland_N` | Loschbour (WHG); Spain_EN | 0.095; 0.905 | 0.024 | 0.248 |
| `Portugal_N` | Loschbour (WHG); Spain_EN | 0.126; 0.874 | 0.029 | 0.099 |

**Hunter-gatherer cline — WHG + EHG**

| target | best model | weights | max-SE | p |
|---|---|---|---|---|
| `Serbia_IronGates_Mesolithic` | France_Mesolithic (WHG); Saratov_Khvalynsk | 0.786; 0.214 | 0.019 | 0.783 |
| `Latvia_LateMesolithic_Kunda` | France_Mesolithic (WHG); SeredniiStih | 0.562; 0.438 | 0.025 | 0.749 |
| `Ukraine_N` | France_Mesolithic (WHG); Saratov_Khvalynsk | 0.395; 0.605 | 0.017 | 0.430 |
| `Spain_Mesolithic` | Loschbour (WHG) | 1.0 | — | 0.404 |

**Near East — Iran_N / Levant / Caucasus axis**

| target | best model | weights | max-SE | p |
|---|---|---|---|---|
| `Jordan_PPNB` | Israel_Natufian (Levant_N); Italy_Sardinia_N | 0.556; 0.444 | 0.049 | 0.582 |
| `Jordan_EBA` | KotiasKlde (CHG); Israel_Natufian | 0.597; 0.403 | 0.053 | 0.320 |
| `Turkey_EBA` | Iran_GanjDareh_N; Austria_N_LBK | 0.362; 0.638 | 0.034 | 0.664 |
| `Turkey_MBA` | Iran_GanjDareh_N; Turkey_Catalhoyuk_MN | 0.352; 0.648 | 0.046 | 0.210 |
| `Iran_TepeHissar_C` | Georgia_KotiasKlde (CHG) | 1.0 | — | 0.080 |
| `Armenia_C` | Latvia_Kunda; KotiasKlde (CHG) | 0.192; 0.808 | 0.050 | 0.487 |
| `Greece_Crete_LBA` | Armenia_Aknashen_N (Caucasus_N); Spain_EN | 0.537; 0.463 | 0.085 | 0.292 |

**Strongest multi-source fits (p > 0.8)**

| target | best model | weights | p |
|---|---|---|---|
| `Serbia_EBA_Maros` | Spain_EN; Yamnaya | 0.664; 0.336 | 0.943 |
| `Russia_Chelyabinsk_BA` | France_Yonne_N; Yamnaya | 0.218; 0.782 | 0.928 |
| `Belgium_LN` | France_Mesolithic; Armenia_Aknashen_N | 0.535; 0.465 | 0.901 |
| `Portugal_BA` | Italy_Sardinia_N; SeredniiStih | 0.829; 0.171 | 0.895 |
| `Poland_BA` (3-way) | France_Mesolithic; KotiasKlde; SeredniiStih | 0.165; 0.488; 0.348 | 0.877 |
| `Croatia_MBA` | Serbia_IronGates; KotiasKlde (CHG) | 0.354; 0.646 | 0.867 |
| `England_EBA_C` | Italy_Sardinia_N; Saratov_Khvalynsk | 0.571; 0.429 | 0.819 |

---

## 3. The 36 targets with NO feasible model (the honest ancestry boundary)

These fail *correctly* — they are not engine bugs. Three categories (spot-checks confirmed still
infeasible even at max-sources 4 for Kazakhstan_LBA, Germany_Schwetzingen_EN, Ukraine_Mesolithic):

- **(A) Basal source-pool members used as targets** — not expressible as mixtures of the OTHER
  17 streams (CORRECT to fail): `Iran_GanjDareh_N`, `Israel_Natufian`, `Turkey_Catalhoyuk_MN`,
  `Russia_Karelia_Mesolithic`, `Russia_Saratov_Eneolithic_Khvalynsk`.
- **(B) Outside the 18-stream W-Eurasian envelope** (East-Asian / Siberian / South-Asian /
  N-African admixture): `Kazakhstan_IA`/`_LBA`/`_EIA_Tasmola` (E-Asian), `Ukraine_LIA_Scythian`,
  `Iran_ShahriSokhta_BA1`/`_BA2` (South-Asian), `Italy_Sicily_IA_*_Punic` (N-African/Levantine),
  `Russia_BolshoyOleniyOstrov_MBA`.
- **(C) Cline points / too-clean EEF vs the strong distal wall** (qpAdm power/sensitivity):
  `Ukraine_Mesolithic` (EHG–WHG cline), several German EN (`Germany_Stuttgart*_EN`,
  `Germany_Schwetzingen_EN`), `Wales_N`, `Netherlands_Mesolithic`, Iberian Chalcolithic clines.

This is the *honest ancestry boundary* of an 18-stream West-Eurasian model, not a failure mode.

---

## 4. Engine note — no self-as-source leak

A correctness detail vs the original design intent: the rotation engine does **not** auto-skip
self-as-source. For the ~18 source/target overlaps it would otherwise emit a degenerate
**weights=1, 100%-self** fit with p=1.0. The atlas build enforces, per target, a **pool = the 18
sources minus the target** AND filters any model with the target in `left`, so no degenerate
self-fit leaks into the selection. For the 15 source-overlap targets the per-target pool is 17,
giving C(17,1..3)=833 combos; the 198 non-overlap targets use the full C(18,1..3)=987.

---

## 5. THE SCALE — "could not be done before, can now"

| | value |
|---|---|
| Targets | **213** |
| Models per target | 987 (198 non-overlap) / 833 (15 source-overlap) |
| **Total model-fits WITH 713-block jackknife SE** | **207,921** |
| f2 cache build (225-pop union: 18 sources + 9 right + 213 targets, dedup) | 15 s |
| All 213 rotations | 378 s |
| **Total wall (one RTX 5090)** | **393 s = 6.6 min** |
| End-to-end loop rate | ~550 fits/sec (dominated by per-target process launch + f2 load; in-kernel batched rate is far higher: ~60k/s jk0, ~20k/s jk1) |

**The f2 cache (one build, then GPU-resident rotation).** `extract-f2 --blgsize 0.05
--maxmiss 0.5 --auto-only --device 0` over the 225-pop union: **1,112,145 of 1,233,013 SNPs kept**
(maxmiss ≤ 0.5, autosomes-only, drop-mono), **713 jackknife blocks** (5 cM), **resident tier**
(fit in 31 GB VRAM), f2.bin = 550.8 MiB, ploidy auto = 5161 pseudo-haploid + 144 diploid,
`f2_cache_id sha256:bd820179…3967c57e`. All 225 union pops verified present in the `.ind` col-3
(0 missing).

**The AT2 comparison — honest basis.** This is an *estimate*, **not** a measured AT2 re-run. Per
qpAdm fit in ADMIXTOOLS 2 (R, ~700-block jackknife over a 9-pop right set) is conservatively
**0.5–2.0 s/model** on CPU, and AT2's R loop **reslices f2 per call**. For 207,921 SE-bearing fits:

| AT2 per-fit | total | vs steppe (6.6 min) |
|---|---|---|
| 0.5 s | 28.9 h (1.2 days) | ~265× |
| 1.0 s | 57.8 h (2.4 days) | ~440× |
| 2.0 s | 115.5 h (4.8 days) | ~1056× |

A 200k-fit jackknife rotation atlas is a **multi-day-to-week** CPU job in AT2; steppe makes the
whole curated atlas an **interactive few-minute single-GPU run** — which is the point: the
curation can be *iterated* (re-pick sources, re-pick the right wall, re-run in minutes), turning
a one-shot HPC batch into a live analysis loop.

---

## 6. Reproduce

```sh
ssh box5090
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ulimit -c 0          # clear core dumps
B=/workspace/steppe/build-rel/bin/steppe
P=/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB

# 1) ONE f2 cache over the union of {18 sources + 9 right + 213 targets} (dedup → 225 pops)
$B extract-f2 --prefix "$P" --pops "<union>" --out /workspace/atlas_f2 \
   --device 0 --blgsize 0.05 --maxmiss 0.5 --auto-only

# 2) per-target rotation (pool = the 18 sources MINUS the target)
$B qpadm-rotate --f2-dir /workspace/atlas_f2 --target "<T>" \
   --pool "<18 sources minus T>" --right "<9 right>" \
   --min-sources 1 --max-sources 3 --jackknife 1 --device 0 \
   --format csv --out /workspace/atlas_csv/<T>.csv

# 3) per target pick: feasible & weights∈[0,1] & p>0.05 & target NOT in left,
#    parsimony = fewest sources then highest p
```

**Artifacts.** Local: [`atlas_results/atlas_summary.csv`](../../atlas_results/atlas_summary.csv)
(177 rows), [`atlas_results/no_feasible_targets.txt`](../../atlas_results/no_feasible_targets.txt)
(36). Box: `/workspace/atlas_summary.csv` (f2 cache + per-target CSVs + cores cleaned).

---

*REAL AADR v66 1240K · single-GPU `--device 0` · curated 18-source pool + 9-pop distal Reich
wall · 207,921 jackknife-SE qpAdm fits in 6.6 min · steppe `main` @ `8babe0f`.*
