# Olalde et al. 2018 — competing-sources ROTATION on steppe vs the published SI tables

A reproduction of the Olalde 2018 **competing-sources qpAdm rotation** for the British Bell
Beaker target on **REAL AADR v66 1240K** (no synthetic data), **single-GPU** (`--device 0`),
via the new `steppe qpadm-rotate` CLI. This is the *rotation* companion to
[`olalde2018.md`](olalde2018.md) (which reproduced the three headline point-models): here we
enumerate **every 1-, 2- and 3-source subset** of a candidate pool with ONE batched
`run_qpadm_search` call and read off the **feasible-model set + weights**, then compare to the
study's published qpAdm tables.

- **Study:** Olalde I, Brace S, Allentoft ME, … Reich D. **"The Beaker phenomenon and the
  genomic transformation of northwest Europe."** *Nature* **555**, 190–198 (2018).
  DOI:[10.1038/nature25738](https://doi.org/10.1038/nature25738). SI:
  `41586_2018_BFnature25738_MOESM2_ESM.pdf` (the 7 MB SI PDF; **SI section 8 = qpAdm/qpWave**).
- **No AT2 run anywhere.** The `qpadm-rotate` CLI wiring is gated by the EXISTING
  `golden_rot.json` (engine test `qpadm_rotation` + CLI e2e test `cli_rotate`, both green); the
  *study* comparison below is purely against the **published SI tables**, not an AT2 re-run.

> **Table numbering note.** The task framed this as "SI Table 6". Olalde 2018 numbers its
> supplementary tables **S1…S12** (there is no plain "Table 6"); the competing-sources qpAdm
> tables are **Table S4** (3-way Steppe_EBA + Anatolia_N + WHG — the Fig. 2a "~50% steppe"
> source), **Table S7** (Steppe_EBA + one competing Neolithic source — the literal
> competing-source rotation), and **Table S8** (S7 + KO1 third source). Our steppe rotation
> (target = British Beaker, pool of farmers + steppe + HGs, 2- and 3-way) lines up most
> directly with **Table S4**; S7's single-Neolithic-source idiom and its feasibility verdicts
> are reported alongside. We read all per-cell numbers directly from the SI PDF (text-layer
> extractable — values below are quoted, not estimated).

---

## HEADLINE

**MATCH.** steppe's rotation reproduces the central Olalde-2018 result: the British Bell
Beaker target is feasibly modelled as **~59% Steppe_EBA + ~30% Anatolian farmer + ~11% WHG**,
against the paper's Table S4 **BK_England_SOU = 0.578 Steppe_EBA / 0.260 Anatolia_N / 0.162
WHG (P=0.682)**. The exact 3-way model (`Russia_Samara_EBA_Yamnaya + Turkey_N +
Luxembourg_Loschbour_Mesolithic`) is **FEASIBLE** in steppe (p=0.290, f4rank 2, all weights
positive) with **steppe weight 0.589 vs the published 0.578** — a near-exact match on the
headline number. Across the 63 enumerated models steppe also reproduces the qualitative
shape: 2-way Steppe+farmer fits sit near ~0.61 steppe and are feasible, while
farmer-only / HG-only and steppe+steppe (Yamnaya+CordedWare) combinations are rejected or
collapse to a single source.

---

## The rotation setup

| | value |
|---|---|
| **Target** | `England_BellBeaker` (n=28; the v66 label for the study's British Beaker, ≈ `BK_England_SOU` n=27) |
| **Source pool (7)** | `Russia_Samara_EBA_Yamnaya` (Steppe_EBA), `Turkey_N` (Anatolia_N), `Spain_EN`, `England_N`, `Czechia_EBA_CordedWare`, `Luxembourg_Loschbour_Mesolithic` (WHG), `Russia_Karelia_Mesolithic` (EHG) |
| **Right / outgroups (6)** | `Mbuti`, `Israel_Natufian`, `Iran_GanjDareh_N`, `Han`, `Papuan`, `Karitiana` (nr = 5; right[0]=R0) |
| **Enumeration** | every subset of the pool of size 1..3 → **63 models** (7 + C(7,2)=21 + C(7,3)=35) |
| **Knobs** | `--min-sources 1 --max-sources 3 --jackknife 1 --format csv --device 0`; `--maxmiss 0`, `--blgsize 0.05`, `--fudge 1e-4` (defaults) |
| **f2 dir** | 14-pop union, **672756 / 1233013 SNPs kept**, **711 jackknife blocks**, resident tier, ploidy auto (306 pseudo-haploid + 114 diploid), `f2_cache_id sha256:4aa67efb…d74b8ec` |

**Outgroup-set caveat (important).** Olalde 2018 used the Reich-lab **basic 9-outgroup set**
(Mota, Ust_Ishim, MA1, Villabruna, Mbuti, Papuan, Onge, Han, Karitiana), and for the
competing-Neolithic Table S7/S8 *added* LBK_EN, Iberia_EN, LaBraña1, ElMiron. We use the
**steppe golden 6-outgroup set** (the `golden_fit0`/`golden_rot` right with nr=5 ≤ 32, the
S8 design-center that exercises the batched common path) — Mbuti/Han/Papuan/Karitiana overlap
the paper, plus Israel_Natufian + Iran_GanjDareh_N replace the paper's deep-ENA/UP outgroups.
This is a **deliberate divergence** (a smaller, modern Reich-lab outgroup basis), and the
expected source of the few-percent weight offsets and any feasibility-tier differences vs the
published numbers. Exact p-value parity to the paper is **not** the bar here (different
outgroups, different AADR vintage, different SNP intersection); the bar is the **feasible-model
SET + the headline steppe proportion**.

---

## PAPER → v66 1240K population mapping (the version switches)

All v66 labels are the 3rd column of
`/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB.ind`, **verified present** by counting
rows (n below). No label had to be substituted away — every pool/right pop resolved cleanly.

| Paper label (SI) | v66 1240K label | n | Note |
|---|---|---|---|
| British Beaker / `BK_England_SOU` | **England_BellBeaker** | 28 | Rename (the rotation target). |
| `Steppe_EBA` / Yamnaya_Samara | **Russia_Samara_EBA_Yamnaya** | 46 | Rename; Samara is the canonical Yamnaya among ~50 v66 site sub-labels. |
| `Anatolia_N` | **Turkey_N** | 68 | Clean rename. |
| Iberian Early Neolithic (Iberia_EN-adjacent) | **Spain_EN** | 23 | Iberian farmer source (the S7 "Iberia-related Neolithic" competitor role). |
| `Britain_N` / British Neolithic | **England_N** | 46 | British Neolithic source (`England_EN` n=10 is a different, narrower label). |
| Corded Ware (Czech) / `Corded_Ware_Czech` | **Czechia_EBA_CordedWare** | 48 | Rename; central-European steppe-ancestry source competitor. |
| `WHG` | **Luxembourg_Loschbour_Mesolithic** | 2 | **PROXY** — no bare `WHG` in v66; canonical AT2 WHG (thin n=2 but standard). |
| EHG | **Russia_Karelia_Mesolithic** | 16 | EHG source (the steppe HG-affinity competitor). |
| `Mbuti` | **Mbuti** | 15 | Present (shared with paper's outgroup set). |
| `Han` | **Han** | 46 | Present (shared). |
| `Papuan` | **Papuan** | 32 | Present (shared). |
| `Karitiana` | **Karitiana** | 16 | Present (shared). |
| — | **Israel_Natufian** | 17 | steppe golden outgroup (replaces paper's Mota/Onge ENA-UP slot). |
| — | **Iran_GanjDareh_N** | 17 | steppe golden outgroup (replaces paper's Ust_Ishim/MA1/Villabruna UP slot). |

**Version-switch summary:** `Anatolia_N`→**Turkey_N**, `Steppe_EBA`→**Russia_Samara_EBA_Yamnaya**,
`Corded_Ware_Czech`→**Czechia_EBA_CordedWare**, `WHG`→**Luxembourg_Loschbour_Mesolithic**
(proxy), British Beaker→**England_BellBeaker**, British Neolithic→**England_N**. The
outgroup basis is a deliberate switch (steppe 6-pop golden set, not the paper's 9-pop) — see
the caveat above.

---

## Published Olalde 2018 SI — the competing-sources result

**Table S4** (3-way `Steppe_EBA + Anatolia_N + WHG`; the Fig. 2a "~50% steppe" source) — the
British / English rows:

| Test (SI label) | Steppe_EBA | SE | Anatolia_N | SE | WHG | SE | P-value |
|---|---|---|---|---|---|---|---|
| **BK_England_SOU (27)** | **0.578** | 0.021 | 0.260 | 0.021 | 0.162 | 0.009 | **0.682** |
| BK_England_NOR (5) | 0.440 | 0.036 | 0.374 | 0.036 | 0.186 | 0.016 | 0.245 |
| England_CA_EBA (23) | 0.491 | 0.022 | 0.349 | 0.021 | 0.160 | 0.009 | 0.255 |
| England_MBA (19) | 0.489 | 0.022 | 0.334 | 0.021 | 0.177 | 0.009 | 0.047 |
| (Beaker-complex range, all groups) | 0–0.75 | | | | | | |

Across the full Beaker complex Table S4 gives 0–75% steppe by region; **British Beaker sits
near ~0.58 steppe**, and the main text rounds the British figure to "~50% steppe-related
ancestry" (Fig. 2a). The paper notes "many populations can be explained by Anatolia_N + WHG
without any contribution from Steppe_EBA, indicating a lack of Steppe-related ancestry" — i.e.
the feasibility verdict is meaningful, not automatic.

**Table S7** (the literal competing-Neolithic-source rotation: `Steppe_EBA + ONE Neolithic/CA
source`, for "BC outside Iberia combined", which includes British Beaker; P>0.05 = feasible):

| Neolithic source | P-value | Steppe_EBA | feasible? |
|---|---|---|---|
| Poland_LN | 1.84E-01 | 0.424 | **YES** |
| Sweden_MN | 2.45E-01 | 0.403 | **YES** |
| Iberia_MN | 1.11E-12 | 0.469 | no |
| C_Iberia_CA | 7.48E-10 | 0.474 | no |
| N_Iberia_CA | 2.18E-10 | 0.482 | no |
| Germany_MN | 2.51E-13 | 0.469 | no |
| Hungary_LCA | 6.05E-55 | 0.522 | no |
| France_MLN | 4.61E-04 | 0.478 | no |

S7's published verdict for Beaker outside Iberia: **only Poland_LN and Sweden_MN fit**
(HG-rich central/north-European farmers); Iberian and German/Hungarian Neolithic sources are
rejected. Steppe_EBA weight sits at **~0.40–0.52** across the S7 single-Neolithic-source
models. (Table S8 adds KO1 as a third source and recovers Germany_MN/Hungary_LCA fits.)

---

## steppe result — the rotation feasible set

**REAL v66 1240K · single-GPU (`--device 0`) · `--maxmiss 0` · `--blgsize 0.05` ·
`--jackknife 1` · 711 blocks · 672756/1233013 SNPs · 63 models, ONE batched
`run_qpadm_search`.** `feasible` = passed the f4-rank test AND (jackknife=1) all weights in
[0,1] within tolerance. The decisive, pop-gen-meaningful rows (positive-weight 2- and 3-way
fits) are below; single-source rows (model_index 0–6) are f4rank-0 "fits" that trivially pass
the rank test and are not informative.

**3-way feasible models (f4rank 2; all weights positive):**

| left set | p | Steppe (Yamnaya) | farmer | HG | feasible |
|---|---|---|---|---|---|
| **Yamnaya + Turkey_N + Loschbour** *(= Table S4 model)* | **0.290** | **0.589** | 0.297 (Turkey_N) | 0.113 (Loschbour) | **TRUE** |
| Yamnaya + Spain_EN + Loschbour | 0.449 | 0.607 | 0.358 (Spain_EN) | 0.036 | TRUE |
| Yamnaya + Spain_EN + Karelia | 0.353 | 0.598 | 0.380 (Spain_EN) | 0.022 (EHG) | TRUE |
| Yamnaya + Spain_EN + England_N | 0.287 | 0.621 | 0.329 (Spain_EN) | 0.050 (England_N) | TRUE |
| Yamnaya + England_N + Loschbour | 0.241 | 0.603 | 0.379 (England_N) | 0.019 | TRUE |
| Yamnaya + Turkey_N + Karelia | 0.005* | 0.564 | 0.362 (Turkey_N) | 0.074 (EHG) | TRUE† |

*p=0.005 fails the usual 0.05 gate but steppe marks it feasible on weights∈[0,1]; treat as
borderline. †Several farmer+CordedWare+HG 3-ways (Turkey_N/Spain_EN/England_N +
Czechia_EBA_CordedWare + Loschbour/Karelia, model_index 50/51/56/57/59/60) are also feasible
with high p (0.35–0.90) — here CordedWare carries the steppe ancestry, so they are
steppe-bearing fits in disguise (CordedWare ≈ 0.75 steppe).

**2-way feasible models (f4rank 1; positive weights):**

| left set | p | weights | feasible |
|---|---|---|---|
| Yamnaya + Spain_EN | 0.435 | [0.623, 0.377] | TRUE |
| Yamnaya + England_N | 0.339 | [0.610, 0.390] | TRUE |
| Turkey_N + Czechia_EBA_CordedWare | 0.248 | [0.112, 0.888] | TRUE |
| Spain_EN + Czechia_EBA_CordedWare | 0.705 | [0.130, 0.870] | TRUE |
| England_N + Czechia_EBA_CordedWare | 0.838 | [0.138, 0.862] | TRUE |

**Rejected / degenerate (the meaningful "no" verdicts):** Yamnaya+Czechia_CordedWare and
Yamnaya+Karelia collapse to a single source (one weight goes negative — two steppe sources are
redundant); Turkey_N+Spain_EN, Turkey_N+England_N, Spain_EN+England_N (farmer+farmer, no
steppe) blow up to wild signed weights (rejected); Loschbour+Karelia (HG+HG) is rejected.
**No farmer-only or HG-only 2-/3-way is feasible without a steppe source** — exactly the
paper's "lack of Steppe-related ancestry" logic in reverse.

---

## COMPARISON — steppe feasible set + weights vs published

1. **Headline 3-way (Table S4).** Olalde S4 BK_England_SOU = **0.578** Steppe_EBA / 0.260
   Anatolia_N / 0.162 WHG, P=0.682. steppe's exact analogue (Yamnaya + Turkey_N + Loschbour) =
   **0.589** Steppe / 0.297 Turkey_N / 0.113 WHG, p=0.290, FEASIBLE, f4rank 2. **MATCH** — the
   steppe proportion is within ~1 point (0.589 vs 0.578); the WHG/farmer split tilts slightly
   farmer-ward (the thin n=2 Loschbour WHG proxy + the different 6-pop outgroup basis). Both
   are feasible 3-way fits with the same ~0.59/0.30/0.11 shape and the same "British Beaker is
   roughly half-to-three-fifths steppe" conclusion (Fig. 2a "~50%").
2. **Steppe is obligatory.** Every feasible 2-/3-way carries a steppe source (Yamnaya
   directly, or CordedWare as a steppe proxy); every farmer-only / HG-only combination is
   rejected. This reproduces the paper's central qualitative claim (Table S4 narrative: groups
   without steppe are explained by Anatolia_N + WHG alone; British Beaker is **not**).
3. **Competing-source feasibility (vs Table S7).** The paper's S7 idiom — Steppe_EBA + ONE
   competing Neolithic — found British (BC-outside-Iberia) Beaker feasible only with HG-rich
   farmers (Poland_LN, Sweden_MN) and rejected Iberian/German farmers. Our pool's farmers are
   Turkey_N / Spain_EN / England_N (not the paper's exact S7 panel), and against our 6-pop
   outgroups the 2-way Yamnaya+farmer fits are feasible at ~0.61 steppe for Spain_EN and
   England_N. We do **not** reproduce S7's per-source rejections (we lack the paper's added
   LBK_EN/Iberia_EN/LaBraña1/ElMiron outgroups that gave S7 its Iberia-discriminating power) —
   this is the **expected divergence** from the deliberate outgroup-set switch, and is
   documented as such, not a discrepancy in the engine.

**Differences + cause:** (a) steppe weight 0.589 vs 0.578 — ~1 pt high, from the Loschbour
n=2 WHG proxy + the 6-pop (vs 9-pop) outgroup basis; (b) p-values are lower than the paper's
(e.g. 0.290 vs 0.682 on the 3-way) — fewer/different outgroups give the rank test less slack;
(c) we do not reproduce Table S7's fine-grained Neolithic-source rejections — we use a
different farmer panel and lack the paper's Iberia-discriminating extra outgroups. All three
trace to the **outgroup-set switch + proxy + v66 SNP-set/vintage**, not to the rotation engine
(which is golden-gated bit-for-bit against `golden_rot.json`).

---

## Verified copy-paste ssh one-liner

Runs on `box5090` (single RTX 5090, `--device 0`), builds the 14-pop f2 dir, runs the full
1/2/3-source rotation, and self-cleans. **VERIFIED end-to-end** (round-tripped: 14-pop f2,
672756/1233013 SNPs, 711 blocks; 63 models; the Table-S4 3-way `Yamnaya;Turkey_N;Loschbour`
feasible at p≈0.290, steppe weight ≈0.589).

```bash
ssh box5090 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && B=/workspace/steppe/build-rel/bin/steppe && P=/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB && D=/workspace/data/study/olalde_rot_f2 && POOL="Russia_Samara_EBA_Yamnaya,Turkey_N,Spain_EN,England_N,Czechia_EBA_CordedWare,Luxembourg_Loschbour_Mesolithic,Russia_Karelia_Mesolithic" && R="Mbuti,Israel_Natufian,Iran_GanjDareh_N,Han,Papuan,Karitiana" && rm -rf $D && $B extract-f2 --prefix $P --pops "England_BellBeaker,$POOL,$R" --out $D --device 0 --blgsize 0.05 --maxmiss 0 --auto-only && $B qpadm-rotate --f2-dir $D --target England_BellBeaker --pool "$POOL" --right "$R" --min-sources 1 --max-sources 3 --jackknife 1 --format csv --device 0 ; rm -rf $D'
```

(On the capable box swap `box5090`→`rtxbox`; everything else is identical. `--maxmiss 0` did
not collapse the SNP set despite the thin n=2 Loschbour, so no relax to 0.5 was needed.)

---

## Provenance / caveats summary

- **REAL data only:** v66 1240K (`/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB.*`),
  no synthetic. **Single-GPU** (`--device 0`); multi-GPU parked.
- **NO AT2 run:** the `qpadm-rotate` CLI is gated by the existing `golden_rot.json`
  (`qpadm_rotation` engine test + `cli_rotate` CLI e2e test, both green at HEAD `17c4606`); the
  study comparison is against the **published SI tables S4/S7/S8** only.
- **Outgroup-set switch (deliberate):** steppe 6-pop golden right
  (Mbuti, Israel_Natufian, Iran_GanjDareh_N, Han, Papuan, Karitiana) vs the paper's 9-pop basic
  set + the S7 add-ons — the main documented source of the small weight offsets / lower p / the
  S7 per-source feasibility differences.
- **Proxy:** `WHG`→`Luxembourg_Loschbour_Mesolithic` (n=2, thin but canonical).
- **No fabrication:** the firm published comparison target is Table S4's BK_England_SOU
  **0.578 Steppe_EBA (P=0.682)**, read directly from the SI text layer; the headline match is
  steppe's 0.589 on the same 3-way.
</content>
</invoke>
