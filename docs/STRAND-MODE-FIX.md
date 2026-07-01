# `--strand-mode` Fix Report — Strand-Ambiguous (Palindromic) SNP Policy

**Project:** steppe (GPU reimplementation of ADMIXTOOLS 2, parity-validated vs AT2 2.0.10)
**Status:** IMPLEMENTED, not committed (working-tree change on branch `phase2-fit-engine`)
**Change set:** 8 files, `+117 / -13`
**Acceptance gate:** PASS — AADR golden bit-identical under the default; HGDP `--strand-mode keep` matches AT2 to ~1e-13

---

## 1. The Divergence and the Fix

### 1.1 The divergence

steppe `extract-f2` **unconditionally dropped** strand-ambiguous / palindromic SNPs
(A/T and C/G allele pairs) — the "drop-not-flip" rule — with **no flag to disable it**.
AT2 **keeps** these SNPs by default.

On modern panels this is ~14–16% of SNPs and was the **dominant out-of-the-box
difference** between the two tools:

| Metric (HGDP, out-of-the-box) | drop vs keep gap |
|---|---|
| f4 estimate | up to ~3.4e-2 |
| f4 z (San quartet) | ~4.06e-2 (max) |
| qpAdm mixture p | ~1.7e-2 |
| qpAdm weight (rel) | ~6.34e-3 |

On a **matched** SNP set (same rows fed to both tools) steppe and AT2 agree to
~2.5e-12 — i.e. the entire discrepancy was **SNP selection**, not numerics. The
unconditional drop was a defensible merge-safety default (palindromic SNPs are a
classic strand-flip corruption source when merging sources), but it made **exact AT2
reproduction impossible**.

### 1.2 The fix — an additive `--strand-mode {drop, keep, flip}` flag

A single new policy knob, threaded CLI → config → filter → Python, with the drop made
**conditional** on the mode:

- **`drop`** (DEFAULT) — DROP strand-ambiguous SNPs. Reproduces the pre-flag behavior
  **bit-identically**. Merge-safety default.
- **`keep`** — do NOT drop palindromes; retain every clean biallelic ACGT SNP.
  Reproduces AT2's default. This is the exact-AT2-reproduction path.
- **`flip`** — documented, accepted, but **not yet implemented** as a frequency-based
  reorientation; currently behaves as `keep` (see §3).

**Multiallelic / non-ACGT SNPs are still dropped unconditionally in every mode** — only
the palindrome half of the predicate is gated, so `keep`/`flip` still yield a clean
biallelic ACGT set.

### 1.3 The core change — one shared `__host__ __device__` site

The strand-ambiguous drop lived at exactly **one** source of truth:
`keep_decision_pooled` in `src/io/filter/snp_summary_reduce.hpp`, which is called by
**both** the host mask-builder (`snp_filter.hpp::snp_keep_decision` →
`build_snp_keep_mask`) **and** the device regime-B keep-mask kernel
(`src/device/cuda/decode_compact_kernel.cu`, `regimeb_keep_mask_kernel`, which passes
`FilterConfig cfg` in). Editing this one line therefore covers host + GPU with no
signature change (`cfg` was already in scope).

```diff
- if (is_multiallelic(ref, alt) || is_strand_ambiguous(ref, alt)) return false;
+ if (is_multiallelic(ref, alt)) return false;
+ if (cfg.strand_mode == StrandMode::Drop && is_strand_ambiguous(ref, alt)) return false;
```

**Why this is parity-safe by construction:** the original `||` short-circuited
`is_multiallelic` **first**, so at `strand_mode == Drop` the two new statements evaluate
to the **exact same boolean** as the original expression. The default-drop keep-mask is
therefore bit-identical for **any** dataset — the flag is purely additive.

### 1.4 Files changed (`+117 / -13`, 8 files)

| File | Role | Change |
|---|---|---|
| `include/steppe/config.hpp` | public config | `enum class StrandMode { Drop, Keep, Flip };` above `FilterConfig`; field `StrandMode strand_mode = StrandMode::Drop;` (trivial scalar enum, device-safe by value) |
| `src/io/filter/snp_summary_reduce.hpp` | **the one compute-logic edit** | split the OR in `keep_decision_pooled` (above); gate the palindrome half on `strand_mode == Drop` |
| `src/io/filter/filter_decision.hpp` | invariant doc | §1 SCOPE reworded `DROP-NOT-FLIP` → drop-by-default, flag-gated (multiallelic always dropped; palindromes only under Drop). Predicates `is_strand_ambiguous` / `is_multiallelic` **untouched** |
| `src/core/config/cli_args.hpp` | CLI args | `std::optional<std::string> strand_mode;` (raw token, mirrors `--tier`; avoids pulling `config.hpp` into `cli_args.hpp`) |
| `src/app/cli_parse.cpp` | CLI parse | `--strand-mode` string option on the `extract-f2` subcommand, next to `--transversions` |
| `src/core/config/config_builder.cpp` | config merge | `take(merged_.strand_mode, args.strand_mode)` in `merge_cli`; `drop|keep|flip → flt.strand_mode` parse + validate in `build()` (unknown token → `InvalidConfig`) |
| `bindings/bind_fstats.cpp` | Python C++ binding | `run_extract_f2_py` gains `const std::string& strand_mode` → `filter.strand_mode`; registered `strand_mode`_a = `"drop"` |
| `bindings/steppe/__init__.py` | Python facade | `extract_f2(..., strand_mode: str = "drop")` passed through to `_core.run_extract_f2`, + docstring |

The predicate helpers in `filter_decision.hpp` are pure classifiers and stay unchanged;
only their **application** is gated. There is exactly one `is_strand_ambiguous` call site
in `src/` (grep-confirmed), so the single-site edit is complete.

---

## 2. Proof — Acceptance Gate

### 2.1 Default (`drop`, flag absent) — AADR golden BIT-IDENTICAL

**Result:** 77/77 ctest PASSED, 0 failed (Release, sm_120, CUDA 13.1; build 247/247
targets incl. the `_core` Python module). `STEPPE_AADR_ROOT=/workspace/data/aadr`.

Includes the load-bearing genotype-path tests:
- **#59 `extract_f2_regimeB_parity`** — raw-AADR-TGENO genotype-extract, host==device
  **exact `memcmp`** of the keep-mask (2.86s). This exercises the changed site directly.
- **#60 `cli_qpadm`**, **#72 `cli_extract_qpadm`**, and the PA / EIGENSTRAT / PLINK /
  ANCESTRYMAP prefix golden tests — all AADR-rooted goldens bit-identical.

**Why the AADR goldens match despite the palindromic drop (confirmed, not assumed):**

1. **The gate IS the genotype-extract path**, not a frozen f2 fixture:
   `tests/reference/test_f2_blocks_equivalence.cu` uses
   `kGenoBase = "v66.p1_HO.aadr.patch.PUB"` under
   `STEPPE_AADR_ROOT=/workspace/data/aadr/raw` and runs the full
   decode → filter → f2 chain. So the changed keep-decision is genuinely exercised.
2. **The AADR HO panel carries ZERO palindromes.** Inspection of
   `/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB.snp` (584,131 SNPs) shows an
   allele-pair histogram that is **entirely** transitions (GA 144,231; CT 120,574;
   AG 120,005; TC 90,494) and transversions (GT 32,780; TG 27,786; CA 27,113;
   AC 21,148) — **zero** self-complementary (A/T, C/G) rows. The Affymetrix Human
   Origins array was designed to exclude palindromes to avoid strand ambiguity. So
   `is_strand_ambiguous()` returns false for all 584,131 rows and the drop fires zero
   times; `drop` and `keep` produce byte-identical f2 there.
3. **Independent of the data**, at `strand_mode == Drop` the split gate collapses to the
   exact original boolean (§1.3), so default-drop is bit-identical for ANY dataset.

Default-drop bit-identity is thus **doubly guaranteed** (data no-op AND boolean identity).

### 2.2 `--strand-mode keep` matches AT2

**Reference recovery (no AT2 re-run for the genome-wide numbers):** AT2 keeps ambiguous
SNPs by default, so the AT2 HGDP numbers already captured in the dataset-generalization
run (`docs/MULTI-DATASET-RESULTS.md` + `wf_abaecc93-5bf`) **are** the keep-reference.
History recorded: steppe dropped **240,788 of 1,539,845** HGDP SNPs (~15.6%); on the
identical non-ambiguous 1,299,057-SNP set steppe==AT2 to **~2.5e-12** (f4 max rel diff
2.455e-12; qpAdm weights 4.448e-13).

**Genome-wide caveat + targeted fill:** the full-genome `HGDP.geno` (1,539,845 SNPs) was
deleted from box5090 after the original dataset test (only `HGDP.snp` /
`HGDPNA.snp` — the 1,539,845 and 1,299,057 counts — survive, confirming 240,788 ambiguous
genome-wide). Validation was therefore done on **chr22** (`HGDP22`, 82,517 SNPs, same 173
individuals / 9 pops) with a **fresh AT2 2.0.10 run on the same data** — a legitimate
targeted fill, since the underlying genome-wide `.geno` is gone.

**Mechanism (exact SNP counts, chr22):**
`steppe extract-f2 --maxmiss 0 --maf 0 --no-drop-mono --ploidy auto` (AT2-matching):

| mode | n_snp_kept | meaning |
|---|---|---|
| `drop` | 70,281 | = exactly the non-ambiguous count |
| `keep` | 82,517 | = all biallelic-ACGT SNPs |
| **delta** | **12,236** | = exactly the independently-counted strand-ambiguous SNPs on chr22 (14.8%, scaling the genome-wide 15.6%) |

AT2 `extract_f2` (default = keep, `maxmiss = 0`, `adjust_pseudohaploid = TRUE`) reported
"82517 SNPs remain" — **identical to steppe `keep`.**

**Numeric agreement (steppe `keep` vs fresh AT2, same 82,517-SNP chr22 set):**

| statistic | max rel diff |
|---|---|
| f4(San,MbutiPygmy;Han,French) est / se / z | 3.2e-12 / 1.9e-13 / 3.4e-12 |
| f4(Han,Uygur;French,Yoruba) est / se / z | 2.9e-14 / 1.0e-13 / 1.3e-13 |
| qpAdm Uygur = Han/French weights | 1.0e-13 / 1.7e-13 |

All ~1e-12…1e-13 — consistent with the history's 2.5e-12.

**The gap this closes (chr22):** steppe `drop` vs AT2(keep) was **13.4%** on f4 Q1
estimate and **2.4%** on the qpAdm Han weight (0.6385 drop vs 0.6233 keep/AT2). Under
`--strand-mode keep` this collapses to **~1e-13**.

**Known residual (pre-existing, unrelated to this flag):** qpAdm weight-SE still differs
(steppe 0.0989 vs AT2 0.1049 on chr22) — the documented jackknife/GLS method nuance. The
**weights themselves** match to ~1e-13; chisq and p match. This flag does not touch it.

### 2.3 Direction clarification (important for the gate)

The ~2.5e-12 "matched set" agreement is steppe-**drop** vs AT2-on-a-pre-stripped-file
(both see 1,299,057) — it validates the **default-drop** path. `--strand-mode keep` must
instead match AT2's **default full-SNP** numbers (the f4/qpAdm keep-reference table),
which §2.2 confirms on chr22. History also records that steppe's earlier "keep all SNPs"
attempt STILL dropped exactly 240,788 — direct evidence there was **no** code path to keep
palindromes before this flag.

---

## 3. `flip` Status and the Remaining General-Dataset Item

### 3.1 `flip` is a documented, accepted, not-yet-implemented token

`flip` parses and validates through the whole chain (CLI + config + `enum
StrandMode::Flip` + Python), but performs **no** frequency-based reorientation. The
keep-gate treats any non-`Drop` mode as "retain", so `flip` currently behaves **exactly
like `keep`** (retains palindromes, no reorientation).

**Why `flip` is deliberately a stub, not a live freq pass:**

- `filter_decision.hpp` §1 SCOPE is an explicit invariant: *"No Q value is ever altered by
  a filter… We do NOT infer strand."* The filters produce **only a keep-mask**, never a
  data rewrite.
- A true frequency-resolved flip must **reorient alleles** — rewrite ref/alt and
  complement Q — against a **reference strand**. That (a) violates the no-alter-Q
  invariant and (b) needs a reference panel that a **single-source** extract does not
  have.
- For a single-source extract there is nothing to reconcile against, so a genuine flip
  **degenerates to keep anyway**. Aliasing `flip → keep` is therefore the correct current
  semantics, not a shortcut.

A future freq-based reorientation lands at the `StrandMode::Flip` enum value with **no
further plumbing** — the multi-source merge path is where a reference-strand freq flip
would actually be meaningful (the TODO anchor).

**Recommendation for `flip` UX:** either emit a one-line warning that `flip` currently
performs no reorientation (behaves as `keep`), or reject it with an explicit
`strand-mode=flip not yet implemented; use keep`. Do not let it silently imply a
reorientation happened.

### 3.2 The remaining general-dataset item (limitations doc)

`drop` + `keep` fully cover the **AT2-parity** goal. The one open item is documenting the
scope so users choose correctly:

- **On a single-source extract:** `keep` = reproduce AT2; `drop` = merge-safe. `flip`
  cannot do better than `keep` and is a stub. This should be stated in the extract-f2
  docs / `--help` and the limitations page.
- **On a multi-source merge** (not this change): palindromic SNPs are a real strand-flip
  corruption source. This is where a true reference-strand `flip` (and stricter default
  policy) is warranted — flagged as future work, not shipped here.
- **Observability (recommended, non-load-bearing for parity):** echo `strand_mode` in the
  `--dry-run` filter line, and record it in the extracted f2-dir `meta.json` (`F2DirMeta`)
  so a produced f2 directory self-documents which mode built it. A unit test pinning that
  palindromes **survive** under `keep` (and are dropped under `drop`) is also recommended —
  currently only the `is_strand_ambiguous` predicate is tested; nothing pins the
  application of the gate.

---

## 4. Recommendation

1. **Ship `drop` + `keep` as-is.** They meet the acceptance gate: AADR golden
   bit-identical under the default, HGDP `keep` matches AT2 to ~1e-13, and the change is
   provably additive (default-drop is the exact original boolean).
2. **Keep the default at `drop`.** This preserves the merge-safety posture and the
   bit-identical golden gate. `keep` is the opt-in exact-AT2-reproduction switch.
3. **Land `flip` as a documented stub aliased to `keep`** with a clear "no reorientation
   performed" signal, and a TODO pointing at the multi-source merge path. Do not
   half-implement a freq flip that would violate the no-alter-Q invariant.
4. **Add the observability + one unit test** in §3.2 before release so the mode is
   self-documenting and the gate application (not just the predicate) is pinned.
5. **Document the scope** in the extract-f2 docs / limitations page: single-source →
   `keep` reproduces AT2, `flip` is future work for multi-source merges.

**Bottom line:** the flag closes the dominant out-of-the-box steppe↔AT2 divergence
(SNP selection, up to ~4% f4 / ~1–2% qpAdm) while leaving every existing result
byte-for-byte unchanged. It is safe to merge behind the default and is the last piece
needed for exact AT2 reproduction on modern (palindrome-carrying) panels.

---

### Provenance

- Genome-wide keep-reference: recovered from history (`docs/MULTI-DATASET-RESULTS.md`,
  `wf_abaecc93-5bf/journal.jsonl`, `agent-acb5cb2c3e921e00a.jsonl`) — **no AT2 re-run.**
- chr22 keep validation: fresh AT2 2.0.10 run on `HGDP22` (legitimate targeted fill; the
  genome-wide `.geno` was deleted from box5090).
- AADR bit-identity: 77/77 ctest under `STEPPE_AADR_ROOT=/workspace/data/aadr`, plus
  direct `.snp` allele-pair histogram (0 palindromes on the HO panel) and the boolean
  identity argument.
- **Nothing committed.** Pre-existing unrelated working-tree modifications
  (`bindings/module.cpp`, `cmd_f4ratio.cpp`, several device `.cu`/`.hpp`) were left
  untouched and are **not** part of this change set.
