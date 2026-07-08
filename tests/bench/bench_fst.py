#!/usr/bin/env python3
"""bench_fst.py — emit the 8-row FST validation+benchmark table (REAL AADR).

main(cfg) -> bench_common.Table. See tests/bench/README.md for the row semantics.
"""
from __future__ import annotations

import os

import numpy as np

import bench_common as bc


def _f(x: str) -> float:
    """Tolerant float parse: steppe emits 'NA' for NaN/invalid sites and summaries."""
    return float("nan") if x in ("NA", "nan", "") else float(x)


PAIR = ("Han", "Papuan")          # the workhorse fit9_ped pair (153 vs 46)
ORACLE_M = 100_000                # thinned SNPs for the numpy-oracle/parity join (fix #5)
MISS_LEVELS = (0, 5, 10, 20)


# --- steppe fst runners ------------------------------------------------------

def steppe_fst_summary(cfg: bc.Cfg, prefix: str, pops: tuple[str, str],
                       out: str) -> tuple[float, dict]:
    wall, _ = bc.run_timed([
        cfg.steppe, "fst", "--prefix", prefix, "--pops", ",".join(pops),
        "--method", "wc", "--format", "tsv", "--device", cfg.device, "--out", out])
    with open(out) as f:
        f.readline()
        t = f.readline().rstrip("\n").split("\t")
    return wall, {"popA": t[0], "popB": t[1], "n_valid": int(t[2]),
                  "fst_ratio": _f(t[3]), "fst_mean": _f(t[4])}


def steppe_fst_per_snp(cfg: bc.Cfg, prefix: str, pops: tuple[str, str],
                       out: str) -> tuple[float, dict[str, tuple]]:
    wall, _ = bc.run_timed([
        cfg.steppe, "fst", "--prefix", prefix, "--pops", ",".join(pops),
        "--method", "wc", "--per-snp", "--format", "tsv", "--device", cfg.device,
        "--out", out])
    rows: dict[str, tuple] = {}
    with open(out) as f:
        head = f.readline().rstrip("\n").split("\t")
        ci = {h: i for i, h in enumerate(head)}
        for line in f:
            t = line.rstrip("\n").split("\t")
            rows[t[ci["snp_id"]]] = (_f(t[ci["fst_num"]]), _f(t[ci["fst_den"]]),
                                     _f(t[ci["fst"]]), int(t[ci["valid"]]))
    return wall, rows


def steppe_fst_allpairs(cfg: bc.Cfg, prefix: str, pops: list[str], out: str) -> float:
    wall, _ = bc.run_timed([
        cfg.steppe, "fst", "--all-pairs", "--prefix", prefix, "--pops", ",".join(pops),
        "--method", "wc", "--format", "tsv", "--device", cfg.device, "--out", out])
    return wall


# --- the 8 rows --------------------------------------------------------------

def main(cfg: bc.Cfg) -> bc.Table:
    tbl = bc.Table("fst")
    sc = cfg.scratch
    hw = "RTX 5090 / EPYC 9654 (nproc=384)"

    # ---- Row 1: parity vs plink2 (live, full fit9_ped Han vs Papuan) --------
    pheno = bc.write_pheno(cfg, cfg.fit9, os.path.join(sc, "fit9_pheno.txt"))
    keep_hp, n_hp = bc.write_keep(cfg, cfg.fit9, list(PAIR), os.path.join(sc, "keep_hp.txt"))
    _, st_snp = steppe_fst_per_snp(cfg, cfg.fit9, PAIR, os.path.join(sc, "fst_parity.tsv"))
    _, _, pv = bc.plink_fst_wc(cfg, cfg.fit9, pheno, keep_hp,
                               os.path.join(sc, "plink_hp"), per_variant=True)
    plink = bc.parse_fst_var(pv)
    diffs = [abs(st_snp[i][2] - plink[i]) for i in plink
             if i in st_snp and st_snp[i][3] == 1 and np.isfinite(st_snp[i][2])]
    maxd = max(diffs) if diffs else float("nan")
    _, st_summ = steppe_fst_summary(cfg, cfg.fit9, PAIR, os.path.join(sc, "fst_summ.tsv"))
    p_summ = bc.parse_fst_summary(os.path.join(sc, "plink_hp.fst.summary"))
    tbl.add(1, "parity_vs_reference",
            "max|WC_FST steppe-plink2| per-SNP (+ genome ratio)",
            f"maxabs={maxd:.2e} over {len(diffs)} sites; "
            f"ratio steppe={st_summ['fst_ratio']:.6f} plink2={p_summ:.6f}",
            "1e-6", "PASS" if (np.isfinite(maxd) and maxd < 1e-6) else "FAIL",
            f"plink2 --fst method=wc report-variants, LIVE; {PAIR[0]} vs {PAIR[1]}, full 584k")

    # ---- Row 2: scaling with K (all-pairs, full 1240K subset) ---------------
    counts = bc.ind_pop_counts(cfg.full)
    ordered = [p for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    Ks = [10, 25, 50] if cfg.quick else [10, 25, 50, 100]
    Mfull = 1_233_013
    kcurve = []
    thr = []
    for K in Ks:
        pops = ordered[:K]
        w = steppe_fst_allpairs(cfg, cfg.full, pops, os.path.join(sc, f"ap_{K}.tsv"))
        kcurve.append((f"K={K}", w))
        thr.append(K * (K - 1) // 2 * Mfull / w)
    tbl.add(2, "scaling_K (all-pairs)", "wall per K (+ pairs*sites/s)",
            bc.fmt_curve(kcurve) + f"; peak={max(thr):.2e} pair*site/s",
            "-", "measured", "full 1240K subset, genome-wide WC matrix")

    # ---- Row 3: scaling with SNP count (thinned fit9_ped, Han vs Papuan) ----
    Ms = [100_000, 300_000, 584_131]
    mcurve = []
    for M in Ms:
        prefix = cfg.fit9 if M == 584_131 else bc.ensure_thinned_bed(cfg, M)
        w, s = steppe_fst_summary(cfg, prefix, PAIR, os.path.join(sc, f"fst_m_{M}.tsv"))
        label = f"{M//1000}k" if M != 584_131 else "584k"
        mcurve.append((f"{label}(nv={s['n_valid']})", w))
    tbl.add(3, "scaling_SNPs", "wall per M SNPs",
            bc.fmt_curve(mcurve), "-", "measured",
            "thinned fit9_ped (SNP set changes -> FST differs; measures scaling, n_valid finite)")

    # ---- Row 4: missingness stress (inject into full fit9 bed) --------------
    mcells = []
    n_valid_seq = []
    all_match = True
    for p in MISS_LEVELS:
        bed = bc.ensure_injected_bed(cfg, p)
        _, s = steppe_fst_summary(cfg, bed, PAIR, os.path.join(sc, f"fst_miss_{p}.tsv"))
        keep_i, _ = bc.write_keep(cfg, bed, list(PAIR), os.path.join(sc, f"keep_miss_{p}.txt"))
        _, summ, _ = bc.plink_fst_wc(cfg, bed, pheno, keep_i,
                                     os.path.join(sc, f"plink_miss_{p}"), per_variant=False)
        pf = bc.parse_fst_summary(summ)
        match = abs(s["fst_ratio"] - pf) < 1e-6
        all_match = all_match and match
        mcells.append(f"{p}%->ratio={s['fst_ratio']:.5f}(nv={s['n_valid']},"
                      f"plink2 d={abs(s['fst_ratio']-pf):.1e})")
        n_valid_seq.append(s["n_valid"])
    monotone = all(n_valid_seq[i] >= n_valid_seq[i + 1] for i in range(len(n_valid_seq) - 1))
    finite = all(np.isfinite(float(c.split("ratio=")[1].split("(")[0])) for c in mcells)
    ok4 = monotone and finite and all_match
    tbl.add(4, "missingness", "n_valid / finite / monotone / ==plink2",
            "; ".join(mcells) + f"; monotone={monotone}",
            "no NaN, ==plink2 <1e-6", "PASS" if ok4 else "flag",
            "missingness injected into real fit9_ped (seed 1)")

    # ---- Row 5: small-sample flag behaviour ---------------------------------
    # Both pops singleton -> WC n_bar=1 -> every site invalid -> n_valid==0 (fix #2).
    cells5 = []
    for N in (1, 2, 5):
        pre = os.path.join(sc, f"smalln_{N}")
        kept = bc.write_relabeled_fam(cfg, pre, {PAIR[0]: N, PAIR[1]: N})
        try:
            _, s = steppe_fst_summary(cfg, pre, PAIR, os.path.join(sc, f"fst_smalln_{N}.tsv"))
            flag = "valid=0(flag)" if s["n_valid"] == 0 else f"valid,n_valid={s['n_valid']}"
            ratio = "nan" if not np.isfinite(s["fst_ratio"]) else f"{s['fst_ratio']:.4f}"
            cells5.append(f"N={N}x2({kept}smp)->{flag},ratio={ratio}")
        except RuntimeError as e:
            cells5.append(f"N={N}->status={type(e).__name__}(no crash)")
    n1_flags = "valid=0" in cells5[0]
    tbl.add(5, "small_sample", "flag/status at N=1/2/5 (both pops)",
            "; ".join(cells5), "no crash; N=1->valid=0", "flag" if n1_flags else "flag",
            "WC n_bar<=1 invalidates the site; N=2,5 valid (high variance)")

    # ---- Row 6: CPU (numpy FP64) vs GPU, thinned bed ------------------------
    thin = bc.ensure_thinned_bed(cfg, ORACLE_M)
    keep_o, _ = bc.write_keep(cfg, thin, list(PAIR), os.path.join(sc, "keep_oracle.txt"))
    raw = bc.plink_recode_A(cfg, thin, keep_o, os.path.join(sc, "fst_oracle"))
    iids, G = bc.load_raw(raw)
    fam = {iid: pop for _, iid, pop in bc.read_fam(thin)}
    ia = [i for i, iid in enumerate(iids) if fam.get(iid) == PAIR[0]]
    ib = [i for i, iid in enumerate(iids) if fam.get(iid) == PAIR[1]]
    _, st_o = steppe_fst_per_snp(cfg, thin, PAIR, os.path.join(sc, "fst_oracle_steppe.tsv"))
    # Order the SNP ids by the thinned .bim so oracle columns align with steppe rows.
    bim_ids = [ln.split()[1] for ln in open(thin + ".bim")]
    orc = bc.wc_fst_oracle(G[ia, :], G[ib, :])
    a6, m6, cnt = [], [], 0
    for j, sid in enumerate(bim_ids):
        if sid in st_o and st_o[sid][3] == 1 and orc["valid"][j]:
            d = abs(orc["fst"][j] - st_o[sid][2])
            a6.append(d)
            cnt += 1
    maxa = max(a6) if a6 else float("nan")
    meana = float(np.mean(a6)) if a6 else float("nan")
    tbl.add(6, "cpu_vs_gpu", "max-abs, mean-abs |numpyFP64 - steppeGPU| per-SNP FST",
            f"maxabs={maxa:.2e}, meanabs={meana:.2e} over {cnt} sites",
            "<1e-9", "PASS" if (np.isfinite(maxa) and maxa < 1e-9) else "FAIL",
            "CPU oracle = numpy FP64 WC (same wc_fst.hpp algebra); "
            "internal CpuBackend gated by ctest test_fst_wc; thinned 100k")

    # ---- Row 7: runtime vs reference (like-for-like single pair) ------------
    w_s, _ = steppe_fst_summary(cfg, cfg.fit9, PAIR, os.path.join(sc, "fst_rt_steppe.tsv"))
    w_p, _, _ = bc.plink_fst_wc(cfg, cfg.fit9, pheno, keep_hp,
                                os.path.join(sc, "plink_rt"), per_variant=False)
    # Contextual all-pairs K=30 throughput datapoint (NOT the head-to-head number).
    w_ap = steppe_fst_allpairs(cfg, cfg.full, ordered[:30], os.path.join(sc, "ap_30.tsv"))
    tbl.add(7, "runtime_vs_reference", "steppe vs plink2 wall (same pair, same bed)",
            f"steppe={w_s:.2f}s / plink2={w_p:.2f}s (single-pair {PAIR[0]}v{PAIR[1]} 584k); "
            f"context: steppe all-pairs K=30 (435 pairs, 1.23M) = {w_ap:.2f}s",
            "-", "measured", f"{hw}; plink2 default threads (all cores)")

    # ---- Row 8: memory (K=50 all-pairs, full 1240K) -------------------------
    cmd = [cfg.steppe, "fst", "--all-pairs", "--prefix", cfg.full, "--pops",
           ",".join(ordered[:50]), "--method", "wc", "--format", "tsv",
           "--device", cfg.device, "--out", os.path.join(sc, "ap_mem.tsv")]
    w8, gpk, gdl, rss = bc.run_with_mem(cmd, device=cfg.device)
    tbl.add(8, "memory", "peak GPU MiB (delta), peak host RSS MiB",
            f"GPU peak={gpk} MiB (delta {gdl} MiB), host RSS={rss:.0f} MiB, wall={w8:.2f}s",
            "<=32607 MiB", "measured",
            "K=50 all-pairs full 1240K; nvidia-smi -lms 200 + /usr/bin/time -v")

    return tbl
