#!/usr/bin/env python3
"""bench_pca.py — emit the 8-row PCA validation+benchmark table (REAL AADR).

main(cfg) -> bench_common.Table. Oracle rows run on the fit9_ped PLINK bed with its REAL
pops (critic fix #1: the full 1240K panel is EIGENSTRAT, unreadable by plink2 --recode A,
so scikit-allel/sklearn can only be driven from the bed). The EIGENSTRAT full panel powers
the steppe-only scaling/memory rows that need no oracle.
"""
from __future__ import annotations

import os
import time

import numpy as np

import bench_common as bc

K = 10
ORACLE_M = 100_000
MISS_LEVELS = (0, 5, 10, 20)


# --- steppe pca runners ------------------------------------------------------

def steppe_pca_coords(cfg: bc.Cfg, prefix: str, pops: list[str], k: int, out: str,
                      precision: str | None = None, check: bool = True):
    cmd = [cfg.steppe, "pca", "--prefix", prefix, "-k", str(k),
           "--format", "tsv", "--device", cfg.device, "--out", out]
    if pops:
        cmd += ["--pops", ",".join(pops)]
    if precision:
        cmd += ["--precision", precision]
    wall, cp = bc.run_timed(cmd, check=check)
    coords: dict[str, np.ndarray] = {}
    n_used = None
    if os.path.exists(out):
        with open(out) as f:
            head = f.readline().rstrip("\n").split("\t")
            pc_cols = [i for i, h in enumerate(head) if h.startswith("PC")]
            s_col = head.index("sample")
            for line in f:
                t = line.rstrip("\n").split("\t")
                if len(t) <= max(pc_cols, default=0):
                    continue
                coords[t[s_col]] = np.array([float(t[i]) for i in pc_cols], dtype=np.float64)
    # n_snp_used is echoed on stderr ("... x N SNPs used ..."); see _n_snp_used().
    return wall, coords, cp


def _n_snp_used(stderr: str) -> str:
    import re
    m = re.search(r"x (\d+) SNPs used", stderr)
    return m.group(1) if m else "?"


def coords_matrix(coords: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    return np.array([coords[s] for s in order if s in coords], dtype=np.float64)


def sign_align_diff(S: np.ndarray, A: np.ndarray, normalize: bool):
    """Per-PC sign-aligned max-abs / mean-abs coord diff. normalize=True scales each PC to
    unit L2 norm first (for oracle-vs-steppe, whose coord SCALE conventions differ)."""
    kk = min(S.shape[1], A.shape[1])
    diffs = []
    for k in range(kk):
        s, a = S[:, k].copy(), A[:, k].copy()
        if normalize:
            ns, na = np.linalg.norm(s), np.linalg.norm(a)
            if ns > 0:
                s = s / ns
            if na > 0:
                a = a / na
        if np.dot(s, a) < 0:
            a = -a
        diffs.append(np.abs(s - a))
    alld = np.concatenate(diffs) if diffs else np.array([np.nan])
    return float(alld.max()), float(alld.mean())


# --- the 8 rows --------------------------------------------------------------

def main(cfg: bc.Cfg) -> bc.Table:
    tbl = bc.Table("pca")
    sc = cfg.scratch
    hw = "RTX 5090 / EPYC 9654 (nproc=384)"

    import allel
    from sklearn.decomposition import PCA as SkPCA

    # ---- Row 1: parity vs scikit-allel/sklearn (live, small N) --------------
    thin = bc.ensure_thinned_bed(cfg, ORACLE_M)          # fit9_ped, all 430 samples
    raw = bc.plink_recode_A(cfg, thin, None, os.path.join(sc, "pca_oracle"))
    iids, G = bc.load_raw(raw)
    Z, n_used = bc.patterson_matrix(G)
    # Build the allel input the same way pca_allel_gate does (mean-filled dosages,
    # variants x samples); patterson_matrix already gives us the standardized Z for eigh.
    with np.errstate(invalid="ignore"):
        p = np.nanmean(G, axis=0) / 2.0
    keep = np.isfinite(p) & (p * (1.0 - p) > 0.0)
    Gf = G[:, keep].copy()
    pk = p[keep]
    fill = np.broadcast_to(2.0 * pk, Gf.shape)
    Gf = np.where(np.isnan(Gf), fill, Gf)
    t0 = time.perf_counter()
    a_coords, a_model = allel.pca(Gf.T, n_components=K, scaler="patterson")
    allel_wall = time.perf_counter() - t0
    a_evr = np.asarray(a_model.explained_variance_ratio_)
    sk = SkPCA(n_components=K).fit_transform(Z)
    sk_r = abs(float(np.corrcoef(a_coords[:, 0], sk[:, 0])[0, 1]))
    w_s, sp, cp = steppe_pca_coords(cfg, thin, [], K, os.path.join(sc, "pca_parity.tsv"))
    common = [iid for iid in iids if iid in sp]
    S = np.array([sp[i] for i in common])
    amap = {iid: a_coords[j, :K] for j, iid in enumerate(iids)}
    A = np.array([amap[i] for i in common])
    rel = bc.subspace_gram_rel(S, A)
    rs = bc.per_pc_abs_r(S, A)
    r_lo = min(rs[:3])
    # steppe var_explained via scree
    _, scp = bc.run_timed([cfg.steppe, "pca", "--prefix", thin, "-k", str(K),
                           "--eigenvalues", "--format", "tsv", "--device", cfg.device,
                           "--out", os.path.join(sc, "pca_scree.tsv")])
    s_ve = []
    with open(os.path.join(sc, "pca_scree.tsv")) as f:
        f.readline()
        for line in f:
            t = line.rstrip("\n").split("\t")
            s_ve.append(float(t[2]))
    ve_diff = max(abs(s_ve[k] - a_evr[k]) for k in range(3))
    ok1 = rel < 5e-3 and r_lo > 0.99
    tbl.add(1, "parity_vs_reference",
            "subspace Gram rel + per-PC |r| (PC1-3) + var_expl diff",
            f"gram_rel={rel:.2e}, |r|min(PC1-3)={r_lo:.5f}, "
            f"var_expl maxdiff(PC1-3)={ve_diff:.2e}, sklearn |r|PC1={sk_r:.5f}",
            "<5e-3 / |r|~1", "PASS" if ok1 else "FAIL",
            f"scikit-allel patterson PCA LIVE ({len(common)} samples, thinned 100k)")

    # ---- Row 2: scaling with N (full 1240K subset) --------------------------
    counts = bc.ind_pop_counts(cfg.full)
    targets = [500, 1000, 2000] if cfg.quick else [500, 1000, 2000, 4000]
    ncurve = []
    for tgt in targets:
        pops, tot = bc.greedy_pops_for_n(counts, tgt)
        w, cds, cp2 = steppe_pca_coords(cfg, cfg.full, pops, K,
                                        os.path.join(sc, f"pca_n_{tgt}.tsv"))
        ncurve.append((f"N={len(cds)}(m={_n_snp_used(cp2.stderr)})", w))
    tbl.add(2, "scaling_N", "wall per N samples (-k 10)",
            bc.fmt_curve(ncurve), "-", "measured",
            "full 1240K --pops subset (greedy top pops); N=actual samples emitted")

    # ---- Row 3: scaling with SNP count (thinned fit9_ped, all pops) ---------
    Ms = [100_000, 300_000, 584_131]
    mcurve = []
    for M in Ms:
        prefix = cfg.fit9 if M == 584_131 else bc.ensure_thinned_bed(cfg, M)
        w, cds, cp3 = steppe_pca_coords(cfg, prefix, [], K,
                                        os.path.join(sc, f"pca_m_{M}.tsv"))
        label = f"{M//1000}k" if M != 584_131 else "584k"
        mcurve.append((f"{label}(m={_n_snp_used(cp3.stderr)})", w))
    tbl.add(3, "scaling_SNPs", "wall per M SNPs (-k 10, 430 samples)",
            bc.fmt_curve(mcurve), "-", "measured", "thinned fit9_ped")

    # ---- Row 4: missingness stress (inject into full fit9 bed) --------------
    cells4 = []
    m_used_seq = []
    all_finite = True
    ve1_band = True
    for p_ in MISS_LEVELS:
        bed = bc.ensure_injected_bed(cfg, p_)
        w, cds, cp4 = steppe_pca_coords(cfg, bed, [], K, os.path.join(sc, f"pca_miss_{p_}.tsv"))
        M = coords_matrix(cds, list(cds.keys()))
        finite = bool(np.isfinite(M).all()) and M.size > 0
        all_finite = all_finite and finite
        m_used = _n_snp_used(cp4.stderr)
        m_used_seq.append(int(m_used) if m_used.isdigit() else -1)
        # PC1 var from scree
        _, _ = bc.run_timed([cfg.steppe, "pca", "--prefix", bed, "-k", str(K),
                             "--eigenvalues", "--format", "tsv", "--device", cfg.device,
                             "--out", os.path.join(sc, f"pca_miss_scree_{p_}.tsv")])
        with open(os.path.join(sc, f"pca_miss_scree_{p_}.tsv")) as f:
            f.readline()
            ve1 = float(f.readline().split("\t")[2])
        if not (0.0 < ve1 < 1.0):
            ve1_band = False
        cells4.append(f"{p_}%->finite={finite},m={m_used},PC1var={ve1:.4f}")
    mono = all(m_used_seq[i] >= m_used_seq[i + 1] for i in range(len(m_used_seq) - 1)
               if m_used_seq[i] >= 0 and m_used_seq[i + 1] >= 0)
    ok4 = all_finite and ve1_band
    tbl.add(4, "missingness", "coords finite / n_snp_used trend / PC1 var band",
            "; ".join(cells4) + f"; m_nonincreasing={mono}",
            "no NaN", "PASS" if ok4 else "flag",
            "missingness injected into real fit9_ped (all 430 samples, seed 1)")

    # ---- Row 5: small-sample flag/status ------------------------------------
    cells5 = []
    ordered_pop = [p for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    for N in (1, 2, 5):
        pre = os.path.join(sc, f"pca_smalln_{N}")
        # take N samples from a single fit9 pop (Han)
        kept = bc.write_relabeled_fam(cfg, pre, {"Han": N})
        k_use = max(1, min(K, N - 1)) if N > 1 else 1
        w, cds, cp5 = steppe_pca_coords(cfg, pre, ["Han"], k_use,
                                        os.path.join(sc, f"pca_smalln_{N}.tsv"), check=False)
        status = "ok" if cp5.returncode == 0 else "rank_deficient/flag"
        if "rank_deficient" in cp5.stderr:
            status = "rank_deficient"
        crash = "CRASH" if cp5.returncode not in (0, 1, 2, 3, 4, 5) else "no-crash"
        cells5.append(f"N={N}(k={k_use},{kept}smp)->rc={cp5.returncode},{status},{crash}")
    no_crash = all("no-crash" in c for c in cells5)
    tbl.add(5, "small_sample", "status/flag at N=1/2/5 (-k<=N-1)",
            "; ".join(cells5), "no crash; rank flag", "flag" if no_crash else "FAIL",
            "single pop subsampled; -k clamped; rank flag vs graceful spectrum")

    # ---- Row 6: CPU vs GPU (precision knob + numpy eigh oracle) -------------
    # (a) fp64 vs emu40 (both GPU) on the row-1 thinned bed.
    w_fp64, sp64, _ = steppe_pca_coords(cfg, thin, [], K,
                                        os.path.join(sc, "pca_fp64.tsv"), precision="fp64")
    common64 = [i for i in iids if i in sp64 and i in sp]
    S64 = np.array([sp64[i] for i in common64])
    Sd = np.array([sp[i] for i in common64])
    a_max, a_mean = sign_align_diff(Sd, S64, normalize=False)
    # (b) numpy FP64 eigh oracle vs steppe default (unit-normalized per PC).
    o_coords, _, _ = bc.pca_oracle_eigh(Z, K)
    omap = {iids[j]: o_coords[j] for j in range(len(iids))}
    commono = [i for i in iids if i in sp]
    So = np.array([sp[i] for i in commono])
    Ao = np.array([omap[i] for i in commono])
    b_max, b_mean = sign_align_diff(So, Ao, normalize=True)
    tbl.add(6, "cpu_vs_gpu", "(a) fp64-vs-emu40 max/mean; (b) numpyFP64 eigh vs GPU max/mean",
            f"(a) emu40-vs-fp64 maxabs={a_max:.2e} meanabs={a_mean:.2e}; "
            f"(b) oracle-vs-GPU maxabs={b_max:.2e} meanabs={b_mean:.2e} (unit-norm/PC)",
            "precision-band", "PASS" if b_max < 1e-2 else "flag",
            "emulated(emu40) vs native FP64 carve-out; oracle=numpy eigh; "
            "internal long-double Jacobi gated by ctest test_pca_eig")

    # ---- Row 7: runtime vs reference ----------------------------------------
    tbl.add(7, "runtime_vs_reference", "steppe vs scikit-allel wall (same N)",
            f"steppe={w_s:.2f}s / scikit-allel={allel_wall:.2f}s "
            f"({len(common)} samples, 100k SNPs); CITED larger-N: full 23,089x1.23M "
            "steppe=6:24 @18GB; fit9 430x584k steppe=2.53s",
            "-", "measured+cited", f"{hw}; sklearn/allel single-thread; larger-N=cited gate")

    # ---- Row 8: memory (N~2000, full 1240K) ---------------------------------
    pops8, _ = bc.greedy_pops_for_n(counts, 2000)
    cmd = [cfg.steppe, "pca", "--prefix", cfg.full, "--pops", ",".join(pops8),
           "-k", str(K), "--format", "tsv", "--device", cfg.device,
           "--out", os.path.join(sc, "pca_mem.tsv")]
    w8, gpk, gdl, rss = bc.run_with_mem(cmd, device=cfg.device)
    tbl.add(8, "memory", "peak GPU MiB (delta), peak host RSS MiB",
            f"GPU peak={gpk} MiB (delta {gdl} MiB), host RSS={rss:.0f} MiB, wall={w8:.2f}s",
            "<=32607 MiB", "measured",
            "N~2000 full 1240K -k 10; nvidia-smi -lms 200 + /usr/bin/time -v")

    return tbl
