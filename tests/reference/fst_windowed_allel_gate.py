#!/usr/bin/env python3
"""fst_windowed_allel_gate.py — gate `steppe fst --windowed` / `--pbs` vs scikit-allel (REAL AADR).

Windowed WC Fst is gated against allel.windowed_weir_cockerham_fst; PBS is gated against a HAND
PBS built from allel's three pairwise bp-window Fst arrays (allel.pbs itself uses Hudson's Fst +
variant-count windows + normed PBSn1 — three deviations from the task's WC + bp-window + raw
Yi-2010 spec, so it must NOT gate steppe directly; the task explicitly permits the hand oracle).

Identical-data bridge (honest — both sides start from the same PLINK bed and the SAME SNP set;
no steppe Fst value is fed to allel):

  1. steppe fst --pbs A,B,C --windowed SIZE:STEP --keep-snps <chrom ids> --emit-kept-snps S.txt
     -> steppe per-window {Fst_AB,Fst_AC,Fst_BC,PBS_A,PBS_B,PBS_C} + the exact SNP set S steppe
     used (its QC survivors on the requested chromosome).
  2. steppe fst --pops A,B --windowed SIZE:STEP --keep-snps S.txt -> the 2-pop --windowed command
     path's per-window Fst (validates that command surface directly).
  3. plink2 --bfile PREFIX --chr C --extract S.txt --keep <A,B,C samples> --recode A -> the exact
     N x |S| 0/1/2/NA dosage matrix on S; reconstruct an allel GenotypeArray (0->(0,0), 1->(0,1),
     2->(1,1), NA->(-1,-1)); pos = the .bim bp of S (position-sorted, == steppe's window axis).
  4. allel.windowed_weir_cockerham_fst(pos, g, [A,B]/[A,C]/[B,C], size, step) -> the three pairwise
     window Fst on the SAME windows (start=pos[0], stop=pos[-1] == steppe's chrom first/last SNP).
  5. hand PBS: T = -log(1 - clip(fst, 0, 1-1e-15)); PBS_A = (T_AB+T_AC-T_BC)/2 (raw Yi 2010).

Compares windows aligned by (start,end); reports the ACTUAL max abs diff for windowed Fst and for
PBS over windows finite in BOTH sides. Gate: < 1e-5.

Usage (box5090, system python has numpy/allel/pandas):
  python3 tests/reference/fst_windowed_allel_gate.py \
      --prefix /workspace/data/aadr/fixtures_eig/v66_fit9_ped --chrom 2 \
      --pops Han,Papuan,Mbuti --size 1000000 --step 1000000 \
      --steppe /workspace/steppe/build-admix-accel/bin/steppe --tmp /workspace/tmp
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import allel


def sh(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_bim_chrom(bim, chrom):
    """id -> bp for one chromosome, plus the position-sorted id list."""
    id2pos = {}
    ids = []
    with open(bim) as f:
        for line in f:
            t = line.split()
            if t[0] == str(chrom):
                id2pos[t[1]] = int(t[3])
                ids.append(t[1])
    return id2pos, ids


def read_fam_pop(fam):
    """(FID, IID) rows + IID -> pop (the .fam phenotype col 6 for these AADR fixtures)."""
    rows = []
    iid2pop = {}
    with open(fam) as f:
        for line in f:
            t = line.split()
            rows.append((t[0], t[1]))
            iid2pop[t[1]] = t[5] if len(t) >= 6 else "NA"
    return rows, iid2pop


def run_steppe_table(steppe, prefix, args, out_tsv):
    cmd = [steppe, "fst", "--prefix", prefix, "--format", "tsv", "--device", "0",
           "--out", out_tsv] + args
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import pandas as pd
    return pd.read_csv(out_tsv, sep="\t")


def build_allel_g(prefix, chrom, keep_ids_file, keep_samples_file, id2pos, tmp, tag):
    """plink2 --recode A on EXACTLY the SNP set steppe kept -> (g, pos, iids, header_ids).

    The SNP set is whatever steppe emitted (--emit-kept-snps), so allel and steppe compute on
    IDENTICAL sites/windows (steppe's default QC keep-set is pop-dependent, so each run's oracle
    must use that run's own survivors)."""
    import pandas as pd
    with open(keep_ids_file) as f:
        S = [x.strip() for x in f if x.strip()]
    raw_prefix = f"{tmp}/gate_{tag}"
    sh(["plink2", "--bfile", prefix, "--chr", str(chrom), "--extract", keep_ids_file,
        "--keep", keep_samples_file, "--recode", "A", "--out", raw_prefix])
    df = pd.read_csv(raw_prefix + ".raw", sep=r"\s+")
    iids = df["IID"].astype(str).tolist()
    snp_cols = list(df.columns[6:])  # FID IID PAT MAT SEX PHENOTYPE then SNPs
    header_ids = [c.rsplit("_", 1)[0] for c in snp_cols]
    if header_ids != S:
        raise SystemExit(f"[{tag}] RAW/S id order mismatch: {header_ids[:3]}... vs {S[:3]}...")
    D = df[snp_cols].to_numpy(dtype=float)  # (n_samp, n_var) dosage, NaN = missing
    Dt = D.T  # (n_var, n_samp)
    mask = np.isnan(Dt)
    a0 = (Dt >= 2).astype("i1")   # dosage 2 -> hom ALT
    a1 = (Dt >= 1).astype("i1")   # dosage >=1 -> at least one ALT
    gt = np.stack([a0, a1], axis=-1)
    gt[mask] = -1
    g = allel.GenotypeArray(gt)
    pos = np.array([id2pos[s] for s in S], dtype=np.int64)
    if not np.all(np.diff(pos) >= 0):
        raise SystemExit(f"[{tag}] positions not sorted ascending")
    return g, pos, iids, header_ids


def pbs_hand(fab, fac, fbc):
    def clamp(f):
        return np.where(np.isnan(f), np.nan, np.clip(f, 0.0, 1.0 - 1e-15))
    t_ab = -np.log(1.0 - clamp(fab))
    t_ac = -np.log(1.0 - clamp(fac))
    t_bc = -np.log(1.0 - clamp(fbc))
    return (0.5 * (t_ab + t_ac - t_bc),
            0.5 * (t_ab + t_bc - t_ac),
            0.5 * (t_ac + t_bc - t_ab))


def max_abs_finite(a, b):
    """max |a-b| over entries finite in BOTH; count finite-in-both; count NaN-semantics
    disagreements (finite on one side, NaN on the other — a silent aggregation/edge mismatch)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    m = fa & fb
    nan_mismatch = int(np.sum(fa != fb))
    if not np.any(m):
        return 0.0, 0, nan_mismatch
    return float(np.max(np.abs(a[m] - b[m]))), int(np.sum(m)), nan_mismatch


def align(steppe_df, allel_windows):
    """Index arrays aligning steppe rows to allel windows by (start, end); asserts identical."""
    s_key = list(zip(steppe_df["start"].astype(int), steppe_df["end"].astype(int)))
    a_key = [(int(w[0]), int(w[1])) for w in allel_windows]
    if s_key != a_key:
        # Report the first divergence for diagnosis.
        n = min(len(s_key), len(a_key))
        first = next((i for i in range(n) if s_key[i] != a_key[i]), n)
        raise SystemExit(
            f"WINDOW MISALIGNMENT: steppe has {len(s_key)} windows, allel {len(a_key)}; "
            f"first diff at index {first}: steppe={s_key[first] if first < len(s_key) else None} "
            f"allel={a_key[first] if first < len(a_key) else None}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--chrom", type=int, required=True)
    ap.add_argument("--pops", required=True, help="A,B,C")
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--step", type=int, default=0, help="defaults to size")
    ap.add_argument("--steppe", required=True)
    ap.add_argument("--tmp", default="/workspace/tmp")
    a = ap.parse_args()
    step = a.step if a.step > 0 else a.size
    pops = a.pops.split(",")
    assert len(pops) == 3, "need three pops A,B,C"
    A, B, C = pops
    os.makedirs(a.tmp, exist_ok=True)

    bim, fam = a.prefix + ".bim", a.prefix + ".fam"
    id2pos, chrom_ids = read_bim_chrom(bim, a.chrom)
    fam_rows, iid2pop = read_fam_pop(fam)
    print(f"[data] chrom {a.chrom}: {len(chrom_ids)} SNPs in .bim; window {a.size}:{step} bp")

    # chrom SNP id list for --keep-snps (restrict steppe to this chromosome).
    chrom_ids_file = f"{a.tmp}/gate_chrom_ids.txt"
    with open(chrom_ids_file, "w") as f:
        f.write("\n".join(chrom_ids) + "\n")

    # plink2 --keep sample lists per pop set.
    def keep_samples(pop_set, path):
        with open(path, "w") as f:
            for fid, iid in fam_rows:
                if iid2pop.get(iid) in pop_set:
                    f.write(f"{fid}\t{iid}\n")
        return path

    keep_ab = keep_samples({A, B}, f"{a.tmp}/gate_keep_ab.txt")
    keep_abc = keep_samples({A, B, C}, f"{a.tmp}/gate_keep_abc.txt")
    winspec = f"{a.size}:{step}"

    def pop_indices(iids, pops):
        return {p: [i for i, iid in enumerate(iids) if iid2pop.get(iid) == p] for p in pops}

    # ===================== WINDOWED-Fst gate (2-pop `--windowed A,B`) =====================
    s_win = f"{a.tmp}/gate_S_win.txt"
    win_df = run_steppe_table(
        a.steppe, a.prefix,
        ["--pops", f"{A},{B}", "--windowed", winspec, "--keep-snps", chrom_ids_file,
         "--emit-kept-snps", s_win],
        f"{a.tmp}/gate_steppe_win.tsv")
    g_ab, pos_ab, iids_ab, _ = build_allel_g(a.prefix, a.chrom, s_win, keep_ab, id2pos, a.tmp,
                                             "win")
    idx_ab = pop_indices(iids_ab, (A, B))
    print(f"[windowed] steppe {A},{B}: {len(win_df)} windows on {g_ab.shape[0]} SNPs; "
          f"allel g={g_ab.shape}")
    fab_w, win_w, cnt_w = allel.windowed_weir_cockerham_fst(
        pos_ab, g_ab, [idx_ab[A], idx_ab[B]], size=a.size, step=step)
    fab_w = np.asarray(fab_w, float)
    align(win_df, win_w)
    d_win, n_win_cmp, nanmm_win = max_abs_finite(win_df["Fst"].to_numpy(float), fab_w)
    d_cnt_win = int(np.max(np.abs(win_df["n_snp"].to_numpy(int) - np.asarray(cnt_w).astype(int))))

    # ===================== PBS gate (3-pop `--pbs A,B,C`, hand WC oracle) =================
    s_pbs = f"{a.tmp}/gate_S_pbs.txt"
    pbs_df = run_steppe_table(
        a.steppe, a.prefix,
        ["--pbs", a.pops, "--windowed", winspec, "--keep-snps", chrom_ids_file,
         "--emit-kept-snps", s_pbs],
        f"{a.tmp}/gate_steppe_pbs.tsv")
    g_abc, pos_abc, iids_abc, _ = build_allel_g(a.prefix, a.chrom, s_pbs, keep_abc, id2pos, a.tmp,
                                                "pbs")
    idx_abc = pop_indices(iids_abc, (A, B, C))
    print(f"[pbs] steppe {A},{B},{C}: {len(pbs_df)} windows on {g_abc.shape[0]} SNPs; "
          f"allel g={g_abc.shape}")

    def wfst(P, Q):
        fst, windows, counts = allel.windowed_weir_cockerham_fst(
            pos_abc, g_abc, [idx_abc[P], idx_abc[Q]], size=a.size, step=step)
        return np.asarray(fst, float), windows, np.asarray(counts)

    fab, win_abc, cnt_abc = wfst(A, B)
    fac, _, _ = wfst(A, C)
    fbc, _, _ = wfst(B, C)
    align(pbs_df, win_abc)

    # steppe's 3 pairwise window Fst (from the PBS table) vs allel on the SAME 3-pop set.
    d_ab, _, mm_ab = max_abs_finite(pbs_df["Fst_AB"].to_numpy(float), fab)
    d_ac, _, mm_ac = max_abs_finite(pbs_df["Fst_AC"].to_numpy(float), fac)
    d_bc, _, mm_bc = max_abs_finite(pbs_df["Fst_BC"].to_numpy(float), fbc)

    # hand PBS from allel's 3 window Fst (raw Yi 2010) vs steppe PBS.
    pa, pb, pc = pbs_hand(fab, fac, fbc)
    d_pa, n_pbs_cmp, mm_pa = max_abs_finite(pbs_df["PBS_A"].to_numpy(float), pa)
    d_pb, _, mm_pb = max_abs_finite(pbs_df["PBS_B"].to_numpy(float), pb)
    d_pc, _, mm_pc = max_abs_finite(pbs_df["PBS_C"].to_numpy(float), pc)
    d_cnt_pbs = int(np.max(np.abs(pbs_df["n_snp"].to_numpy(int) - np.asarray(cnt_abc).astype(int))))

    max_fst = max(d_win, d_ab, d_ac, d_bc)
    max_pbs = max(d_pa, d_pb, d_pc)
    d_cnt = max(d_cnt_win, d_cnt_pbs)
    nan_mismatch = max(nanmm_win, mm_ab, mm_ac, mm_bc, mm_pa, mm_pb, mm_pc)

    print("\n===== RESULTS =====")
    print(f"n_snp match steppe-vs-allel counts: windowed max|diff|={d_cnt_win}  "
          f"pbs max|diff|={d_cnt_pbs}")
    print(f"NaN-semantics disagreements (finite one side / NaN other): {nan_mismatch}")
    print(f"windowed Fst max|diff|:  --windowed cmd={d_win:.3e} (n={n_win_cmp})  "
          f"| PBS-table AB={d_ab:.3e} AC={d_ac:.3e} BC={d_bc:.3e}  => MAX {max_fst:.6e}")
    print(f"PBS max|diff|:           A={d_pa:.3e} B={d_pb:.3e} C={d_pc:.3e} (n={n_pbs_cmp})  "
          f"=> MAX {max_pbs:.6e}")
    tol = 1e-5
    ok = (max_fst < tol) and (max_pbs < tol) and (d_cnt == 0) and (nan_mismatch == 0)
    print(f"\nGATE (<{tol:g}): {'PASS' if ok else 'FAIL'}  "
          f"[max_fst={max_fst:.3e}, max_pbs={max_pbs:.3e}, n_snp_diff={d_cnt}]")
    print(f"RESULT max_abs_fst_diff={max_fst:.6e} max_abs_pbs_diff={max_pbs:.6e} "
          f"n_snp_diff={d_cnt} pass={int(ok)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
