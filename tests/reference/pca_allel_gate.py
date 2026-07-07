#!/usr/bin/env python3
"""pca_allel_gate.py — freeze the scikit-allel PCA golden for `steppe pca` (REAL AADR).

The identical-data bridge (independent, honest — both sides start from the raw genotype
file; no steppe number is fed to allel):

  1. plink2 --bfile <prefix> [--keep <same samples>] --recode A  -> .raw (0/1/2/NA), the
     exact N x M dosage matrix.
  2. Per-SNP p = nanmean(G, axis=0)/2. Drop SNPs with (~isfinite(p)) | (p*(1-p) == 0) — the
     SAME deterministic filter steppe applies (all-missing AND monomorphic; critic fix #2:
     nanmean of an all-NaN column is NaN and NaN*(1-NaN)==0 is False, so all-missing must be
     caught by the isfinite guard, not the p*(1-p) guard). Mean-fill the survivors' missing
     entries with the per-SNP mean 2p (NEVER allel's to_n_alt(fill=0), which centers missing
     to -2p/std). Assert no NaN survives into allel.
  3. allel.pca(Gk_filled.T, n_components=K, scaler='patterson')  (allel wants variants x
     samples), cross-checked against sklearn PCA on the Patterson-standardized matrix.

Freezes tests/reference/goldens/pca/{allel_coords.tsv, allel_eigenvalues.tsv,
aadr_pca_meta.txt}. Optionally runs `steppe pca` and reports the rotation/sign-invariant
subspace agreement (Gram) + per-PC |Pearson r| + explained-variance ratios.

Run in a numpy<2 venv (mandatory for scikit-allel); do NOT touch the box system numpy:
  python3 -m venv /workspace/tmp/pca_oracle_venv
  /workspace/tmp/pca_oracle_venv/bin/pip install 'numpy<2' scikit-allel scikit-learn
  TMPDIR=/workspace/tmp /workspace/tmp/pca_oracle_venv/bin/python \
      tests/reference/pca_allel_gate.py --bed <prefix> --pops A,B,C --k 10 \
      --steppe <build>/steppe --out tests/reference/goldens/pca
"""
import argparse
import os
import subprocess
import sys

import numpy as np


def load_fam(prefix):
    fam = prefix + ".fam"
    fids, iids = [], []
    with open(fam) as f:
        for line in f:
            t = line.split()
            if len(t) >= 2:
                fids.append(t[0])
                iids.append(t[1])
    return fids, iids


def make_keep(prefix, pops, tmp):
    """Write a plink2 --keep FID IID list for the samples whose FID (population) is wanted."""
    fids, iids = load_fam(prefix)
    want = set(pops)
    keep = tmp + "/keep.txt"
    n = 0
    with open(keep, "w") as f:
        for fid, iid in zip(fids, iids):
            if not pops or fid in want:
                f.write(f"{fid}\t{iid}\n")
                n += 1
    return keep, n


def run_plink_recodeA(prefix, keep, tmp):
    out = tmp + "/pca_012"
    cmd = ["plink2", "--bfile", prefix, "--recode", "A", "--out", out]
    if keep:
        cmd += ["--keep", keep]
    subprocess.run(cmd, check=True)
    return out + ".raw"


def load_raw(raw_path):
    """Return (iids, G[n_samples, n_snp] float with NaN=missing)."""
    with open(raw_path) as f:
        header = f.readline().split()
    # plink2 .raw: FID IID PAT MAT SEX PHENOTYPE then one column per SNP.
    ncol = len(header)
    iid_col = header.index("IID")
    iids = []
    rows = []
    with open(raw_path) as f:
        f.readline()
        for line in f:
            t = line.split()
            if len(t) < ncol:
                continue
            iids.append(t[iid_col])
            vals = []
            for x in t[6:]:
                vals.append(np.nan if x == "NA" else float(x))
            rows.append(vals)
    return iids, np.asarray(rows, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", required=True, help="PLINK .bed prefix (also .bim/.fam)")
    ap.add_argument("--pops", default="", help="comma pops (FID) to keep; empty = ALL")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--steppe", default="", help="steppe binary to also run + compare")
    ap.add_argument("--out", required=True, help="golden output dir (tests/.../goldens/pca)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    import allel  # noqa: E402  (numpy<2 must already be importable)
    from sklearn.decomposition import PCA  # noqa: E402

    tmp = os.environ.get("TMPDIR", "/tmp")
    os.makedirs(args.out, exist_ok=True)
    pops = [p for p in args.pops.split(",") if p] if args.pops else []

    keep, n_keep = (make_keep(args.bed, pops, tmp) if pops else (None, 0))
    raw = run_plink_recodeA(args.bed, keep, tmp)
    iids, G = load_raw(raw)
    print(f"loaded {G.shape[0]} samples x {G.shape[1]} SNPs from {raw}")

    with np.errstate(invalid="ignore"):
        p = np.nanmean(G, axis=0) / 2.0
    keep_snp = np.isfinite(p) & (p * (1.0 - p) > 0.0)   # critic fix #2
    Gk = G[:, keep_snp]
    pk = p[keep_snp]
    n_used = int(keep_snp.sum())
    print(f"kept {n_used} polymorphic non-empty SNPs (dropped {G.shape[1] - n_used})")

    # Mean-fill missing to 2p (centers to 0 after Patterson centering); assert no NaN.
    fill = np.broadcast_to(2.0 * pk, Gk.shape)
    Gk = np.where(np.isnan(Gk), fill, Gk)
    assert np.isfinite(Gk).all(), "NaN survived into the allel PCA input"

    K = args.k
    coords, model = allel.pca(Gk.T, n_components=K, scaler="patterson")
    ev = np.asarray(model.explained_variance_)
    evr = np.asarray(model.explained_variance_ratio_)

    # sklearn cross-check on the Patterson-standardized matrix (report |r| PC1).
    Z = (Gk - 2.0 * pk) / np.sqrt(pk * (1.0 - pk))
    sk = PCA(n_components=K).fit_transform(Z)
    r_pc1 = np.corrcoef(coords[:, 0], sk[:, 0])[0, 1]
    print(f"sklearn cross-check PC1 |r| vs allel = {abs(r_pc1):.6f}")

    # Freeze the goldens.
    with open(args.out + "/allel_coords.tsv", "w") as f:
        f.write("sample\t" + "\t".join(f"PC{k+1}" for k in range(K)) + "\n")
        for i, iid in enumerate(iids):
            f.write(iid + "\t" + "\t".join(f"{coords[i, k]:.10g}" for k in range(K)) + "\n")
    with open(args.out + "/allel_eigenvalues.tsv", "w") as f:
        f.write("pc_index\teigenvalue\tvar_explained\n")
        for k in range(K):
            f.write(f"{k+1}\t{ev[k]:.10g}\t{evr[k]:.10g}\n")
    with open(args.out + "/aadr_pca_meta.txt", "w") as f:
        f.write(args.bed + "\n")
        f.write((",".join(pops) if pops else "ALL") + "\n")
        f.write(str(K) + "\n")
    print(f"froze goldens under {args.out}")

    if not args.steppe:
        print("RESULT: goldens frozen (no --steppe compare requested)")
        return 0

    # Optional: run steppe and report the subspace agreement.
    sp_tsv = tmp + "/steppe_pca_gate.tsv"
    cmd = [args.steppe, "pca", "--prefix", args.bed, "-k", str(K), "--format", "tsv",
           "--device", args.device, "--out", sp_tsv]
    if pops:
        cmd += ["--pops", ",".join(pops)]
    subprocess.run(cmd, check=True)

    sp = {}
    with open(sp_tsv) as f:
        head = f.readline().rstrip("\n").split("\t")
        pc_cols = [i for i, h in enumerate(head) if h.startswith("PC")]
        s_col = head.index("sample")
        for line in f:
            t = line.rstrip("\n").split("\t")
            sp[t[s_col]] = np.asarray([float(t[i]) for i in pc_cols[:K]])
    allel_map = {iid: coords[i, :K] for i, iid in enumerate(iids)}
    common = [s for s in sp if s in allel_map]
    S = np.asarray([sp[s] for s in common])
    A = np.asarray([allel_map[s] for s in common])
    print(f"joined {len(common)} samples")

    Gs, Ga = S @ S.T, A @ A.T
    rel = np.abs(Gs - Ga).max() / (np.sqrt((Ga * Ga).mean()) + 1e-30)
    print(f"subspace Gram rel = {rel:.3e}")
    for k in range(K):
        r = abs(np.corrcoef(S[:, k], A[:, k])[0, 1])
        print(f"  PC{k+1:2d} |r|={r:.6f}  var_explained={evr[k]:.5f}")
    ok = rel < 5e-3
    print("RESULT:", "PASS" if ok else "FAIL", f"(subspace rel={rel:.3e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
