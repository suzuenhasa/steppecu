#!/usr/bin/env /venv/main/bin/python
"""
aadr/02_build_matrix.py  — build the per-population reference-allele frequency
matrices (Q, V, N) for the f2 emulated-FP64 accuracy test, as a SELF-CONTAINED
pure-NumPy reader.  The box has NO conda / eigensoft / plink — only python+numpy
at /venv/main — so this script parses AADR PACKEDANCESTRYMAP directly.

Run under /venv/main:
    /venv/main/bin/python 02_build_matrix.py [opts]

INPUTS  (AADR PACKEDANCESTRYMAP triple <prefix>.{geno,snp,ind})
  --prefix   : path prefix to the AADR triple.  If omitted, the script searches
               /workspace/data/aadr/raw for a usable <prefix>.{geno,snp,ind}
               triple and uses the first it finds.
  --raw      : directory to search for a triple when --prefix is not given
               (default /workspace/data/aadr/raw).

POPULATION SELECTION
  --pops a,b,c  : comma-separated list of EXACT .ind population labels.
  --list-pops   : print every population with its individual count, then exit
                  (so the user can pick exact labels).
  --auto-top K  : pick the K populations with the most individuals.
  (default)     : a curated set of closely-related clusters (multi-European,
                  multi-East-Asian, some African) to maximize f2 (p_i - p_j)^2
                  catastrophic-cancellation stress.  AADR labels vary (suffixes
                  like .DG/.SG/.HO); if fewer than MIN_CURATED_MATCH of the
                  curated names match the real .ind labels, FALL BACK to
                  --auto-top 40 with a loud warning.

SNP FILTERING
  --snp-cap K   : take the first K retained SNPs (in file order).  Default: none.
  --maf-min m   : drop SNPs whose pooled mean ref-freq across selected pops is
                  < m or > 1-m.  Default 0.0 (no filter).
  SNPs with no data in ANY selected pop are always dropped.

OUTPUT (SHARED BINARY FORMAT — producer here and the CUDA spike loader agree)
  <outdir>/Q.f64, V.f64, N.f64  — raw little-endian float64, .tofile() of a
                                  C-contiguous numpy array of shape (M, P).
  <outdir>/shape.txt            — single ASCII line "P M\n".
  <outdir>/meta.json            — {P, M, pops, n_indiv_per_pop, source, ...}.

  LAYOUT:  P = number of POPULATIONS, M = number of SNPs.  We write a
  C-contiguous numpy array A of shape (M, P) = (n_snp, n_pop) with A[s, i] =
  value for (pop i, snp s).  Then A.tofile() places element (pop i, snp s) at
  flat index  P*s + i = i + P*s — i.e. read back as a cuBLAS COLUMN-MAJOR
  [P x M] matrix with leading dimension ld = P, element (row = pop i, col =
  snp s) is at i + P*s.  Identical convention on both sides.

  SEMANTICS:
    Q[s, i] = frequency in [0,1] of the FIXED .snp col-5 REFERENCE allele at SNP
              s in pop i (the .geno value counts copies of that reference allele,
              so it is automatically consistent across pops — we never re-polarize
              to the minor allele).
    N[s, i] = non-missing HAPLOID count = 2 * (non-missing diploid individuals).
    V[s, i] = 1.0 if N[s,i] > 0 else 0.0.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Curated closely-related-cluster set: many low-FST pairs for maximal f2
# cancellation.  Labels follow AADR/Human-Origins conventions (col 3 of .ind);
# exact spellings/suffixes vary by release, hence the fuzzy-match fallback.
CURATED_POPS = [
    # ---- Europe (dense, low-FST) ----
    "French", "Sardinian", "Tuscan", "Spanish", "Basque", "Orcadian",
    "Italian_North", "Norwegian", "Greek", "Bulgarian", "Czech", "Russian",
    # ---- East Asia (dense, low-FST) ----
    "Han", "Japanese", "Korean", "Dai", "She", "Tujia", "Miao", "Lahu",
    # ---- Africa (mix of related & divergent) ----
    "Yoruba", "Mandenka", "Mbuti", "BantuKenya", "Esan", "Luhya",
]
MIN_CURATED_MATCH = 5     # below this many matched curated pops -> auto-top fallback
AUTOTOP_FALLBACK = 40     # K used when the curated set fails to match


# ---------------------------------------------------------------------------
# .ind / .snp readers
def read_ind(path):
    if not os.path.exists(path):
        print(f"ERROR: .ind not found: {path}", file=sys.stderr)
        sys.exit(1)
    ids, sexes, pops = [], [], []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            ids.append(t[0])
            sexes.append(t[1] if len(t) > 1 else "U")
            pops.append(t[2] if len(t) > 2 else "Unknown")
    return ids, sexes, pops


def count_snp_lines(path):
    """Count SNP records in the .snp file (one SNP per non-blank line)."""
    if not os.path.exists(path):
        print(f"ERROR: .snp not found: {path}", file=sys.stderr)
        sys.exit(1)
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.split():
                n += 1
    return n


# ---------------------------------------------------------------------------
# PACKEDANCESTRYMAP packed .geno header
def read_geno_header(path):
    """
    Parse the EIGENSOFT packedancestrymap .geno ASCII header (first record).

    The header begins with the magic 'GENO' followed by ASCII fields; the two
    decimal integers after the magic are nIndiv and nSNP.  Returns
    (n_indiv_hdr, n_snp_hdr, rlen) where rlen = max(48, ceil(n_indiv/4)).
    """
    if not os.path.exists(path):
        print(f"ERROR: .geno not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "rb") as fh:
        head = fh.read(48)
    if len(head) < 4 or head[:4] != b"GENO":
        raise ValueError(
            f"not a packedancestrymap .geno file (bad magic; expected 'GENO', "
            f"got {head[:4]!r}) in {path}")
    # Header text after 'GENO': nIndiv nSNP <hash> <hash> ...  Parse the first
    # two whitespace-separated decimal integers.
    txt = head[4:].decode("latin-1", errors="ignore")
    ints = []
    for tok in txt.replace("\x00", " ").split():
        if tok.isdigit():
            ints.append(int(tok))
        if len(ints) >= 2:
            break
    if len(ints) < 2:
        raise ValueError(f"could not parse nIndiv/nSNP from .geno header: {txt!r}")
    n_indiv_hdr, n_snp_hdr = ints[0], ints[1]
    rlen = max(48, (n_indiv_hdr + 3) // 4)
    return n_indiv_hdr, n_snp_hdr, rlen


# ---------------------------------------------------------------------------
# Pure-numpy packed .geno reader, CHUNKED over SNP records to bound peak RAM.
def iter_geno_chunks(path, n_indiv, n_snp, rlen, chunk_snps=20000):
    """
    Yield (s0, G_chunk) where G_chunk is an int8 array of shape
    (n_rows, n_indiv): 0/1/2 = copies of the REFERENCE (.snp col5) allele,
    -1 = missing, for SNP rows [s0, s0 + n_rows).

    EIGENSOFT packedancestrymap layout:
      * first record (rlen bytes) = ASCII header, skipped.
      * one record per SNP, rlen bytes each.
      * genotypes packed 4-per-byte, 2 bits each, MSB-first (sample 0 in bits
        7-6, sample 1 in bits 5-4, ...).
      * raw 2-bit value 3 == missing (-> -1); 0/1/2 = ref-allele copies.
    """
    shifts = np.array([6, 4, 2, 0], dtype=np.uint8)        # MSB-first bit order
    itemsize = rlen
    with open(path, "rb") as fh:
        for s0 in range(0, n_snp, chunk_snps):
            n_rows = min(chunk_snps, n_snp - s0)
            # +1 record for the header, which precedes record s0.
            fh.seek((s0 + 1) * itemsize, os.SEEK_SET)
            raw = np.fromfile(fh, dtype=np.uint8, count=n_rows * itemsize)
            if raw.size != n_rows * itemsize:
                raise ValueError(
                    f"geno short read at SNP {s0}: got {raw.size} bytes, "
                    f"expected {n_rows * itemsize} (rlen={rlen}, n_rows={n_rows})")
            raw = raw.reshape(n_rows, rlen)
            vals = (raw[:, :, None] >> shifts) & 3          # (n_rows, rlen, 4)
            vals = vals.reshape(n_rows, rlen * 4)[:, :n_indiv]
            G = vals.astype(np.int8)
            G[G == 3] = -1
            yield s0, G


# ---------------------------------------------------------------------------
def discover_prefix(raw_dir):
    """Find a usable <prefix>.{geno,snp,ind} triple under raw_dir."""
    if not os.path.isdir(raw_dir):
        print(f"ERROR: raw dir not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)
    genos = sorted(glob.glob(os.path.join(raw_dir, "*.geno")))
    for g in genos:
        pref = g[: -len(".geno")]
        if os.path.exists(pref + ".snp") and os.path.exists(pref + ".ind"):
            return pref
    # also allow .geno symlink with sibling .snp/.ind of a different stem name
    print(f"ERROR: no complete <prefix>.{{geno,snp,ind}} triple found in {raw_dir}",
          file=sys.stderr)
    if genos:
        print(f"       (.geno files present: {[os.path.basename(g) for g in genos]}, "
              f"but missing matching .snp/.ind)", file=sys.stderr)
    sys.exit(1)


def build_numpy(prefix, use_pops, pops_arr, n_indiv, n_snp, rlen, chunk_snps):
    """
    Accumulate per-(snp,pop) ref-allele count AC and haploid count AN by
    streaming the packed .geno in SNP chunks (bounds peak RAM).

    Returns (Q, N, V, n_indiv_per_pop) where Q/N/V are (n_snp, n_pop) snp-major.
    Missing genotypes are excluded from BOTH numerator (AC) and denominator (AN).
    """
    K = len(use_pops)
    # Boolean selection masks per pop, over the FULL individual axis.
    sel_masks = [(pops_arr == p) for p in use_pops]
    n_indiv_per_pop = {p: int(m.sum()) for p, m in zip(use_pops, sel_masks)}

    AC = np.zeros((n_snp, K), dtype=np.int64)   # ref-allele count (numerator)
    AN = np.zeros((n_snp, K), dtype=np.int64)   # haploid count = 2*non-missing diploids

    for s0, G in iter_geno_chunks(prefix + ".geno", n_indiv, n_snp, rlen, chunk_snps):
        n_rows = G.shape[0]
        valid = (G >= 0)                                   # (n_rows, n_indiv)
        ref_count = np.where(valid, G, 0).astype(np.int64) # missing -> 0 in AC
        for k, m in enumerate(sel_masks):
            if not m.any():
                continue
            AC[s0:s0 + n_rows, k] = ref_count[:, m].sum(axis=1)
            AN[s0:s0 + n_rows, k] = 2 * valid[:, m].sum(axis=1)
        if (s0 // chunk_snps) % 10 == 0:
            print(f"==> [02][numpy] processed SNPs {s0 + n_rows}/{n_snp}", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        Q = np.where(AN > 0, AC / AN, 0.0)
    V = (AN > 0).astype(np.float64)
    N = AN.astype(np.float64)
    return Q, N, V, n_indiv_per_pop


# ---------------------------------------------------------------------------
def drop_empty_and_cap(Q, N, V, snp_cap, maf_min):
    """
    Q,N,V are (n_snp, n_pop).  Drop pops with no data, drop SNPs with no data in
    ANY selected pop, apply pooled MAF floor, then optional snp cap (first kept,
    in file order).  Returns (Q, N, V, pop_has_data_mask, snp_idx).
    """
    pop_has_data = V.sum(axis=0) > 0
    Q = Q[:, pop_has_data]; N = N[:, pop_has_data]; V = V[:, pop_has_data]

    # SNP must have data in >=1 kept pop.
    snp_keep = V.sum(axis=1) > 0

    # pooled ref-allele freq across kept pops with data (for MAF filter / stats)
    AC = (Q * N).sum(axis=1)
    AN = N.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.where(AN > 0, AC / AN, 0.0)

    if maf_min and maf_min > 0.0:
        maf = np.minimum(pooled, 1.0 - pooled)
        snp_keep = snp_keep & (maf >= maf_min)

    snp_idx = np.nonzero(snp_keep)[0]
    if snp_cap is not None and snp_cap >= 0:
        snp_idx = snp_idx[:snp_cap]

    Q = Q[snp_idx, :]; N = N[snp_idx, :]; V = V[snp_idx, :]
    return Q, N, V, pop_has_data, snp_idx


# ---------------------------------------------------------------------------
def resolve_pops(args, pops):
    """
    Decide the population label list to keep.  Returns (use_pops, mode_desc).
    `pops` is the parallel per-individual population list from the .ind.
    """
    # counts per label, descending by count
    labels, counts = np.unique(np.array(pops), return_counts=True)
    order = np.argsort(-counts)
    labels = labels[order]; counts = counts[order]
    count_of = dict(zip(labels.tolist(), counts.tolist()))

    if args.list_pops:
        print(f"==> [02] {len(labels)} populations in {args.prefix}.ind "
              f"(label  n_individuals):", flush=True)
        for lab, c in zip(labels.tolist(), counts.tolist()):
            print(f"  {lab}\t{c}")
        sys.exit(0)

    present = set(labels.tolist())

    if args.pops:
        requested = [p.strip() for p in args.pops.split(",") if p.strip()]
        use = [p for p in requested if p in present]
        missing = [p for p in requested if p not in present]
        if missing:
            print(f"==> [02] WARNING: {len(missing)} requested pops absent: "
                  f"{', '.join(missing)}", flush=True)
        if not use:
            print("ERROR: none of the requested --pops are present in this dataset",
                  file=sys.stderr)
            sys.exit(1)
        return use, f"--pops ({len(use)} matched)"

    if args.auto_top is not None:
        K = min(args.auto_top, len(labels))
        use = labels[:K].tolist()
        return use, f"--auto-top {args.auto_top} (kept {len(use)})"

    # default: curated set with fuzzy-suffix matching, else auto-top fallback.
    def match_curated(name):
        # exact, or AADR-suffixed variant (e.g. French.DG / French.SG / French.HO)
        cands = [p for p in present
                 if p == name or p.split(".")[0] == name or p.startswith(name + ".")]
        if cands:
            # prefer the variant with the most individuals
            return max(cands, key=lambda p: count_of.get(p, 0))
        return None

    matched, seen = [], set()
    for name in CURATED_POPS:
        m = match_curated(name)
        if m is not None and m not in seen:
            matched.append(m); seen.add(m)

    if len(matched) >= MIN_CURATED_MATCH:
        print(f"==> [02] curated set matched {len(matched)} pops "
              f"(of {len(CURATED_POPS)} curated names)", flush=True)
        return matched, f"curated ({len(matched)} matched)"

    print(f"==> [02] WARNING: only {len(matched)} of {len(CURATED_POPS)} curated "
          f"pop names matched the real .ind labels (AADR labels vary, e.g. .DG/"
          f".SG/.HO suffixes) — FALLING BACK to --auto-top {AUTOTOP_FALLBACK}.",
          file=sys.stderr)
    K = min(AUTOTOP_FALLBACK, len(labels))
    return labels[:K].tolist(), f"auto-top-{AUTOTOP_FALLBACK} fallback (kept {K})"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None,
                    help="path prefix to AADR <prefix>.{geno,snp,ind}; "
                         "if omitted, auto-discovered under --raw")
    ap.add_argument("--raw", default="/workspace/data/aadr/raw",
                    help="dir to search for a triple when --prefix is omitted")
    ap.add_argument("--pops", default=None,
                    help="comma-separated EXACT .ind population labels")
    ap.add_argument("--list-pops", action="store_true",
                    help="print every population with its individual count and exit")
    ap.add_argument("--auto-top", type=int, default=None,
                    help="pick the K populations with the most individuals")
    ap.add_argument("--snp-cap", type=int, default=None,
                    help="take only the first K retained SNPs (file order)")
    ap.add_argument("--maf-min", type=float, default=0.0,
                    help="drop SNPs whose pooled ref-freq is < maf or > 1-maf")
    ap.add_argument("--chunk-snps", type=int, default=20000,
                    help="number of SNP records processed per chunk (RAM control)")
    ap.add_argument("--outdir", default="/workspace/data/aadr/derived")
    args = ap.parse_args()

    if args.maf_min < 0.0 or args.maf_min > 0.5:
        print(f"ERROR: --maf-min must be in [0, 0.5], got {args.maf_min}", file=sys.stderr)
        sys.exit(1)
    if args.snp_cap is not None and args.snp_cap < 0:
        print(f"ERROR: --snp-cap must be >= 0, got {args.snp_cap}", file=sys.stderr)
        sys.exit(1)
    if args.auto_top is not None and args.auto_top <= 0:
        print(f"ERROR: --auto-top must be > 0, got {args.auto_top}", file=sys.stderr)
        sys.exit(1)
    if args.chunk_snps <= 0:
        print(f"ERROR: --chunk-snps must be > 0, got {args.chunk_snps}", file=sys.stderr)
        sys.exit(1)

    # resolve prefix (fail loud if raw files missing)
    if args.prefix is None:
        args.prefix = discover_prefix(args.raw)
        print(f"==> [02] auto-discovered prefix: {args.prefix}", flush=True)
    for ext in (".geno", ".snp", ".ind"):
        if not os.path.exists(args.prefix + ext):
            print(f"ERROR: missing AADR input {args.prefix + ext}", file=sys.stderr)
            sys.exit(1)

    # read .ind / .snp metadata and the .geno header
    ids, sexes, pops = read_ind(args.prefix + ".ind")
    n_indiv = len(ids)
    n_snp_file = count_snp_lines(args.prefix + ".snp")
    n_indiv_hdr, n_snp_hdr, rlen = read_geno_header(args.prefix + ".geno")
    print(f"==> [02] .ind individuals={n_indiv}  .snp SNPs={n_snp_file}  "
          f".geno header: nIndiv={n_indiv_hdr} nSNP={n_snp_hdr} rlen={rlen}",
          flush=True)
    if n_indiv_hdr != n_indiv:
        print(f"==> [02] WARNING: .geno header nIndiv={n_indiv_hdr} != .ind count "
              f"{n_indiv}; trusting the .ind count.", file=sys.stderr)
    if n_snp_hdr != n_snp_file:
        print(f"==> [02] WARNING: .geno header nSNP={n_snp_hdr} != .snp count "
              f"{n_snp_file}; trusting the .geno header.", file=sys.stderr)
    n_snp = n_snp_hdr     # the .geno record count is authoritative for the packing

    # validate the .geno data length against rlen*n_snp before streaming
    data_bytes = os.path.getsize(args.prefix + ".geno") - rlen   # minus header rec
    if data_bytes != rlen * n_snp:
        print(f"==> [02] WARNING: .geno data bytes={data_bytes} != rlen*nSNP="
              f"{rlen * n_snp}; the file may be truncated or rlen mis-derived.",
              file=sys.stderr)

    pops_arr = np.array(pops)
    use_pops, mode_desc = resolve_pops(args, pops)   # may sys.exit on --list-pops
    print(f"==> [02] population selection mode: {mode_desc}", flush=True)
    print(f"==> [02] selected pops ({len(use_pops)}): {', '.join(use_pops)}", flush=True)

    Q, N, V, n_indiv_per_pop = build_numpy(
        args.prefix, use_pops, pops_arr, n_indiv, n_snp, rlen, args.chunk_snps)

    print(f"==> [02] pre-filter: n_snp={Q.shape[0]} n_pop={Q.shape[1]}", flush=True)
    Q, N, V, pop_mask, snp_idx = drop_empty_and_cap(
        Q, N, V, args.snp_cap, args.maf_min)
    kept_pops = [p for p, keep in zip(use_pops, pop_mask) if keep]

    # Spec naming: P = #populations, M = #SNPs.
    M = Q.shape[0]     # number of SNPs (rows of the snp-major array)
    P = Q.shape[1]     # number of pops (cols of the snp-major array)
    print(f"==> [02] post-filter: P(pops)={P}  M(SNPs)={M}", flush=True)
    if P == 0 or M == 0:
        print("ERROR: nothing survived filtering (P or M is zero)", file=sys.stderr)
        sys.exit(1)
    if len(kept_pops) != P:
        print(f"ERROR: internal: kept_pops ({len(kept_pops)}) != P ({P})", file=sys.stderr)
        sys.exit(1)

    # ---- write WITHOUT transposing (see LAYOUT in the module docstring) ------
    # Q/N/V are C-contiguous (M=n_snp, P=n_pop).  .tofile() places element
    # (pop i, snp s) = A[s, i] at flat index P*s + i = i + P*s, matching the
    # SHARED FORMAT's cuBLAS column-major [P x M] (ld = P) convention.
    Qout = np.ascontiguousarray(Q, dtype="<f8")   # (M, P) = (n_snp, n_pop)
    Vout = np.ascontiguousarray(V, dtype="<f8")
    Nout = np.ascontiguousarray(N, dtype="<f8")
    assert Qout.shape == (M, P) and Qout.flags["C_CONTIGUOUS"]
    assert Vout.flags["C_CONTIGUOUS"] and Nout.flags["C_CONTIGUOUS"]

    # ---- invariants required by the SHARED FORMAT ----------------------------
    assert np.all((Qout >= 0.0) & (Qout <= 1.0)), "Q must be in [0,1]"
    assert np.all(np.isin(Vout, (0.0, 1.0))), "V must be in {0,1}"
    assert np.all(Nout >= 0.0), "N must be >= 0"
    assert np.all(Vout[Nout > 0] == 1.0) and np.all(Vout[Nout == 0] == 0.0), \
        "V must equal (N>0)"

    os.makedirs(args.outdir, exist_ok=True)
    Qout.tofile(os.path.join(args.outdir, "Q.f64"))
    Vout.tofile(os.path.join(args.outdir, "V.f64"))
    Nout.tofile(os.path.join(args.outdir, "N.f64"))

    # shape.txt: single ASCII line "P M\n" for trivial C fscanf.
    with open(os.path.join(args.outdir, "shape.txt"), "w") as fh:
        fh.write(f"{P} {M}\n")

    meta = {
        "P": int(P),                       # number of POPULATIONS
        "M": int(M),                       # number of SNPs
        "layout": "float64 [P x M] column-major (element (pop i, snp s) at i + P*s); "
                  "== numpy C-contiguous (M, P) .tofile(); cuBLAS column-major ld=P",
        "pops": kept_pops,
        "n_indiv_per_pop": {p: int(n_indiv_per_pop.get(p, 0)) for p in kept_pops},
        "source": "AADR PACKEDANCESTRYMAP (pure-numpy reader); prefix=" + args.prefix
                  + "; ref allele = fixed .snp col5 (consistent across pops, no "
                    "per-pop re-polarization); selection=" + mode_desc,
        "snp_cap": (int(args.snp_cap) if args.snp_cap is not None else None),
        "maf_min": float(args.maf_min),
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # ---- sanity stats / echo -------------------------------------------------
    valid_mask = Vout > 0
    mean_freq = float(Qout[valid_mask].mean()) if valid_mask.any() else float("nan")
    mean_missing = float(1.0 - Vout.mean())
    mean_hapN = float(Nout[valid_mask].mean()) if valid_mask.any() else float("nan")
    print("==> [02] SANITY")
    print("----------------------------------------")
    print(f"  P (pops)            = {P}")
    print(f"  M (SNPs)            = {M}")
    print(f"  matched pops        = {', '.join(kept_pops)}")
    print(f"  mean ref-allele freq (over valid) = {mean_freq:.6f}")
    print(f"  mean missingness    = {mean_missing:.6f}")
    print(f"  mean haploid N (over valid)       = {mean_hapN:.2f}")
    print(f"  wrote: {args.outdir}/Q.f64 V.f64 N.f64 shape.txt meta.json")
    print(f"  bytes per .f64 file = {P * M * 8}  (= P*M*8)")
    print("----------------------------------------")
    print("==> [02] DONE")


if __name__ == "__main__":
    main()
