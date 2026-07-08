#!/usr/bin/env python3
"""roh_panel_hdf5_to_eigenstrat.py — stage the hapROH 1000G phased reference panel.

A one-time HOST stager (NOT a steppe runtime reader). Converts the Zenodo
`1000g1240khdf5.tar.gz` (record 4992532) per-chromosome HDF5 into the EIGENSTRAT triple
`steppe roh --ref-panel` consumes. Each phased 1000G haplotype becomes its OWN
pseudo-haploid EIGENSTRAT column following the EIGENSTRAT "count of the REFERENCE allele"
convention (a REF haplotype -> code '2' = hom-REF, an ALT haplotype -> code '0'), matching
steppe's decode and the AADR pseudo-haploid target so the copying emission compares target
and panel alleles on the SAME axis. This yields K = 2 * n_individuals copying states with
NO heterozygous-code loss (a diploid GT sum would collapse phased haplotypes).

Per chromosome it writes <out>.chrN.{geno,snp,ind}:
  .snp : `rsID  chrom  genpos(Morgan, from variants/MAP)  physpos(POS)  REF  ALT`
  .ind : one row per HAPLOTYPE (`<sample>_A` / `<sample>_B`), pop from --meta or "REF"
  .geno: unpacked ASCII, one char per (variant, haplotype-column) in {0,2}

HDF5 layout consumed (hapROH panel): calldata/GT [n_var, n_ind, 2] phased int8,
variants/POS, variants/MAP (Morgan), variants/REF, variants/ALT, samples.

Disk discipline (box5090 ~27GB, TIGHT): stage a chromosome SUBSET with --chroms and
document it in the gate; the concordance gate then runs pip-hapROH on the SAME subset.

Requires: h5py, numpy (a numpy<2 env is fine). Example:
  python3 tools/roh_panel_hdf5_to_eigenstrat.py \
      --hdf5-dir /workspace/tmp/1000g1240khdf5 --hdf5-template 'chr{ch}.hdf5' \
      --chroms 3,8,20 --out /workspace/tmp/panel/1000g --meta meta.csv
"""
import argparse
import os
import sys


def load_meta(path):
    """sample -> pop label, from a CSV/TSV with a 'sample'/'iid' col + a pop col."""
    if not path:
        return {}
    import csv

    m = {}
    with open(path, newline="") as fh:
        sniff = fh.read(4096)
        fh.seek(0)
        delim = "\t" if sniff.count("\t") >= sniff.count(",") else ","
        rdr = csv.DictReader(fh, delimiter=delim)
        cols = {c.lower(): c for c in (rdr.fieldnames or [])}
        skey = cols.get("sample") or cols.get("iid") or cols.get("id")
        pkey = (
            cols.get("pop")
            or cols.get("population")
            or cols.get("super_pop")
            or cols.get("superpop")
            or cols.get("clst")
        )
        if not skey or not pkey:
            sys.exit(f"--meta {path}: need a sample column and a pop column")
        for row in rdr:
            m[str(row[skey])] = str(row[pkey])
    return m


def stage_chrom(h5path, chrom, out_prefix, meta, max_ind):
    import h5py
    import numpy as np

    with h5py.File(h5path, "r") as f:
        gt = f["calldata/GT"]  # [n_var, n_ind, 2]
        pos = np.asarray(f["variants/POS"])
        gmap = np.asarray(f["variants/MAP"])  # Morgan
        ref = np.asarray(f["variants/REF"])
        alt = f["variants/ALT"]
        alt = np.asarray(alt[:, 0] if alt.ndim == 2 else alt)
        samples = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in np.asarray(f["samples"])
        ]
        n_var, n_ind = gt.shape[0], gt.shape[1]
        if max_ind and max_ind < n_ind:
            n_ind = max_ind
        samples = samples[:n_ind]

        def dc(x):
            return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

        # .ind : 2 haplotype rows per individual.
        with open(f"{out_prefix}.chr{chrom}.ind", "w") as ind:
            for s in samples:
                pop = meta.get(s, "REF")
                ind.write(f"{s}_A U {pop}\n")
                ind.write(f"{s}_B U {pop}\n")

        # .snp
        with open(f"{out_prefix}.chr{chrom}.snp", "w") as snp:
            for v in range(n_var):
                snp.write(
                    f"rs{chrom}_{int(pos[v])} {chrom} {float(gmap[v]):.8f} "
                    f"{int(pos[v])} {dc(ref[v])} {dc(alt[v])}\n"
                )

        # .geno : unpacked ASCII, 2*n_ind chars/row (hap0,hap1 per individual), {0,2}.
        # Chunked variant reads keep the resident footprint small.
        #
        # POLARITY: EIGENSTRAT genotype = COUNT OF THE REFERENCE allele (steppe
        # decode_af.hpp / the AADR pseudo-haploid target both use this convention). So a
        # phased haplotype carrying the REFERENCE allele (HDF5 GT bit 0) is written as the
        # pseudo-hom-REF code '2' (two REF copies), and an ALT haplotype (GT bit 1) as '0'.
        # Writing GT bit -> {0:'0',1:'2'} would INVERT the target/panel allele axis (every
        # copying state would mismatch and no ROH would ever be called).
        chunk = 4096
        lut = {0: "2", 1: "0"}
        with open(f"{out_prefix}.chr{chrom}.geno", "w") as geno:
            for v0 in range(0, n_var, chunk):
                v1 = min(v0 + chunk, n_var)
                block = np.asarray(gt[v0:v1, :n_ind, :])  # [b, n_ind, 2]
                for r in range(block.shape[0]):
                    row = block[r]
                    chars = []
                    for i in range(n_ind):
                        a0 = int(row[i, 0])
                        a1 = int(row[i, 1])
                        chars.append(lut.get(a0, "9"))
                        chars.append(lut.get(a1, "9"))
                    geno.write("".join(chars))
                    geno.write("\n")
    return n_var, 2 * n_ind


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5-dir", required=True, help="dir holding the per-chromosome HDF5")
    ap.add_argument("--hdf5-template", default="chr{ch}.hdf5",
                    help="per-chrom filename template, {ch} substituted (default chr{ch}.hdf5)")
    ap.add_argument("--chroms", default="1-22",
                    help="chromosomes to stage, e.g. '3,8,20' or '1-22' (default 1-22)")
    ap.add_argument("--out", required=True, help="output EIGENSTRAT prefix (writes <out>.chrN.*)")
    ap.add_argument("--meta", default="", help="OPTIONAL sample->pop metadata CSV/TSV")
    ap.add_argument("--max-ind", type=int, default=0,
                    help="cap reference INDIVIDUALS per chromosome (0 = all; K = 2*max-ind)")
    args = ap.parse_args()

    chroms = []
    for tok in args.chroms.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            chroms.extend(range(int(a), int(b) + 1))
        elif tok:
            chroms.append(int(tok))

    meta = load_meta(args.meta)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    total_sites = 0
    for ch in chroms:
        h5path = os.path.join(args.hdf5_dir, args.hdf5_template.format(ch=ch))
        if not os.path.exists(h5path):
            print(f"[skip] chr{ch}: {h5path} not found", file=sys.stderr)
            continue
        nv, ncol = stage_chrom(h5path, ch, args.out, meta, args.max_ind)
        total_sites += nv
        print(f"[ok] chr{ch}: {nv} sites x {ncol} haplotype columns -> "
              f"{args.out}.chr{ch}.{{geno,snp,ind}}", file=sys.stderr)
    print(f"[done] staged {total_sites} sites across {len(chroms)} requested chromosome(s)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
