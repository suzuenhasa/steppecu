#!/usr/bin/env python3
"""Transform READv2's Read_Results.tsv into steppe's frozen READv2 row schema.

Frozen schema (scope adna-relatedness-roh-scope.md §1a), one row per UNORDERED pair,
sampleA<sampleB lexicographic:

    sampleA  sampleB  n_windows  n_overlap_sites  P0_mean  P0_norm  degree  z

Column mapping Read_Results.tsv -> steppe schema:
    sampleA,sampleB   <- PairIndividuals split on ',', canonicalized (lexicographic)
    n_windows         <- DERIVED run-level constant: number of non-empty 5 Mbp autosomal
                         blocks the READv2 run tiled (READv2 emits no per-pair window count;
                         steppe's SNP-count windowing is a separate axis reconciled in Phase 1).
    n_overlap_sites   <- OverlapNSNPs
    P0_mean           <- Nonnormalized_P0            (raw per-window mean-mismatch proportion)
    P0_norm           <- P0_mean  (READv2's column name; this is the MEDIAN-normalized value)
    degree            <- Rel collapsed to {identical,first,second,unrelated}
                         (READv2 'Third Degree' and 'Unrelated/Consistent with Third Degree'
                          -> unrelated; raw Rel preserved in Read_Results.raw.tsv)
    z                 <- Zup (Z_upper); 'NA' where READv2 reports NA (unrelated pairs)

Usage: make_fixture.py --results Read_Results.tsv --bim readv2_subset.bim --out readv2_oracle.tsv
"""
import argparse, csv, math

def rel_to_enum(rel):
    r=rel.strip()
    if r.startswith("IdenticalTwins"): return "identical"
    if r=="First Degree": return "first"
    if r=="Second Degree": return "second"
    if r=="Third Degree": return "unrelated"
    if r.startswith("Unrelated"): return "unrelated"
    raise SystemExit(f"unknown Rel token: {rel!r}")

def count_5mbp_blocks(bim, window=5_000_000):
    blocks=set()
    with open(bim) as f:
        for line in f:
            t=line.split()
            if len(t)<4: continue
            chrom=t[0]; bp=int(t[3])
            blocks.add((chrom, bp//window))
    return len(blocks)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",required=True)
    ap.add_argument("--bim",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    n_windows=count_5mbp_blocks(a.bim)

    rows=[]
    with open(a.results) as f:
        r=csv.DictReader(f,delimiter="\t")
        for row in r:
            s1,s2=row["PairIndividuals"].split(",")
            sA,sB=sorted((s1.strip(),s2.strip()))
            rows.append({
                "sampleA":sA, "sampleB":sB,
                "n_windows":str(n_windows),
                "n_overlap_sites":row["OverlapNSNPs"].strip(),
                "P0_mean":row["Nonnormalized_P0"].strip(),
                "P0_norm":row["P0_mean"].strip(),
                "degree":rel_to_enum(row["Rel"]),
                "z":row["Zup"].strip(),
            })
    rows.sort(key=lambda d:(d["sampleA"],d["sampleB"]))

    cols=["sampleA","sampleB","n_windows","n_overlap_sites","P0_mean","P0_norm","degree","z"]
    with open(a.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,delimiter="\t")
        w.writeheader()
        for d in rows: w.writerow(d)
    print(f"wrote {len(rows)} pairs -> {a.out}  (n_windows={n_windows})")

if __name__=="__main__":
    main()
