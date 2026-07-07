#!/usr/bin/env python3
"""Li-Stephens localanc (Phase 3) REAL-data concordance harness.

Scores steppe's per-SNP called ancestry (argmax of the `steppe paint --face localanc`
posterior) against an independent local-ancestry caller (FLARE, or RFMix v2 as a
fallback) on the SAME phased input, and reports the per-SNP called-ancestry concordance
(argmax-label agreement %). REAL data only; NO hard-coded expected ancestry — the admixed
sample's tract structure is whatever the reference tool independently calls. There is NO
pass/fail threshold: the honest measured % is the deliverable.

This is NOT a CTest gate (it needs the external tool + real phased panels). Run it by hand
on box5090 after producing the steppe output and the reference-tool output.

INPUT PINNING (critical — see the spec §8b/§8c critic fix):
  Both tools MUST read the SAME phased haplotypes (one phased source), with the SAME
  ref/alt polarization and the SAME donor/recipient individuals (recipient excluded from
  the donor panel). Align steppe and the tool by genomic position (chrom:pos_bp) only
  AFTER confirming the recipient allele coding matches; otherwise the number is confounded
  by phase-arm / polarization divergence, not the localanc statistic. Because the phase-arm
  labelling is arbitrary between tools, we try BOTH haplotype-to-haplotype pairings and
  report the better (max-of-2) — reported honestly, not silently.

Usage:
  localanc_concordance.py --steppe steppe_localanc.csv \
      (--flare flare_out.anc.vcf.gz | --rfmix out.msp.tsv) \
      [--label-map CEU=EUR,YRI=AFR] [--recip-index 0]

Outputs a short report to stdout:
  concordance_vs_reference_tool = <pct>%  (or BLOCKED if the tool output is absent)
  per-label breakdown, aligned-SNP count, hap-pairing chosen.
"""
import argparse
import csv
import gzip
import sys
from collections import defaultdict


def open_maybe_gz(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def parse_label_map(s):
    """`CEU=EUR,YRI=AFR` -> {'CEU':'EUR','YRI':'AFR'} (identity if empty)."""
    m = {}
    if not s:
        return m
    for kv in s.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            m[k.strip()] = v.strip()
    return m


def norm(label, label_map):
    return label_map.get(label, label)


def load_steppe(path, label_map):
    """steppe long-format CSV -> {recipient: {(chrom,pos): called_label}}.

    Header: recipient,snp_id,chrom,pos_bp,genpos_cM,ancestry_label,posterior.
    Collapses the P posterior rows per (recipient, chrom:pos) to the argmax label.
    """
    # recipient -> (chrom,pos) -> best (posterior, label)
    best = defaultdict(dict)
    with open(path, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            recip = row["recipient"]
            chrom = int(float(row["chrom"]))
            pos = int(float(row["pos_bp"]))
            lab = norm(row["ancestry_label"], label_map)
            p = float(row["posterior"])
            key = (chrom, pos)
            cur = best[recip].get(key)
            if cur is None or p > cur[0]:
                best[recip][key] = (p, lab)
    return {r: {k: v[1] for k, v in d.items()} for r, d in best.items()}


def load_flare(path, label_map):
    """FLARE ancestry VCF -> per-sample per-haplotype called ancestry.

    FLARE writes ANCESTRY as a VCF meta line mapping index->population and per-record
    AN1/AN2 FORMAT fields (the ancestry-of-origin index per haplotype). Returns
    (anc_names, {sample: {hap: {(chrom,pos): label}}}) with hap in {0,1}.
    """
    anc_names = {}  # index -> population name
    # sample -> hap(0/1) -> (chrom,pos) -> label
    calls = {0: defaultdict(dict), 1: defaultdict(dict)}
    samples = []
    with open_maybe_gz(path) as f:
        for line in f:
            if line.startswith("##"):
                # ##ANCESTRY=<...> or ##ancestry lines vary by FLARE version; parse a
                # key=index style if present (e.g. ##ANCESTRY=<AFR=0,EUR=1>).
                if "ANCESTRY" in line.upper() and "=" in line:
                    body = line.strip().split("=<", 1)[-1].rstrip(">\n")
                    for kv in body.split(","):
                        if "=" in kv:
                            name, idx = kv.split("=", 1)
                            try:
                                anc_names[int(idx)] = name.strip()
                            except ValueError:
                                pass
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            chrom_s, pos_s = parts[0], parts[1]
            try:
                chrom = int(chrom_s.replace("chr", ""))
            except ValueError:
                continue
            pos = int(pos_s)
            fmt = parts[8].split(":")
            try:
                i1 = fmt.index("AN1")
                i2 = fmt.index("AN2")
            except ValueError:
                continue
            for si, sample in enumerate(samples):
                g = parts[9 + si].split(":")
                if max(i1, i2) >= len(g):
                    continue
                for hap, idx_field in ((0, g[i1]), (1, g[i2])):
                    try:
                        aidx = int(idx_field)
                    except ValueError:
                        continue
                    name = anc_names.get(aidx, str(aidx))
                    calls[hap][sample][(chrom, pos)] = norm(name, label_map)
    return anc_names, calls, samples


def load_rfmix(path, label_map):
    """RFMix v2 .msp.tsv -> per-sample per-haplotype windowed calls.

    Header line 2 is `#chm spos epos sgpos egpos n snps <sample>.0 <sample>.1 ...`.
    Each row is a window [spos,epos); the sample columns hold the subpop INDEX. The first
    comment line maps subpop index->name. Returns (subpops, windows) where windows is a
    list of (chrom, spos, epos, {(sample,hap): label}).
    """
    subpops = {}  # index -> name
    col_hap = []  # list of (sample, hap)
    windows = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#Subpopulation order") or ("Subpopulation" in line and ":" in line):
                # e.g. `#Subpopulation order/codes: AFR=0	EUR=1`
                for tok in line.replace(",", "\t").split():
                    if "=" in tok:
                        name, idx = tok.split("=", 1)
                        try:
                            subpops[int(idx)] = name.strip().lstrip("#")
                        except ValueError:
                            pass
                continue
            if line.startswith("#chm") or line.lower().startswith("#chrom"):
                cols = line.rstrip("\n").split("\t")[6:]
                for c in cols:
                    if c.endswith(".0") or c.endswith(".1"):
                        s, h = c.rsplit(".", 1)
                        col_hap.append((s, int(h)))
                    else:
                        col_hap.append((c, 0))
                continue
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            try:
                chrom = int(parts[0].replace("chr", ""))
                spos = int(parts[1])
                epos = int(parts[2])
            except ValueError:
                continue
            call = {}
            for ci, (s, h) in enumerate(col_hap):
                v = parts[6 + ci]
                try:
                    call[(s, h)] = norm(subpops.get(int(v), v), label_map)
                except (ValueError, IndexError):
                    pass
            windows.append((chrom, spos, epos, call))
    return subpops, col_hap, windows


def rfmix_call_at(windows, sample, hap, chrom, pos):
    for wchrom, spos, epos, call in windows:
        if wchrom == chrom and spos <= pos < epos:
            return call.get((sample, hap))
    return None


def score_pairing(steppe_calls, tool_at_pos):
    """steppe_calls: {(chrom,pos): label}; tool_at_pos: fn (chrom,pos)->label|None.
    Returns (matched, total, per_label{label:[match,total]})."""
    matched = 0
    total = 0
    per_label = defaultdict(lambda: [0, 0])
    for (chrom, pos), slab in steppe_calls.items():
        tlab = tool_at_pos(chrom, pos)
        if tlab is None:
            continue
        total += 1
        per_label[slab][1] += 1
        if slab == tlab:
            matched += 1
            per_label[slab][0] += 1
    return matched, total, per_label


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steppe", required=True, help="steppe --face localanc CSV output")
    ap.add_argument("--flare", help="FLARE ancestry VCF (.vcf or .vcf.gz)")
    ap.add_argument("--rfmix", help="RFMix v2 .msp.tsv output")
    ap.add_argument("--label-map", default="",
                    help="comma list mapping tool/steppe labels to a shared space, "
                         "e.g. CEU=EUR,YRI=AFR")
    ap.add_argument("--recip-index", type=int, default=0,
                    help="which steppe recipient (by emit order) to score; steppe emits "
                         "each haplotype column as its own recipient row")
    ap.add_argument("--sample", default=None,
                    help="reference-tool sample id to compare against (default: first)")
    args = ap.parse_args()

    label_map = parse_label_map(args.label_map)

    steppe = load_steppe(args.steppe, label_map)
    recips = list(steppe.keys())
    if not recips:
        print("concordance_vs_reference_tool = BLOCKED (no steppe rows parsed)")
        return 1
    if args.recip_index >= len(recips):
        print(f"concordance_vs_reference_tool = BLOCKED (recip-index {args.recip_index} "
              f">= {len(recips)} recipients)")
        return 1
    recip_key = recips[args.recip_index]
    steppe_calls = steppe[recip_key]
    print(f"steppe recipient scored: {recip_key} ({len(steppe_calls)} SNPs)")

    if not args.flare and not args.rfmix:
        # No reference tool provided/installed -> honest BLOCKED + a tract-contiguity sanity
        # read of steppe's own calls (argmax should form contiguous tracts, not per-SNP noise).
        print("concordance_vs_reference_tool = BLOCKED "
              "(no --flare / --rfmix output supplied — install the tool or pass its output)")
        ordered = sorted(steppe_calls.items())
        switches = sum(1 for i in range(1, len(ordered)) if ordered[i][1] != ordered[i - 1][1])
        n = len(ordered)
        print(f"tract sanity: {switches} ancestry switches over {n} SNPs "
              f"(mean tract ~{(n / (switches + 1)):.0f} SNPs) — "
              f"contiguous tracts expected, per-SNP noise would switch ~half the sites")
        return 0

    # Build the two candidate hap-labellings of the reference tool for this sample, then
    # try both steppe-hap-to-tool-hap pairings and take the better (phase-arm arbitrary).
    if args.flare:
        _, calls, samples = load_flare(args.flare, label_map)
        sample = args.sample or (samples[0] if samples else None)
        if sample is None:
            print("concordance_vs_reference_tool = BLOCKED (no samples in FLARE VCF)")
            return 1
        print(f"reference tool: FLARE, sample {sample}")
        tool_haps = [
            (lambda c, p, h=h: calls[h][sample].get((c, p))) for h in (0, 1)
        ]
    else:
        _, _, windows = load_rfmix(args.rfmix, label_map)
        # discover sample id
        sample = args.sample
        if sample is None:
            for _, _, _, call in windows:
                if call:
                    sample = next(iter(call))[0]
                    break
        print(f"reference tool: RFMix v2, sample {sample}")
        tool_haps = [
            (lambda c, p, h=h: rfmix_call_at(windows, sample, h, c, p)) for h in (0, 1)
        ]

    best = None
    for hi, tool_at in enumerate(tool_haps):
        matched, total, per_label = score_pairing(steppe_calls, tool_at)
        if total == 0:
            continue
        pct = 100.0 * matched / total
        if best is None or pct > best[0]:
            best = (pct, hi, matched, total, per_label)

    if best is None:
        print("concordance_vs_reference_tool = BLOCKED (no aligned SNPs between steppe and tool)")
        return 1

    pct, hi, matched, total, per_label = best
    print(f"hap pairing chosen: steppe-hap -> tool-hap {hi} (max-of-2, phase-arm arbitrary)")
    print(f"aligned SNPs: {total}")
    print(f"concordance_vs_reference_tool = {pct:.2f}%  ({matched}/{total})")
    print("per-label breakdown:")
    for lab in sorted(per_label):
        m, t = per_label[lab]
        frac = (100.0 * m / t) if t else 0.0
        print(f"  {lab:>8}: {frac:6.2f}%  ({m}/{t})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
