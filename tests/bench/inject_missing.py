#!/usr/bin/env python3
"""inject_missing.py — set a fraction p of CALLED genotypes in a PLINK .bed to missing.

The one sanctioned synthetic operation in the harness: real fit9_ped genotypes are kept,
only a deterministic random subset of called calls is flipped to the PLINK missing 2-bit
code 0b01. .bim/.fam are NOT touched by inject() — the caller copies them so col6 pops and
--pops selection keep working at query time (bench_common.ensure_injected_bed).

PLINK1 .bed layout: 3 magic bytes (0x6c 0x1b 0x01 = SNP-major), then per SNP
ceil(n_samples/4) bytes, each byte packing 4 samples' 2-bit codes low-bits-first. The 2-bit
codes on disk are 00=hom-A1, 01=MISSING, 10=het, 11=hom-A2 (steppe kBedToCanon={2,miss,het,0}).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

_MAGIC = bytes([0x6C, 0x1B, 0x01])
_MISSING = 0b01


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def inject(in_prefix: str, out_prefix: str, p: float, seed: int) -> tuple[int, int]:
    """Rewrite in_prefix.bed -> out_prefix.bed with fraction p of called genotypes set
    missing. Returns (n_snps, n_flipped)."""
    n_snps = _count_lines(in_prefix + ".bim")
    n_samp = _count_lines(in_prefix + ".fam")
    bps = (n_samp + 3) // 4

    data = np.fromfile(in_prefix + ".bed", dtype=np.uint8)
    if len(data) < 3 or data[0] != 0x6C or data[1] != 0x1B or data[2] != 0x01:
        raise ValueError(f"{in_prefix}.bed is not a SNP-major PLINK1 .bed")
    body = data[3:].reshape(n_snps, bps)

    # Unpack the 4 codes per byte -> (n_snps, bps, 4), then a sample view of the first
    # n_samp slots (the tail slots in the last byte are padding and stay untouched).
    codes = np.zeros((n_snps, bps, 4), dtype=np.uint8)
    for k in range(4):
        codes[:, :, k] = (body >> (2 * k)) & 0b11
    flat = codes.reshape(n_snps, bps * 4)
    view = flat[:, :n_samp]

    called = view != _MISSING
    rng = np.random.default_rng(seed)
    draw = rng.random(view.shape) < p
    flip = called & draw
    view[flip] = _MISSING
    flat[:, :n_samp] = view

    codes = flat.reshape(n_snps, bps, 4)
    newbody = np.zeros((n_snps, bps), dtype=np.uint8)
    for k in range(4):
        newbody |= (codes[:, :, k] & 0b11) << (2 * k)

    with open(out_prefix + ".bed", "wb") as f:
        f.write(_MAGIC)
        newbody.reshape(-1).tofile(f)
    return n_snps, int(flip.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input PLINK prefix")
    ap.add_argument("--out", required=True, help="output PLINK prefix (.bed written)")
    ap.add_argument("--p", type=float, required=True, help="missing fraction 0..1")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    n, f = inject(a.inp, a.out, a.p, a.seed)
    print(f"injected p={a.p} into {n} SNPs; flipped {f} calls -> {a.out}.bed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
