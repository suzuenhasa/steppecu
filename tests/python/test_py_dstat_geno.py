"""qpDstat Part-B acceptance gate: reproduce the AT2 GENOTYPE-PATH normalized-D golden
(golden_fit0_dstat_geno.csv, 60 quadruple rows) THROUGH the Python API.

The proof the steppe.dstat binding is correct: the golden est/se/z must flow read_ind(union)
-> name-resolve -> decode -> the per-SNP D kernel -> num/den block-jackknife -> result->Python,
exercising every Part-B binding seam. NO synthetic data — steppe reads the raw HO TGENO prefix
(STEPPE_AADR_ROOT/raw/v66.p1_HO.aadr.patch.PUB), the lossless transcode of the convertf-PA
v66_HO_pa that AT2 read to make the golden; AT2/steppe both read only the 9-pop union (indvec).

The golden was made via admixtools::qpdstat(pref=the PA prefix, the 60 fit0 quadruples,
f4mode=FALSE) -> qpdstat_geno (allsnps=TRUE, blgsize=0.05) — the NORMALIZED D (est ~±0.06),
DISTINCT from the f2-path f4 golden (~10x smaller); z = est/se and p = 2*(1-Phi(|z|)) ARE the
AT2 D sign/Z/p convention. The THREE parity pins (forced diploid / autosomes-only /
allsnps=TRUE finiteness) are PINNED inside run_dstat.

TIERS (achievable): est/se/z at rtol 1e-6 (the decode + block components pinned to ~1e-15 on
box5090). SKIPs cleanly when no CUDA device is visible OR the PA prefix is absent.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from conftest import maybe_skip_no_gpu

_AADR_ROOT = Path(os.environ.get("STEPPE_AADR_ROOT", "/workspace/data/aadr"))
# AT2 generated the golden from the convertf-PA v66_HO_pa (PACKEDANCESTRYMAP / SNP-major GENO);
# steppe's reader is TGENO (individual-major) only, so we read the SAME UNDERLYING DATA from the
# raw HO TGENO prefix (convertf is a lossless transcode of these IDENTICAL genotypes — same
# 27594 ind / 584131 SNP axes, same .ind/.snp — so the D is bit-identical).
_PA_PREFIX = _AADR_ROOT / "raw" / "v66.p1_HO.aadr.patch.PUB"

_GOLDEN_CSV = (
    Path(__file__).resolve().parents[1]
    / "reference"
    / "goldens"
    / "at2"
    / "csv"
    / "golden_fit0_dstat_geno.csv"
)


def _close(got, want, rtol, atol):
    return abs(got - want) <= atol + rtol * abs(want)


def _require_prefix():
    for ext in (".geno", ".snp", ".ind"):
        if not (_PA_PREFIX.parent / (_PA_PREFIX.name + ext)).is_file():
            pytest.skip(f"PA genotype prefix absent: {_PA_PREFIX}{ext}")


def _load_golden():
    if not _GOLDEN_CSV.is_file():
        pytest.skip(f"genotype-path dstat golden absent: {_GOLDEN_CSV}")
    rows = []
    with open(_GOLDEN_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "pop1": r["pop1"],
                    "pop2": r["pop2"],
                    "pop3": r["pop3"],
                    "pop4": r["pop4"],
                    "est": float(r["est"]),
                    "se": float(r["se"]),
                    "z": float(r["z"]),
                    "p": float(r["p"]),
                }
            )
    return rows


def _run_dstat(steppe_mod, quartets):
    try:
        return steppe_mod.dstat(str(_PA_PREFIX), quartets)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise


# ---- GATE (1): the genotype-path D THROUGH the Python API reproduces the golden -------
def test_dstat_geno_reproduces_golden(steppe_mod):
    _require_prefix()
    golden = _load_golden()
    assert len(golden) == 60, len(golden)
    quartets = [(g["pop1"], g["pop2"], g["pop3"], g["pop4"]) for g in golden]
    res = _run_dstat(steppe_mod, quartets)

    assert res.status.name == "OK"
    assert len(res) == 60
    got = {
        (res.pop1[i], res.pop2[i], res.pop3[i], res.pop4[i]): (
            res.est[i],
            res.se[i],
            res.z[i],
            res.p[i],
        )
        for i in range(len(res))
    }
    for g in golden:
        key = (g["pop1"], g["pop2"], g["pop3"], g["pop4"])
        assert key in got, key
        est, se, z, p = got[key]
        assert _close(est, g["est"], 1e-6, 1e-9), (key, est, g["est"])
        assert _close(se, g["se"], 1e-6, 1e-9), (key, se, g["se"])
        assert _close(z, g["z"], 1e-6, 1e-9), (key, z, g["z"])


# ---- GATE (2): the dstat DataFrame has the golden schema (the D-output convention) ----
def test_dstat_geno_dataframe(steppe_mod):
    pytest.importorskip("pandas")
    _require_prefix()
    golden = _load_golden()
    quartets = [(g["pop1"], g["pop2"], g["pop3"], g["pop4"]) for g in golden[:5]]
    try:
        df = steppe_mod.dstat(str(_PA_PREFIX), quartets, as_dataframe=True)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise

    assert list(df.columns) == ["pop1", "pop2", "pop3", "pop4", "est", "se", "z", "p"]
    assert len(df) == 5
    for i, g in enumerate(golden[:5]):
        assert df["pop1"][i] == g["pop1"]
        assert _close(df["est"][i], g["est"], 1e-6, 1e-9)


# ---- GATE (3): the single-quadruple sanity (the CLI-mirrored row 1) ------------------
def test_dstat_geno_single_quadruple(steppe_mod):
    _require_prefix()
    golden = _load_golden()
    g0 = golden[0]
    res = _run_dstat(steppe_mod, [(g0["pop1"], g0["pop2"], g0["pop3"], g0["pop4"])])
    assert len(res) == 1
    assert _close(res.est[0], g0["est"], 1e-6, 1e-9)
    assert _close(res.se[0], g0["se"], 1e-6, 1e-9)
    assert _close(res.z[0], g0["z"], 1e-6, 1e-9)


# ---- GATE (4): the genotype-path D is DISTINCT from (larger than) the f2-path f4 -----
# The normalized D (~±0.06) is ~10x the f2-path f4 (~±0.006), confirming this is the
# NORMALIZED magnitude, not the bare f4 (Part A). Spot-check the known high-z row 4.
def test_dstat_geno_is_normalized_not_f4(steppe_mod):
    _require_prefix()
    golden = _load_golden()
    # Row 4 (England_BellBeaker,Turkey_N,Han,Israel_Natufian): est ~0.0455, z ~24.3.
    row4 = next(
        g
        for g in golden
        if (g["pop1"], g["pop2"], g["pop3"], g["pop4"])
        == ("England_BellBeaker", "Turkey_N", "Han", "Israel_Natufian")
    )
    res = _run_dstat(steppe_mod, [(row4["pop1"], row4["pop2"], row4["pop3"], row4["pop4"])])
    assert _close(res.est[0], row4["est"], 1e-6, 1e-9)
    assert abs(res.est[0]) > 0.04  # the normalized magnitude, not the ~0.006 f4
    assert abs(res.z[0]) > 20.0
