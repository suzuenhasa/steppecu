"""DATES acceptance gate: reproduce the DATES reference golden (admixture date in generations)
THROUGH the Python API (steppe.dates).

The proof the steppe.dates binding is correct: the inferred date must flow decode-front-end ->
the cuFFT autocorrelation LD engine -> the exp-decay fit + leave-one-chromosome jackknife ->
result -> Python, exercising every binding seam. NO synthetic data — steppe reads the raw HO
TGENO prefix (STEPPE_AADR_ROOT/raw/v66.p1_HO.aadr.patch.PUB), the SAME underlying genotypes the
DATES reference dated (it decoded TGENO to packedancestrymap; steppe reads TGENO losslessly).

The reference golden (tests/reference/goldens/dates/aadr_PUR_CEU_YRI): target PUR, sources
CEU+YRI -> date = 9.742 generations, SE = 0.317 (literature-consistent PUR European-African
admixture ~9-12 gen / colonial era). The covariance decay is computed with NO host O(M^2)
SNP-pair loop (the ALDER FFT trick). The raw TGENO carries 584131 SNPs with the cM map vs the
reference's convertf-filtered 579720, so the DATE is reproduced within rtol 2% and the SE within
atol 0.10.

SKIPs cleanly when no CUDA device is visible OR the raw AADR prefix is absent.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import maybe_skip_no_gpu

_AADR_ROOT = Path(os.environ.get("STEPPE_AADR_ROOT", "/workspace/data/aadr"))
_PREFIX = _AADR_ROOT / "raw" / "v66.p1_HO.aadr.patch.PUB"

# The DATES reference golden (aadr_PUR_CEU_YRI).
_GOLDEN_DATE = 9.742
_GOLDEN_SE = 0.317


def _require_prefix():
    for ext in (".geno", ".snp", ".ind"):
        if not (_PREFIX.parent / (_PREFIX.name + ext)).is_file():
            pytest.skip(f"raw AADR TGENO prefix absent: {_PREFIX}{ext}")


def _run(steppe_mod):
    try:
        return steppe_mod.dates(str(_PREFIX), "PUR", "CEU", "YRI", device=0)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise


def test_dates_reproduces_golden(steppe_mod):
    _require_prefix()
    d = _run(steppe_mod)
    assert d["status"] == "ok", d["status"]

    date = d["date_gen"]
    se = d["se"]
    assert date == pytest.approx(_GOLDEN_DATE, rel=0.02), (date, _GOLDEN_DATE)
    assert abs(se - _GOLDEN_SE) <= 0.10, (se, _GOLDEN_SE)


def test_dates_curve_shape(steppe_mod):
    _require_prefix()
    d = _run(steppe_mod)
    # The binned covariance-decay curve: cM vs the normalized correlation, parallel arrays.
    cm = d["curve_cm"]
    corr = d["curve_corr"]
    assert len(cm) == len(corr)
    assert len(cm) > 100  # ~995 bins (maxdis 100 cM / 0.1 cM bin, minus the trimmed tail)
    # The curve is strictly increasing in distance and decays (early bins > late bins).
    assert cm[0] < cm[-1]
    assert corr[0] > corr[-1]


def test_dates_source_order_is_date_neutral(steppe_mod):
    """The weight wt = freq(source1) - freq(source2) flips sign when the sources swap, which
    scales the decay AMPLITUDE (not the rate), so the DATE is the same up to the fit tolerance."""
    _require_prefix()
    a = _run(steppe_mod)
    try:
        b = steppe_mod.dates(str(_PREFIX), "PUR", "YRI", "CEU", device=0)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise
    assert a["date_gen"] == pytest.approx(b["date_gen"], rel=0.02), (a["date_gen"], b["date_gen"])
