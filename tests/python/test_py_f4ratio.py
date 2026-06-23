"""Standalone f4-ratio acceptance gate (fit-engine §6): reproduce the fixture-matched AT2
qpf4ratio golden (golden_fit0_f4ratio_readf2.csv, 10 5-tuple rows) THROUGH the Python API. The
FIVE-tuple clone of test_py_f4.py / test_py_f3.py.

The proof the steppe.f4ratio binding is correct: the golden alpha/se/z must flow read_f2 ->
name-resolve -> upload -> run_f4ratio -> result->Python -> (DataFrame), exercising every
binding seam. NO synthetic data — it stages the COMMITTED real 9-pop golden f2 fixture into a
temp f2-dir (the same recipe test_py_f4 uses), then runs through the GPU.

The golden was made via admixtools::qpf4ratio(read_f2(the 9-pop maxmiss=0 f2 fixture)) so a
standalone f4-ratio over the SAME committed fixture matches it.

TIERS: alpha/se/z ALL TIGHT (rtol 1e-6). SKIPs cleanly when no CUDA device is visible.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from conftest import maybe_skip_no_gpu

POPS_9 = [
    "England_BellBeaker",
    "Czechia_EBA_CordedWare",
    "Turkey_N",
    "Mbuti",
    "Israel_Natufian",
    "Iran_GanjDareh_N",
    "Han",
    "Papuan",
    "Karitiana",
]

_GOLDEN_CSV = (
    Path(__file__).resolve().parents[1]
    / "reference"
    / "goldens"
    / "at2"
    / "csv"
    / "golden_fit0_f4ratio_readf2.csv"
)


def _close(got, want, rtol, atol):
    return abs(got - want) <= atol + rtol * abs(want)


def _load_golden():
    if not _GOLDEN_CSV.is_file():
        pytest.skip(f"f4-ratio golden absent: {_GOLDEN_CSV}")
    rows = []
    with open(_GOLDEN_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "pop1": r["pop1"],
                    "pop2": r["pop2"],
                    "pop3": r["pop3"],
                    "pop4": r["pop4"],
                    "pop5": r["pop5"],
                    "alpha": float(r["alpha"]),
                    "se": float(r["se"]),
                    "z": float(r["z"]),
                }
            )
    return rows


def _tuples(golden):
    return [
        (g["pop1"], g["pop2"], g["pop3"], g["pop4"], g["pop5"]) for g in golden
    ]


def _run_f4ratio(steppe_mod, stage_f2_dir):
    golden = _load_golden()
    assert len(golden) == 10, len(golden)
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    try:
        res = steppe_mod.f4ratio(f2, _tuples(golden))
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise
    return golden, res


# ---- GATE (1): f4-ratio THROUGH the Python API reproduces the golden -----------------
def test_f4ratio_reproduces_golden(steppe_mod, stage_f2_dir):
    golden, res = _run_f4ratio(steppe_mod, stage_f2_dir)

    assert res.status.name == "OK"
    assert len(res) == 10
    # Key by the 5-tuple name (the steppe order == input order == golden order, but key on
    # names for robustness).
    got = {
        (res.pop1[i], res.pop2[i], res.pop3[i], res.pop4[i], res.pop5[i]): (
            res.alpha[i],
            res.se[i],
            res.z[i],
        )
        for i in range(len(res))
    }
    for g in golden:
        key = (g["pop1"], g["pop2"], g["pop3"], g["pop4"], g["pop5"])
        assert key in got, key
        alpha, se, z = got[key]
        assert _close(alpha, g["alpha"], 1e-6, 1e-9), (key, alpha, g["alpha"])
        assert _close(se, g["se"], 1e-6, 1e-9), (key, se, g["se"])
        assert _close(z, g["z"], 1e-6, 1e-9), (key, z, g["z"])


# ---- GATE (2): the f4-ratio DataFrame has the golden schema --------------------------
def test_f4ratio_dataframe(steppe_mod, stage_f2_dir):
    pd = pytest.importorskip("pandas")
    golden, _ = _run_f4ratio(steppe_mod, stage_f2_dir)
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    try:
        df = steppe_mod.f4ratio(f2, _tuples(golden[:5]), as_dataframe=True)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise

    assert list(df.columns) == [
        "pop1",
        "pop2",
        "pop3",
        "pop4",
        "pop5",
        "alpha",
        "se",
        "z",
    ]
    assert len(df) == 5
    for i, g in enumerate(golden[:5]):
        assert df["pop1"][i] == g["pop1"]
        assert _close(df["alpha"][i], g["alpha"], 1e-6, 1e-9)


# ---- GATE (3): the single-tuple sanity (the CLI-mirrored row 1) ----------------------
def test_f4ratio_single_tuple(steppe_mod, stage_f2_dir):
    golden, _ = _run_f4ratio(steppe_mod, stage_f2_dir)
    g0 = golden[0]
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    try:
        res = steppe_mod.f4ratio(
            f2, [(g0["pop1"], g0["pop2"], g0["pop3"], g0["pop4"], g0["pop5"])]
        )
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise

    assert len(res) == 1
    assert _close(res.alpha[0], g0["alpha"], 1e-6, 1e-9)
    assert _close(res.se[0], g0["se"], 1e-6, 1e-9)
    assert _close(res.z[0], g0["z"], 1e-6, 1e-9)
