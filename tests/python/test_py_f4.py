"""Standalone f4 acceptance gate (fit-engine §6): reproduce the REGENERATED, fixture-
matched AT2 f4 golden (golden_fit0_f4_readf2.csv, 60 quartet rows) THROUGH the Python API.

The proof the steppe.f4 binding is correct: the golden est/se/z/p must flow read_f2 ->
name-resolve -> upload -> run_f4 -> result->Python -> (DataFrame), exercising every binding
seam. NO synthetic data — it stages the COMMITTED real 9-pop golden f2 fixture into a temp
f2-dir (the same recipe test_py_qpadm uses), then runs through the GPU.

The golden was made via admixtools::f4(read_f2(the 9-pop maxmiss=0 f2 fixture)) so a
standalone f4 over the SAME committed fixture matches it (the old golden_fit0_f4.csv carried
the qpadm-internal ALLSNPS normalization, a constant 1.1131993890x off the read_f2 fixture).

TIERS: est/se/z/p ALL TIGHT (rtol 1e-6) — the regen cross-check measured max rel delta
1.36e-12. SKIPs cleanly when no CUDA device is visible (cli-bindings.md §7).
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
    / "golden_fit0_f4_readf2.csv"
)


def _close(got, want, rtol, atol):
    return abs(got - want) <= atol + rtol * abs(want)


def _load_golden():
    if not _GOLDEN_CSV.is_file():
        pytest.skip(f"regenerated f4 golden absent: {_GOLDEN_CSV}")
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


def _run_f4(steppe_mod, stage_f2_dir):
    golden = _load_golden()
    assert len(golden) == 60, len(golden)
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    quartets = [(g["pop1"], g["pop2"], g["pop3"], g["pop4"]) for g in golden]
    try:
        res = steppe_mod.f4(f2, quartets)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise
    return golden, res


# ---- GATE (1): f4 THROUGH the Python API reproduces the regenerated golden -----------
def test_f4_reproduces_golden(steppe_mod, stage_f2_dir):
    golden, res = _run_f4(steppe_mod, stage_f2_dir)

    assert res.status.name == "OK"
    assert len(res) == 60
    # Key by the quartet name 4-tuple (the steppe order == input order == golden order,
    # but key on names for robustness).
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
        assert _close(p, g["p"], 1e-6, 1e-12), (key, p, g["p"])


# ---- GATE (2): the f4 DataFrame has the golden schema --------------------------------
def test_f4_dataframe(steppe_mod, stage_f2_dir):
    pd = pytest.importorskip("pandas")
    golden, _ = _run_f4(steppe_mod, stage_f2_dir)
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    quartets = [(g["pop1"], g["pop2"], g["pop3"], g["pop4"]) for g in golden[:5]]
    try:
        df = steppe_mod.f4(f2, quartets, as_dataframe=True)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise

    assert list(df.columns) == ["pop1", "pop2", "pop3", "pop4", "est", "se", "z", "p"]
    assert len(df) == 5
    for i, g in enumerate(golden[:5]):
        assert df["pop1"][i] == g["pop1"]
        assert _close(df["est"][i], g["est"], 1e-6, 1e-9)


# ---- GATE (3): the single-quartet sanity (the CLI-mirrored row 1) --------------------
def test_f4_single_quartet(steppe_mod, stage_f2_dir):
    golden, _ = _run_f4(steppe_mod, stage_f2_dir)
    g0 = golden[0]
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    try:
        res = steppe_mod.f4(f2, [(g0["pop1"], g0["pop2"], g0["pop3"], g0["pop4"])])
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise

    assert len(res) == 1
    assert _close(res.est[0], g0["est"], 1e-6, 1e-9)
    assert _close(res.se[0], g0["se"], 1e-6, 1e-9)
    assert _close(res.z[0], g0["z"], 1e-6, 1e-9)
    assert _close(res.p[0], g0["p"], 1e-6, 1e-12)
