"""qpDstat Part-A acceptance gate: reproduce the REGENERATED, fixture-matched AT2 qpdstat
golden (golden_fit0_qpdstat_readf2.csv, 60 quadruple rows) THROUGH the Python API.

The proof the steppe.qpdstat binding is correct: the golden est/se/z/p must flow read_f2 ->
name-resolve -> upload -> run_qpdstat (== run_f4 on the f2-data path) -> result->Python ->
(DataFrame), exercising every binding seam. NO synthetic data — it stages the COMMITTED real
9-pop golden f2 fixture into a temp f2-dir (the same recipe test_py_f4 uses), then runs through
the GPU.

The golden was made via admixtools::qpdstat(read_f2(the 9-pop maxmiss=0 f2 fixture), the SAME
60 quadruples as the f4 golden, f4mode=TRUE) and est is CONFIRMED byte-identical to the f4
golden: the qpdstat f2-path == f4 (f4mode is a no-op without per-SNP genotypes), where z =
est/se and p = 2*(1-Phi(|z|)) ARE the AT2 D-stat sign/Z/p convention. The normalized-D
MAGNITUDE (per-SNP genotypes) is Part B (a genotype prefix), not yet implemented.

TIERS: est/se/z/p ALL TIGHT (rtol 1e-6) — the qpdstat f2-path == f4, whose regen cross-check
measured max rel delta 1.36e-12. SKIPs cleanly when no CUDA device is visible (cli-bindings §7).
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
    / "golden_fit0_qpdstat_readf2.csv"
)


def _close(got, want, rtol, atol):
    return abs(got - want) <= atol + rtol * abs(want)


def _load_golden():
    if not _GOLDEN_CSV.is_file():
        pytest.skip(f"regenerated qpdstat golden absent: {_GOLDEN_CSV}")
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


def _run_qpdstat(steppe_mod, stage_f2_dir):
    golden = _load_golden()
    assert len(golden) == 60, len(golden)
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    quartets = [(g["pop1"], g["pop2"], g["pop3"], g["pop4"]) for g in golden]
    try:
        res = steppe_mod.qpdstat(f2, quartets)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise
    return golden, res


# ---- GATE (1): qpdstat THROUGH the Python API reproduces the regenerated golden ------
def test_qpdstat_reproduces_golden(steppe_mod, stage_f2_dir):
    golden, res = _run_qpdstat(steppe_mod, stage_f2_dir)

    assert res.status.name == "OK"
    assert len(res) == 60
    # Key by the quadruple name 4-tuple (the steppe order == input order == golden order,
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


# ---- GATE (2): the qpdstat DataFrame has the golden schema (== f4: D-output convention) -
def test_qpdstat_dataframe(steppe_mod, stage_f2_dir):
    pytest.importorskip("pandas")
    golden, _ = _run_qpdstat(steppe_mod, stage_f2_dir)
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    quartets = [(g["pop1"], g["pop2"], g["pop3"], g["pop4"]) for g in golden[:5]]
    try:
        df = steppe_mod.qpdstat(f2, quartets, as_dataframe=True)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise

    assert list(df.columns) == ["pop1", "pop2", "pop3", "pop4", "est", "se", "z", "p"]
    assert len(df) == 5
    for i, g in enumerate(golden[:5]):
        assert df["pop1"][i] == g["pop1"]
        assert _close(df["est"][i], g["est"], 1e-6, 1e-9)


# ---- GATE (3): the single-quadruple sanity (the CLI-mirrored row 1) ------------------
def test_qpdstat_single_quadruple(steppe_mod, stage_f2_dir):
    golden, _ = _run_qpdstat(steppe_mod, stage_f2_dir)
    g0 = golden[0]
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    try:
        res = steppe_mod.qpdstat(f2, [(g0["pop1"], g0["pop2"], g0["pop3"], g0["pop4"])])
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise

    assert len(res) == 1
    assert _close(res.est[0], g0["est"], 1e-6, 1e-9)
    assert _close(res.se[0], g0["se"], 1e-6, 1e-9)
    assert _close(res.z[0], g0["z"], 1e-6, 1e-9)
    assert _close(res.p[0], g0["p"], 1e-6, 1e-12)


# ---- GATE (4): the qpdstat f2-path == f4 (byte-identical est on the SAME quadruples) -
def test_qpdstat_equals_f4(steppe_mod, stage_f2_dir):
    golden, _ = _run_qpdstat(steppe_mod, stage_f2_dir)
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    quartets = [(g["pop1"], g["pop2"], g["pop3"], g["pop4"]) for g in golden]
    try:
        d_res = steppe_mod.qpdstat(f2, quartets)
        f4_res = steppe_mod.f4(f2, quartets)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise

    assert len(d_res) == len(f4_res) == 60
    for i in range(len(d_res)):
        # The f2-path qpdstat IS f4 (same run_f4 seam) — exactly equal, no tolerance slack.
        assert d_res.est[i] == f4_res.est[i]
        assert d_res.se[i] == f4_res.se[i]
        assert d_res.z[i] == f4_res.z[i]
        assert d_res.p[i] == f4_res.p[i]
