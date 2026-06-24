"""Single-graph qpGraph acceptance gate THROUGH the Python API: reproduce the AT2
golden (golden_qpgraph_score.csv / golden_qpgraph_weights.csv; admixtools 2.0.10) for the
committed WELL-IDENTIFIED topology.

The proof the steppe.qpgraph binding is correct: the golden score + admix weight must flow
read_f2 -> edge-list -> upload f2 RESIDENT -> run_qpgraph (the IDEA-1 fleet on-device) ->
result -> Python. NO synthetic data — it stages the COMMITTED real-AADR afprod=FALSE f2
fixture (f2_qpgraph_9pop.bin, the SAME f2 admixtools::qpgraph(read_f2(...)) reads) into a
temp f2-dir.

TIERS: score TIGHT-ish (1e-4 — the converged optimum, not step-identical L-BFGS-B), the
admix weight keyed by parent NAME (pSteppe->aCW) at 1e-5. SKIPs cleanly when no CUDA device.
"""
from __future__ import annotations

import pytest

from conftest import maybe_skip_no_gpu

# The fixture P-axis pop order (== the f2_qpgraph_9pop.bin layout).
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

# The committed golden topology (golden_qpgraph_topology.csv).
EDGES = [
    ("R", "Mbuti"), ("R", "nOOA"), ("nOOA", "Papuan"), ("nOOA", "nEAS"),
    ("nEAS", "Han"), ("nEAS", "Karitiana"), ("nOOA", "nWE"), ("nWE", "Israel_Natufian"),
    ("nWE", "nAnat"), ("nAnat", "Turkey_N"), ("nAnat", "England_BellBeaker"),
    ("nWE", "pIran"), ("pIran", "Iran_GanjDareh_N"), ("nEAS", "pSteppe"),
    ("pSteppe", "aCW"), ("pIran", "aCW"), ("aCW", "Czechia_EBA_CordedWare"),
]

GOLDEN_SCORE = 80.0674246076313
GOLDEN_W_STEPPE_TO_CW = 0.153483829987482   # pSteppe -> aCW
GOLDEN_W_IRAN_TO_CW = 0.846516170012518     # pIran -> aCW


def _close(got, want, rtol, atol):
    return abs(got - want) <= atol + rtol * abs(want)


def _run(steppe_mod, stage_f2_dir):
    d = stage_f2_dir("f2_qpgraph_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    try:
        return steppe_mod.qpgraph(f2, EDGES)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise


def _named_weight(res, frm, to):
    af = list(res._d["admix_from"])
    at = list(res._d["admix_to"])
    w = list(res._d["weight"])
    for i in range(len(w)):
        if af[i] == frm and at[i] == to:
            return w[i]
    # the other parent of the same admix node carries 1 - weight.
    for i in range(len(w)):
        if at[i] == to:
            return 1.0 - w[i]
    raise AssertionError(f"no admix edge {frm}->{to}")


# ---- GATE (1): qpgraph THROUGH the Python API reproduces the golden ------------------
def test_qpgraph_reproduces_golden(steppe_mod, stage_f2_dir):
    res = _run(steppe_mod, stage_f2_dir)
    assert res.status.name == "OK"
    assert _close(res.score, GOLDEN_SCORE, 0.0, 1e-4), (res.score, GOLDEN_SCORE)
    w_steppe = _named_weight(res, "pSteppe", "aCW")
    w_iran = _named_weight(res, "pIran", "aCW")
    assert _close(w_steppe, GOLDEN_W_STEPPE_TO_CW, 0.0, 1e-5), (w_steppe, GOLDEN_W_STEPPE_TO_CW)
    assert _close(w_iran, GOLDEN_W_IRAN_TO_CW, 0.0, 1e-5), (w_iran, GOLDEN_W_IRAN_TO_CW)
    # the identifiability witness: a TIGHT restart spread (a unique optimum).
    assert res.restart_spread < 1e-4, res.restart_spread


# ---- GATE (2): the fitted-edges / weights DataFrames have the expected schema --------
def test_qpgraph_dataframes(steppe_mod, stage_f2_dir):
    pytest.importorskip("pandas")
    res = _run(steppe_mod, stage_f2_dir)
    edges = res.edges
    assert list(edges.columns) == ["from", "to", "length"]
    assert len(edges) == 15  # 17 edges - 2 admixture edges
    # every fitted drift length is finite and (constrained=True default) non-negative.
    for v in edges["length"]:
        assert v >= -1e-9, v
    weights = res.weights
    assert list(weights.columns) == ["from", "to", "weight", "low", "high"]
    assert len(weights) >= 1
