"""M(py-1) ACCEPTANCE GATE: reproduce a real-AADR AT2 golden THROUGH the Python API.

The proof the nanobind binding is correct: the golden values must flow read_f2 ->
name-resolve -> upload -> run_qpadm -> result->Python -> DataFrame, exercising every
binding seam. NO synthetic data — it stages the COMMITTED real golden f2 fixtures into a
temp f2-dir (the same recipe test_cli_qpadm.cpp uses), then runs through the GPU.

Two-tier tolerances (test_cli_qpadm.cpp:302-352): weights/chisq TIGHT (rtol 1e-6), dof/
f4rank EXACT, se/z/p LOOSE (rtol 1e-3). The staged-from-.bin path reproduces the
f2-OBJECT-path golden values (golden_fit0.json `fixture_f2_object_path`), NOT the
directory-path values (the documented read-arg caveat) — exactly what the C++ CLI gate
asserts.

SKIPs cleanly when no CUDA device is visible (cli-bindings.md §7).
"""
from __future__ import annotations

import math

import pytest

from conftest import maybe_skip_no_gpu, read_raw_fixture

# The golden_fit0 9-pop model (golden_fit0.json metadata; index order).
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
TARGET = "England_BellBeaker"
LEFT = ["Czechia_EBA_CordedWare", "Turkey_N"]
RIGHT = ["Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"]

# f2-OBJECT-path golden_fit0 (the binary fixture values; test_cli_qpadm.cpp:456-467).
G_WEIGHT = [0.868755109981416, 0.131244890018584]
G_P = 0.411881081897742
G_CHISQ = 3.95682062790988
G_DOF = 4
G_F4RANK = 1
G_RD_F4RANK = [1, 0]
G_RD_DOF = [4, 10]
G_RD_CHISQ = [3.95682062790988, 1474.03320584515]


def _close(got, want, rtol, atol):
    return abs(got - want) <= atol + rtol * abs(want)


def _run_fit0(steppe_mod, stage_f2_dir):
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    try:
        res = steppe_mod.qpadm(f2, target=TARGET, left=LEFT, right=RIGHT)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise
    return f2, res


# ---- GATE (1): qpadm THROUGH the Python API reproduces golden_fit0 -------------------
def test_qpadm_reproduces_golden_fit0(steppe_mod, stage_f2_dir):
    _, res = _run_fit0(steppe_mod, stage_f2_dir)

    assert res.status.name == "OK"
    # weights TIGHT (rtol 1e-6), via the .weight list and the pandas frame.
    assert len(res.weight) == 2
    for got, want in zip(res.weight, G_WEIGHT):
        assert _close(got, want, 1e-6, 1e-12), (got, want)

    # scalars: chisq TIGHT, p LOOSE, dof/f4rank EXACT.
    assert _close(res.chisq, G_CHISQ, 1e-6, 1e-9), res.chisq
    assert _close(res.p, G_P, 1e-3, 1e-9), res.p
    assert res.dof == G_DOF
    assert res.f4rank == G_F4RANK
    assert res.feasible is True

    # rankdrop EXACT rows + dof, chisq TIGHT.
    rd = res._d
    assert list(rd["rankdrop_f4rank"]) == G_RD_F4RANK
    assert list(rd["rankdrop_dof"]) == G_RD_DOF
    for got, want in zip(rd["rankdrop_chisq"], G_RD_CHISQ):
        assert _close(got, want, 1e-6, 1e-9), (got, want)


def test_qpadm_weights_dataframe(steppe_mod, stage_f2_dir):
    pd = pytest.importorskip("pandas")
    _, res = _run_fit0(steppe_mod, stage_f2_dir)

    w = res.weights
    assert list(w.columns) == ["target", "left", "weight", "se", "z"]
    assert len(w) == 2
    assert list(w["left"]) == LEFT
    assert (w["target"] == TARGET).all()
    for got, want in zip(w["weight"], G_WEIGHT):
        assert _close(got, want, 1e-6, 1e-12)
    # se/z present (run_qpadm always computes SE) -> finite, near-golden (LOOSE).
    assert all(math.isfinite(v) for v in w["se"])

    rdf = res.rankdrop
    assert list(rdf.columns) == [
        "f4rank", "dof", "chisq", "p", "dofdiff", "chisqdiff", "p_nested",
    ]
    pdf = res.popdrop
    assert list(pdf.columns) == ["pat", "wt", "dof", "f4rank", "chisq", "p", "feasible"]
    assert pdf["feasible"].dtype == bool


# ---- GATE (2): the f2 numpy array round-trips / matches the raw C++ fixture ----------
def test_f2_numpy_roundtrip(steppe_mod, stage_f2_dir, fixtures_dir):
    np = pytest.importorskip("numpy")
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))

    arr = f2.to_numpy()
    assert arr.shape == (9, 9, 710)
    assert arr.dtype == np.float64
    assert arr.flags.f_contiguous

    # Each slab is symmetric (f2 is symmetric per fstats.hpp).
    for b in (0, 1, 355, 709):
        slab = arr[:, :, b]
        assert np.allclose(slab, slab.T, rtol=0, atol=0)

    # Values BIT-EQUAL the raw fixture doubles (col-major i + P*j + P*P*b).
    P, nb, _bs, flat = read_raw_fixture(fixtures_dir / "f2_fit0_9pop.bin")
    assert (P, nb) == (9, 710)
    ref = np.asarray(flat, dtype=np.float64).reshape((9, 9, 710), order="F")
    assert np.array_equal(arr, ref)

    # Handle metadata matches.
    assert f2.P == 9
    assert f2.n_block == 710
    assert list(f2.pops) == POPS_9
    assert len(f2.block_sizes) == 710


# ---- GATE (3): qpwave + qpadm_search return sensible structured results --------------
def test_qpwave_structure(steppe_mod, stage_f2_dir):
    # golden_qpwave M1: left[0]=England_BellBeaker (ref), then CordedWare, Turkey_N.
    # f2-object-path gated values (test_qpwave_parity.cu): est_rank=1, f4rank=1,
    # rankdrop chisq {3.95682.., 1474.0332..}.
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    left = ["England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N"]
    try:
        wave = steppe_mod.qpwave(f2, left=left, right=RIGHT)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise

    assert wave.status.name == "OK"
    assert wave.f4rank == 1
    assert wave.est_rank == 1
    rd = wave._d
    assert list(rd["rankdrop_f4rank"]) == [1, 0]
    assert list(rd["rankdrop_dof"]) == [4, 10]
    for got, want in zip(rd["rankdrop_chisq"], [3.95682062790988, 1474.03320584515]):
        assert _close(got, want, 1e-6, 1e-9), (got, want)


def test_qpadm_search_structure(steppe_mod, stage_f2_dir):
    # A small search: 2 explicit left-source sets sharing target/right. Returns one
    # result per model in input order; the first model == golden_fit0.
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    models = [
        ["Czechia_EBA_CordedWare", "Turkey_N"],
        {"left": ["Czechia_EBA_CordedWare", "Iran_GanjDareh_N"]},
    ]
    try:
        results = steppe_mod.qpadm_search(f2, target=TARGET, models=models, right=RIGHT)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise

    assert len(results) == 2
    assert [r.model_index for r in results] == [0, 1]
    # model 0 == golden_fit0 (TIGHT weights).
    m0 = results[0]
    assert m0.f4rank == G_F4RANK
    for got, want in zip(m0.weight, G_WEIGHT):
        assert _close(got, want, 1e-6, 1e-12), (got, want)
    assert m0.feasible is True
    # every row carries a real status (no exception for domain outcomes).
    for r in results:
        assert r.status.name in {
            "OK", "RANK_DEFICIENT", "NON_SPD_COVARIANCE", "CHISQ_UNDEFINED",
        }


def test_search_dataframe(steppe_mod, stage_f2_dir):
    pd = pytest.importorskip("pandas")
    d = stage_f2_dir("f2_fit0_9pop.bin", POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    models = [LEFT, ["Czechia_EBA_CordedWare", "Iran_GanjDareh_N"]]
    try:
        df = steppe_mod.qpadm_search(
            f2, target=TARGET, models=models, right=RIGHT, as_dataframe=True
        )
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise

    assert len(df) == 2
    for col in ("model_index", "target", "left", "p", "chisq", "f4rank", "feasible", "status"):
        assert col in df.columns
    assert list(df["model_index"]) == [0, 1]
