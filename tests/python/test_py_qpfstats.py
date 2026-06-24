"""qpfstats acceptance gate: reproduce the AT2 GENOTYPE-PATH smoothed-f2 TENSOR golden
(golden_qpfstats_geno.csv, 9x9x711 upper-tri incl diag) THROUGH the Python API.

The proof the steppe.qpfstats binding is correct: the smoothed per-block f2 must flow
read_ind(sorted pops) -> decode -> the dstat-numerator engine over the full f2/f3/f4 popcomb
set -> the on-device shared-factor smoothing solve -> scatter/recenter -> the F2Blocks handle
-> Python, exercising every binding seam. NO synthetic data — steppe reads the raw HO TGENO
prefix (STEPPE_AADR_ROOT/raw/v66.p1_HO.aadr.patch.PUB), the SAME genotypes AT2 read to make the
golden via admixtools::qpfstats(pref, the 9 fit0 pops, include_f2/f3/f4=TRUE).

TIER: every (i,j,block) entry at rtol 1e-6 (atol 1e-9). The numerator path matches the
qpDstat-B genotype golden to ~1e-15; the smoothing solve (EmulatedFp64 syrk/gemm + native
potrf/trsm) lands well inside 1e-6. The all-NaN block (536) is matched as 0. SKIPs cleanly
when no CUDA device is visible OR the prefix / golden is absent.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from conftest import maybe_skip_no_gpu

_AADR_ROOT = Path(os.environ.get("STEPPE_AADR_ROOT", "/workspace/data/aadr"))
_PREFIX = _AADR_ROOT / "raw" / "v66.p1_HO.aadr.patch.PUB"

_GOLDEN_CSV = (
    Path(__file__).resolve().parents[1]
    / "reference"
    / "goldens"
    / "at2"
    / "csv"
    / "golden_qpfstats_geno.csv"
)

# The 9 fit0 pops (sorted ASC == the golden dimnames order == steppe's internal sort).
_POPS = [
    "Czechia_EBA_CordedWare",
    "England_BellBeaker",
    "Han",
    "Iran_GanjDareh_N",
    "Israel_Natufian",
    "Karitiana",
    "Mbuti",
    "Papuan",
    "Turkey_N",
]


def _close(got, want, rtol, atol):
    return abs(got - want) <= atol + rtol * abs(want)


def _require_prefix():
    for ext in (".geno", ".snp", ".ind"):
        if not (_PREFIX.parent / (_PREFIX.name + ext)).is_file():
            pytest.skip(f"genotype prefix absent: {_PREFIX}{ext}")


def _load_golden():
    if not _GOLDEN_CSV.is_file():
        pytest.skip(f"qpfstats golden absent: {_GOLDEN_CSV}")
    rows = []
    with open(_GOLDEN_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["i"]), int(r["j"]), int(r["block"]), float(r["f2"])))
    return rows


def _run(steppe_mod):
    try:
        return steppe_mod.qpfstats(str(_PREFIX), pops=_POPS)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise


def test_qpfstats_reproduces_golden_tensor(steppe_mod):
    _require_prefix()
    golden = _load_golden()
    assert len(golden) == 45 * 711, len(golden)

    f2 = _run(steppe_mod)  # an F2Blocks handle
    assert f2.P == 9
    assert f2.n_block == 711
    arr = f2.to_numpy()  # (P, P, n_block) F-contiguous float64
    assert arr.shape == (9, 9, 711)

    n_checked = 0
    for i, j, b, want in golden:
        got = float(arr[i, j, b])
        assert _close(got, want, 1e-6, 1e-9), (i, j, b, got, want)
        # symmetry: arr[j,i,b] == arr[i,j,b]
        assert abs(float(arr[j, i, b]) - got) <= 1e-12, (i, j, b)
        n_checked += 1
    assert n_checked == 45 * 711


def test_qpfstats_diagonal_is_zero(steppe_mod):
    _require_prefix()
    f2 = _run(steppe_mod)
    arr = f2.to_numpy()
    for i in range(9):
        for b in (0, 100, 536, 710):  # incl the all-NaN block 536
            assert arr[i, i, b] == 0.0, (i, b)


def test_qpfstats_all_nan_block_matches_golden(steppe_mod):
    """The all-NaN block (0-based 535 == R block 536; no SNP valid in all 4 pops of any comb)
    has PRE-recenter b=0 (the AT2 nan_chunk policy), so AFTER recentering its off-diagonal
    entries collapse to the per-pair constant bglob - f2(f2blocks)$est (NOT zero — the recenter
    shift is added to every block). It must match the golden element-wise (the recentered
    constant), and the diagonal stays 0. This is the parity-exact all-NaN handling."""
    _require_prefix()
    golden = {(i, j, b): v for i, j, b, v in _load_golden()}
    f2 = _run(steppe_mod)
    arr = f2.to_numpy()
    for i in range(9):
        assert arr[i, i, 535] == 0.0, i  # diagonal stays 0
        for j in range(i + 1, 9):
            assert _close(float(arr[i, j, 535]), golden[(i, j, 535)], 1e-6, 1e-9), (i, j)


def test_qpfstats_out_dir_is_readable(steppe_mod, tmp_path):
    """The smoothed f2 dir round-trips through read_f2 (so qpadm/f4 consume it)."""
    _require_prefix()
    out = tmp_path / "smoothed"
    try:
        path = steppe_mod.qpfstats(str(_PREFIX), pops=_POPS, out=str(out))
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise
    assert str(path) == str(out)
    reloaded = steppe_mod.read_f2(str(out))
    assert reloaded.P == 9
    assert reloaded.n_block == 711
    assert reloaded.pops == _POPS  # F2Blocks.pops is a property (list), not a method
