"""steppe — GPU qpAdm / qpWave (M(py-1) nanobind bindings).

A thin, pandas-friendly facade over the compiled extension ``steppe._core``. The
compiled module does ONLY numpy<->span marshalling, name->index resolution, options
fill, and result->flat-dict shaping (no pandas link); ALL DataFrame / dataclass shaping
lives HERE (cli-bindings.md §5, line 392-393). ``pandas`` is a LAZY, soft dependency:
``import steppe`` works without it; ``.to_dataframe()`` / the ``.weights`` etc. accessors
import it on demand and raise a clear error if it is missing.

steppe is a GPU product (memory cpu-is-test-only): there is NO CPU runtime path. A fit on
a no-CUDA box raises a clear "no CUDA device" error from the compiled layer.
"""
from __future__ import annotations

import enum
import math
from typing import Any, Optional

from . import _core  # the compiled nanobind extension (steppe/_core*.so)

__all__ = [
    "Status",
    "F2Blocks",
    "QpAdmResult",
    "QpWaveResult",
    "F4Result",
    "read_f2",
    "qpadm",
    "qpwave",
    "f4",
    "qpadm_search",
]


class Status(enum.Enum):
    """Mirror of the C++ steppe::Status taxonomy (error.hpp). Domain outcomes
    (RANK_DEFICIENT / NON_SPD_COVARIANCE / CHISQ_UNDEFINED) are VALUES on a result,
    never exceptions — filter a search on ``status``, not on ``p``."""

    OK = "ok"
    DEVICE_OOM = "device_oom"
    RANK_DEFICIENT = "rank_deficient"
    NON_SPD_COVARIANCE = "non_spd_covariance"
    CHISQ_UNDEFINED = "chisq_undefined"
    INVALID_CONFIG = "invalid_config"

    @classmethod
    def _from(cls, s: str) -> "Status":
        return cls(s)


def _require_pandas():
    try:
        import pandas as pd  # noqa: WPS433 (lazy by design — soft dep)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "this method needs pandas; install it (`pip install pandas`). "
            "pandas is an optional steppe dependency, imported only by the "
            "DataFrame accessors."
        ) from exc
    return pd


def _model_feasible(weight: list[float]) -> bool:
    """AT2 res$ whole-model feasibility = every weight in [0, 1] (result_emit.cpp:60).
    Empty weights (a domain-failed model) => not feasible."""
    if not weight:
        return False
    return all(0.0 <= w <= 1.0 for w in weight)


def _rotation_feasible(d: dict) -> bool:
    """The rotation per-row feasibility: prefer the popdrop FULL-model row (index 0,
    byte-faithful to what run_qpadm_search recorded; result_emit.cpp:261), falling back
    to the canonical weights-in-[0,1] screen when popdrop is empty."""
    pf = d.get("popdrop_feasible") or []
    if pf:
        return bool(pf[0])
    return _model_feasible(d.get("weight") or [])


class QpAdmResult:
    """A single qpAdm fit result, pandas-shaped (admixr parity). Built from the flat
    dict the compiled layer returns; the DataFrames mirror AT2's $weights/$rankdrop/
    $popdrop column names verbatim (interop-usecases.md §2)."""

    def __init__(self, d: dict, *, target: str, left: list[str]):
        self._d = d
        self.target = target
        self.left = list(left)

        self.weight: list[float] = list(d["weight"])
        self.se: list[float] = list(d["se"])  # empty => SE not computed (-> NaN)
        self.z: list[float] = list(d["z"])
        self.p: float = d["p"]
        self.chisq: float = d["chisq"]
        self.dof: int = d["dof"]
        self.est_rank: int = d["est_rank"]
        self.f4rank: int = d["f4rank"]
        self.status: Status = Status._from(d["status"])
        self.precision: str = d["precision"]
        self.model_index: int = d["model_index"]
        self.feasible: bool = _rotation_feasible(d)

    @property
    def weights(self):  # -> pandas.DataFrame [target, left, weight, se, z]
        pd = _require_pandas()
        n = len(self.weight)
        have_se = len(self.se) == n
        se = self.se if have_se else [math.nan] * n
        z = self.z if (len(self.z) == n) else [math.nan] * n
        return pd.DataFrame(
            {
                "target": [self.target] * n,
                "left": list(self.left),
                "weight": list(self.weight),
                "se": list(se),
                "z": list(z),
            }
        )

    @property
    def rankdrop(self):  # [f4rank, dof, chisq, p, dofdiff, chisqdiff, p_nested]
        pd = _require_pandas()
        d = self._d
        return pd.DataFrame(
            {
                "f4rank": list(d["rankdrop_f4rank"]),
                "dof": list(d["rankdrop_dof"]),
                "chisq": list(d["rankdrop_chisq"]),
                "p": list(d["rankdrop_p"]),
                "dofdiff": list(d["rankdrop_dofdiff"]),
                "chisqdiff": list(d["rankdrop_chisqdiff"]),
                "p_nested": list(d["rankdrop_p_nested"]),
            }
        )

    @property
    def popdrop(self):  # [pat, wt, dof, f4rank, chisq, p, feasible]
        pd = _require_pandas()
        d = self._d
        return pd.DataFrame(
            {
                "pat": list(d["popdrop_pat"]),
                "wt": list(d["popdrop_wt"]),
                "dof": list(d["popdrop_dof"]),
                "f4rank": list(d["popdrop_f4rank"]),
                "chisq": list(d["popdrop_chisq"]),
                "p": list(d["popdrop_p"]),
                "feasible": list(d["popdrop_feasible"]),
            }
        )

    def __repr__(self) -> str:
        return (
            f"QpAdmResult(target={self.target!r}, left={self.left!r}, "
            f"weight={self.weight!r}, p={self.p!r}, f4rank={self.f4rank}, "
            f"feasible={self.feasible}, status={self.status.name})"
        )


class QpWaveResult:
    """A qpWave rank-sufficiency sweep result, pandas-shaped (QpWaveResult; qpadm.hpp)."""

    def __init__(self, d: dict, *, left: list[str]):
        self._d = d
        self.left = list(left)
        self.f4rank: int = d["f4rank"]
        self.est_rank: int = d["est_rank"]
        self.status: Status = Status._from(d["status"])
        self.precision: str = d["precision"]

    @property
    def rankdrop(self):  # [f4rank, dof, chisq, p, dofdiff, chisqdiff, p_nested]
        pd = _require_pandas()
        d = self._d
        return pd.DataFrame(
            {
                "f4rank": list(d["rankdrop_f4rank"]),
                "dof": list(d["rankdrop_dof"]),
                "chisq": list(d["rankdrop_chisq"]),
                "p": list(d["rankdrop_p"]),
                "dofdiff": list(d["rankdrop_dofdiff"]),
                "chisqdiff": list(d["rankdrop_chisqdiff"]),
                "p_nested": list(d["rankdrop_p_nested"]),
            }
        )

    @property
    def per_rank(self):  # ascending-r sweep: [rank, chisq, dof, p]
        pd = _require_pandas()
        d = self._d
        n = len(d["rank_chisq"])
        return pd.DataFrame(
            {
                "rank": list(range(n)),
                "chisq": list(d["rank_chisq"]),
                "dof": list(d["rank_dof"]),
                "p": list(d["rank_p"]),
            }
        )

    def __repr__(self) -> str:
        return (
            f"QpWaveResult(left={self.left!r}, f4rank={self.f4rank}, "
            f"est_rank={self.est_rank}, status={self.status.name})"
        )


class F4Result:
    """A standalone f4 result table (one row per quartet), pandas-shaped. Built from the
    flat dict of parallel arrays the compiled layer returns; ``.table`` is a tidy DataFrame
    with the golden columns pop1,pop2,pop3,pop4,est,se,z,p (admixtools::f4 parity)."""

    def __init__(self, d: dict):
        self._d = d
        self.pop1: list[str] = list(d["pop1"])
        self.pop2: list[str] = list(d["pop2"])
        self.pop3: list[str] = list(d["pop3"])
        self.pop4: list[str] = list(d["pop4"])
        self.est: list[float] = list(d["est"])
        self.se: list[float] = list(d["se"])
        self.z: list[float] = list(d["z"])
        self.p: list[float] = list(d["p"])
        self.status: Status = Status._from(d["status"])
        self.precision: str = d["precision"]

    @property
    def table(self):  # -> pandas.DataFrame [pop1, pop2, pop3, pop4, est, se, z, p]
        pd = _require_pandas()
        return pd.DataFrame(
            {
                "pop1": list(self.pop1),
                "pop2": list(self.pop2),
                "pop3": list(self.pop3),
                "pop4": list(self.pop4),
                "est": list(self.est),
                "se": list(self.se),
                "z": list(self.z),
                "p": list(self.p),
            }
        )

    def __len__(self) -> int:
        return len(self.est)

    def __repr__(self) -> str:
        return f"F4Result(n_quartets={len(self.est)}, status={self.status.name})"


class F2Blocks:
    """An opaque f2-dir handle: the host f2 tensor + the P pop labels (P-axis order).
    Wraps the compiled ``_core.F2Handle``; build_resources is cached on the handle so the
    precompute-once / fit-many path reuses one Resources across fits (ADR-0005)."""

    def __init__(self, handle: "_core.F2Handle"):
        self._h = handle

    @property
    def pops(self) -> list[str]:
        return list(self._h.pops)

    @property
    def P(self) -> int:
        return self._h.P

    @property
    def n_block(self) -> int:
        return self._h.n_block

    @property
    def block_sizes(self) -> list[int]:
        return list(self._h.block_sizes)

    @property
    def device(self) -> int:
        return self._h.device

    def to_numpy(self):
        """The f2 tensor as a numpy float64 array, F-contiguous, shape (P, P, n_block).
        ``arr[:, :, b]`` is slab b (column-major within, no silent transpose). A COPY."""
        return self._h._f2_numpy()

    def vpair_to_numpy(self):
        """The vpair tensor as a numpy float64 array, F-contiguous (P, P, n_block). COPY.
        (The staged-from-.bin fixtures carry zero vpair; a real extract-f2 dir carries the
        true pairwise-valid counts.)"""
        return self._h._vpair_numpy()

    def __repr__(self) -> str:
        return f"F2Blocks(P={self.P}, n_block={self.n_block}, device={self.device})"


def read_f2(directory: str, *, device: int = 0) -> F2Blocks:
    """Load an f2-dir (``f2.bin`` STPF2BK1 + ``pops.txt``) into an opaque F2Blocks handle.
    Does NOT upload to the GPU — the upload happens per fit call (mirroring the CLI)."""
    return F2Blocks(_core.read_f2(str(directory), device))


def qpadm(
    f2: F2Blocks,
    *,
    target: str,
    left: list[str],
    right: list[str],
    rank: int = -1,
    fudge: float = 1e-4,
    als_iterations: int = 20,
    rank_alpha: float = 0.05,
    allow_negative_weights: bool = True,
) -> QpAdmResult:
    """Single-model qpAdm GLS fit on the GPU. Defaults match QpAdmOptions so a bare
    call reproduces the AT2 goldens. Unknown pop names raise a clean KeyError."""
    d = _core.run_qpadm(
        f2._h,
        target,
        list(left),
        list(right),
        rank,
        fudge,
        als_iterations,
        rank_alpha,
        allow_negative_weights,
    )
    return QpAdmResult(d, target=target, left=left)


def qpwave(
    f2: F2Blocks,
    *,
    left: list[str],
    right: list[str],
    fudge: float = 1e-4,
    rank_alpha: float = 0.05,
) -> QpWaveResult:
    """qpWave rank-sufficiency sweep on the GPU. ``left[0]`` is the reference row;
    there is NO target argument (the distinguishing qpWave invocation)."""
    d = _core.run_qpwave(f2._h, list(left), list(right), fudge, rank_alpha)
    return QpWaveResult(d, left=left)


def f4(
    f2: F2Blocks,
    quartets: list[Any],
    *,
    as_dataframe: bool = False,
):
    """Standalone f4(p1,p2;p3,p4) statistic on the GPU — est/se/z/p per quartet (NO ALS /
    NO rank; the AT2 weighted block-jackknife f4 + the jackknife-diagonal SE).

    ``quartets`` is a list where each entry is a ``(p1, p2, p3, p4)`` name tuple/list (or a
    ``{"pop1":..,"pop2":..,"pop3":..,"pop4":..}`` dict). Returns an ``F4Result`` (or, when
    ``as_dataframe=True``, the tidy ``F4Result.table`` DataFrame: pop1,pop2,pop3,pop4,
    est,se,z,p). Unknown pop names raise a clean KeyError."""
    quads: list[tuple[str, str, str, str]] = []
    for q in quartets:
        if isinstance(q, dict):
            quads.append((q["pop1"], q["pop2"], q["pop3"], q["pop4"]))
        else:
            p1, p2, p3, p4 = q  # exactly 4 names (raises on a malformed tuple)
            quads.append((p1, p2, p3, p4))

    d = _core.run_f4(f2._h, quads)
    res = F4Result(d)
    return res.table if as_dataframe else res


def qpadm_search(
    f2: F2Blocks,
    *,
    target: str,
    models: list[Any],
    right: list[str],
    rank: int = -1,
    fudge: float = 1e-4,
    als_iterations: int = 20,
    rank_alpha: float = 0.05,
    allow_negative_weights: bool = True,
    jackknife: str = "all",
    as_dataframe: bool = False,
):
    """Batched qpAdm over a list of candidate left-source sets (shared target/right).

    ``models`` is a list where each entry is either a ``list[str]`` of left sources or a
    ``{"left": [...]}`` dict (the explicit-models form; the pool/min/max rotate sugar is
    a future add). Returns a list of ``QpAdmResult`` in input order, or — when
    ``as_dataframe=True`` — ONE tidy pandas DataFrame, one row per model.
    """
    lefts: list[list[str]] = []
    for m in models:
        if isinstance(m, dict):
            lefts.append(list(m["left"]))
        else:
            lefts.append(list(m))

    dicts = _core.run_qpadm_search(
        f2._h,
        target,
        lefts,
        list(right),
        rank,
        fudge,
        als_iterations,
        rank_alpha,
        allow_negative_weights,
        jackknife,
    )
    results = [
        QpAdmResult(d, target=target, left=lefts[i]) for i, d in enumerate(dicts)
    ]
    if not as_dataframe:
        return results
    return _search_dataframe(results, target=target, right=right)


def _search_dataframe(results: list[QpAdmResult], *, target: str, right: list[str]):
    """One tidy DataFrame, one row per model in input order: model_index, target, left,
    right_n, p, chisq, dof, f4rank, feasible, status, weight (semicolon-joined)."""
    pd = _require_pandas()
    right_n = max(len(right) - 1, 0)  # nr convention: right[0] == R0.
    rows = []
    for r in results:
        rows.append(
            {
                "model_index": r.model_index,
                "target": target,
                "left": ";".join(r.left),
                "right_n": right_n,
                "p": r.p,
                "chisq": r.chisq,
                "dof": r.dof,
                "f4rank": r.f4rank,
                "feasible": r.feasible,
                "status": r.status.value,
                "weight": ";".join(repr(w) for w in r.weight),
            }
        )
    return pd.DataFrame(rows)
