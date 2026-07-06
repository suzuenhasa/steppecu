"""steppe — GPU qpAdm / qpWave (M(py-1) nanobind bindings).

A thin, pandas-friendly facade over the compiled extension ``steppe._core``. The
compiled module does ONLY numpy<->span marshalling, name->index resolution, options
fill, and result->flat-dict shaping (no pandas link); ALL DataFrame / dataclass shaping
lives HERE (cli-bindings.md §5, line 392-393). ``pandas`` is a LAZY, soft dependency:
``import steppe`` works without it; ``.to_dataframe()`` / the ``.weights`` etc. accessors
import it on demand and raise a clear error if it is missing.

steppe is a GPU product (memory cpu-is-test-only): there is NO CPU runtime path. A fit on
a no-CUDA box raises a clear "no CUDA device" error from the compiled layer.

This module's docstrings are the source for the generated Python API reference: the
opt-in ``docs-python`` CMake target (``STEPPE_BUILD_DOCS``) runs ``pdoc -o <out>
bindings/steppe`` over this facade (the C++ headers are the sibling Doxygen ``docs``
target). See docs/api/README.md.
"""
from __future__ import annotations

import enum
import math
from typing import Any, Optional


def _preload_cuda_runtime() -> None:
    """Best-effort: make the CUDA 13 runtime loadable WITHOUT the user setting LD_LIBRARY_PATH.

    The compiled ``_core`` has DT_NEEDED on libcudart.so.13 / libcublas.so.13 /
    libcusolver.so.12 / libcufft.so.12; if those aren't on the loader path, ``import _core``
    fails opaquely. Preloading them (RTLD_GLOBAL) from the usual CUDA install dirs — the wheel
    equivalent of the CLI launcher — lets ``_core``'s dlopen resolve against the already-loaded
    libraries (CUDA's own libs carry a ``$ORIGIN`` RUNPATH, so their siblings resolve too). A
    silent no-op when they're already loadable or genuinely absent (the import guard reports)."""
    import ctypes
    import glob
    import os

    needed = ("libcudart.so.13", "libcublas.so.13", "libcusolver.so.12", "libcufft.so.12")
    dirs = []
    env = os.environ.get("STEPPE_CUDA_LIB")
    if env:
        dirs.append(env)
    dirs.append("/usr/local/cuda/lib64")
    dirs += sorted(glob.glob("/usr/local/cuda-13*/lib64"), reverse=True)
    try:  # the pip `nvidia-*-cu13` redistributable wheels, if installed alongside
        import site

        roots = list(getattr(site, "getsitepackages", list)()) + [site.getusersitepackages()]
        for root in roots:
            dirs += glob.glob(os.path.join(root, "nvidia", "*", "lib"))
    except Exception:  # noqa: BLE001 - site probing is best-effort
        pass

    for name in needed:
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)  # already on the loader path? done.
            continue
        except OSError:
            pass
        for d in dirs:
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                try:
                    ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    continue


_preload_cuda_runtime()

try:
    from . import _core  # the compiled nanobind extension (steppe/_core*.so)

    _CORE_AVAILABLE = True  # the real extension loaded — GPU compute is usable
except ImportError as _exc:
    # `import steppe` must still succeed when the CUDA-13 extension can't load AT ALL (no CUDA
    # runtime on the box) so the GPU-FREE surface keeps working: the f2 .rds converter
    # (export_f2_rds / import_f2_rds / the `steppe-rds` CLI) and `__version__`. Any COMPUTE call
    # then raises a clear error instead of an opaque AttributeError on a missing module. (When
    # the runtime IS present but no GPU device is, `_core` loads fine and the device fault
    # surfaces at call time — this path is unchanged.)
    _core_load_error = _exc  # keep it: the `except ... as _exc` name is cleared at block end

    class _CoreUnavailable:
        """Stand-in for a `_core` that failed to import: every attribute access re-raises the
        original ImportError with guidance, so GPU calls fail loudly while the pure-Python
        converter (which never touches `_core`) still works."""

        def __getattr__(self, _name):
            raise ImportError(
                "steppe._core (the compiled CUDA-13 extension) could not be loaded, so GPU "
                f"compute is unavailable: {_core_load_error}. steppe needs a CUDA-13 runtime "
                "(and a GPU to run). The f2 .rds converter (steppe-rds / export_f2_rds / "
                "import_f2_rds) is GPU-free and does NOT require this."
            ) from _core_load_error

    _core = _CoreUnavailable()  # type: ignore[assignment]
    _CORE_AVAILABLE = False  # only the GPU-free surface (the .rds converter) is usable

try:
    from importlib.metadata import version as _pkg_version

    # An installed wheel reports the version scikit-build-core derived from CMakeLists.txt
    # project(VERSION ...) (pyproject [project] dynamic = ["version"]) — the single source (D2).
    __version__ = _pkg_version("steppe")
except Exception:  # not installed as a distribution (e.g. an in-tree dev import)
    # Non-release SENTINEL for the in-tree dev-import path only: deliberately NOT a real
    # version, so a stale fallback can never impersonate a release (D2).
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "Status",
    "F2Blocks",
    "QpAdmResult",
    "QpWaveResult",
    "F4Result",
    "F3Result",
    "F4RatioResult",
    "read_f2",
    "extract_f2",
    "qpfstats",
    "qpadm",
    "qpwave",
    "qpgraph",
    "QpGraphResult",
    "qpgraph_search",
    "QpGraphSearchResult",
    "f4",
    "f3",
    "f4ratio",
    "qpdstat",
    "dstat",
    "dates",
    "DatesResult",
    "qpadm_search",
    "export_f2_rds",
    "import_f2_rds",
    "list_caches",
    "cache_info",
    "verify_cache",
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
    def _from(cls, s: str) -> Status:
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


class F3Result:
    """A standalone f3 result table (one row per triple), pandas-shaped. The THREE-column
    clone of F4Result (drop pop4). Built from the flat dict of parallel arrays the compiled
    layer returns; ``.table`` is a tidy DataFrame with the golden columns pop1,pop2,pop3,
    est,se,z,p (admixtools::f3 parity)."""

    def __init__(self, d: dict):
        self._d = d
        self.pop1: list[str] = list(d["pop1"])
        self.pop2: list[str] = list(d["pop2"])
        self.pop3: list[str] = list(d["pop3"])
        self.est: list[float] = list(d["est"])
        self.se: list[float] = list(d["se"])
        self.z: list[float] = list(d["z"])
        self.p: list[float] = list(d["p"])
        self.status: Status = Status._from(d["status"])
        self.precision: str = d["precision"]

    @property
    def table(self):  # -> pandas.DataFrame [pop1, pop2, pop3, est, se, z, p]
        pd = _require_pandas()
        return pd.DataFrame(
            {
                "pop1": list(self.pop1),
                "pop2": list(self.pop2),
                "pop3": list(self.pop3),
                "est": list(self.est),
                "se": list(self.se),
                "z": list(self.z),
                "p": list(self.p),
            }
        )

    def __len__(self) -> int:
        return len(self.est)

    def __repr__(self) -> str:
        return f"F3Result(n_triples={len(self.est)}, status={self.status.name})"


class F4RatioResult:
    """A standalone f4-ratio result table (one row per 5-tuple), pandas-shaped. The FIVE-column
    clone of F4Result (add pop5; alpha replaces est, NO p column). Built from the flat dict of
    parallel arrays the compiled layer returns; ``.table`` is a tidy DataFrame with the golden
    columns pop1,pop2,pop3,pop4,pop5,alpha,se,z (admixtools::qpf4ratio parity)."""

    def __init__(self, d: dict):
        self._d = d
        self.pop1: list[str] = list(d["pop1"])
        self.pop2: list[str] = list(d["pop2"])
        self.pop3: list[str] = list(d["pop3"])
        self.pop4: list[str] = list(d["pop4"])
        self.pop5: list[str] = list(d["pop5"])
        self.alpha: list[float] = list(d["alpha"])
        self.se: list[float] = list(d["se"])
        self.z: list[float] = list(d["z"])
        self.status: Status = Status._from(d["status"])
        self.precision: str = d["precision"]

    @property
    def table(self):  # -> pandas.DataFrame [pop1..pop5, alpha, se, z]
        pd = _require_pandas()
        return pd.DataFrame(
            {
                "pop1": list(self.pop1),
                "pop2": list(self.pop2),
                "pop3": list(self.pop3),
                "pop4": list(self.pop4),
                "pop5": list(self.pop5),
                "alpha": list(self.alpha),
                "se": list(self.se),
                "z": list(self.z),
            }
        )

    def __len__(self) -> int:
        return len(self.alpha)

    def __repr__(self) -> str:
        return f"F4RatioResult(n_tuples={len(self.alpha)}, status={self.status.name})"


class F2Blocks:
    """An opaque f2-dir handle: the host f2 tensor + the P pop labels (P-axis order).
    Wraps the compiled ``_core.F2Handle``; build_resources is cached on the handle so the
    precompute-once / fit-many path reuses one Resources across fits (ADR-0005)."""

    def __init__(self, handle: _core.F2Handle):
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


def extract_f2(
    prefix: Any,
    *,
    pops: list[str],
    out: Optional[str] = None,
    device: int = 0,
    blgsize: float = 0.05,
    maf: float = 0.0,
    maxmiss: float = 0.0,
    autosomes_only: bool = True,
    drop_monomorphic: bool = False,
    transversions_only: bool = False,
    ploidy: str = "auto",
    strand_mode: str = "drop",
    precision: Optional[str] = None,
):
    """Build an f2_blocks tensor from a genotype prefix on the GPU (M(py-2) extract-f2).

    Reads ``<prefix>.{geno,snp,ind}`` directly and runs the SAME decode -> filter ->
    assign_blocks -> tiered f2 compute -> to_host chain the CLI ``extract-f2`` command runs
    (no duplicated compute). ``pops`` is the Explicit population subset (the P axis is that
    selection SORTED ASC by label = ``pops.txt`` order). Defaults match the AT2 ``extract_f2``
    convention so a bare extract reproduces the golden: ``autosomes_only=True`` (AT2
    ``auto_only`` default), ``maxmiss=0`` (the AT2 POPULATION-axis coverage = the global
    intersection, NOT the sample-axis predicate), per-sample ``ploidy="auto"`` (AT2
    ``adjust_pseudohaploid``), ``blgsize=0.05`` Morgans.

    ``strand_mode`` sets the strand-ambiguous (palindromic A/T, C/G) SNP policy:
    ``"drop"`` (default) drops them (merge-safety default; reproduces the frozen
    behavior bit-identically), ``"keep"`` retains them (reproduces ADMIXTOOLS 2's
    default, which keeps ambiguous SNPs — use this for exact-AT2 reproduction on a
    panel that still carries palindromes), ``"flip"`` is a documented
    not-yet-implemented token (currently behaves like ``"keep"``; no frequency-based
    strand reorientation is performed). Multiallelic SNPs are dropped regardless.

    TWO RETURN MODES (capsule/path idiom, NOT a giant disk round-trip):
      * ``out`` is None (default): returns an :class:`F2Blocks` handle wrapping the host f2
        tensor + labels (no disk write); feed it straight to :func:`qpadm` / :func:`f4` etc.
      * ``out`` is a directory path: writes an STPF2BK1 f2-dir there (``f2.bin`` + ``pops.txt``
        + ``meta.json``) and returns the path string (then ``read_f2(out)`` reloads it).

    ``precision`` selects the f2-GEMM arithmetic; tokens match the CLI ``--precision`` set
    (canonical ``emu40|emu32|fp64|tf32`` plus documented aliases): None (default) -> emulated
    FP64 40-bit, the f2 default; ``"emu40"``/``"emu"``/``"emulated_fp64"`` -> emulated FP64
    40-bit; ``"emu32"``/``"emulated_fp64_32"`` -> emulated FP64 32-bit;
    ``"fp64"``/``"native"`` -> native FP64 oracle; ``"tf32"`` -> TF32. GPU-only: no CUDA
    device raises a clear ValueError. An unknown pop name or a missing genotype file raises;
    every-SNP-filtered raises."""
    h = _core.run_extract_f2(
        str(prefix),
        list(pops),
        "" if out is None else str(out),
        device,
        blgsize,
        maf,
        maxmiss,
        autosomes_only,
        drop_monomorphic,
        transversions_only,
        ploidy,
        strand_mode,
        precision,
    )
    # out= None -> a raw _core.F2Handle (wrap it); out= path -> the path string (return it).
    if out is None:
        return F2Blocks(h)
    return h


def qpfstats(
    prefix: Any,
    *,
    pops: list[str],
    out: Optional[str] = None,
    device: int = 0,
    blgsize: float = 0.05,
    precision: Optional[str] = None,
):
    """Genotype-path JOINT f2 SMOOTHER on the GPU (admixtools::qpfstats).

    Reads ``<prefix>.{geno,snp,ind}`` directly, drives the qpDstat-B genotype-f4 numerator
    engine over the FULL f2/f3/f4 population-comb set, runs the on-device shared-factor
    smoothing regression, and returns a SMOOTHED per-block f2 tensor that ``qpadm`` / ``f4``
    consume like any extract-f2 cache. ``pops`` is the smoothing pop set (SORTED ASC internally
    = the AT2 dimnames order = ``pops.txt`` order); ``blgsize`` is MORGANS (AT2 default 0.05).

    TWO RETURN MODES (the same idiom as :func:`extract_f2`):
      * ``out`` is None (default): returns an :class:`F2Blocks` handle wrapping the smoothed f2
        tensor + labels (no disk write); feed it straight to :func:`qpadm` / :func:`f4` etc.
      * ``out`` is a directory path: writes an STPF2BK1 f2-dir there and returns the path
        string (then ``read_f2(out)`` reloads it).

    ``precision`` selects the matmul-substep arithmetic (the smoothing SYRK/GEMM); tokens match
    the CLI ``--precision`` set (canonical ``emu40|emu32|fp64|tf32`` plus documented aliases):
    None (default) -> emulated FP64 40-bit, the fit default; ``"emu40"``/``"emu"``/
    ``"emulated_fp64"`` -> emulated FP64 40-bit; ``"emu32"``/``"emulated_fp64_32"`` -> emulated
    FP64 32-bit; ``"fp64"``/``"native"`` -> native FP64. The Cholesky/solve is native FP64 (the
    cancellation carve-out). GPU-only: no CUDA device raises."""
    h = _core.run_qpfstats(
        str(prefix),
        list(pops),
        "" if out is None else str(out),
        device,
        blgsize,
        precision,
    )
    if out is None:
        return F2Blocks(h)
    return h


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


class QpGraphResult:
    """The single-graph qpGraph fit result (the productized IDEA-1 fleet). ``.score`` is
    the GLS fit score; ``.weights`` the per-admixture-node mixture weights; ``.edges`` the
    fitted drift edge lengths (both as tidy DataFrames when pandas is available)."""

    def __init__(self, d: dict):
        self._d = d

    @property
    def score(self) -> float:
        return float(self._d["score"])

    @property
    def restart_spread(self) -> float:
        return float(self._d["restart_spread"])

    @property
    def status(self) -> Status:
        return Status._from(self._d["status"])

    @property
    def worst_residual_z(self) -> float:
        return float(self._d["worst_residual_z"])

    @property
    def weights(self):  # -> DataFrame [from, to, weight, low, high]
        pd = _require_pandas()
        return pd.DataFrame({
            "from": list(self._d["admix_from"]),
            "to": list(self._d["admix_to"]),
            "weight": list(self._d["weight"]),
            "low": list(self._d["weight_lo"]),
            "high": list(self._d["weight_hi"]),
        })

    @property
    def edges(self):  # -> DataFrame [from, to, length]
        pd = _require_pandas()
        return pd.DataFrame({
            "from": list(self._d["edge_from"]),
            "to": list(self._d["edge_to"]),
            "length": list(self._d["edge_length"]),
        })

    def __repr__(self) -> str:
        return (f"QpGraphResult(score={self.score:.6f}, "
                f"nadmix={len(self._d['weight'])}, "
                f"nedge={len(self._d['edge_length'])}, status={self.status.name})")


def qpgraph(
    f2: F2Blocks,
    edges: list[Any],
    *,
    numstart: int = 10,
    fudge: float = 1e-4,
    diag_f3: float = 1e-5,
    constrained: bool = True,
) -> QpGraphResult:
    """Fit a FIXED admixture graph to the f2 on the GPU (the productized IDEA-1 fleet,
    on-device). ``edges`` is a list of ``(parent, child)`` name pairs (the admixtools edge
    list); the leaves must be f2 populations.

    NOTE: qpGraph uses the AT2 ``afprod=FALSE`` f2 (DIFFERENT from qpadm's ``afprod=TRUE``);
    the ``f2`` handle must come from ``read_f2`` of an afprod=FALSE f2 dir."""
    edge_pairs: list[tuple[str, str]] = []
    for e in edges:
        if isinstance(e, dict):
            edge_pairs.append((str(e["from"]), str(e["to"])))
        else:
            edge_pairs.append((str(e[0]), str(e[1])))
    d = _core.run_qpgraph(f2._h, edge_pairs, numstart, fudge, diag_f3, constrained)
    return QpGraphResult(d)


class QpGraphSearchResult:
    """The qpGraph TOPOLOGY-SEARCH result (oracle C: exhaustive bounded enumeration). The
    search scores EVERY rooted topology on the bounded leaf set (nadmix in {0..max_nadmix})
    in ONE heterogeneous-fleet GPU launch and returns the deterministic GLOBAL-BEST argmin.

    ``.best_score`` is the global-best (min over all candidates) GLS fit score; ``.best_edges``
    the argmin topology (an admixtools edge list); ``.best_hash`` its canonical graph_hash (the
    isomorphism key). ``.n_candidates`` is the exhaustive count scored (== admixtools
    generate_all_graphs). ``.candidates`` is the FULL per-candidate scored vector (one row per
    enumerated topology, in deterministic enumeration order — the same data the argmin reduces
    over), as a tidy DataFrame when pandas is available."""

    def __init__(self, d: dict):
        self._d = d

    @property
    def n_trees(self) -> int:
        return int(self._d["n_trees"])

    @property
    def n_admix1(self) -> int:
        return int(self._d["n_admix1"])

    @property
    def n_candidates(self) -> int:
        return int(self._d["n_candidates"])

    @property
    def best_score(self) -> float:
        return float(self._d["best_score"])

    @property
    def second_best_score(self) -> float:
        return float(self._d["second_best_score"])

    @property
    def best_nadmix(self) -> int:
        return int(self._d["best_nadmix"])

    @property
    def best_hash(self) -> int:
        return int(self._d["best_hash"])

    @property
    def best_edges(self) -> list[tuple[str, str]]:
        return [(str(p), str(c)) for p, c in self._d["best_edges"]]

    @property
    def heuristic_recovered(self) -> bool:
        return bool(self._d["heuristic_recovered"])

    @property
    def fit_all_wall_ms(self) -> float:
        return float(self._d["fit_all_wall_ms"])

    @property
    def topologies_per_s(self) -> float:
        return float(self._d["topologies_per_s"])

    @property
    def argmin(self) -> int:
        """The index of the global-best candidate within ``.candidates`` (the argmin row).
        ``candidates[argmin]`` is the global-best score/hash the search returned; -1 if the
        per-candidate vector was not populated (no candidate scored)."""
        scores = self._d.get("cand_score") or []
        if not scores:
            return -1
        best = scores[0]
        idx = 0
        for i, s in enumerate(scores):
            if s < best:
                best = s
                idx = i
        return idx

    @property
    def candidates(self):  # -> DataFrame [nadmix, hash, score, restart_spread]
        """The FULL per-candidate scored vector (every successfully-fit topology, in the
        deterministic enumeration order: trees then admix1), as a tidy DataFrame with columns
        nadmix, hash, score, restart_spread. The argmin over ``score`` is the global-best."""
        pd = _require_pandas()
        return pd.DataFrame(
            {
                "nadmix": list(self._d.get("cand_nadmix") or []),
                "hash": list(self._d.get("cand_hash") or []),
                "score": list(self._d.get("cand_score") or []),
                "restart_spread": list(self._d.get("cand_restart_spread") or []),
            }
        )

    def __repr__(self) -> str:
        return (
            f"QpGraphSearchResult(n_candidates={self.n_candidates}, "
            f"best_score={self.best_score:.6f}, best_nadmix={self.best_nadmix}, "
            f"heuristic_recovered={self.heuristic_recovered}, "
            f"status={Status.OK.name})"
        )


def qpgraph_search(
    f2: F2Blocks,
    *,
    pops: list[str],
    max_nadmix: int = 1,
    numstart: int = 10,
    fudge: float = 1e-4,
    diag_f3: float = 1e-5,
    constrained: bool = True,
    run_heuristic: bool = True,
) -> QpGraphSearchResult:
    """qpGraph TOPOLOGY SEARCH on the GPU (oracle C: exhaustive bounded enumeration; the
    heterogeneous-topology fleet fits ALL candidates in ONE launch).

    Exhaustively enumerates every rooted topology on the bounded ``pops`` leaf set (nadmix in
    ``{0..max_nadmix}``; this reproduces admixtools ``generate_all_graphs`` 1:1) and returns
    the deterministic GLOBAL-BEST: the argmin over the per-candidate clean single-graph GLS
    fit scores. The host does only the cheap enumeration + the argmin — NEVER a per-candidate
    host fit. ``pops`` is the leaf pop-set (each must be an f2 population; >= 3 required).

    Mirrors :func:`qpadm_search`: the per-candidate FIT options (``numstart`` / ``fudge`` /
    ``diag_f3`` / ``constrained``) match the single-graph :func:`qpgraph` fit so a candidate's
    score reproduces ``qpgraph(best_edges).score``. ``run_heuristic`` additionally runs the
    hill-climb and verifies it recovers the exhaustive global-best (the falsifiable v1 gate).

    NOTE: qpGraph uses the AT2 ``afprod=FALSE`` f2 (DIFFERENT from qpadm's ``afprod=TRUE``);
    the ``f2`` handle must come from ``read_f2`` of an afprod=FALSE f2 dir.

    Returns a :class:`QpGraphSearchResult` — ``.best_score`` / ``.best_edges`` / ``.best_hash``
    are the global-best; ``.n_candidates`` the exhaustive count; ``.candidates`` the full
    per-candidate scored vector; ``.argmin`` the global-best row in it. GPU-only: no CUDA
    device raises a clear ValueError; an unknown pop name raises."""
    d = _core.run_qpgraph_search(
        f2._h,
        list(pops),
        max_nadmix,
        numstart,
        fudge,
        diag_f3,
        constrained,
        run_heuristic,
    )
    return QpGraphSearchResult(d)


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


def qpdstat(
    f2: F2Blocks,
    quartets: list[Any],
    *,
    as_dataframe: bool = False,
):
    """D-statistic / f4 over the f2-data path on the GPU — est/se/z/p per quartet (qpDstat
    Part A). The ``--f2-dir`` qpdstat path reports f4 (the AT2 f2-path convention:
    ``admixtools::qpdstat(f2dir, f4mode=TRUE)`` is byte-identical to ``f4mode=FALSE`` and to
    ``f4``, since f4mode is a no-op without per-SNP genotypes), where z = est/se and
    p = 2*(1-Phi(|z|)) ARE the AT2 D-stat sign/Z/p convention. NO new compute and NO new
    result type — it returns an ``F4Result`` (or, when ``as_dataframe=True``, the tidy
    ``F4Result.table`` DataFrame: pop1,pop2,pop3,pop4,est,se,z,p).

    ``quartets`` is a list where each entry is a ``(p1, p2, p3, p4)`` name tuple/list (or a
    ``{"pop1":..,"pop2":..,"pop3":..,"pop4":..}`` dict) = the QUADRUPLE input. The
    normalized-D MAGNITUDE needs a genotype prefix — use ``steppe.dstat`` (the genotype-path
    normalized-D, Part B; or the CLI ``qpdstat --prefix PREFIX.{geno,snp,ind}``). Unknown pop
    names raise a clean KeyError."""
    quads: list[tuple[str, str, str, str]] = []
    for q in quartets:
        if isinstance(q, dict):
            quads.append((q["pop1"], q["pop2"], q["pop3"], q["pop4"]))
        else:
            p1, p2, p3, p4 = q  # exactly 4 names (raises on a malformed tuple)
            quads.append((p1, p2, p3, p4))

    d = _core.run_qpdstat(f2._h, quads)
    res = F4Result(d)
    return res.table if as_dataframe else res


def dstat(
    prefix: Any,
    quartets: list[Any],
    *,
    blgsize: float = 0.05,
    device: int = 0,
    as_dataframe: bool = False,
):
    """Genotype-path NORMALIZED-D on the GPU — est/se/z/p per quadruple (qpDstat Part B).

    UNLIKE :func:`qpdstat` (which reads an f2-dir and reports f4), this reads the GENOTYPE
    TRIPLE ``<prefix>.{geno,snp,ind}`` directly: D = mean_snp(num)/mean_snp(den),
    num=(a-b)(c-d), den=(a+b-2ab)(c+d-2cd) over per-SNP allele frequencies, block-jackknifed
    (the AT2 ``qpdstat_geno`` allsnps=TRUE / f4mode=FALSE convention). Forced diploid +
    autosomes-only + allsnps=TRUE are pinned (the AT2 genotype-path parity). The est is the
    NORMALIZED D (~±0.06), distinct from the f2-path f4 (~10x smaller); z = est/se and
    p = 2*(1-Phi(|z|)) ARE the AT2 D sign/Z/p convention.

    ``prefix`` is the genotype prefix (``<prefix>.geno/.snp/.ind`` must exist). ``quartets``
    is a list where each entry is a ``(p1, p2, p3, p4)`` name tuple/list (or a
    ``{"pop1":..,"pop2":..,"pop3":..,"pop4":..}`` dict). ``blgsize`` is the jackknife block
    size in MORGANS (AT2 default 0.05). Returns an ``F4Result`` (or, when
    ``as_dataframe=True``, the tidy DataFrame: pop1,pop2,pop3,pop4,est,se,z,p). Unknown pop
    names raise a clean KeyError; a missing genotype file raises a ValueError."""
    quads: list[tuple[str, str, str, str]] = []
    for q in quartets:
        if isinstance(q, dict):
            quads.append((q["pop1"], q["pop2"], q["pop3"], q["pop4"]))
        else:
            p1, p2, p3, p4 = q  # exactly 4 names (raises on a malformed tuple)
            quads.append((p1, p2, p3, p4))

    d = _core.run_dstat(str(prefix), quads, blgsize, device)
    res = F4Result(d)
    return res.table if as_dataframe else res


class DatesResult:
    """A single DATES admixture-dating result (the time since admixture, in generations).
    Built from the flat dict ``_core.run_dates`` returns — the lone Python facade outlier,
    now a typed sibling of QpAdmResult/F4Result et al. ``date_gen`` is the generations since
    admixture, ``se`` the leave-one-chromosome block-jackknife standard error, and
    ``curve_cm``/``curve_corr`` the binned covariance-decay curve (cM vs the normalized
    correlation; ``.curve`` is the tidy two-column DataFrame view). The underlying dict stays
    reachable via ``.raw`` and dict-style ``res["date_gen"]`` indexing for back-compat."""

    def __init__(self, d: dict):
        self._d = d
        self.target: str = d["target"]
        self.source1: str = d["source1"]
        self.source2: str = d["source2"]
        self.date_gen: float = d["date_gen"]
        self.se: float = d["se"]
        self.fit_error_sd: float = d["fit_error_sd"]
        # The compiled layer emits a coarse "ok"/"error" string here (NOT the full Status
        # taxonomy), so keep it a plain str — Status("error") has no enum member.
        self.status: str = d["status"]
        self.curve_cm: list[float] = list(d["curve_cm"])
        self.curve_corr: list[float] = list(d["curve_corr"])

    @property
    def raw(self) -> dict:
        """The underlying ``_core.run_dates`` dict (back-compat escape hatch)."""
        return self._d

    @property
    def curve(self):  # -> pandas.DataFrame [cm, corr]
        """The binned covariance-decay curve as a tidy DataFrame: cM vs the normalized
        correlation (the ``curve_cm``/``curve_corr`` parallel arrays)."""
        pd = _require_pandas()
        return pd.DataFrame({"cm": list(self.curve_cm), "corr": list(self.curve_corr)})

    # --- dict back-compat: nothing that indexed the old dict (res["date_gen"]) breaks. ---
    def __getitem__(self, key: str):
        return self._d[key]

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def get(self, key: str, default=None):
        return self._d.get(key, default)

    def keys(self):
        return self._d.keys()

    def __repr__(self) -> str:
        return (
            f"DatesResult(target={self.target!r}, source1={self.source1!r}, "
            f"source2={self.source2!r}, date_gen={self.date_gen!r}, se={self.se!r}, "
            f"status={self.status!r})"
        )


def dates(
    prefix: Any,
    target: str,
    source1: str,
    source2: str,
    *,
    device: int = 0,
) -> DatesResult:
    """Admixture DATING on the GPU — the DATES tool (the time since admixture, in generations).

    Reads the GENOTYPE TRIPLE ``<prefix>.{geno,snp,ind}`` directly (NOT the f2 cache) and
    infers the date from the weighted ancestry-covariance decay vs genetic distance: the
    covariance between SNP pairs decays as ``A*exp(-lambda*d)+c`` with the genetic distance d,
    and ``lambda`` is the generations since admixture. The covariance curve is computed with
    the ALDER FFT reformulation (the cuFFT autocorrelation LD engine) — NO host O(M^2)
    SNP-pair loop. ``target`` is the admixed population; ``source1``/``source2`` are the two
    reference sources (the weight is ``freq(source1) - freq(source2)``). The ``.snp`` MUST
    carry a real cM genetic map.

    Returns a :class:`DatesResult` with typed attributes ``target``/``source1``/``source2``,
    ``date_gen`` (the generations since admixture), ``se`` (the leave-one-chromosome
    block-jackknife standard error), ``fit_error_sd``, ``status``, and the
    ``curve_cm``/``curve_corr`` binned covariance-decay curve (cM vs the normalized
    correlation; ``.curve`` is the tidy DataFrame view). The original dict stays reachable via
    ``res.raw`` and dict-style ``res["date_gen"]`` indexing. A missing genotype file raises a
    ValueError."""
    return DatesResult(_core.run_dates(str(prefix), target, source1, source2, device))


def f3(
    f2: F2Blocks,
    triples: list[Any],
    *,
    as_dataframe: bool = False,
):
    """Standalone f3(C;A,B) statistic on the GPU — est/se/z/p per triple (NO ALS / NO rank;
    the AT2 weighted block-jackknife f3 + the jackknife-diagonal SE). The THREE-tuple clone
    of f4.

    ``triples`` is a list where each entry is a ``(C, A, B)`` name tuple/list (or a
    ``{"pop1":..,"pop2":..,"pop3":..}`` dict; pop1=C is the apex/outgroup/target). Returns an
    ``F3Result`` (or, when ``as_dataframe=True``, the tidy ``F3Result.table`` DataFrame:
    pop1,pop2,pop3,est,se,z,p). Outgroup-f3 = f3(Outgroup;A,B) (shared drift); admixture-f3 =
    f3(Target;Src1,Src2) (negative ⇒ admixture). Unknown pop names raise a clean KeyError."""
    trips: list[tuple[str, str, str]] = []
    for t in triples:
        if isinstance(t, dict):
            trips.append((t["pop1"], t["pop2"], t["pop3"]))
        else:
            c, a, b = t  # exactly 3 names (raises on a malformed tuple)
            trips.append((c, a, b))

    d = _core.run_f3(f2._h, trips)
    res = F3Result(d)
    return res.table if as_dataframe else res


def f4ratio(
    f2: F2Blocks,
    tuples: list[Any],
    *,
    as_dataframe: bool = False,
):
    """Standalone f4-ratio admixture proportion on the GPU — alpha/se/z per 5-tuple (NO ALS /
    NO rank; the AT2 qpf4ratio weighted block-jackknife OF THE RATIO). The FIVE-tuple sibling
    of f4/f3.

    alpha = f4(p1,p2;p3,p4) / f4(p1,p2;p5,p4): the shared pops across num/den are p1,p2,p4;
    only the 3rd slot swaps (p3 in the numerator, p5 in the denominator).

    ``tuples`` is a list where each entry is a ``(p1, p2, p3, p4, p5)`` name tuple/list (or a
    ``{"pop1":..,"pop2":..,"pop3":..,"pop4":..,"pop5":..}`` dict). Returns an ``F4RatioResult``
    (or, when ``as_dataframe=True``, the tidy ``F4RatioResult.table`` DataFrame:
    pop1,pop2,pop3,pop4,pop5,alpha,se,z). Unknown pop names raise a clean KeyError."""
    quins: list[tuple[str, str, str, str, str]] = []
    for t in tuples:
        if isinstance(t, dict):
            quins.append((t["pop1"], t["pop2"], t["pop3"], t["pop4"], t["pop5"]))
        else:
            p1, p2, p3, p4, p5 = t  # exactly 5 names (raises on a malformed tuple)
            quins.append((p1, p2, p3, p4, p5))

    d = _core.run_f4ratio(f2._h, quins)
    res = F4RatioResult(d)
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


# =======================================================================================
# ADMIXTOOLS 2 <-> steppe f2-cache converter (optional; pure-Python, NO GPU / NO CUDA).
#
# EXPORT a steppe STPF2BK1 f2 cache to an AT2 read_f2() `.rds` directory so a user can
# verify steppe's fits inside ADMIXTOOLS 2 / R; IMPORT the reverse. This is on-disk FORMAT
# TRANSLATION only — steppe's f2 values + jackknife-block partition are already identical to
# AT2's on the same blgsize/setblocks partition (see docs/featuredesign/rds-converter.md).
# WRITING AT2's numeric-matrix-with-dimnames `.rds` is unsupported by librdata/pyreadr, so
# export uses the hand-rolled stdlib serializer in `_rds`; import reads via pyreadr (the
# optional `[rds]` extra — `pip install steppe[rds]`). These are THIN wrappers: the whole
# implementation lives in `_rds` (which has ZERO `_core` dependency), so the converter is
# importable + testable without a CUDA build.
# =======================================================================================


def export_f2_rds(f2, out_dir, *, counts="ones", write_ap=False):
    """Export a steppe f2 cache to an ADMIXTOOLS 2 ``read_f2()`` ``.rds`` directory.

    The stated goal of the converter: after this call, ``admixtools::read_f2(out_dir)`` in R
    loads a ``[P, P, n_block]`` array whose off-diagonal equals steppe's f2, so a user can
    re-run ``qpadm`` / ``qpgraph`` / ``f4`` inside ADMIXTOOLS 2 and compare to steppe's native
    fit. Pure on-disk translation — NO GPU, NO CUDA, and no third-party dependency (the
    hand-rolled RDS serializer is stdlib-only).

    ``f2`` is an :class:`F2Blocks` handle (from :func:`read_f2` / :func:`extract_f2`). The
    output layout AT2 expects: one subdir **per pop**; each unordered pair
    ``<p1>/<p2>_f2.rds`` (keyed under the C-locale-smaller ``p1``) is a gzip'd R numeric
    matrix ``[n_block, 2]`` with ``colnames c("f2","counts")`` (col 1 = f2); the diagonal
    self-pairs are all ``0.0`` (AT2's convention — f2 is off-diagonal-only for f4/qpAdm); plus
    a top-level ``block_lengths_f2.rds`` integer vector of the per-block SNP counts.

    ``counts`` selects the second matrix column: ``"ones"`` (default) writes ``1.0`` — AT2's
    own f2 family stores ``1.0`` there, so this reproduces a native AT2 cache; ``"vpair"``
    writes steppe's per-block pairwise-valid counts instead (only meaningful if the handle
    carries real ``vpair`` — a ``.bin``-staged cache carries zeros). ``write_ap`` (emit the
    parallel ``_ap`` allele-pair family so true ``vpair`` survives a round-trip) is not yet
    implemented; the ``f2`` family alone is sufficient for qpAdm / qpGraph / f4. Returns the
    output directory path."""
    from . import _rds  # lazy: keeps `import steppe` free of the converter machinery

    return _rds.export_f2_rds(f2, out_dir, counts=counts, write_ap=write_ap)


def import_f2_rds(rds_dir, out_dir, *, type="f2"):
    """Import an ADMIXTOOLS 2 ``read_f2()`` ``.rds`` directory to a steppe STPF2BK1 f2 cache.

    Reads AT2's per-pair matrices with ``pyreadr`` (the optional ``[rds]`` extra —
    ``pip install steppe[rds]``) and writes ``<out_dir>/{f2.bin,pops.txt,meta.json}`` that
    :func:`read_f2` reloads. Pops are the immediate subdirs, sorted with Python ``sorted()``
    (byte order == the C-locale AT2 dimnames order) as the P-axis; each unordered pair is read
    from ``<p1>/<p2>_f2.rds`` (col 1 = f2) and mirrored into ``f2[i,j]`` == ``f2[j,i]``; the
    diagonal is zeroed (AT2's self-pair convention).

    IMPORT is ``vpair``-lossy by design: AT2's f2 ``counts`` column is ``1.0``, not real
    pairwise-valid counts, so ``vpair`` cannot be recovered from an f2-only ``.rds``. The
    writer fills ``vpair`` with a NONZERO sentinel (the ``block_lengths`` broadcast), never
    zeros — zeros would trip steppe/AT2's ``vpair==0`` missing-block detector and break
    ``maxmiss>0``. Off-diagonal f2 (all qpAdm/qpGraph/f4 need) is exact. ``type`` must be
    ``"f2"`` (the ``_ap``/``_fst`` families are not imported). Returns the output path."""
    from . import _rds  # lazy: pulls pyreadr only on the import path

    return _rds.import_f2_rds(rds_dir, out_dir, type=type)


def list_caches(root=None, *, use_index=False):
    """List the STPF2BK1 f2 caches under ``root`` (default: ``$STEPPE_F2_DIR`` or cwd).

    Returns one lightweight dict per cache (``path``, ``P``, ``n_block``, ``precision_tag``,
    ``cache_id``, ``bytes``, …), read header-only so a tree of multi-GB caches is cheap to
    scan. GPU-free — the whole implementation is stdlib in :mod:`steppe._cache`. ``use_index``
    memoizes the scan in ``~/.cache/steppe`` (invalidated by f2.bin mtime + size)."""
    from . import _cache  # lazy: keeps `import steppe` free of the manager machinery

    return _cache.list_caches(root, use_index=use_index)


def cache_info(f2_dir):
    """The full parsed record for one f2 cache: header-derived ``P``/``n_block``, the on-disk
    vs expected f2.bin size, and the raw ``meta.json`` provenance. GPU-free."""
    from . import _cache

    return _cache.cache_info(f2_dir)


def verify_cache(f2_dir, *, check_sources=False):
    """True iff the cache's ``f2.bin`` payload + ``pops.txt`` labels re-hash to the content-
    address the extract writer stamped (``f2_cache_id`` / ``pops_sha256``); a cache carrying
    no stored id is not a failure. Does not attest the ``meta.json`` record. GPU-free."""
    from . import _cache

    return _cache.verify_cache(f2_dir, check_sources=check_sources)
