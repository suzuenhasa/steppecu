"""Pytest fixtures for the M(py-1) nanobind gate.

Provides: the goldens/fixtures path fixtures, a raw-.bin -> STPF2BK1 f2-dir staging helper
(the same recipe test_cli_qpadm.cpp uses), and a device-present SKIP guard (every steppe
reference test SKIPs cleanly when no CUDA device is visible; cli-bindings.md §7). The built
``steppe`` package is put on sys.path by the CTest env (STEPPE_BINDINGS_DIR), or located
relative to the build tree as a fallback so a bare ``pytest tests/python`` from a build
also works.
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDENS = _REPO_ROOT / "tests" / "reference" / "goldens" / "at2"
_FIXTURES = _GOLDENS / "fixtures"


def _put_bindings_on_path() -> None:
    """Add the directory CONTAINING the built ``steppe`` package to sys.path.

    Priority: STEPPE_BINDINGS_DIR (set by the CTest wiring), else search the usual build
    dirs for ``bindings/steppe`` with a compiled ``_core*.so`` next to ``__init__.py``."""
    env = os.environ.get("STEPPE_BINDINGS_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    for build in ("build-rel", "build", "build-ci"):
        candidates.append(_REPO_ROOT / build / "bindings")
    # The source-tree package dir (its __init__.py) — used once _core*.so is copied in.
    candidates.append(_REPO_ROOT / "bindings")

    for c in candidates:
        pkg = Path(c) / "steppe"
        if pkg.is_dir() and any(pkg.glob("_core*.so")):
            p = str(Path(c).resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
            return
    # Last resort: let the import fail with a clear message in the test.


_put_bindings_on_path()


@pytest.fixture(scope="session")
def steppe_mod():
    """Import the built steppe bindings, or SKIP if unbuilt."""
    try:
        import steppe  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - build dependent
        pytest.skip(f"steppe bindings not importable (build with -DSTEPPE_BUILD_PYTHON=ON): {exc}")
    # `import steppe` now succeeds even without the CUDA `_core` (the GPU-free .rds converter
    # surface stays usable); this fixture promises a BUILT steppe, so skip when _core didn't load.
    if not getattr(steppe, "_CORE_AVAILABLE", True):
        pytest.skip("steppe._core (the CUDA-13 extension) not built/loadable")
    return steppe


@pytest.fixture(scope="session")
def goldens_dir() -> Path:
    if not _GOLDENS.is_dir():
        pytest.skip(f"goldens dir absent: {_GOLDENS}")
    return _GOLDENS


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    if not _FIXTURES.is_dir():
        pytest.skip(f"fixtures dir absent: {_FIXTURES}")
    return _FIXTURES


# ---- raw .bin reader + STPF2BK1 writer (mirror test_cli_qpadm.cpp:66-145) ------------
# Raw fixture layout (NO vpair): int32 P, int32 nb, int32[nb] block_sizes, f64[P*P*nb] f2
# (column-major i + P*j + P*P*b). The STPF2BK1 on-disk dir: a 64-byte header, then the f2
# region, then the vpair region (zeros — the fit reads block_sizes, not vpair), then the
# block_sizes int32 trailer (f2_disk_format.hpp).

_F2_MAGIC = b"STPF2BK1"
_F2_VERSION = 1
_F2_DTYPE_FP64 = 1
_F2_HEADER_SIZE = 64


def read_raw_fixture(path: Path):
    """Return (P, nb, block_sizes:list[int], f2:list[float] flat col-major)."""
    with open(path, "rb") as f:
        P, nb = struct.unpack("<ii", f.read(8))
        if P <= 0 or nb <= 0:
            raise ValueError(f"bad fixture header P={P} nb={nb}")
        block_sizes = list(struct.unpack(f"<{nb}i", f.read(4 * nb)))
        n = P * P * nb
        f2 = list(struct.unpack(f"<{n}d", f.read(8 * n)))
    return P, nb, block_sizes, f2


def write_f2_dir(out_dir: Path, P: int, nb: int, block_sizes, f2, pops):
    """Write an STPF2BK1 f2-dir (f2.bin + pops.txt + meta.json) from raw fields."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slab_bytes = P * P * nb * 8
    f2_offset = _F2_HEADER_SIZE
    vpair_offset = f2_offset + slab_bytes
    block_sizes_offset = vpair_offset + slab_bytes

    # 64-byte header: magic[8] version:u32 dtype:u32 P:i32 n_block:i32
    #   f2_offset:u64 vpair_offset:u64 block_sizes_offset:u64 reserved[16]
    header = struct.pack(
        "<8sIIiiQQQ16s",
        _F2_MAGIC,
        _F2_VERSION,
        _F2_DTYPE_FP64,
        P,
        nb,
        f2_offset,
        vpair_offset,
        block_sizes_offset,
        b"\x00" * 16,
    )
    assert len(header) == _F2_HEADER_SIZE, len(header)

    with open(out_dir / "f2.bin", "wb") as o:
        o.write(header)
        o.write(struct.pack(f"<{P * P * nb}d", *f2))
        o.write(b"\x00" * slab_bytes)  # vpair zeros
        o.write(struct.pack(f"<{nb}i", *block_sizes))

    with open(out_dir / "pops.txt", "w") as p:
        for s in pops:
            p.write(s + "\n")

    with open(out_dir / "meta.json", "w") as m:
        m.write(
            '{\n  "format": "STPF2BK1",\n  "P": %d,\n  "n_block": %d,\n'
            '  "provenance": "py test fixture (real-AADR golden f2)"\n}\n' % (P, nb)
        )


@pytest.fixture
def stage_f2_dir(tmp_path, fixtures_dir):
    """Factory: stage a committed raw .bin fixture + pop labels into a temp f2-dir,
    returning its path. The staged dir reproduces the f2-OBJECT-path golden values
    (read_f2(dir) of the binary fixture), exactly like the C++ CLI gate."""

    def _stage(fixture_name: str, pops: list[str], subdir: str = "f2dir") -> Path:
        P, nb, block_sizes, f2 = read_raw_fixture(fixtures_dir / fixture_name)
        if len(pops) != P:
            raise ValueError(f"{fixture_name}: {len(pops)} pops but P={P}")
        out = tmp_path / subdir
        write_f2_dir(out, P, nb, block_sizes, f2, pops)
        return out

    return _stage


def maybe_skip_no_gpu(exc: Exception) -> None:
    """SKIP cleanly when a fit raised because no CUDA device is visible (the GPU-product
    no-device fault), re-raise anything else."""
    msg = str(exc).lower()
    if "no cuda device" in msg or "cuda" in msg and "device" in msg:
        pytest.skip(f"no CUDA device available: {exc}")
    raise exc
