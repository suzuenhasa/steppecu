#!/usr/bin/env python3
"""QUICK-START: read_f2 -> qpadm -> inspect, through the steppe Python facade.

A living API canary for the Python surface. The whole fit is three facade calls::

    import steppe
    f2  = steppe.read_f2(f2_dir, device=0)
    res = steppe.qpadm(f2, target=TARGET, left=LEFT, right=RIGHT)
    print(res.weights); print(f"p={res.p}")

``res.weights`` is a pandas DataFrame [target, left, weight, se, z]; ``res.p`` is the
model tail-p (a float). steppe is a GPU product: this runs on a CUDA box only (the fit
raises a clear "no CUDA device" message off-GPU). NO synthetic data — it points at the
committed real-AADR golden_fit0 f2 fixture.

USAGE
  # zero-arg default: stage the committed golden_fit0 fixture into a temp f2-dir and fit
  python examples/python/quickstart_qpadm.py

  # point at an existing f2-dir (e.g. the box's real /workspace/data/aadr/f2_fit0_corrected)
  python examples/python/quickstart_qpadm.py /path/to/f2-dir

  # ONLY stage the committed fixture into <dir> (for the C++ quickstart) and exit
  python examples/python/quickstart_qpadm.py --stage /path/to/out-dir

Expected output for the staged golden_fit0 fixture (the f2-OBJECT-path values; see
examples/README.md to self-check): weight ~= [0.868755, 0.131245], p ~= 0.4119.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

# ---- The real-AADR golden_fit0 9-pop model (golden_fit0.json metadata, index order) ----
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

# repo root = examples/python/quickstart_qpadm.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests" / "reference" / "goldens" / "at2" / "fixtures" / "f2_fit0_9pop.bin"

# STPF2BK1 on-disk constants (src/device/f2_disk_format.hpp; mirrors tests/python/conftest.py).
_F2_MAGIC = b"STPF2BK1"
_F2_VERSION = 1
_F2_DTYPE_FP64 = 1
_F2_HEADER_SIZE = 64


def _put_bindings_on_path() -> None:
    """Make ``import steppe`` work from a build tree without installing the wheel.

    Honors STEPPE_BINDINGS_DIR (the CTest wiring), else searches the usual build dirs for
    a built ``steppe`` package (``_core*.so`` next to ``__init__.py``)."""
    candidates = []
    env = os.environ.get("STEPPE_BINDINGS_DIR")
    if env:
        candidates.append(Path(env))
    for build in ("build-rel", "build", "build-ci"):
        candidates.append(_REPO_ROOT / build / "bindings")
    candidates.append(_REPO_ROOT / "bindings")
    for c in candidates:
        pkg = Path(c) / "steppe"
        if pkg.is_dir() and any(pkg.glob("_core*.so")):
            p = str(Path(c).resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
            return


def _read_raw_fixture(path: Path):
    """Raw fixture layout (NO vpair): int32 P, int32 nb, int32[nb] block_sizes,
    f64[P*P*nb] f2 (column-major i + P*j + P*P*b). Returns (P, nb, block_sizes, f2)."""
    with open(path, "rb") as f:
        P, nb = struct.unpack("<ii", f.read(8))
        if P <= 0 or nb <= 0:
            raise ValueError(f"bad fixture header P={P} nb={nb}")
        block_sizes = list(struct.unpack(f"<{nb}i", f.read(4 * nb)))
        n = P * P * nb
        f2 = list(struct.unpack(f"<{n}d", f.read(8 * n)))
    return P, nb, block_sizes, f2


def stage_f2_dir(out_dir: Path) -> Path:
    """Stage the committed golden_fit0 .bin fixture into an STPF2BK1 f2-dir (f2.bin +
    pops.txt + meta.json) at ``out_dir`` — the same recipe tests/python/conftest.py uses.
    The staged dir reproduces the f2-OBJECT-path golden values (read_f2(dir) of the binary
    fixture). Returns ``out_dir``."""
    if not _FIXTURE.is_file():
        raise FileNotFoundError(f"committed golden fixture absent: {_FIXTURE}")
    P, nb, block_sizes, f2 = _read_raw_fixture(_FIXTURE)
    if P != len(POPS_9):
        raise ValueError(f"fixture P={P} but {len(POPS_9)} pop labels")

    out_dir.mkdir(parents=True, exist_ok=True)
    slab_bytes = P * P * nb * 8
    f2_offset = _F2_HEADER_SIZE
    vpair_offset = f2_offset + slab_bytes
    block_sizes_offset = vpair_offset + slab_bytes

    # 64-byte header: magic[8] version:u32 dtype:u32 P:i32 n_block:i32
    #   f2_offset:u64 vpair_offset:u64 block_sizes_offset:u64 reserved[16].
    header = struct.pack(
        "<8sIIiiQQQ16s",
        _F2_MAGIC, _F2_VERSION, _F2_DTYPE_FP64, P, nb,
        f2_offset, vpair_offset, block_sizes_offset, b"\x00" * 16,
    )
    assert len(header) == _F2_HEADER_SIZE, len(header)

    with open(out_dir / "f2.bin", "wb") as o:
        o.write(header)
        o.write(struct.pack(f"<{P * P * nb}d", *f2))
        o.write(b"\x00" * slab_bytes)  # vpair zeros (the fit reads block_sizes, not vpair)
        o.write(struct.pack(f"<{nb}i", *block_sizes))
    with open(out_dir / "pops.txt", "w") as p:
        for s in POPS_9:
            p.write(s + "\n")
    with open(out_dir / "meta.json", "w") as m:
        m.write(
            '{\n  "format": "STPF2BK1",\n  "P": %d,\n  "n_block": %d,\n'
            '  "provenance": "examples/quickstart (committed real-AADR golden_fit0 f2)"\n}\n'
            % (P, nb)
        )
    return out_dir


def run_fit(f2_dir: str) -> int:
    """The three-call facade fit: read_f2 -> qpadm -> print(weights) + print(p)."""
    _put_bindings_on_path()
    try:
        import steppe
    except Exception as exc:  # bindings absent
        print(
            "steppe not importable — install the wheel (`pip install steppe`) or build from "
            f"source with -DSTEPPE_BUILD_PYTHON=ON: {exc}",
            file=sys.stderr,
        )
        return 1

    f2 = steppe.read_f2(f2_dir, device=0)
    try:
        res = steppe.qpadm(f2, target=TARGET, left=LEFT, right=RIGHT)
    except Exception as exc:  # the GPU-product no-device fault surfaces here
        print(f"qpadm failed: {exc}", file=sys.stderr)
        return 1

    print(f"qpadm  target={TARGET}  f2-dir={f2_dir}  status={res.status.name}")
    try:
        print(res.weights.to_string(index=False))  # DataFrame [target, left, weight, se, z]
    except ImportError:
        # pandas is a soft dependency; fall back to the raw lists so the canary still runs.
        for name, w in zip(LEFT, res.weight):
            print(f"  {name:<26} {w:12.6f}")
    print(f"p={res.p}  chisq={res.chisq}  dof={res.dof}  f4rank={res.f4rank}")
    return 0


def main(argv: list[str]) -> int:
    args = argv[1:]
    if args and args[0] == "--stage":
        if len(args) != 2:
            print("usage: quickstart_qpadm.py --stage <out-dir>", file=sys.stderr)
            return 2
        out = stage_f2_dir(Path(args[1]))
        print(f"staged golden_fit0 f2-dir -> {out}")
        return 0

    if args:  # an explicit f2-dir to point at (e.g. a real extract-f2 dir on the box)
        return run_fit(args[0])

    # zero-arg default: stage the committed golden fixture into a temp dir and fit it.
    with tempfile.TemporaryDirectory(prefix="steppe_fit0_") as tmp:
        d = stage_f2_dir(Path(tmp) / "f2dir")
        return run_fit(str(d))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
