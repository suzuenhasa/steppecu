"""Hand-rolled R ``.rds`` (de)serializer for the ADMIXTOOLS 2 ⇄ steppe f2 converter.

WHY this exists: ADMIXTOOLS 2's ``read_f2()`` reads each unordered pop-pair as an R
numeric **matrix** ``[n_block, 2]`` with ``dim``/``dimnames`` attributes (col 1 = f2, col 2
= counts), gzip-compressed. Every off-the-shelf writer (``librdata``, ``pyreadr``) can only
emit a column-oriented R **data.frame** — none can write a matrix-with-dimnames, and
``librdata`` has no gzip writer at all. AT2's target shape is tiny and fully specified, so
this module writes it exactly with pure stdlib (``struct`` big-endian XDR + ``gzip``).

The output is BYTE-IDENTICAL to what ``base::saveRDS(m, f, compress="gzip", version=2)``
produces in R 4.3.3 for the same matrix (verified against a reference generated on box5090
— R 4.3.3 + admixtools 2.0.10). XDR + gzip are lossless, so f2 values are bit-exact.

RDS XDR format v2 layout (all integers 4-byte big-endian, doubles 8-byte big-endian IEEE754):

    "X\n"                      the XDR-format marker
    int   2                    serialization format version (v2)
    int   0x00040303           writer R version (4.3.3 — informational; readRDS ignores it)
    int   0x00020300           minimum reader R version (2.3.0 — v2 reads on any modern R)
    <object>

Each object begins with a packed *flags* int: ``type | (levels<<12) | is_object<<8 |
has_attr<<9 | has_tag<<10``. A REALSXP with attributes then serializes: length, the doubles,
then its ATTRIB pairlist (``dim`` INTSXP[2], ``dimnames`` VECSXP[NULL, STRSXP[colnames]]),
terminated by the NILVALUE pseudo-object (0x000000fe).

Reading uses ``pyreadr`` (librdata) — lazily imported with a clear
``pip install steppe[rds]`` message when it is absent (reading is the only side that needs a
third-party dep; writing is stdlib-only).
"""
from __future__ import annotations

import gzip
import json
import os
import struct
from collections.abc import Sequence
from typing import Optional

# --- R SEXPTYPE codes (Rinternals.h) ---------------------------------------------------
_NILSXP = 0
_SYMSXP = 1
_LISTSXP = 2      # a pairlist (CONS cell) — how ATTRIB is serialized
_CHARSXP = 9      # a single string's characters
_INTSXP = 13
_REALSXP = 14
_STRSXP = 16      # a character vector (of CHARSXPs)
_VECSXP = 19      # a generic list (dimnames is a VECSXP)
_NILVALUE_SXP = 254  # the 0x000000fe pseudo-object == R's NULL in the stream

# --- packed-flag bits (serialize.c PackFlags) ------------------------------------------
_HAS_ATTR = 1 << 9   # 0x200
_HAS_TAG = 1 << 10   # 0x400
# CHARSXP "levels" carry the encoding; ASCII_MASK == 1<<6 == 64, stored in bits 12+ .
_ASCII_LEVELS = 64
_CHARSXP_FLAGS = _CHARSXP | (_ASCII_LEVELS << 12)  # 0x00040009 — matches R for ASCII

# --- RDS v2 header fields (match base::saveRDS(version=2) in R 4.3.3, byte-for-byte) ----
_FMT_VERSION = 2
_WRITER_VERSION = 0x00040303   # R 4.3.3 packed (major*65536 + minor*256 + patch)
_MIN_READER_VERSION = 0x00020300  # R 2.3.0 — the floor for format v2


def _i32(n: int) -> bytes:
    """A 4-byte big-endian (XDR) signed int."""
    return struct.pack(">i", int(n))


def _f64(x: float) -> bytes:
    """An 8-byte big-endian IEEE-754 double (lossless; preserves NaN/Inf bit patterns)."""
    return struct.pack(">d", float(x))


def _rds_header() -> bytes:
    """The fixed 15-byte RDS-XDR-v2 preamble (``X\\n`` + the three version ints)."""
    return b"X\n" + _i32(_FMT_VERSION) + _i32(_WRITER_VERSION) + _i32(_MIN_READER_VERSION)


def _charsxp(s: str) -> bytes:
    """A CHARSXP: the ASCII-flagged type word, the length, then the raw bytes.

    Every string this module writes (``dim``, ``dimnames``, ``f2``, ``counts``) is ASCII, so
    the ASCII encoding flag is always correct (and byte-identical to R's own output)."""
    raw = s.encode("ascii")
    return _i32(_CHARSXP_FLAGS) + _i32(len(raw)) + raw


def _attr_tag(name: str) -> bytes:
    """Open an ATTRIB pairlist cell: ``LISTSXP|HAS_TAG`` then the SYMSXP tag ``name``.

    A symbol serializes as the SYMSXP type word followed by its printname CHARSXP. Writing a
    fresh SYMSXP each time (never a REFSXP) is valid: R's reader installs and ref-tables each
    one as it reads, and we simply never emit a back-reference to it."""
    return _i32(_LISTSXP | _HAS_TAG) + _i32(_SYMSXP) + _charsxp(name)


def _gzip_write(path: str, payload: bytes) -> None:
    """gzip ``payload`` to ``path`` with a fixed mtime so repeat exports are byte-stable.

    readRDS / pyreadr auto-detect the gzip magic (1f 8b) and decompress transparently; the
    exact compressed bytes are irrelevant (gzip is lossless), so a fixed-mtime stream keeps
    the output deterministic without affecting readability."""
    # filename="" (not None) so GzipFile does NOT embed the on-disk basename into the header —
    # that would leak the path AND make the bytes depend on the filename (breaking determinism).
    with open(path, "wb") as fh:
        with gzip.GzipFile(filename="", fileobj=fh, mode="wb", mtime=0) as gz:
            gz.write(payload)


def _write_rds_matrix(
    path: str,
    col1_f2: Sequence[float],
    col2_counts: Sequence[float],
    colnames: Sequence[str] = ("f2", "counts"),
) -> None:
    """Write an AT2 per-pair f2 matrix ``[n, 2]`` as a gzip'd RDS-XDR-v2 file.

    ``col1_f2`` / ``col2_counts`` are the two length-``n`` columns (R stores a matrix
    column-major, so col 1 is emitted first, then col 2); ``colnames`` labels the columns
    (``dimnames[[2]]``; row names are NULL). The result reads back identically in
    ``base::readRDS`` and ``admixtools::read_f2``."""
    n = len(col1_f2)
    if len(col2_counts) != n:
        raise ValueError(f"column length mismatch: {n} f2 vs {len(col2_counts)} counts")
    if len(colnames) != 2:
        raise ValueError(f"expected 2 column names, got {len(colnames)}")

    buf = bytearray()
    buf += _rds_header()
    # REALSXP payload (a length-2n vector; the [n,2] shape lives in the `dim` attribute).
    buf += _i32(_REALSXP | _HAS_ATTR)
    buf += _i32(2 * n)
    for v in col1_f2:
        buf += _f64(v)
    for v in col2_counts:
        buf += _f64(v)
    # ATTRIB pairlist cell 1: dim = INTSXP c(nrow, ncol).
    buf += _attr_tag("dim")
    buf += _i32(_INTSXP) + _i32(2) + _i32(n) + _i32(2)
    # ATTRIB pairlist cell 2: dimnames = VECSXP(NULL, STRSXP(colnames)).
    buf += _attr_tag("dimnames")
    buf += _i32(_VECSXP) + _i32(2)
    buf += _i32(_NILVALUE_SXP)            # rownames = NULL
    buf += _i32(_STRSXP) + _i32(2)        # colnames character vector
    buf += _charsxp(colnames[0])
    buf += _charsxp(colnames[1])
    # Terminate the ATTRIB pairlist (CDR = NULL).
    buf += _i32(_NILVALUE_SXP)

    _gzip_write(path, bytes(buf))


def _write_rds_int_vector(path: str, ints: Sequence[int]) -> None:
    """Write an integer vector (e.g. ``block_lengths_f2``) as a gzip'd RDS-XDR-v2 file.

    A plain INTSXP with no attributes — byte-identical to ``saveRDS(as.integer(v), version=2)``."""
    buf = bytearray()
    buf += _rds_header()
    buf += _i32(_INTSXP)
    buf += _i32(len(ints))
    for v in ints:
        buf += _i32(v)
    _gzip_write(path, bytes(buf))


# ---------------------------------------------------------------------------------------
# Reading — via pyreadr (librdata). Lazily imported so the base wheel stays dependency-free
# and only the IMPORT direction (AT2 -> steppe) pulls the optional `[rds]` extra.
# ---------------------------------------------------------------------------------------
def _require_pyreadr():
    try:
        import pyreadr  # noqa: WPS433 (lazy by design — the optional [rds] extra)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "reading .rds files needs pyreadr, which is not installed. "
            "Install the optional extra: `pip install steppe[rds]`. "
            "(Export/writing .rds needs no extra dependency.)"
        ) from exc
    return pyreadr


def _read_rds_matrix(path: str):
    """Read an AT2 per-pair matrix ``[n, 2]`` via pyreadr; return ``(col1_f2, col2_counts)``.

    Both columns come back as ``float64`` numpy arrays. Columns are taken **positionally**
    (col 0 = f2, col 1 = counts) so the read is robust to the ``dimnames`` label casing."""
    import numpy as np  # local: numpy is a base dep, but keep the module import-light

    pyreadr = _require_pyreadr()
    result = pyreadr.read_r(str(path))
    df = result[None]
    col1 = np.asarray(df.iloc[:, 0], dtype=np.float64)
    col2 = (
        np.asarray(df.iloc[:, 1], dtype=np.float64)
        if df.shape[1] > 1
        else np.ones(len(col1), dtype=np.float64)
    )
    return col1, col2


def _read_rds_int_vector(path: str) -> list:
    """Read an integer vector (e.g. ``block_lengths_f2``) via pyreadr; return a ``list[int]``."""
    import numpy as np

    pyreadr = _require_pyreadr()
    result = pyreadr.read_r(str(path))
    df = result[None]
    return [int(v) for v in np.asarray(df.iloc[:, 0])]


def _decode_rds_bytes(path: str) -> Optional[bytes]:
    """Return the gunzipped RDS payload of ``path`` (test/debug helper; no pyreadr needed)."""
    with open(path, "rb") as fh:
        head = fh.read(2)
        fh.seek(0)
        if head == b"\x1f\x8b":
            return gzip.decompress(fh.read())
        return fh.read()


# ---------------------------------------------------------------------------------------
# The converter proper. Lives HERE (not __init__.py) so it has ZERO dependency on the
# compiled `_core`: the public `steppe.export_f2_rds` / `steppe.import_f2_rds` are thin
# delegators to these, and they are importable + testable in a CUDA-free lane. Pure Python:
# numpy (a base dep) + the stdlib serializer above + pyreadr (only on the import path).
# ---------------------------------------------------------------------------------------

# STPF2BK1 on-disk format (src/device/f2_disk_format.hpp): a 64-byte header, then the f2
# region (P*P*n_block FP64, column-major i + P*j + P*P*b), then the vpair region (same
# shape), then the block_sizes int32 trailer. Single home for the constants the writer stamps.
_F2_MAGIC = b"STPF2BK1"
_F2_VERSION = 1
_F2_DTYPE_FP64 = 1
_F2_HEADER_SIZE = 64


def _write_stpf2bk1(out_dir, P, n_block, block_sizes, f2, vpair, pops):
    """Write an STPF2BK1 f2-dir (``f2.bin`` + ``pops.txt`` + ``meta.json``) from numpy fields.

    Mirrors the reference recipe in tests/python/conftest.py (and test_cli_qpadm.cpp), but
    numpy-vectorized and — for the IMPORT path — writes a NONZERO ``vpair`` (the caller passes
    the block_sizes-broadcast sentinel; zeros would trip the ``vpair==0`` missing-block
    detector). ``f2`` / ``vpair`` are ``(P, P, n_block)`` arrays flattened column-major
    (``order="F"`` -> i + P*j + P*P*b, byte-identical to F2BlockTensor)."""
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    slab_bytes = P * P * n_block * 8
    f2_offset = _F2_HEADER_SIZE
    vpair_offset = f2_offset + slab_bytes
    block_sizes_offset = vpair_offset + slab_bytes

    header = struct.pack(
        "<8sIIiiQQQ16s",
        _F2_MAGIC,
        _F2_VERSION,
        _F2_DTYPE_FP64,
        P,
        n_block,
        f2_offset,
        vpair_offset,
        block_sizes_offset,
        b"\x00" * 16,
    )
    assert len(header) == _F2_HEADER_SIZE, len(header)

    f2_bytes = np.asarray(f2, dtype=np.float64).reshape(-1, order="F").astype("<f8").tobytes()
    vp_bytes = np.asarray(vpair, dtype=np.float64).reshape(-1, order="F").astype("<f8").tobytes()
    bs_bytes = np.asarray(block_sizes, dtype="<i4").tobytes()

    with open(os.path.join(out_dir, "f2.bin"), "wb") as o:
        o.write(header)
        o.write(f2_bytes)
        o.write(vp_bytes)
        o.write(bs_bytes)

    with open(os.path.join(out_dir, "pops.txt"), "w") as p:
        for s in pops:
            p.write(s + "\n")

    meta = {
        "format": "STPF2BK1",
        "P": int(P),
        "n_block": int(n_block),
        "provenance": (
            "imported from an ADMIXTOOLS 2 read_f2 .rds dir "
            "(import_f2_rds; vpair = block_sizes sentinel)"
        ),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as m:
        m.write(json.dumps(meta, indent=2) + "\n")


def export_f2_rds(f2, out_dir, *, counts="ones", write_ap=False):
    """Implementation of ``steppe.export_f2_rds`` (see that wrapper for the public docstring).

    ``f2`` is any object exposing the F2Blocks accessors ``pops`` / ``P`` / ``n_block`` /
    ``block_sizes`` / ``to_numpy()`` (and ``vpair_to_numpy()`` when ``counts='vpair'``)."""
    if write_ap:
        raise NotImplementedError(
            "write_ap=True (the _ap allele-pair family) is not implemented; the f2 family "
            "alone is sufficient for qpAdm/qpGraph/f4. Omit write_ap or pass write_ap=False."
        )
    counts = str(counts).lower()
    if counts not in ("ones", "vpair"):
        raise ValueError(f"counts must be 'ones' or 'vpair', got {counts!r}")

    pops = list(f2.pops)
    P = int(f2.P)
    n_block = int(f2.n_block)
    block_sizes = [int(b) for b in f2.block_sizes]
    arr = f2.to_numpy()  # (P, P, n_block) FP64; arr[i,j,:] is the f2 series for pops i,j
    name_to_idx = {p: i for i, p in enumerate(pops)}
    sorted_pops = sorted(pops)  # Python sorted() == C-locale byte order == AT2 dimnames order

    vp = f2.vpair_to_numpy() if counts == "vpair" else None
    ones = [1.0] * n_block
    zeros = [0.0] * n_block

    os.makedirs(out_dir, exist_ok=True)
    # A subdir for EVERY pop — AT2 read_f2 does list.dirs(recursive=FALSE) to derive `pops`,
    # so even the last-alphabetical pop needs a dir (holding only its zero self-pair).
    for p in pops:
        os.makedirs(os.path.join(out_dir, p), exist_ok=True)

    for a in range(P):
        for b in range(a, P):
            p1 = sorted_pops[a]
            p2 = sorted_pops[b]
            i = name_to_idx[p1]
            j = name_to_idx[p2]
            col1 = zeros if a == b else arr[i, j, :]  # diagonal self-pair == 0.0
            if counts == "vpair":
                col2 = zeros if a == b else vp[i, j, :]
            else:
                col2 = ones
            path = os.path.join(out_dir, p1, f"{p2}_f2.rds")
            _write_rds_matrix(path, col1, col2, ("f2", "counts"))

    _write_rds_int_vector(os.path.join(out_dir, "block_lengths_f2.rds"), block_sizes)
    return str(out_dir)


def import_f2_rds(rds_dir, out_dir, *, type="f2"):
    """Implementation of ``steppe.import_f2_rds`` (see that wrapper for the public docstring)."""
    import numpy as np  # noqa: WPS433

    if type != "f2":
        raise ValueError(f"only type='f2' is supported, got {type!r}")

    pops = sorted(
        name
        for name in os.listdir(rds_dir)
        if os.path.isdir(os.path.join(rds_dir, name))
    )
    P = len(pops)
    if P == 0:
        raise ValueError(f"no population subdirs found in {rds_dir!r}")

    bl_path = os.path.join(rds_dir, "block_lengths_f2.rds")
    if not os.path.exists(bl_path):
        raise FileNotFoundError(
            f"block_lengths_f2.rds missing in {rds_dir!r} (AT2 read_f2 requires it)"
        )
    block_sizes = _read_rds_int_vector(bl_path)
    n_block = len(block_sizes)

    f2 = np.zeros((P, P, n_block), dtype=np.float64)
    for a in range(P):
        for b in range(a, P):
            if a == b:
                continue  # diagonal self-pair is 0.0 by AT2 convention (leave the zeros)
            p1, p2 = pops[a], pops[b]  # sorted -> p1 is the C-locale-smaller key
            path = os.path.join(rds_dir, p1, f"{p2}_f2.rds")
            if not os.path.exists(path):
                raise FileNotFoundError(f"expected pair file missing: {path!r}")
            col1, _ = _read_rds_matrix(path)
            if len(col1) != n_block:
                raise ValueError(
                    f"{path!r}: {len(col1)} blocks but block_lengths has {n_block}"
                )
            f2[a, b, :] = col1
            f2[b, a, :] = col1

    # vpair: the nonzero sentinel (block_sizes broadcast over every pair) — see the docstring.
    vpair = np.empty((P, P, n_block), dtype=np.float64)
    for bi, bs in enumerate(block_sizes):
        vpair[:, :, bi] = float(bs)

    _write_stpf2bk1(out_dir, P, n_block, block_sizes, f2, vpair, pops)
    return str(out_dir)
