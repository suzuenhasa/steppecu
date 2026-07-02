"""Round-trip gates for the ADMIXTOOLS 2 <-> steppe f2 `.rds` converter (rds-converter.md §7).

This is a PURE-PYTHON, GPU-free feature (on-disk format translation), so these gates need
neither a CUDA device nor R — they run in a CUDA-free CI lane:

 (1) SERIALIZER FIDELITY (no dep at all): the hand-rolled `_rds` writer must be byte-for-byte
     identical to `base::saveRDS(m, f, compress="gzip", version=2)`. The reference bytes below
     were captured from R 4.3.3 on box5090 (a [3,2] matrix and a length-3 integer vector), so
     this asserts against *real R output* without needing R installed.

 (2) EXPORT re-read (needs pyreadr): export a small steppe f2 cache -> re-read every
     `<p1>/<p2>_f2.rds` with pyreadr and assert col 1 == the steppe f2 slice BIT-EXACT, the
     diagonal self-pairs are 0.0, `counts` == 1.0, `block_lengths_f2.rds` == block_sizes, and a
     subdir exists for EVERY pop (AT2 read_f2 derives `pops` from list.dirs).

 (3) IMPORT round-trip (needs pyreadr): export -> import -> re-read the STPF2BK1 `f2.bin` and
     assert the off-diagonal survives bit-exact, the diagonal is zeroed, and `vpair` carries the
     NONZERO block_sizes sentinel (never zeros — that would trip the missing-block detector).

The end-to-end "verified in ADMIXTOOLS 2" proof (admixtools::read_f2 + qpadm/f4 on the exported
cache) lives in tests/r/verify_export_rds.R (box5090-only). A `steppe_mod`-gated variant of (2)
using the committed 9-pop fixture runs when the compiled bindings are present.
"""
from __future__ import annotations

import gzip
import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RDS_SRC = _REPO_ROOT / "bindings" / "steppe" / "_rds.py"

# Byte-for-byte reference output of R 4.3.3 saveRDS(..., compress="gzip", version=2), gunzipped.
# matrix: matrix(c(0.1,0.2,0.3, 10,20,30), 3, 2); colnames c("f2","counts")
_R_REF_MATRIX = bytes(
    int(b, 16)
    for b in (
        "58 0a 00 00 00 02 00 04 03 03 00 02 03 00 00 00 "
        "02 0e 00 00 00 06 3f b9 99 99 99 99 99 9a 3f c9 "
        "99 99 99 99 99 9a 3f d3 33 33 33 33 33 33 40 24 "
        "00 00 00 00 00 00 40 34 00 00 00 00 00 00 40 3e "
        "00 00 00 00 00 00 00 00 04 02 00 00 00 01 00 04 "
        "00 09 00 00 00 03 64 69 6d 00 00 00 0d 00 00 00 "
        "02 00 00 00 03 00 00 00 02 00 00 04 02 00 00 00 "
        "01 00 04 00 09 00 00 00 08 64 69 6d 6e 61 6d 65 "
        "73 00 00 00 13 00 00 00 02 00 00 00 fe 00 00 00 "
        "10 00 00 00 02 00 04 00 09 00 00 00 02 66 32 00 "
        "04 00 09 00 00 00 06 63 6f 75 6e 74 73 00 00 00 fe"
    ).split()
)
# integer vector as.integer(c(100,200,300))
_R_REF_INTVEC = bytes(
    int(b, 16)
    for b in (
        "58 0a 00 00 00 02 00 04 03 03 00 02 03 00 00 00 "
        "00 0d 00 00 00 03 00 00 00 64 00 00 00 c8 00 00 01 2c"
    ).split()
)


def _load_rds():
    """Load bindings/steppe/_rds.py standalone (NO compiled _core), so BOTH the serializer and
    the whole converter (export_f2_rds / import_f2_rds live in _rds) are exercised without a
    steppe build and without polluting the global `steppe` module for the other test files."""
    spec = importlib.util.spec_from_file_location("steppe_rds_under_test", _RDS_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeF2:
    """A minimal duck-typed F2Blocks stand-in (the converter uses only these accessors),
    with a DELIBERATELY UNSORTED, mixed-case pop list to exercise the C-locale keying and the
    name->index remap; a symmetric tensor with a NONZERO diagonal (the steppe het correction
    the export must zero)."""

    def __init__(self, seed: int = 0):
        self.pops = ["Zebra", "Alpha", "mango", "Beta"]  # unsorted; upper < lower in C-locale
        self.P = 4
        self.n_block = 3
        self.block_sizes = [215, 408, 412]
        rng = np.random.default_rng(seed)
        a = np.zeros((self.P, self.P, self.n_block), dtype=np.float64)
        for b in range(self.n_block):
            m = rng.random((self.P, self.P))
            m = (m + m.T) / 2.0
            np.fill_diagonal(m, 5.0 + b)  # nonzero diagonal -> must be exported as 0.0
            a[:, :, b] = m
        self._a = np.asfortranarray(a)

    def to_numpy(self):
        return self._a.copy()

    def vpair_to_numpy(self):
        vp = np.zeros((self.P, self.P, self.n_block), dtype=np.float64)
        for b, bs in enumerate(self.block_sizes):
            vp[:, :, b] = bs
        return vp


# ---- GATE (1): serializer is byte-for-byte identical to R saveRDS(version=2) ----------
def test_rds_matrix_byte_exact_vs_R(tmp_path):
    rds = _load_rds()
    p = tmp_path / "m.rds"
    rds._write_rds_matrix(str(p), [0.1, 0.2, 0.3], [10.0, 20.0, 30.0], ("f2", "counts"))
    with open(p, "rb") as f:
        payload = gzip.decompress(f.read())
    assert payload == _R_REF_MATRIX, (len(payload), len(_R_REF_MATRIX))


def test_rds_int_vector_byte_exact_vs_R(tmp_path):
    rds = _load_rds()
    p = tmp_path / "v.rds"
    rds._write_rds_int_vector(str(p), [100, 200, 300])
    with open(p, "rb") as f:
        payload = gzip.decompress(f.read())
    assert payload == _R_REF_INTVEC, (len(payload), len(_R_REF_INTVEC))


def test_rds_write_is_deterministic(tmp_path):
    """Fixed-mtime gzip -> repeat exports of the same matrix are byte-identical on disk."""
    rds = _load_rds()
    a, b = tmp_path / "a.rds", tmp_path / "b.rds"
    rds._write_rds_matrix(str(a), [1.0, 2.0], [1.0, 1.0], ("f2", "counts"))
    rds._write_rds_matrix(str(b), [1.0, 2.0], [1.0, 1.0], ("f2", "counts"))
    assert a.read_bytes() == b.read_bytes()


# ---- GATE (2): pyreadr re-reads what we wrote (matrix + int vector) --------------------
def test_rds_pyreadr_roundtrip(tmp_path):
    pytest.importorskip("pyreadr")
    rds = _load_rds()
    mp = tmp_path / "m.rds"
    rds._write_rds_matrix(str(mp), [0.1, 0.2, 0.3], [7.0, 8.0, 9.0], ("f2", "counts"))
    col1, col2 = rds._read_rds_matrix(str(mp))
    assert np.array_equal(col1, [0.1, 0.2, 0.3])
    assert np.array_equal(col2, [7.0, 8.0, 9.0])
    vp = tmp_path / "v.rds"
    rds._write_rds_int_vector(str(vp), [11, 22, 33])
    assert rds._read_rds_int_vector(str(vp)) == [11, 22, 33]


def test_rds_preserves_special_floats(tmp_path):
    """NaN / +-Inf survive the XDR big-endian double round-trip (a serializer-correctness risk)."""
    pytest.importorskip("pyreadr")
    rds = _load_rds()
    p = tmp_path / "s.rds"
    vals = [float("inf"), float("-inf"), 1.5e-300, -2.5e300]
    rds._write_rds_matrix(str(p), vals, [1.0, 1.0, 1.0, 1.0], ("f2", "counts"))
    col1, _ = rds._read_rds_matrix(str(p))
    assert col1[0] == float("inf") and col1[1] == float("-inf")
    assert col1[2] == 1.5e-300 and col1[3] == -2.5e300


# ---- GATE (3): export re-read with pyreadr (the "CI proxy, no R" export gate) ----------
def test_export_reread_bitexact(tmp_path):
    pytest.importorskip("pyreadr")
    import pyreadr

    rds = _load_rds()
    h = _FakeF2()
    out = tmp_path / "exported"
    rds.export_f2_rds(h, str(out))

    # A subdir for EVERY pop (read_f2 derives `pops` from list.dirs).
    for p in h.pops:
        assert (out / p).is_dir(), p
    assert (out / "block_lengths_f2.rds").is_file()

    arr = h.to_numpy()
    name_to_idx = {p: i for i, p in enumerate(h.pops)}
    sorted_pops = sorted(h.pops)  # C-locale byte order
    for a in range(h.P):
        for b in range(a, h.P):
            p1, p2 = sorted_pops[a], sorted_pops[b]
            path = out / p1 / f"{p2}_f2.rds"
            assert path.is_file(), path  # keyed under the C-locale-smaller pop
            df = pyreadr.read_r(str(path))[None]
            assert list(df.columns) == ["f2", "counts"]
            col1 = df["f2"].to_numpy()
            assert np.all(df["counts"].to_numpy() == 1.0)
            if a == b:
                assert np.all(col1 == 0.0), ("diagonal must be 0.0", p1)
            else:
                i, j = name_to_idx[p1], name_to_idx[p2]
                assert np.array_equal(col1, arr[i, j, :]), (p1, p2)

    bl = pyreadr.read_r(str(out / "block_lengths_f2.rds"))[None]
    assert [int(v) for v in bl.iloc[:, 0]] == h.block_sizes


def test_export_keys_under_smaller_pop(tmp_path):
    """The pair file lives under the C-locale-smaller pop only (no duplicate under the larger)."""
    rds = _load_rds()
    h = _FakeF2()
    out = tmp_path / "exported"
    rds.export_f2_rds(h, str(out))
    # 'Alpha' < 'Beta' -> Alpha/Beta_f2.rds exists, Beta/Alpha_f2.rds does not.
    assert (out / "Alpha" / "Beta_f2.rds").is_file()
    assert not (out / "Beta" / "Alpha_f2.rds").exists()
    # lowercase 'mango' is the C-locale-largest -> its subdir holds only its own diagonal.
    assert list((out / "mango").glob("*_f2.rds")) == [out / "mango" / "mango_f2.rds"]


def test_export_validates_args(tmp_path):
    rds = _load_rds()
    h = _FakeF2()
    with pytest.raises(ValueError):
        rds.export_f2_rds(h, str(tmp_path / "o1"), counts="bogus")
    with pytest.raises(NotImplementedError):
        rds.export_f2_rds(h, str(tmp_path / "o2"), write_ap=True)


# ---- GATE (4): export -> import round-trip (off-diagonal bit-exact; vpair sentinel) ----
def test_export_import_roundtrip(tmp_path):
    pytest.importorskip("pyreadr")
    rds = _load_rds()
    h = _FakeF2()
    rds_dir = tmp_path / "exported"
    bin_dir = tmp_path / "reimported"
    rds.export_f2_rds(h, str(rds_dir))
    rds.import_f2_rds(str(rds_dir), str(bin_dir))

    P, nb = h.P, h.n_block
    with open(bin_dir / "f2.bin", "rb") as f:
        magic, ver, dt, hP, hnb, f2o, vpo, bso = struct.unpack("<8sIIiiQQQ", f.read(48))
        assert magic == b"STPF2BK1" and hP == P and hnb == nb
        f.seek(f2o)
        f2 = np.frombuffer(f.read(P * P * nb * 8), dtype="<f8").reshape((P, P, nb), order="F")
        f.seek(vpo)
        vp = np.frombuffer(f.read(P * P * nb * 8), dtype="<f8").reshape((P, P, nb), order="F")
        f.seek(bso)
        bs = np.frombuffer(f.read(nb * 4), dtype="<i4")

    # imported pops are the sorted subdir names.
    ipops = (bin_dir / "pops.txt").read_text().split()
    assert ipops == sorted(h.pops)

    arr = h.to_numpy()
    name_to_idx = {p: i for i, p in enumerate(h.pops)}
    for a in range(P):
        for b in range(P):
            if a == b:
                assert np.all(f2[a, a, :] == 0.0)  # diagonal zeroed
            else:
                i, j = name_to_idx[ipops[a]], name_to_idx[ipops[b]]
                assert np.array_equal(f2[a, b, :], arr[i, j, :]), (a, b)

    assert np.array_equal(bs, np.asarray(h.block_sizes, dtype="<i4"))
    # vpair must be the NONZERO block_sizes sentinel, never zeros.
    assert np.all(vp != 0.0)
    for bi, bsz in enumerate(h.block_sizes):
        assert np.all(vp[:, :, bi] == bsz)


def test_import_requires_block_lengths(tmp_path):
    pytest.importorskip("pyreadr")
    rds = _load_rds()
    h = _FakeF2()
    rds_dir = tmp_path / "exported"
    rds.export_f2_rds(h, str(rds_dir))
    (rds_dir / "block_lengths_f2.rds").unlink()
    with pytest.raises(FileNotFoundError):
        rds.import_f2_rds(str(rds_dir), str(tmp_path / "bin"))


# ---- GATE (5): the build-gated variant on the COMMITTED 9-pop fixture (no GPU needed) --
_POPS_9 = [
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


def test_export_committed_fixture(steppe_mod, stage_f2_dir, tmp_path):
    """read_f2 (host-only) the committed real-AADR 9-pop f2 fixture -> export -> pyreadr re-read
    bit-exact. Exercises the true F2Blocks handle path end-to-end (skips if pyreadr absent)."""
    pytest.importorskip("pyreadr")
    import pyreadr

    if not hasattr(steppe_mod, "export_f2_rds"):
        pytest.skip("built steppe predates export_f2_rds")
    d = stage_f2_dir("f2_fit0_9pop.bin", _POPS_9)
    f2 = steppe_mod.read_f2(str(d))
    out = tmp_path / "exported9"
    steppe_mod.export_f2_rds(f2, str(out))

    arr = f2.to_numpy()
    name_to_idx = {p: i for i, p in enumerate(f2.pops)}
    sorted_pops = sorted(f2.pops)
    # spot-check a handful of off-diagonal pairs bit-exact.
    for (a, b) in [(0, 1), (0, 8), (3, 6), (2, 7)]:
        p1, p2 = sorted_pops[a], sorted_pops[b]
        df = pyreadr.read_r(str(out / p1 / f"{p2}_f2.rds"))[None]
        i, j = name_to_idx[p1], name_to_idx[p2]
        assert np.array_equal(df["f2"].to_numpy(), arr[i, j, :]), (p1, p2)
    bl = pyreadr.read_r(str(out / "block_lengths_f2.rds"))[None]
    assert [int(v) for v in bl.iloc[:, 0]] == list(f2.block_sizes)


# ---- GATE (6): the GPU-free STPF2BK1 reader + the `steppe-rds` CLI (main) --------------
def _stage_stpf2bk1(rds, h, out_dir):
    """Write an STPF2BK1 f2-dir on disk from a _FakeF2, so the CLI's disk reader has an input."""
    rds._write_stpf2bk1(
        str(out_dir), h.P, h.n_block, h.block_sizes, h.to_numpy(), h.vpair_to_numpy(), h.pops
    )


def test_read_stpf2bk1_inverts_writer(tmp_path):
    """_read_stpf2bk1 (the GPU-free disk reader the CLI uses) is the exact inverse of
    _write_stpf2bk1: pops / P / n_block / block_sizes / f2 / vpair all round-trip bit-exact."""
    rds = _load_rds()
    h = _FakeF2()
    src = tmp_path / "src_f2"
    _stage_stpf2bk1(rds, h, src)
    hh = rds._read_stpf2bk1(str(src))
    assert hh.pops == h.pops
    assert (hh.P, hh.n_block) == (h.P, h.n_block)
    assert hh.block_sizes == h.block_sizes
    assert np.array_equal(hh.to_numpy(), h.to_numpy())
    assert np.array_equal(hh.vpair_to_numpy(), h.vpair_to_numpy())


def test_cli_export_then_import(tmp_path):
    """`steppe-rds export` (GPU-free, via _read_stpf2bk1) writes an AT2 .rds dir, then
    `steppe-rds import` round-trips it back to STPF2BK1 with the off-diagonal f2 bit-exact."""
    pytest.importorskip("pyreadr")
    rds = _load_rds()
    h = _FakeF2()
    src = tmp_path / "src_f2"
    _stage_stpf2bk1(rds, h, src)
    rds_dir = tmp_path / "exported"
    back = tmp_path / "reimported"

    assert rds.main(["export", str(src), str(rds_dir)]) == 0
    assert (rds_dir / "block_lengths_f2.rds").exists()
    for p in h.pops:  # AT2 read_f2 derives `pops` from list.dirs -> a subdir per pop
        assert (rds_dir / p).is_dir()

    assert rds.main(["import", str(rds_dir), str(back)]) == 0
    hh = rds._read_stpf2bk1(str(back))
    arr = h.to_numpy()
    name_to_idx = {p: i for i, p in enumerate(h.pops)}
    f2 = hh.to_numpy()
    for a in range(h.P):
        for b in range(h.P):
            if a == b:
                assert np.all(f2[a, a, :] == 0.0)  # diagonal zeroed
            else:
                i, j = name_to_idx[hh.pops[a]], name_to_idx[hh.pops[b]]
                assert np.array_equal(f2[a, b, :], arr[i, j, :]), (a, b)


def test_cli_bad_args_and_missing_input(tmp_path):
    """argparse misuse exits nonzero (SystemExit); a missing input dir is a handled error
    (return 1, no traceback) not a crash."""
    rds = _load_rds()
    with pytest.raises(SystemExit):  # no subcommand (required=True)
        rds.main([])
    with pytest.raises(SystemExit):  # unknown subcommand
        rds.main(["frobnicate", "a", "b"])
    assert rds.main(["export", str(tmp_path / "nope"), str(tmp_path / "out")]) == 1
