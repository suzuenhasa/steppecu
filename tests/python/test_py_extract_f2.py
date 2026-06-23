"""M(py-2) ACCEPTANCE GATE: build an f2_blocks tensor from a genotype prefix THROUGH the
Python extract_f2 binding, on REAL AADR (no synthetic data).

The proof the steppe.extract_f2 binding is correct: the genotype TRIPLE must flow
read_ind(Explicit{pops}) -> decode -> filter (pop-axis maxmiss) -> assign_blocks -> the
tiered f2 compute -> to_host -> the F2Handle / the STPF2BK1 dir, exercising every M(py-2)
binding seam. It runs the SAME chain the CLI extract-f2 command runs (steppe::run_extract_f2),
so a clean extract reproduces the AT2 extract_f2 convention.

SOURCE PREFIX (B3, resolved): steppe's GenoReader read_tile is TGENO-only (individual-major;
geno_reader.hpp/eigenstrat_format.hpp). The convertf-PA v66_HO_pa is PACKEDANCESTRYMAP
(SNP-major GENO) — NOT readable by the decode path. So the extract reads the SAME raw HO
TGENO prefix the genotype-path dstat test reads (STEPPE_AADR_ROOT/raw/v66.p1_HO.aadr.patch.PUB),
the lossless transcode of the convertf-PA the AT2 goldens were generated from.

THE PARITY LAW SELF-CONSISTENCY (the central M(py-2) gate, the prompt's ~1e-12 check):
extract_f2(out=None) -> an in-memory F2Handle and extract_f2(out=DIR) + read_f2(DIR) -> a
disk-round-tripped F2Handle must carry BIT-IDENTICAL f2 tensors (both materialize the SAME
compute_f2_blocks_multigpu_tiered().to_host(); the STPF2BK1 dir is a byte-exact serialization).
Asserted at rtol 1e-12 (in practice array_equal). SKIPs cleanly when no CUDA device is visible
OR the TGENO prefix is absent (the genotype-path data-absent guard).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import maybe_skip_no_gpu

_AADR_ROOT = Path(os.environ.get("STEPPE_AADR_ROOT", "/workspace/data/aadr"))
# steppe's reader is TGENO (individual-major) only; the convertf-PA v66_HO_pa is SNP-major
# PACKEDANCESTRYMAP and is NOT read by the decode path, so we read the raw HO TGENO prefix
# (the lossless transcode of the SAME genotypes — same ind/snp axes; B3 resolved).
_TGENO_PREFIX = _AADR_ROOT / "raw" / "v66.p1_HO.aadr.patch.PUB"

# A SMALL real pop subset from the 9-pop golden union (these are present in the HO prefix;
# the genotype-path dstat golden uses this exact union). 3 pops keeps the extract fast.
SMALL_POPS = ["England_BellBeaker", "Turkey_N", "Mbuti"]


def _require_prefix():
    for ext in (".geno", ".snp", ".ind"):
        if not (_TGENO_PREFIX.parent / (_TGENO_PREFIX.name + ext)).is_file():
            pytest.skip(f"TGENO genotype prefix absent: {_TGENO_PREFIX}{ext}")


def _extract(steppe_mod, **kw):
    try:
        return steppe_mod.extract_f2(str(_TGENO_PREFIX), pops=SMALL_POPS, **kw)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise


# ---- GATE (1): extract_f2(out=None) builds a usable F2Blocks handle -------------------
def test_extract_f2_in_memory_handle(steppe_mod):
    _require_prefix()
    f2 = _extract(steppe_mod)
    # The P axis is the Explicit selection SORTED ASC by label (= pops.txt order).
    assert f2.P == 3
    assert list(f2.pops) == sorted(SMALL_POPS)
    assert f2.n_block > 0
    assert len(f2.block_sizes) == f2.n_block
    # The f2 tensor is the FP64 [P, P, n_block] tensor, F-contiguous, symmetric per slab.
    np = pytest.importorskip("numpy")
    arr = f2.to_numpy()
    assert arr.shape == (3, 3, f2.n_block)
    assert arr.dtype == np.float64
    assert arr.flags.f_contiguous
    # Each slab is symmetric (f2 is symmetric per fstats.hpp) up to FP round-off — a
    # freshly-COMPUTED slab is not bit-symmetric (the (i,j) vs (j,i) GEMM accumulations
    # differ in the last ULP), unlike a slab read back from a fixture.
    for b in (0, f2.n_block // 2, f2.n_block - 1):
        slab = arr[:, :, b]
        assert np.allclose(slab, slab.T, rtol=1e-9, atol=1e-12)


# ---- GATE (2): the PARITY LAW — out=None matches a written-then-read dir to ~1e-12 ----
# The central M(py-2) self-consistency proof: the in-memory handle and a disk-round-tripped
# handle carry BIT-IDENTICAL f2 (both materialize the same tiered to_host(); STPF2BK1 is a
# byte-exact serialization). The prompt's ~1e-12 gate (in practice array_equal).
def test_extract_f2_matches_read_f2_roundtrip(steppe_mod, tmp_path):
    np = pytest.importorskip("numpy")
    _require_prefix()

    f2_mem = _extract(steppe_mod)               # out=None -> in-memory F2Handle.
    out_dir = tmp_path / "extracted_f2dir"
    try:
        path = steppe_mod.extract_f2(
            str(_TGENO_PREFIX), pops=SMALL_POPS, out=str(out_dir)
        )
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise
    assert str(path) == str(out_dir)
    f2_disk = steppe_mod.read_f2(str(path))     # reload the STPF2BK1 dir.

    # Same shape + same labels.
    assert f2_disk.P == f2_mem.P
    assert list(f2_disk.pops) == list(f2_mem.pops)
    assert f2_disk.n_block == f2_mem.n_block
    assert list(f2_disk.block_sizes) == list(f2_mem.block_sizes)

    a = f2_mem.to_numpy()
    b = f2_disk.to_numpy()
    # ~1e-12 (the parity law) — in practice the bytes are identical.
    assert np.allclose(a, b, rtol=1e-12, atol=1e-12)
    assert np.array_equal(a, b)
    # The vpair (REAL pairwise-valid counts, not zeros) round-trips too.
    assert np.array_equal(f2_mem.vpair_to_numpy(), f2_disk.vpair_to_numpy())


# ---- GATE (3): the extracted f2 drives a qpadm fit (the extract -> fit chain works) ---
def test_extract_f2_feeds_qpadm(steppe_mod):
    _require_prefix()
    # A 4-pop extract so a 1-source qpadm has a valid left + right.
    pops = ["England_BellBeaker", "Turkey_N", "Mbuti", "Han"]
    try:
        f2 = steppe_mod.extract_f2(str(_TGENO_PREFIX), pops=pops)
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise
    assert f2.P == 4
    try:
        res = steppe_mod.qpadm(
            f2, target="England_BellBeaker", left=["Turkey_N"], right=["Mbuti", "Han"]
        )
    except Exception as exc:  # noqa: BLE001
        maybe_skip_no_gpu(exc)
        raise
    # A real fit produces a real status + a weight per source (a domain outcome rides on
    # status; no exception). The single source weight is ~1.0 (the trivial 1-source fit).
    assert res.status.name in {
        "OK", "RANK_DEFICIENT", "NON_SPD_COVARIANCE", "CHISQ_UNDEFINED",
    }
    assert len(res.weight) == 1


# ---- GATE (4): an unknown pop name raises a clean error (the fault contract) ----------
def test_extract_f2_unknown_pop_raises(steppe_mod):
    _require_prefix()
    with pytest.raises(Exception) as ei:
        try:
            steppe_mod.extract_f2(
                str(_TGENO_PREFIX), pops=["England_BellBeaker", "NoSuchPopXYZ"]
            )
        except Exception as exc:  # noqa: BLE001
            maybe_skip_no_gpu(exc)
            raise
    assert "NoSuchPopXYZ" in str(ei.value) or "unknown" in str(ei.value).lower()
