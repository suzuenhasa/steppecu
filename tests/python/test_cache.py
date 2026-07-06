"""py_cache — GPU-free steppe-cache manager tests (run under the py_qpadm pytest lane).

The oracle is deterministic self-consistency against the committed real-AADR golden
docs/examples/f2_9pop/: the parse must reproduce the fields the cache carries, and
verify's re-hash must equal the content-address the extract writer stamped."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import steppe
from steppe import _cache

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "docs" / "examples" / "f2_9pop"

# The committed golden's exact content-address (docs/examples/f2_9pop/meta.json).
F2_CACHE_ID = "sha256:1bc49e1c764b7652443a6636b813758be486b456fd783832f9cfb0fddf242a85"
POPS_SHA = "sha256:895fe01ab1f8eb773ebd4f541149b470f94c5bd746c2a44def1fc70103a514d6"


@pytest.fixture
def example():
    if not (EXAMPLE / "f2.bin").exists():
        pytest.skip(f"example cache not present: {EXAMPLE}")
    return str(EXAMPLE)


def test_header_only_parse(example):
    info = _cache._read_header_only(example)
    assert info["P"] == 9
    assert info["n_block"] == 710
    assert len(info["pops"]) == 9
    assert info["f2_bin_size"] == 923064
    # the writer's size arithmetic is byte-exact for this real cache
    assert info["f2_bin_size"] == info["expected_f2_bin_size"]


def test_show_json_reproduces_meta(example):
    rec = _cache._record(_cache._read_header_only(example))
    assert rec["P"] == 9 and rec["n_block"] == 710
    assert rec["size_ok"] is True
    m = rec["meta"]
    assert m["precision_tag"] == "emu"
    assert m["blgsize_cm"] == 5
    assert m["n_snp_kept"] == 351539
    assert m["f2_cache_id"] == F2_CACHE_ID


def test_verify_passes_on_golden(example):
    assert _cache.main(["verify", example]) == 0
    # the stored content-address re-hashes correctly today
    assert _cache._sha256_file(str(EXAMPLE / "f2.bin")) == F2_CACHE_ID
    assert _cache._sha256_file(str(EXAMPLE / "pops.txt")) == POPS_SHA


def test_verify_fails_on_truncated(example, tmp_path):
    dst = tmp_path / "trunc"
    shutil.copytree(example, dst)
    f2 = dst / "f2.bin"
    f2.write_bytes(f2.read_bytes()[:-1])  # drop one byte -> size + hash both diverge
    assert _cache.main(["verify", str(dst)]) == 1


def test_minimal_meta_tolerated(example, tmp_path, capsys):
    # a cache like the .rds-import path writes: full payload, minimal 4-field meta, no cache_id
    dst = tmp_path / "minimal"
    dst.mkdir()
    shutil.copy(EXAMPLE / "f2.bin", dst / "f2.bin")
    shutil.copy(EXAMPLE / "pops.txt", dst / "pops.txt")
    (dst / "meta.json").write_text(json.dumps(
        {"format": "STPF2BK1", "P": 9, "n_block": 710, "provenance": "imported"}))
    assert _cache.main(["show", str(dst)]) == 0
    assert _cache.main(["ls", str(tmp_path)]) == 0
    # verify has no stored id to compare against -> reports, does NOT fail
    assert _cache.main(["verify", str(dst)]) == 0
    assert "no stored id" in capsys.readouterr().out


def test_pops_lists_labels(example, capsys):
    assert _cache.main(["pops", example]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 9


# --- Phase 3: the public facade (steppe.list_caches / cache_info / verify_cache) ---

def _make_cache(dst: Path, meta: dict) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(EXAMPLE / "f2.bin", dst / "f2.bin")
    shutil.copy(EXAMPLE / "pops.txt", dst / "pops.txt")
    (dst / "meta.json").write_text(json.dumps(meta))


def test_public_facade(example):
    rec = steppe.cache_info(example)
    assert rec["P"] == 9 and rec["meta"]["precision_tag"] == "emu"
    assert steppe.verify_cache(example) is True


def test_list_caches_facade(example, tmp_path):
    _make_cache(tmp_path / "a", {"format": "STPF2BK1", "P": 9, "n_block": 710, "f2_cache_id": F2_CACHE_ID})
    _make_cache(tmp_path / "b", {"format": "STPF2BK1", "P": 9, "n_block": 710})
    caches = {os.path.basename(c["path"]): c for c in steppe.list_caches(str(tmp_path))}
    assert set(caches) == {"a", "b"}
    assert caches["a"]["cache_id"] == F2_CACHE_ID
    assert caches["b"]["cache_id"] is None  # minimal meta -> no stored id
    assert all(c["P"] == 9 and c["n_block"] == 710 for c in caches.values())


def test_verify_cache_facade_fails_truncated(example, tmp_path):
    dst = tmp_path / "trunc"
    shutil.copytree(example, dst)
    (dst / "f2.bin").write_bytes((dst / "f2.bin").read_bytes()[:-1])
    assert steppe.verify_cache(str(dst)) is False


def test_index_invalidates_on_change(example, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    croot = tmp_path / "caches"
    _make_cache(croot / "c1", {"format": "STPF2BK1", "P": 9, "n_block": 710, "n_snp_kept": 111})
    first = _cache.list_caches(str(croot), use_index=True)
    assert len(first) == 1 and first[0]["n_snp_kept"] == 111
    idx = json.loads((tmp_path / "xdg" / "steppe" / "cache_index.json").read_text())
    assert len(idx) == 1
    _make_cache(croot / "c1", {"format": "STPF2BK1", "P": 9, "n_block": 710, "n_snp_kept": 999})
    f2 = croot / "c1" / "f2.bin"
    f2.write_bytes(f2.read_bytes() + b"\x00")  # size change -> index entry invalidates
    assert _cache.list_caches(str(croot), use_index=True)[0]["n_snp_kept"] == 999


# --- Phase 2: the dataset half (datasets presence-detect + get wiring) ---

def test_datasets_detects_present_panel(tmp_path, capsys):
    (tmp_path / "aadr_1240K").mkdir()
    (tmp_path / "aadr_1240K" / "v66.p1_1240K.aadr.geno").write_text("stub")
    assert _cache.main(["datasets", "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1240K" in out and "yes" in out  # present
    assert "HO" in out and "no" in out       # absent


def test_get_wires_to_download_script(tmp_path, monkeypatch):
    stub = tmp_path / "stub.sh"
    stub.write_text('#!/usr/bin/env bash\necho "got: $@"\nexit 0\n')
    monkeypatch.setenv("STEPPE_AADR_SCRIPT", str(stub))
    # no network: get shells out to the (stubbed) script and passes panel + outdir through
    assert _cache.main(["get", "HO", str(tmp_path / "out")]) == 0
