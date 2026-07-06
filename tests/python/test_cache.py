"""py_cache — GPU-free steppe-cache manager tests (run under the py_qpadm pytest lane).

The oracle is deterministic self-consistency against the committed real-AADR golden
docs/examples/f2_9pop/: the parse must reproduce the fields the cache carries, and
verify's re-hash must equal the content-address the extract writer stamped."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

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
