"""``steppe-cache`` — list, inspect, and verify f2 caches (STPF2BK1 dirs). GPU-free.

A cache is a directory holding ``f2.bin`` (a 64-byte header + the f2 / vpair slabs +
an int32 block-sizes trailer), ``pops.txt``, and an optional ``meta.json``. This tool
does no compute and parses nothing new: it reads the header, ``pops.txt``, and
whatever ``meta.json`` carries, and re-hashes ``f2.bin`` / ``pops.txt`` against the
content-address the extract writer already stamped (``f2_cache_id`` / ``pops_sha256``).

Design notes (from the scope doc + its critic pass):
  * The ONLY parse primitive is header-only (:func:`_read_header_only`) — it never
    touches the multi-GB f2/vpair payload, so ``ls`` over a tree of big caches is cheap
    and ``show`` / ``pops`` / ``verify`` never pay a full-slab read.
  * ``meta.json`` is an OPEN, best-effort record (a cache may carry the full schema-v1
    provenance, the minimal 4-field import meta, or none) — render whatever keys are
    present, ``-`` for the rest, never crash.
  * ``verify`` attests the ``f2.bin`` payload + ``pops.txt`` labels. When a cache has no
    stored ``f2_cache_id`` (the ``.rds``-import path writes none), verify recomputes and
    reports it as informational, not a failure. The ``meta.json`` record itself is not
    self-hashable and is not attested.

GPU-free by construction: stdlib only (no numpy, no compiled ``_core``, no CUDA) — the
same reason ``steppe-rds`` runs without a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from typing import Optional

# STPF2BK1 header — mirrors bindings/steppe/_rds.py and src/device/f2_disk_format.hpp.
_F2_MAGIC = b"STPF2BK1"
_F2_DTYPE_FP64 = 1
_F2_HEADER_SIZE = 64
_HEADER_FMT = "<8sIIiiQQQ16s"
_DASH = "-"


def _expected_f2_bytes(P: int, n_block: int) -> int:
    """The f2.bin size the writer produces: 64-byte header + f2 slab + vpair slab +
    int32 block_sizes trailer (f2_dir_writer.cpp: slab = P*P*n_block*8, two of them)."""
    return _F2_HEADER_SIZE + 2 * P * P * n_block * 8 + 4 * n_block


def _read_header_only(f2_dir: str) -> dict:
    """The sole parse primitive: read the 64-byte header + pops.txt + meta.json (if any),
    and STOP — never read the f2/vpair payload. Header P/n_block are authoritative."""
    f2_path = os.path.join(f2_dir, "f2.bin")
    with open(f2_path, "rb") as fh:
        header = fh.read(_F2_HEADER_SIZE)
    if len(header) != _F2_HEADER_SIZE:
        raise ValueError(f"{f2_path!r}: truncated header ({len(header)} < {_F2_HEADER_SIZE} bytes)")
    magic, _version, dtype, P, n_block, _f2_off, _vp_off, _bs_off, _res = struct.unpack(
        _HEADER_FMT, header)
    if magic != _F2_MAGIC:
        raise ValueError(f"{f2_path!r}: bad magic {magic!r} (expected {_F2_MAGIC!r})")
    if dtype != _F2_DTYPE_FP64:
        raise ValueError(f"{f2_path!r}: unsupported dtype {dtype} (only FP64={_F2_DTYPE_FP64})")

    pops = []
    pops_path = os.path.join(f2_dir, "pops.txt")
    if os.path.exists(pops_path):
        with open(pops_path) as ph:
            pops = [ln.strip() for ln in ph if ln.strip()]

    meta: dict = {}
    meta_path = os.path.join(f2_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as m:
                meta = json.load(m)
        except (OSError, ValueError):
            meta = {}

    f2_size = os.path.getsize(f2_path)
    return {
        "path": f2_dir,
        "P": P,
        "n_block": n_block,
        "pops": pops,
        "meta": meta,
        "f2_bin_size": f2_size,
        "expected_f2_bin_size": _expected_f2_bytes(P, n_block),
    }


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Streamed content hash, formatted ``sha256:<hex>`` to match the stored id."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024.0 or unit == "T":
            return f"{int(size)}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}T"


def _short_id(cache_id: Optional[str]) -> str:
    if not cache_id:
        return _DASH
    return cache_id.split(":", 1)[-1][:8]


def _dir_bytes(f2_dir: str) -> int:
    total = 0
    for name in ("f2.bin", "pops.txt", "meta.json"):
        p = os.path.join(f2_dir, name)
        if os.path.exists(p):
            total += os.path.getsize(p)
    return total


def _find_caches(root: str):
    """Every directory under ROOT that contains an f2.bin (a cache is defined by it)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if "f2.bin" in filenames:
            yield dirpath


def _default_root() -> str:
    """Honor $STEPPE_F2_DIR (the env the fit commands read) else the current directory."""
    return os.environ.get("STEPPE_F2_DIR") or os.getcwd()


# ---------------------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------------------

def cmd_ls(root: str, long_: bool, sort_: str) -> int:
    rows = []
    for d in _find_caches(root):
        try:
            info = _read_header_only(d)
        except (OSError, ValueError) as exc:
            print(f"steppe-cache: skipping {d!r}: {exc}", file=sys.stderr)
            continue
        meta = info["meta"]
        rows.append({
            "path": os.path.relpath(d, root),
            "P": info["P"],
            "n_block": info["n_block"],
            "snp_kept": meta.get("n_snp_kept", _DASH),
            "prec": meta.get("precision_tag", _DASH),
            "blgsize": meta.get("blgsize_cm", _DASH),
            "bytes": _dir_bytes(d),
            "cache_id": _short_id(meta.get("f2_cache_id")),
            "source": meta.get("pop_selection", _DASH),
        })
    if not rows:
        print(f"steppe-cache: no STPF2BK1 caches found under {root!r}", file=sys.stderr)
        return 0

    key = {
        "size": lambda r: r["bytes"],
        "pops": lambda r: r["P"],
        "blocks": lambda r: r["n_block"],
        "mtime": lambda r: os.path.getmtime(os.path.join(root, r["path"])),
    }.get(sort_)
    if key is not None:
        rows.sort(key=key, reverse=(sort_ in ("size", "mtime")))
    else:
        rows.sort(key=lambda r: r["path"])

    header = f"{'PATH':<28}{'P':>4}{'N_BLOCK':>9}{'SNP_KEPT':>11}{'PREC':>6}{'BLGSIZE':>9}{'SIZE':>9}  CACHE_ID"
    print(header)
    for r in rows:
        print(f"{r['path']:<28}{r['P']:>4}{r['n_block']:>9}{str(r['snp_kept']):>11}"
              f"{str(r['prec']):>6}{str(r['blgsize']):>9}{_human_size(r['bytes']):>9}  {r['cache_id']}")
        if long_:
            print(f"    source: {r['source']}")
    return 0


def _record(info: dict) -> dict:
    """The machine-readable record for ``show --json``: header-derived facts + raw meta."""
    return {
        "path": info["path"],
        "P": info["P"],
        "n_block": info["n_block"],
        "n_pops": len(info["pops"]),
        "f2_bin_size": info["f2_bin_size"],
        "expected_f2_bin_size": info["expected_f2_bin_size"],
        "size_ok": info["f2_bin_size"] == info["expected_f2_bin_size"],
        "meta": info["meta"],
    }


def cmd_show(f2_dir: str, as_json: bool) -> int:
    info = _read_header_only(f2_dir)
    if as_json:
        print(json.dumps(_record(info), indent=2))
        return 0
    size_ok = info["f2_bin_size"] == info["expected_f2_bin_size"]
    print(f"Cache: {info['path']}")
    print(f"  P (populations):  {info['P']}   (from f2.bin header)")
    print(f"  n_block:          {info['n_block']}   (from f2.bin header)")
    print(f"  pops.txt labels:  {len(info['pops'])}")
    mark = "OK" if size_ok else "MISMATCH"
    print(f"  f2.bin size:      {info['f2_bin_size']} bytes "
          f"(expected {info['expected_f2_bin_size']})  [{mark}]")
    meta = info["meta"]
    if not meta:
        print("  meta.json:        (none)")
        return 0
    print("  --- meta.json ---")
    for k, v in meta.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, separators=(",", ":"))
        print(f"  {k + ':':<26}{v}")
    return 0


def cmd_pops(f2_dir: str) -> int:
    pops_path = os.path.join(f2_dir, "pops.txt")
    with open(pops_path) as ph:
        for ln in ph:
            ln = ln.strip()
            if ln:
                print(ln)
    return 0


def cmd_verify(f2_dir: str, check_sources: bool) -> int:
    info = _read_header_only(f2_dir)
    meta = info["meta"]
    checks = []  # (label, status: ok|fail|info, detail)

    # 1. size vs the writer's arithmetic
    if info["f2_bin_size"] == info["expected_f2_bin_size"]:
        checks.append(("f2.bin size", "ok", f"{info['f2_bin_size']} bytes"))
    else:
        checks.append(("f2.bin size", "fail",
                       f"{info['f2_bin_size']} != expected {info['expected_f2_bin_size']}"))

    # 2. header vs meta P/n_block (only if meta carries them)
    for field in ("P", "n_block"):
        if field in meta:
            if meta[field] == info[field]:
                checks.append((f"header {field}", "ok", f"{info[field]}"))
            else:
                checks.append((f"header {field}", "fail",
                               f"header {info[field]} != meta {meta[field]}"))

    # 3. content-address re-hash (G1: report-only when there is no stored reference)
    computed_f2 = _sha256_file(os.path.join(f2_dir, "f2.bin"))
    stored_f2 = meta.get("f2_cache_id")
    if stored_f2:
        checks.append(("f2_cache_id", "ok" if computed_f2 == stored_f2 else "fail",
                       computed_f2 if computed_f2 == stored_f2 else f"{computed_f2} != stored {stored_f2}"))
    else:
        checks.append(("f2_cache_id", "info", f"no stored id; computed {computed_f2}"))

    pops_path = os.path.join(f2_dir, "pops.txt")
    if os.path.exists(pops_path):
        computed_pops = _sha256_file(pops_path)
        stored_pops = meta.get("pops_sha256")
        if stored_pops:
            checks.append(("pops_sha256", "ok" if computed_pops == stored_pops else "fail",
                           computed_pops if computed_pops == stored_pops else f"{computed_pops} != stored {stored_pops}"))
        else:
            checks.append(("pops_sha256", "info", f"no stored id; computed {computed_pops}"))

    # 4. optional source-file hashes (off by default; absent files are info, not fail)
    if check_sources:
        src = meta.get("source", {})
        if src.get("source_hash_computed"):
            base = os.path.dirname(os.path.abspath(f2_dir))
            for key, kind in (("geno", "geno"), ("snp", "snp"), ("ind", "ind")):
                name = src.get(kind)
                stored = src.get(f"{kind}_sha256")
                if not name:
                    continue
                path = name if os.path.isabs(name) else os.path.join(base, name)
                if not os.path.exists(path):
                    checks.append((f"source {kind}", "info", f"{name} not present"))
                elif stored:
                    got = _sha256_file(path)
                    checks.append((f"source {kind}", "ok" if got == stored else "fail",
                                   got if got == stored else f"{got} != stored {stored}"))
        else:
            checks.append(("sources", "info", "source_hash_computed is false"))

    print(f"verify {info['path']}:")
    for label, status, detail in checks:
        tag = {"ok": "OK  ", "fail": "FAIL", "info": "--  "}[status]
        print(f"  [{tag}] {label:<14} {detail}")
    print("  (verify attests the f2.bin payload + pops.txt labels; the meta.json record itself is not hashable.)")

    failed = [c for c in checks if c[1] == "fail"]
    if failed:
        print(f"steppe-cache: verify FAILED ({len(failed)} check(s))", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    """``steppe-cache`` — the GPU-free f2-cache manager CLI (the ``[project.scripts]`` entry).

        steppe-cache ls [ROOT]         scan for STPF2BK1 caches (default: $STEPPE_F2_DIR or cwd)
        steppe-cache show <DIR>        pretty-print one cache (--json for the parsed record)
        steppe-cache pops <DIR>        the population labels, one per line
        steppe-cache verify <DIR>      re-hash f2.bin/pops.txt vs the stored content-address

    Pure on-disk inspection — no GPU. Returns a process exit code (0 ok, 1 on error / a
    failed integrity check)."""
    parser = argparse.ArgumentParser(
        prog="steppe-cache",
        description="List, inspect, and verify steppe f2 caches (STPF2BK1 dirs). No GPU.",
    )
    sub = parser.add_subparsers(dest="mode", required=True, metavar="{ls,show,pops,verify}")

    p_ls = sub.add_parser("ls", help="scan a root for STPF2BK1 caches and tabulate them")
    p_ls.add_argument("root", nargs="?", default=None,
                      help="directory to scan (default: $STEPPE_F2_DIR or the current dir)")
    p_ls.add_argument("--long", action="store_true", help="add the pop-selection source line")
    p_ls.add_argument("--sort", choices=("path", "size", "pops", "blocks", "mtime"),
                      default="path", help="row ordering (default: path)")

    p_show = sub.add_parser("show", help="pretty-print one cache's header + meta.json")
    p_show.add_argument("dir", help="the f2 cache directory (holds f2.bin + pops.txt)")
    p_show.add_argument("--json", action="store_true", help="emit the parsed record as JSON")

    p_pops = sub.add_parser("pops", help="print the population labels (reads pops.txt)")
    p_pops.add_argument("dir", help="the f2 cache directory")

    p_verify = sub.add_parser("verify", help="re-hash and check integrity vs the stored id")
    p_verify.add_argument("dir", help="the f2 cache directory")
    p_verify.add_argument("--check-sources", action="store_true",
                          help="also re-hash geno/snp/ind if source_hash_computed (may be large)")

    args = parser.parse_args(argv)
    try:
        if args.mode == "ls":
            return cmd_ls(args.root or _default_root(), args.long, args.sort)
        if args.mode == "show":
            return cmd_show(args.dir, args.json)
        if args.mode == "pops":
            return cmd_pops(args.dir)
        if args.mode == "verify":
            return cmd_verify(args.dir, args.check_sources)
    except (OSError, ValueError) as exc:
        print(f"steppe-cache: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # `python -m steppe._cache ...` (PYTHONPATH boxes) — same entry.
    sys.exit(main())
