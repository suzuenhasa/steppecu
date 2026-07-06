"""``steppe-cache`` — list, inspect, verify f2 caches and manage AADR datasets. GPU-free.

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
  * The dataset half (``datasets`` / ``get``) WRAPS ``docs/download-aadr.sh`` — it does
    not re-implement the Harvard Dataverse resolve/MD5 logic.

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

# The AADR panels docs/download-aadr.sh understands (with approximate genotype sizes).
_AADR_PANELS = ("1240K", "HO", "2M")
_AADR_APPROX = {"1240K": "~7G", "HO": "~4G", "2M": "~12G"}


# ---------------------------------------------------------------------------------------
# Parse primitives (header-only; never touch the f2/vpair payload)
# ---------------------------------------------------------------------------------------

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


def _summary(info: dict) -> dict:
    """The lightweight per-cache record for ``ls`` / :func:`list_caches` (no payload)."""
    meta = info["meta"]
    return {
        "path": os.path.abspath(info["path"]),
        "P": info["P"],
        "n_block": info["n_block"],
        "n_snp_kept": meta.get("n_snp_kept"),
        "precision_tag": meta.get("precision_tag"),
        "blgsize_cm": meta.get("blgsize_cm"),
        "cache_id": meta.get("f2_cache_id"),
        "pop_selection": meta.get("pop_selection"),
        "bytes": _dir_bytes(info["path"]),
    }


def _record(info: dict) -> dict:
    """The machine-readable record for ``show --json`` / :func:`cache_info`."""
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


# ---------------------------------------------------------------------------------------
# Integrity check (shared by the CLI verify and the verify_cache facade)
# ---------------------------------------------------------------------------------------

def _verify_checks(f2_dir: str, check_sources: bool) -> list:
    """Build the integrity check list: (label, status: ok|fail|info, detail)."""
    info = _read_header_only(f2_dir)
    meta = info["meta"]
    checks = []

    if info["f2_bin_size"] == info["expected_f2_bin_size"]:
        checks.append(("f2.bin size", "ok", f"{info['f2_bin_size']} bytes"))
    else:
        checks.append(("f2.bin size", "fail",
                       f"{info['f2_bin_size']} != expected {info['expected_f2_bin_size']}"))

    for field in ("P", "n_block"):
        if field in meta:
            if meta[field] == info[field]:
                checks.append((f"header {field}", "ok", f"{info[field]}"))
            else:
                checks.append((f"header {field}", "fail",
                               f"header {info[field]} != meta {meta[field]}"))

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

    if check_sources:
        src = meta.get("source", {})
        if src.get("source_hash_computed"):
            base = os.path.dirname(os.path.abspath(f2_dir))
            for kind in ("geno", "snp", "ind"):
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

    return checks


# ---------------------------------------------------------------------------------------
# Opt-in scan index (off by default; scan-on-demand stays the source of truth)
# ---------------------------------------------------------------------------------------

def _index_path() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "steppe", "cache_index.json")


def _load_index() -> dict:
    try:
        with open(_index_path()) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_index(index: dict) -> None:
    path = _index_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(index, fh)
    except OSError:
        pass  # the index is a best-effort memo; a write failure never breaks ls


# ---------------------------------------------------------------------------------------
# Public facade (GPU-free; mirrors export_f2_rds / import_f2_rds in __init__.py)
# ---------------------------------------------------------------------------------------

def list_caches(root: Optional[str] = None, use_index: bool = False) -> list:
    """Return a lightweight record per STPF2BK1 cache under ``root`` (default:
    $STEPPE_F2_DIR or cwd). With ``use_index``, memoize the header scan in
    ``~/.cache/steppe/cache_index.json``, invalidated by f2.bin mtime + size."""
    root = root or _default_root()
    index = _load_index() if use_index else None
    out = []
    for d in _find_caches(root):
        try:
            if index is not None:
                key = os.path.abspath(d)
                st = os.stat(os.path.join(d, "f2.bin"))
                ent = index.get(key)
                if ent and ent.get("mtime") == st.st_mtime and ent.get("size") == st.st_size:
                    out.append(ent["summary"])
                    continue
                summary = _summary(_read_header_only(d))
                index[key] = {"mtime": st.st_mtime, "size": st.st_size, "summary": summary}
                out.append(summary)
            else:
                out.append(_summary(_read_header_only(d)))
        except (OSError, ValueError):
            continue
    if index is not None:
        _save_index(index)
    return out


def cache_info(f2_dir: str) -> dict:
    """The full parsed record for one cache (header facts + raw meta.json)."""
    return _record(_read_header_only(f2_dir))


def verify_cache(f2_dir: str, check_sources: bool = False) -> bool:
    """True iff every present integrity check passes (a missing stored id is not a
    failure). Attests the f2.bin payload + pops.txt labels, not the meta.json record."""
    return not any(c[1] == "fail" for c in _verify_checks(f2_dir, check_sources))


# ---------------------------------------------------------------------------------------
# AADR dataset half (wrap docs/download-aadr.sh; do not re-implement)
# ---------------------------------------------------------------------------------------

def _panel_geno(panel: str) -> str:
    return f"v66.p1_{panel}.aadr.geno"


def _panel_present(root: str, panel: str) -> Optional[str]:
    name = _panel_geno(panel)
    for d in (root, os.path.join(root, f"aadr_{panel}")):
        if os.path.exists(os.path.join(d, name)):
            return d
    return None


def _find_download_script() -> Optional[str]:
    """Locate docs/download-aadr.sh: $STEPPE_AADR_SCRIPT, else repo-relative, else cwd/docs."""
    env = os.environ.get("STEPPE_AADR_SCRIPT")
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "..", "..", "docs", "download-aadr.sh"),
              os.path.join(os.getcwd(), "docs", "download-aadr.sh")):
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


# ---------------------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------------------

def cmd_ls(root: str, long_: bool, sort_: str, use_index: bool) -> int:
    caches = list_caches(root, use_index=use_index)
    if not caches:
        print(f"steppe-cache: no STPF2BK1 caches found under {root!r}", file=sys.stderr)
        return 0

    key = {
        "size": lambda c: c["bytes"],
        "pops": lambda c: c["P"],
        "blocks": lambda c: c["n_block"],
        "mtime": lambda c: os.path.getmtime(os.path.join(c["path"], "f2.bin")),
    }.get(sort_)
    if key is not None:
        caches.sort(key=key, reverse=(sort_ in ("size", "mtime")))
    else:
        caches.sort(key=lambda c: c["path"])

    print(f"{'PATH':<28}{'P':>4}{'N_BLOCK':>9}{'SNP_KEPT':>11}{'PREC':>6}{'BLGSIZE':>9}{'SIZE':>9}  CACHE_ID")
    for c in caches:
        rel = os.path.relpath(c["path"], root)
        print(f"{rel:<28}{c['P']:>4}{c['n_block']:>9}{str(c['n_snp_kept'] or _DASH):>11}"
              f"{str(c['precision_tag'] or _DASH):>6}{str(c['blgsize_cm'] or _DASH):>9}"
              f"{_human_size(c['bytes']):>9}  {_short_id(c['cache_id'])}")
        if long_:
            print(f"    source: {c['pop_selection'] or _DASH}")
    return 0


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
    checks = _verify_checks(f2_dir, check_sources)
    print(f"verify {f2_dir}:")
    for label, status, detail in checks:
        tag = {"ok": "OK  ", "fail": "FAIL", "info": "--  "}[status]
        print(f"  [{tag}] {label:<14} {detail}")
    print("  (verify attests the f2.bin payload + pops.txt labels; the meta.json record itself is not hashable.)")
    failed = [c for c in checks if c[1] == "fail"]
    if failed:
        print(f"steppe-cache: verify FAILED ({len(failed)} check(s))", file=sys.stderr)
        return 1
    return 0


def cmd_datasets(root: str) -> int:
    print(f"{'PANEL':<8}{'SIZE~':>8}  PRESENT  PATH")
    for panel in _AADR_PANELS:
        loc = _panel_present(root, panel)
        print(f"{panel:<8}{_AADR_APPROX[panel]:>8}  {'yes' if loc else 'no':<7}  {loc or _DASH}")
    return 0


def cmd_get(panel: str, outdir: Optional[str]) -> int:
    if panel not in _AADR_PANELS:
        print(f"steppe-cache: unknown panel {panel!r} (choose {'|'.join(_AADR_PANELS)})",
              file=sys.stderr)
        return 2
    script = _find_download_script()
    if script is None:
        print("steppe-cache: download-aadr.sh not found. Set $STEPPE_AADR_SCRIPT or run it "
              "directly from the repo (docs/download-aadr.sh).", file=sys.stderr)
        return 1
    import subprocess
    cmd = ["bash", script, panel] + ([outdir] if outdir else [])
    print(f"steppe-cache: running {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main(argv=None):
    """``steppe-cache`` — the GPU-free f2-cache + AADR-dataset manager CLI.

        steppe-cache ls [ROOT]         scan for STPF2BK1 caches (default: $STEPPE_F2_DIR or cwd)
        steppe-cache show <DIR>        pretty-print one cache (--json for the parsed record)
        steppe-cache pops <DIR>        the population labels, one per line
        steppe-cache verify <DIR>      re-hash f2.bin/pops.txt vs the stored content-address
        steppe-cache datasets          which AADR panels (1240K|HO|2M) are present locally
        steppe-cache get <PANEL>       fetch a panel (wraps docs/download-aadr.sh; needs bash)

    Pure on-disk inspection — no GPU. Returns a process exit code (0 ok, 1 on error / a
    failed integrity check)."""
    parser = argparse.ArgumentParser(
        prog="steppe-cache",
        description="List, inspect, and verify steppe f2 caches, and manage AADR datasets. No GPU.",
    )
    sub = parser.add_subparsers(dest="mode", required=True,
                                metavar="{ls,show,pops,verify,datasets,get}")

    p_ls = sub.add_parser("ls", help="scan a root for STPF2BK1 caches and tabulate them")
    p_ls.add_argument("root", nargs="?", default=None,
                      help="directory to scan (default: $STEPPE_F2_DIR or the current dir)")
    p_ls.add_argument("--long", action="store_true", help="add the pop-selection source line")
    p_ls.add_argument("--sort", choices=("path", "size", "pops", "blocks", "mtime"),
                      default="path", help="row ordering (default: path)")
    p_ls.add_argument("--index", action="store_true",
                      help="memoize the scan in ~/.cache/steppe (invalidated by mtime+size)")

    p_show = sub.add_parser("show", help="pretty-print one cache's header + meta.json")
    p_show.add_argument("dir", help="the f2 cache directory (holds f2.bin + pops.txt)")
    p_show.add_argument("--json", action="store_true", help="emit the parsed record as JSON")

    p_pops = sub.add_parser("pops", help="print the population labels (reads pops.txt)")
    p_pops.add_argument("dir", help="the f2 cache directory")

    p_verify = sub.add_parser("verify", help="re-hash and check integrity vs the stored id")
    p_verify.add_argument("dir", help="the f2 cache directory")
    p_verify.add_argument("--check-sources", action="store_true",
                          help="also re-hash geno/snp/ind if source_hash_computed (may be large)")

    p_ds = sub.add_parser("datasets", help="which AADR panels are known + present locally")
    p_ds.add_argument("--dir", default=None, help="root to look under (default: cwd)")

    p_get = sub.add_parser("get", help="fetch an AADR panel (wraps docs/download-aadr.sh)")
    p_get.add_argument("panel", choices=_AADR_PANELS, help="the AADR panel to fetch")
    p_get.add_argument("outdir", nargs="?", default=None,
                       help="output directory (default: ./aadr_<PANEL>)")

    args = parser.parse_args(argv)
    try:
        if args.mode == "ls":
            return cmd_ls(args.root or _default_root(), args.long, args.sort, args.index)
        if args.mode == "show":
            return cmd_show(args.dir, args.json)
        if args.mode == "pops":
            return cmd_pops(args.dir)
        if args.mode == "verify":
            return cmd_verify(args.dir, args.check_sources)
        if args.mode == "datasets":
            return cmd_datasets(args.dir or os.getcwd())
        if args.mode == "get":
            return cmd_get(args.panel, args.outdir)
    except (OSError, ValueError) as exc:
        print(f"steppe-cache: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # `python -m steppe._cache ...` (PYTHONPATH boxes) — same entry.
    sys.exit(main())
