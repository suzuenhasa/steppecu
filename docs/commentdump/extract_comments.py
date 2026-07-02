#!/usr/bin/env python3
"""Dump comment-carrying lines from the steppe source tree into per-file text files.

For every source file under the repo (skipping build/, docs/, .git, data/, etc.), this writes a
MIRRORED file under docs/commentdump/ holding just the lines that carry a comment, tagged with
their original line numbers. Language-aware:

  C / C++ / CUDA (.c .cc .cpp .cxx .h .hpp .hh .cu .cuh):  `//` line comments + `/* ... */` blocks
  Python / CMake / shell / toml / yaml (.py .cmake .sh .toml .cfg .yml .yaml, CMakeLists.txt):  `#`

It also writes docs/commentdump/INDEX.txt: every dumped file with its comment-line count, total
lines, and comment density (sorted densest-first — handy for spotting over-commented files).

Heuristic, not a real parser: a `//`/`#` inside a string literal is still counted (rare here),
and Python docstrings are NOT treated as comments (they're strings). Good enough for a dump.

Usage:
  python3 docs/commentdump/extract_comments.py               # regenerate (cleans stale output first)
  python3 docs/commentdump/extract_comments.py --no-line-numbers
  python3 docs/commentdump/extract_comments.py --keep        # keep previous output (no clean)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SCRIPT = Path(__file__).resolve()
OUT = SCRIPT.parent                 # docs/commentdump/
REPO = OUT.parents[1]               # docs/commentdump -> docs -> repo root

# Directory names skipped anywhere in the path (build artifacts, docs, vendored, data, local).
SKIP_DIRS = {
    ".git", "build", "build-rel", "build-ci", "docs", ".claude", "experiments",
    "agentscripts", "__pycache__", ".cache", ".vscode", ".idea", "data", "aadr", "node_modules",
}
C_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".cu", ".cuh"}
HASH_EXT = {".py", ".cmake", ".sh", ".toml", ".cfg", ".yml", ".yaml"}
HASH_NAMES = {"CMakeLists.txt"}


def family(p: Path):
    if p.suffix in C_EXT:
        return "c"
    if p.suffix in HASH_EXT or p.name in HASH_NAMES:
        return "hash"
    return None


def comment_lines_c(lines):
    """Yield (lineno, text) for lines carrying a // or /* ... */ comment (naive block tracking)."""
    in_block = False
    for i, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        is_comment = False
        if in_block:
            is_comment = True
            if "*/" in line:
                in_block = False
        else:
            if "/*" in line:
                is_comment = True
                if "*/" not in line.split("/*", 1)[1]:  # block stays open past this line
                    in_block = True
            if "//" in line:
                is_comment = True
        if is_comment and line.strip():
            yield i, line


def comment_lines_hash(lines):
    """Yield (lineno, text) for lines containing a # comment."""
    for i, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        if "#" in line and line.strip():
            yield i, line


def iter_sources(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if family(p):
            yield p


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dump comment lines per source file.")
    ap.add_argument("--no-line-numbers", action="store_true", help="omit original line numbers")
    ap.add_argument("--keep", action="store_true", help="do not delete previous output first")
    args = ap.parse_args(argv)

    if not args.keep:  # wipe prior generated output, but never this script
        for item in OUT.iterdir():
            if item.resolve() == SCRIPT:
                continue
            shutil.rmtree(item) if item.is_dir() else item.unlink()

    index = []
    total_c = total_t = n_files = 0
    for src in iter_sources(REPO):
        fam = family(src)
        lines = src.read_text(errors="replace").splitlines(keepends=True)
        extractor = comment_lines_c if fam == "c" else comment_lines_hash
        hits = list(extractor(lines))
        if not hits:
            continue
        rel = src.relative_to(REPO)
        dest = OUT / (str(rel) + ".txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w") as f:
            f.write(f"# comments from {rel}  ({len(hits)} comment lines / {len(lines)} total)\n\n")
            for ln, s in hits:
                f.write(f"{s}\n" if args.no_line_numbers else f"{ln:>6}: {s}\n")
        pct = 100.0 * len(hits) / len(lines) if lines else 0.0
        index.append((str(rel), len(hits), len(lines), pct))
        total_c += len(hits)
        total_t += len(lines)
        n_files += 1

    index.sort(key=lambda r: r[3], reverse=True)  # densest-first
    overall = 100.0 * total_c / total_t if total_t else 0.0
    with (OUT / "INDEX.txt").open("w") as f:
        f.write(f"# steppe comment dump — {n_files} files, {total_c} comment lines / "
                f"{total_t} total ({overall:.1f}% overall)\n")
        f.write("# sorted by comment density (comment lines / total lines)\n\n")
        f.write(f"{'density':>8}  {'cmts':>6} {'total':>6}  file\n")
        for rel, c, t, pct in index:
            f.write(f"{pct:>7.1f}%  {c:>6} {t:>6}  {rel}\n")

    print(f"wrote {n_files} comment-dump files under {OUT}")
    print(f"total: {total_c} comment lines / {total_t} source lines ({overall:.1f}%)")
    print(f"index: {OUT / 'INDEX.txt'}")


if __name__ == "__main__":
    main()
