#!/usr/bin/env python3
"""run_bench.py — entrypoint for the steppe FST/PCA validation+benchmark harness.

Bootstraps the numpy<2 venv (once, cached), re-execs itself under that interpreter so the
scikit-allel/sklearn oracles import, warms the page cache, runs bench_fst/bench_pca, and
writes out/{fst,pca}_table.{md,tsv} + out/bench_meta.json. SKIPs cleanly (exit 0) if the
GPU or the data are absent, matching the existing ctest gates.

Bare invocation (no CMake):
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 TMPDIR=/workspace/tmp \
  python3 tests/bench/run_bench.py \
    --steppe /workspace/steppe/build-recheck/bin/steppe \
    --fit9   /workspace/data/aadr/fixtures_eig/v66_fit9_ped \
    --full   /workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB \
    --out    tests/bench/out
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

VENV = os.environ.get("STEPPE_BENCH_VENV", "/workspace/tmp/pca_oracle_venv")
_SENTINEL = "STEPPE_BENCH_IN_VENV"


def bootstrap_venv() -> str:
    """Create the numpy<2 venv (numpy + scikit-allel + scikit-learn) if absent. Cached."""
    py = os.path.join(VENV, "bin", "python")
    if os.path.exists(py):
        return py
    print(f"[run_bench] bootstrapping numpy<2 venv at {VENV} (one-time, ~1-2 min)...",
          flush=True)
    subprocess.run([sys.executable, "-m", "venv", VENV], check=True)
    subprocess.run([py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
    subprocess.run([py, "-m", "pip", "install", "--quiet",
                    "numpy<2", "scikit-allel", "scikit-learn"], check=True)
    return py


def have_gpu() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return False


def data_present(args) -> bool:
    ok = os.path.exists(args.steppe)
    ok = ok and all(os.path.exists(args.fit9 + e) for e in (".bed", ".bim", ".fam"))
    ok = ok and all(os.path.exists(args.full + e) for e in (".geno", ".snp", ".ind"))
    return ok


def gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader", "-i", "0"],
            text=True).strip()
    except Exception:
        return "unknown"


def git_commit(repo: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steppe", default="/workspace/steppe/build-recheck/bin/steppe")
    ap.add_argument("--fit9", default="/workspace/data/aadr/fixtures_eig/v66_fit9_ped")
    ap.add_argument("--full",
                    default="/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--feature", choices=["fst", "pca", "both"], default="both")
    ap.add_argument("--device", default="0")
    ap.add_argument("--quick", action="store_true",
                    help="smaller top-K/N scaling points (faster smoke run)")
    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Pre-venv gates: skip cleanly (exit 0) like the ctest gates.
    if not have_gpu():
        print("[run_bench] SKIP: no CUDA GPU visible (exit 0).")
        return 0
    if not data_present(args):
        print("[run_bench] SKIP: steppe binary or AADR data absent (exit 0).")
        return 0

    # Re-exec under the venv interpreter so numpy<2 / scikit-allel / sklearn import.
    if os.environ.get(_SENTINEL) != "1":
        py = bootstrap_venv()
        env = dict(os.environ)
        env[_SENTINEL] = "1"
        env.setdefault("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
        os.execve(py, [py, os.path.abspath(__file__)] + sys.argv[1:], env)

    # --- Under the venv from here ------------------------------------------
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench_common as bc

    tmp = os.environ.get("TMPDIR", "/tmp")
    scratch = os.path.join(tmp, "bench_scratch")
    # Clean scratch first (disk-tight), but KEEP the venv (it lives elsewhere).
    if os.path.isdir(scratch):
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    cfg = bc.Cfg(steppe=args.steppe, fit9=args.fit9, full=args.full, out=args.out,
                 scratch=scratch, device=args.device, seed=args.seed, quick=args.quick)

    # Warm the page cache (untimed) so the first scaling point is not I/O-skewed.
    print("[run_bench] warming page cache...", flush=True)
    for path in (cfg.full + ".geno", cfg.fit9 + ".bed"):
        try:
            with open(path, "rb") as f:
                while f.read(1 << 24):
                    pass
        except OSError:
            pass

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "steppe_binary": args.steppe,
        "gpu": gpu_name(),
        "cpu": "AMD EPYC 9654 96-Core (nproc=384)",
        "device": args.device,
        "quick": args.quick,
        "seed": args.seed,
        "venv": VENV,
        "panels": {
            "fit9_ped": args.fit9 + " (PLINK, 430 samples x 584,131 variants, 9 pops)",
            "full_1240K": args.full + " (EIGENSTRAT, 23,089 samples x 1,233,013 SNPs)",
        },
        "label_policy": {
            "plink2": "LIVE (fast C) — parity, missingness cross-check, runtime, M-curve",
            "scikit-allel/sklearn": "LIVE only at N<=~514; larger-N reference numbers CITED",
            "synthetic": "ONLY missingness injected into real fit9_ped genotypes",
        },
        "tables": {},
    }
    try:
        version = subprocess.check_output([args.steppe, "--version"],
                                          text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        version = "unknown"
    meta["steppe_version"] = version

    rc = 0
    if args.feature in ("fst", "both"):
        import bench_fst
        print("[run_bench] FST table...", flush=True)
        t = bench_fst.main(cfg)
        t.write(args.out)
        meta["tables"]["fst"] = [r.__dict__ for r in sorted(t.rows, key=lambda x: x.idx)]
        fails = [r.idx for r in t.rows if r.status == "FAIL"]
        if fails:
            print(f"[run_bench] FST FAIL rows: {fails}")
            rc = 1
        print(t.to_md())

    if args.feature in ("pca", "both"):
        import bench_pca
        print("[run_bench] PCA table...", flush=True)
        t = bench_pca.main(cfg)
        t.write(args.out)
        meta["tables"]["pca"] = [r.__dict__ for r in sorted(t.rows, key=lambda x: x.idx)]
        fails = [r.idx for r in t.rows if r.status == "FAIL"]
        if fails:
            print(f"[run_bench] PCA FAIL rows: {fails}")
            rc = 1
        print(t.to_md())

    with open(os.path.join(args.out, "bench_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[run_bench] wrote tables + bench_meta.json under {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
