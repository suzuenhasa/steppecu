#!/usr/bin/env python3
"""bench_common.py — shared library for the steppe FST/PCA validation+benchmark harness.

No side effects on import. Provides:
  * binary/data discovery + a Cfg dataclass,
  * run_timed() / run_with_mem() (wall via perf_counter; GPU peak via a background
    nvidia-smi sampler; host peak RSS via /usr/bin/time -v),
  * plink2 wrappers (plink_fst_wc, plink_thin, plink_recode_A),
  * .fam(col6) / .ind(col3) pop-table parsers + greedy_pops_for_n(),
  * a vectorized plink2 .raw loader (NO pure-Python per-value loop — critic fix #5),
  * the numpy-FP64 Weir-Cockerham FST oracle (same algebra as core/internal/wc_fst.hpp),
  * the Patterson-standardize + numpy.linalg.eigh PCA oracle,
  * idempotent scratch builders: ensure_thinned_bed() / ensure_injected_bed() /
    write_relabeled_fam() (shared by both features so nothing is rebuilt twice),
  * the Row/Table emitter (TSV + markdown; status in {PASS,FAIL,measured,cited,flag}).

Run under the numpy<2 venv (numpy + scikit-allel + scikit-learn importable).
"""
from __future__ import annotations

import glob
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# --- Config ------------------------------------------------------------------

@dataclass
class Cfg:
    steppe: str                    # steppe binary
    fit9: str                      # PLINK .bed prefix (fit9_ped)
    full: str                      # EIGENSTRAT prefix (full 1240K)
    out: str                       # output dir
    scratch: str                   # scratch dir (thinned/injected beds, .raw, venv output)
    device: str = "0"
    seed: int = 1
    quick: bool = False
    plink2: str = "plink2"


def steppe_env() -> dict:
    env = dict(os.environ)
    env.setdefault("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
    return env


# --- Timing / memory ---------------------------------------------------------

def run_timed(cmd: list[str], env: Optional[dict] = None,
              check: bool = True) -> tuple[float, subprocess.CompletedProcess]:
    """Run cmd, return (wall_seconds, CompletedProcess). Wall via perf_counter."""
    env = env or steppe_env()
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if check and cp.returncode != 0:
        raise RuntimeError(
            f"command failed ({cp.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{cp.stdout[-2000:]}\nstderr:\n{cp.stderr[-2000:]}")
    return wall, cp


class _GpuSampler:
    """Background nvidia-smi sampler (memory.used, MiB) at 200 ms per the box note."""

    def __init__(self, device: str = "0", interval: float = 0.2):
        self.device = device
        self.interval = interval
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self.samples: list[int] = []

    def _sample_once(self) -> Optional[int]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", self.device],
                text=True, stderr=subprocess.DEVNULL)
            return int(out.strip().splitlines()[0])
        except Exception:
            return None

    def _loop(self):
        while not self._stop.is_set():
            v = self._sample_once()
            if v is not None:
                self.samples.append(v)
            self._stop.wait(self.interval)

    def __enter__(self):
        base = self._sample_once()
        self.baseline = base if base is not None else 0
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2.0)

    @property
    def peak(self) -> int:
        return max(self.samples) if self.samples else self.baseline


def run_with_mem(cmd: list[str], env: Optional[dict] = None,
                 device: str = "0") -> tuple[float, int, int, float]:
    """Run cmd under /usr/bin/time -v while sampling GPU memory.

    Returns (wall_s, gpu_peak_mib, gpu_delta_mib, host_rss_mib).
    """
    env = env or steppe_env()
    timev = ["/usr/bin/time", "-v"] + cmd
    with _GpuSampler(device) as g:
        t0 = time.perf_counter()
        cp = subprocess.run(timev, env=env, capture_output=True, text=True)
        wall = time.perf_counter() - t0
    if cp.returncode != 0:
        raise RuntimeError(
            f"timed command failed ({cp.returncode}): {' '.join(cmd)}\n{cp.stderr[-2000:]}")
    rss_kb = 0
    for line in cp.stderr.splitlines():
        if "Maximum resident set size" in line:
            rss_kb = int(line.rsplit(":", 1)[1].strip())
    rss_mib = rss_kb / 1024.0
    return wall, g.peak, max(0, g.peak - g.baseline), rss_mib


# --- plink2 wrappers ---------------------------------------------------------

def plink_thin(cfg: Cfg, prefix: str, m: int, out_prefix: str) -> str:
    """--thin-count M --make-bed (samples unchanged); caller restores the .fam."""
    run_timed([cfg.plink2, "--bfile", prefix, "--thin-count", str(m),
               "--seed", str(cfg.seed), "--make-bed", "--out", out_prefix])
    return out_prefix


def plink_recode_A(cfg: Cfg, prefix: str, keep: Optional[str], out_prefix: str) -> str:
    cmd = [cfg.plink2, "--bfile", prefix, "--recode", "A", "--out", out_prefix]
    if keep:
        cmd += ["--keep", keep]
    run_timed(cmd)
    return out_prefix + ".raw"


def plink_fst_wc(cfg: Cfg, prefix: str, pheno: str, keep: str,
                 out_prefix: str, per_variant: bool = True) -> tuple[float, str, Optional[str]]:
    """plink2 --fst POP method=wc [report-variants]. Returns (wall_s, summary, var|None)."""
    cmd = [cfg.plink2, "--bfile", prefix, "--keep", keep, "--pheno", pheno,
           "--fst", "POP", "method=wc"]
    if per_variant:
        cmd += ["report-variants"]
    cmd += ["--out", out_prefix]
    wall, _ = run_timed(cmd)
    summ = out_prefix + ".fst.summary"
    var = None
    hits = glob.glob(out_prefix + ".*.fst.var")
    if hits:
        var = hits[0]
    return wall, summ, var


def parse_fst_summary(path: str) -> float:
    """Read plink2 <out>.fst.summary -> WC_FST (single pair)."""
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        col = head.index("HUDSON_FST") if "HUDSON_FST" in head else head.index("WC_FST")
        line = f.readline().rstrip("\n").split("\t")
    return float(line[col])


def parse_fst_var(path: str) -> dict[str, float]:
    """Read plink2 <out>.<A>.<B>.fst.var -> {ID: WC_FST}."""
    out: dict[str, float] = {}
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        id_col = head.index("ID")
        fst_col = head.index("WC_FST") if "WC_FST" in head else head.index("HUDSON_FST")
        for line in f:
            t = line.rstrip("\n").split("\t")
            v = t[fst_col]
            if v in ("nan", "NA", ""):
                continue
            out[t[id_col]] = float(v)
    return out


# --- Pop-table parsers -------------------------------------------------------

_FAM_IGNORE = {"9", "-9", "0"}


def read_fam(prefix: str) -> list[tuple[str, str, str]]:
    """Return [(FID, IID, POP)] from PLINK .fam col1/col2/col6 (steppe read_fam semantics)."""
    rows = []
    with open(prefix + ".fam") as f:
        for line in f:
            t = line.split()
            if len(t) < 6:
                continue
            rows.append((t[0], t[1], t[5]))
    return rows


def fam_pop_counts(prefix: str) -> dict[str, int]:
    c: dict[str, int] = {}
    for _, _, pop in read_fam(prefix):
        if pop in _FAM_IGNORE:
            continue
        c[pop] = c.get(pop, 0) + 1
    return c


def read_ind(prefix: str) -> list[tuple[str, str]]:
    """Return [(IID, POP)] from EIGENSTRAT .ind col1/col3."""
    rows = []
    with open(prefix + ".ind") as f:
        for line in f:
            t = line.split()
            if len(t) < 3:
                continue
            rows.append((t[0], t[2]))
    return rows


def ind_pop_counts(prefix: str) -> dict[str, int]:
    c: dict[str, int] = {}
    for _, pop in read_ind(prefix):
        c[pop] = c.get(pop, 0) + 1
    return c


def greedy_pops_for_n(counts: dict[str, int], target: int) -> tuple[list[str], int]:
    """Pick pops by descending N until cumulative samples >= target. Returns (pops, total)."""
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    picked, tot = [], 0
    for pop, n in ordered:
        picked.append(pop)
        tot += n
        if tot >= target:
            break
    return picked, tot


# --- Scratch builders (idempotent; shared across both features) --------------

def write_pheno(cfg: Cfg, prefix: str, path: str) -> str:
    """Categorical plink2 phenotype '#FID\\tIID\\tPOP' from .fam col6 (header mandatory)."""
    with open(path, "w") as f:
        f.write("#FID\tIID\tPOP\n")
        for fid, iid, pop in read_fam(prefix):
            f.write(f"{fid}\t{iid}\t{pop}\n")
    return path


def write_keep(cfg: Cfg, prefix: str, pops: list[str], path: str) -> tuple[str, int]:
    """plink2 --keep FID IID list for samples whose col6 pop is in `pops`."""
    want = set(pops)
    n = 0
    with open(path, "w") as f:
        for fid, iid, pop in read_fam(prefix):
            if pop in want:
                f.write(f"{fid}\t{iid}\n")
                n += 1
    return path, n


def ensure_thinned_bed(cfg: Cfg, m: int) -> str:
    """--thin-count M bed with the ORIGINAL fit9 .fam/.bim restored (critic fix #3:
    thinning keeps the sample set/order so the untouched .fam preserves col6 pops).
    Idempotent — skips if already built."""
    out = os.path.join(cfg.scratch, f"thin_{m}")
    if os.path.exists(out + ".bed") and os.path.exists(out + ".fam"):
        return out
    plink_thin(cfg, cfg.fit9, m, out)
    # Restore col6-bearing .fam byte-for-byte; .bim SNP order comes from the thin.
    shutil.copyfile(cfg.fit9 + ".fam", out + ".fam")
    return out


def ensure_injected_bed(cfg: Cfg, p_pct: int) -> str:
    """Copy of the FULL fit9 bed with p% of called genotypes set to missing (critic fix
    #7: inject into the full bed, .bim/.fam untouched so col6 + --pops work at query time).
    Idempotent. Delegates the .bed rewrite to inject_missing.inject()."""
    import inject_missing
    out = os.path.join(cfg.scratch, f"miss_{p_pct}")
    if os.path.exists(out + ".bed"):
        return out
    if p_pct == 0:
        shutil.copyfile(cfg.fit9 + ".bed", out + ".bed")
    else:
        inject_missing.inject(cfg.fit9, out, p_pct / 100.0, cfg.seed)
    shutil.copyfile(cfg.fit9 + ".bim", out + ".bim")
    shutil.copyfile(cfg.fit9 + ".fam", out + ".fam")
    return out


def write_relabeled_fam(cfg: Cfg, out_prefix: str, keep_pop_n: dict[str, int]) -> int:
    """Build a bed at out_prefix that shares fit9's .bed/.bim but with a hand-edited .fam
    (critic fix #3): the first N samples of each pop in keep_pop_n keep their col6 label;
    EVERY other sample's col6 is set to -9 (steppe ignores the {-9,9,0} sentinels). No
    plink2 --make-bed, so there is no categorical-col6 round-trip risk. Returns kept count."""
    # Symlink the big genotype/variant files (never copied — disk-tight).
    for ext in (".bed", ".bim"):
        dst = out_prefix + ext
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(cfg.fit9 + ext), dst)
    seen: dict[str, int] = {}
    kept = 0
    with open(out_prefix + ".fam", "w") as f:
        for fid, iid, pop in read_fam(cfg.fit9):
            want = keep_pop_n.get(pop, 0)
            take = pop in keep_pop_n and seen.get(pop, 0) < want
            if take:
                seen[pop] = seen.get(pop, 0) + 1
                kept += 1
                f.write(f"{fid}\t{iid}\t0\t0\t0\t{pop}\n")
            else:
                f.write(f"{fid}\t{iid}\t0\t0\t0\t-9\n")
    return kept


# --- Vectorized .raw loader (critic fix #5: no per-value Python loop) ---------

def load_raw(raw_path: str) -> tuple[list[str], np.ndarray]:
    """Return (iids, G[n_samples, n_snp] float64 with NaN=missing) from a plink2 .raw.

    The only Python-level loop is over the few-hundred SAMPLE rows; the token->float
    conversion is a single vectorized astype on the whole matrix."""
    with open(raw_path) as f:
        header = f.readline().split()
        iid_col = header.index("IID")
        ncol = len(header)
        iids: list[str] = []
        cells: list[list[str]] = []
        for line in f:
            t = line.split()
            if len(t) < ncol:
                continue
            iids.append(t[iid_col])
            cells.append(t[6:])
    arr = np.array(cells, dtype="U3")          # (n_samples, n_snp) as strings
    arr[arr == "NA"] = "nan"
    G = arr.astype(np.float64)
    return iids, G


# --- Weir-Cockerham FST oracle (numpy FP64; core/internal/wc_fst.hpp algebra) -

def wc_fst_oracle(GA: np.ndarray, GB: np.ndarray) -> dict[str, np.ndarray]:
    """Per-SNP WC (r=2). GA/GB are (nA,M)/(nB,M) dosage matrices, NaN=missing.
    Returns dict with num, den, fst (NaN on invalid), valid — matching wc_finalize."""
    def stats(G):
        v = np.isfinite(G)
        n = v.sum(0).astype(np.float64)
        ac = np.nansum(np.where(v, G, 0.0), 0)
        het = (G == 1.0).sum(0).astype(np.float64)   # NaN==1 is False -> excluded
        return n, ac, het

    n1, ac1, het1 = stats(GA)
    n2, ac2, het2 = stats(GB)
    M = GA.shape[1]
    num = np.zeros(M)
    den = np.zeros(M)
    fst = np.full(M, np.nan)
    valid = np.zeros(M, dtype=bool)

    with np.errstate(invalid="ignore", divide="ignore"):
        n_sum = n1 + n2
        n_bar = 0.5 * n_sum
        n_c = n_sum - (n1 * n1 + n2 * n2) / n_sum
        base = (n1 > 0) & (n2 > 0) & (n_bar > 1.0) & (n_c > 0)

        p1 = ac1 / (2.0 * n1)
        p2 = ac2 / (2.0 * n2)
        h1 = het1 / n1
        h2 = het2 / n2
        p_bar = (n1 * p1 + n2 * p2) / n_sum
        h_bar = (n1 * h1 + n2 * h2) / n_sum
        dp1 = p1 - p_bar
        dp2 = p2 - p_bar
        s2 = (n1 * dp1 * dp1 + n2 * dp2 * dp2) / n_bar
        pq = p_bar * (1.0 - p_bar)
        nm1 = n_bar - 1.0
        a = (n_bar / n_c) * (s2 - (1.0 / nm1) * (pq - 0.5 * s2 - 0.25 * h_bar))
        b = (n_bar / nm1) * (pq - 0.5 * s2 - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar)
        c = 0.5 * h_bar
        d = a + b + c
        good = base & np.isfinite(d) & (d != 0.0)
        num[good] = a[good]
        den[good] = d[good]
        fst[good] = a[good] / d[good]
        valid[good] = True
    return {"num": num, "den": den, "fst": fst, "valid": valid}


# --- Patterson PCA oracle (numpy.linalg.eigh) --------------------------------

def patterson_matrix(G: np.ndarray) -> tuple[np.ndarray, int]:
    """Patterson-standardize (Z = (G-2p)/sqrt(p(1-p)), missing mean-filled to 2p),
    dropping SNPs with ~isfinite(p) | p(1-p)==0. Returns (Z[n_samples,n_used], n_used)."""
    with np.errstate(invalid="ignore"):
        p = np.nanmean(G, axis=0) / 2.0
    keep = np.isfinite(p) & (p * (1.0 - p) > 0.0)
    Gk = G[:, keep]
    pk = p[keep]
    fill = np.broadcast_to(2.0 * pk, Gk.shape)
    Gk = np.where(np.isnan(Gk), fill, Gk)
    Z = (Gk - 2.0 * pk) / np.sqrt(pk * (1.0 - pk))
    return Z, int(keep.sum())


def pca_oracle_eigh(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k PCA via eigh of the sample GRM (Z Z^T). Returns (coords[N,k], eigvals[k],
    var_explained[k]). coords = eigenvectors (sign/scale irrelevant for the |r| compare)."""
    grm = Z @ Z.T
    w, V = np.linalg.eigh(grm)              # ascending
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    total = np.sum(np.clip(w, 0, None))
    kk = min(k, V.shape[1])
    coords = V[:, :kk]
    eig = w[:kk]
    ve = eig / total if total > 0 else np.zeros(kk)
    return coords, eig, ve


def subspace_gram_rel(S: np.ndarray, A: np.ndarray) -> float:
    """Rotation/sign-invariant subspace agreement: |SSᵀ - AAᵀ|_max / rms(AAᵀ)."""
    Gs, Ga = S @ S.T, A @ A.T
    return float(np.abs(Gs - Ga).max() / (np.sqrt((Ga * Ga).mean()) + 1e-30))


def per_pc_abs_r(S: np.ndarray, A: np.ndarray) -> list[float]:
    kk = min(S.shape[1], A.shape[1])
    out = []
    for k in range(kk):
        if np.std(S[:, k]) == 0 or np.std(A[:, k]) == 0:
            out.append(float("nan"))
        else:
            out.append(abs(float(np.corrcoef(S[:, k], A[:, k])[0, 1])))
    return out


# --- Output tables -----------------------------------------------------------

@dataclass
class Row:
    idx: int
    dimension: str
    metric: str
    value: str
    tol: str
    status: str
    notes: str


@dataclass
class Table:
    feature: str
    rows: list[Row] = field(default_factory=list)

    def add(self, idx, dimension, metric, value, tol, status, notes):
        self.rows.append(Row(idx, dimension, metric, str(value), str(tol),
                             str(status), str(notes)))

    def to_tsv(self) -> str:
        cols = ["#", "dimension", "metric", "value", "tol/expected", "status", "notes"]
        lines = ["\t".join(cols)]
        for r in sorted(self.rows, key=lambda x: x.idx):
            lines.append("\t".join([str(r.idx), r.dimension, r.metric, r.value,
                                    r.tol, r.status, r.notes]))
        return "\n".join(lines) + "\n"

    def to_md(self) -> str:
        head = (f"# steppe {self.feature.upper()} validation & benchmark table\n\n"
                "| # | dimension | metric | value | tol/expected | status | notes |\n"
                "|---|-----------|--------|-------|--------------|--------|-------|\n")
        body = []
        for r in sorted(self.rows, key=lambda x: x.idx):
            def esc(s):
                return s.replace("|", "\\|").replace("\n", "<br>")
            body.append(f"| {r.idx} | {esc(r.dimension)} | {esc(r.metric)} | {esc(r.value)} "
                        f"| {esc(r.tol)} | {r.status} | {esc(r.notes)} |")
        return head + "\n".join(body) + "\n"

    def write(self, out_dir: str):
        base = os.path.join(out_dir, f"{self.feature}_table")
        with open(base + ".tsv", "w") as f:
            f.write(self.to_tsv())
        with open(base + ".md", "w") as f:
            f.write(self.to_md())


def fmt_curve(points: list[tuple[str, float]]) -> str:
    return ", ".join(f"{k}->{v:.2f}s" for k, v in points)
