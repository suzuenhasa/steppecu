# /workspace/data/aadr/aadr/03_run.sh
#!/usr/bin/env bash
# aadr/03_run.sh — invoke the f2 emulation-accuracy spike in LOAD mode against the
# derived matrices produced by 02_build_matrix.py.
#
# The spike loader is wired up separately (it mmaps the four files); this script only
# documents the contract and calls the binary. The derived dir must contain:
#   Q.f64  V.f64  N.f64  meta.json
# all little-endian float64, each a C-contiguous numpy (M, P) = (SNPs, pops) array =>
# element (pop i, snp s) at flat index  i + P*s  == a cuBLAS COLUMN-MAJOR [P x M] matrix
# with leading dimension ld = P (P = #populations, M = #SNPs).
set -euo pipefail

echo "==> [03_run] starting"

DERIVED=/workspace/data/aadr/derived
VENV=/venv/main
# Location of the built spike binary (override via $SPIKE_BIN). Adjust to your build tree.
SPIKE_BIN="${SPIKE_BIN:-/workspace/f2_emu/build/f2_emu_spike}"

command -v stat >/dev/null 2>&1 || { echo "ERROR: stat not found on PATH" >&2; exit 1; }
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: python venv not found at ${VENV}/bin/python" >&2
  exit 1
fi

# --- preflight: the four artifacts must exist ------------------------------
for f in Q.f64 V.f64 N.f64 meta.json; do
  if [[ ! -f "${DERIVED}/${f}" ]]; then
    echo "ERROR: missing ${DERIVED}/${f} — run 02_build_matrix.py first" >&2
    exit 1
  fi
done

# --- preflight: byte sizes match P*M*8 from meta.json ----------------------
# meta.json: P = #populations, M = #SNPs. Each matrix is P*M float64 = P*M*8 bytes.
read -r P M < <("${VENV}/bin/python" - "$DERIVED/meta.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
P, M = int(m["P"]), int(m["M"])
assert P > 0 and M > 0, f"P,M must be positive, got P={P} M={M}"
print(P, M)
PY
)
EXP=$(( P * M * 8 ))
echo "==> [03_run] meta.json: P(pops)=${P}  M(SNPs)=${M}  expected bytes/file=${EXP}"
for f in Q.f64 V.f64 N.f64; do
  sz=$(stat -c %s "${DERIVED}/${f}")
  if [[ "${sz}" != "${EXP}" ]]; then
    echo "ERROR: ${f} is ${sz} bytes, expected ${EXP} (P*M*8)" >&2
    exit 1
  fi
  echo "==> [03_run] ${f}: ${sz} bytes OK"
done

# --- call the spike in load mode -------------------------------------------
if [[ ! -x "${SPIKE_BIN}" ]]; then
  echo "NOTE: spike binary not found/executable at ${SPIKE_BIN}." >&2
  echo "      The spike loader is wired separately; set SPIKE_BIN to its path, e.g.:" >&2
  echo "        SPIKE_BIN=/workspace/f2_emu/build/f2_emu_spike bash aadr/03_run.sh" >&2
  echo "      Intended invocation:" >&2
  echo "        ${SPIKE_BIN} --load ${DERIVED}" >&2
  exit 1
fi

echo "==> [03_run] exec: ${SPIKE_BIN} --load ${DERIVED}"
exec "${SPIKE_BIN}" --load "${DERIVED}"
