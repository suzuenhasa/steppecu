# /workspace/data/aadr/aadr/00_setup.sh
#!/usr/bin/env bash
# aadr/00_setup.sh — install tooling for the AADR f2-emulation data-prep pipeline.
# Idempotent: re-running only installs what is missing, then echoes versions.
#
# Tooling rationale (from research):
#   - eigensoft (convertf)  : PACKEDANCESTRYMAP -> PACKEDPED conversion
#   - plink (1.9)           : --freq --within --keep-allele-order --keep-clusters => .frq.strat
#   - plink2                : convenience / future
#   - numpy in /venv/main   : the pure-numpy packed reader + matrix writer (02_build_matrix.py)
set -euo pipefail

echo "==> [00_setup] starting"

WORK=/workspace/data/aadr
VENV=/venv/main
CONDA_ENV=aadr

mkdir -p "${WORK}/raw" "${WORK}/derived" "${WORK}/work"

# --- locate a mamba/conda front-end -----------------------------------------
PKG=""
if command -v mamba >/dev/null 2>&1; then
  PKG=mamba
elif command -v conda >/dev/null 2>&1; then
  PKG=conda
else
  echo "ERROR: neither mamba nor conda found on PATH" >&2
  exit 1
fi
echo "==> [00_setup] using package manager: ${PKG} ($(command -v "${PKG}"))"

# --- create / populate the bioinformatics env (idempotent) ------------------
# Use a dedicated env so the EIGENSOFT/PLINK toolchain does not collide with /venv/main.
# `conda env list` prints the env *path* in the last column; the NAME is the basename.
if "${PKG}" env list | awk 'NF && $1 !~ /^#/ {n=$1; sub(/\*$/,"",n); print n}' \
     | grep -qx "${CONDA_ENV}"; then
  echo "==> [00_setup] conda env '${CONDA_ENV}' already exists; ensuring packages present"
else
  echo "==> [00_setup] creating conda env '${CONDA_ENV}'"
  "${PKG}" create -y -n "${CONDA_ENV}" -c conda-forge -c bioconda python=3.11
fi

# `install` is idempotent: already-satisfied specs are a no-op.
echo "==> [00_setup] installing eigensoft plink plink2 into '${CONDA_ENV}'"
"${PKG}" install -y -n "${CONDA_ENV}" -c conda-forge -c bioconda eigensoft plink plink2

# Resolve the env prefix robustly. `conda env list --json` is the only stable parse;
# fall back to a column parse if --json is unavailable. The env line may carry a
# trailing '*' (active marker) on the name, never on the path, so take the LAST field.
ENV_PREFIX=""
if "${PKG}" env list --json >/dev/null 2>&1; then
  ENV_PREFIX="$("${PKG}" env list --json \
    | "${VENV}/bin/python" -c 'import json,sys,os; \
envs=json.load(sys.stdin)["envs"]; \
m=[e for e in envs if os.path.basename(e)==sys.argv[1]]; \
print(m[0] if m else "")' "${CONDA_ENV}" 2>/dev/null || true)"
fi
if [[ -z "${ENV_PREFIX}" ]]; then
  ENV_PREFIX="$("${PKG}" env list \
    | awk -v e="${CONDA_ENV}" 'NF && $1 !~ /^#/ {n=$1; sub(/\*$/,"",n); if(n==e) print $NF}')"
fi
if [[ -z "${ENV_PREFIX}" || ! -d "${ENV_PREFIX}" ]]; then
  echo "ERROR: could not resolve prefix for env '${CONDA_ENV}'" >&2
  exit 1
fi
echo "==> [00_setup] env prefix: ${ENV_PREFIX}"

# Fail loud if the toolchain binaries are not where we expect them.
for b in convertf plink plink2; do
  if [[ ! -x "${ENV_PREFIX}/bin/${b}" ]]; then
    echo "ERROR: expected binary not found/executable: ${ENV_PREFIX}/bin/${b}" >&2
    exit 1
  fi
done

# --- ensure numpy in the project venv ---------------------------------------
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: python venv not found at ${VENV}/bin/python" >&2
  exit 1
fi
echo "==> [00_setup] ensuring numpy in ${VENV}"
if ! "${VENV}/bin/python" -c "import numpy" >/dev/null 2>&1; then
  "${VENV}/bin/python" -m pip install --upgrade pip
  "${VENV}/bin/python" -m pip install numpy
else
  echo "==> [00_setup] numpy already present in ${VENV}"
fi

# --- echo versions ----------------------------------------------------------
echo "==> [00_setup] VERSIONS"
echo "----------------------------------------"
# convertf prints its version banner on stderr when run with no args; capture line 1.
"${ENV_PREFIX}/bin/convertf" 2>&1 | head -n1 || true
"${ENV_PREFIX}/bin/plink"  --version 2>&1 | head -n1 || true
"${ENV_PREFIX}/bin/plink2" --version 2>&1 | head -n1 || true
"${VENV}/bin/python" -c "import sys,numpy; print('python', sys.version.split()[0]); print('numpy', numpy.__version__)"
echo "----------------------------------------"

# Persist the resolved env prefix for downstream scripts (02 uses convertf+plink).
echo "${ENV_PREFIX}" > "${WORK}/work/.conda_env_prefix"
echo "==> [00_setup] wrote env prefix to ${WORK}/work/.conda_env_prefix"

echo "==> [00_setup] DONE"
