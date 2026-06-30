#!/usr/bin/env bash
#
# regenerate_goldens.sh — the single, version-asserting entry point that
# reproduces the committed AT2 / DATES parity goldens from the real AADR data.
#
# This is a MANUAL, OPT-IN tool of record. It does NOT run in CI and it does NOT
# auto-overwrite the committed goldens: a dry preflight is the default; actual
# regeneration requires the explicit --apply flag. It exists so that "how were
# these goldens produced" has ONE answer instead of eight loose R scripts plus
# prose in tests/reference/goldens/at2/README.md.
#
# WHAT IT DOES (in dependency order):
#   0. Preflight — HARD-FAIL unless the PINNED toolchain is present:
#        admixtools 2.0.10, R 4.3.3, convertf v8621, and the AADR prefix exists.
#   1. convertf TGENO -> PACKEDANCESTRYMAP. The raw v66 .geno is TGENO, which
#      admixtools 2.0.10 does NOT support and silently MISREADS (the documented
#      corrupt-golden bug: 500848 SNPs / weights [0.559,0.441]). convertf v8621
#      converts it to PACKEDANCESTRYMAP, which AT2 reads correctly. See the
#      CORRECTION note in tests/reference/goldens/at2/README.md.
#   2. Run the AT2 R generators in order (a thin orchestrator — the script bodies
#      are reused verbatim, never inlined or forked here).
#   3. Document the DATES goldens (reproduced from the committed par.dates files).
#   4. Remind that the committed parity tests are the acceptance gate and MUST be
#      re-run after any regeneration.
#
# ENV OVERRIDES:
#   AADR_ROOT  dir holding the AADR data on the build box   (default below)
#   OUT        committed goldens dir to (re)generate into    (default: in-repo)
#   STEPPE_REGEN_ALLOW_CONVERTF_UNKNOWN_VERSION=1
#              proceed with a WARNING if convertf will not report a version
#              (it still HARD-FAILS on a DETECTED mismatch).
#
# USAGE:
#   bash scripts/regenerate_goldens.sh            # preflight + print the plan (dry)
#   bash scripts/regenerate_goldens.sh --apply    # actually regenerate (box5090)
#
set -euo pipefail

# --- Pinned toolchain (the §12 PARITY-LAW reproduction record) ----------------
readonly PIN_ADMIXTOOLS="2.0.10"
readonly PIN_R="4.3.3"
readonly PIN_CONVERTF="8621"

# --- Paths --------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AADR_ROOT="${AADR_ROOT:-/workspace/data/aadr}"
OUT="${OUT:-${REPO_ROOT}/tests/reference/goldens/at2}"

# Raw v66 TGENO triple (what AT2 misreads) and the convertf-PA correction.
readonly RAW_PREFIX="${AADR_ROOT}/raw/v66.p1_HO.aadr.patch.PUB"
readonly PA_PREFIX="${AADR_ROOT}/converted_pa/v66_HO_pa"

readonly R_SCRIPT_DIR="${OUT}/scripts"
readonly DATES_DIR="${REPO_ROOT}/tests/reference/goldens/dates"

APPLY=0
for arg in "$@"; do
    case "${arg}" in
        --apply) APPLY=1 ;;
        -h|--help)
            grep -E '^# ' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "regenerate_goldens.sh: unknown argument '${arg}'" >&2; exit 2 ;;
    esac
done

# --- Helpers ------------------------------------------------------------------
fail()  { echo "FATAL: $*" >&2; exit 1; }
note()  { echo ">> $*"; }
run() {
    # Echo every command, then run it only under --apply (else dry-run).
    echo "+ $*"
    if [[ "${APPLY}" -eq 1 ]]; then
        "$@"
    fi
}

assert_eq_version() {
    # assert_eq_version <label> <found> <pinned>
    local label="$1" found="$2" pinned="$3"
    if [[ "${found}" != "${pinned}" ]]; then
        fail "${label} version mismatch: found '${found}', require '${pinned}'. \
A different version silently produces a DIVERGENT golden — refusing to proceed."
    fi
    note "${label} ${found} (pinned ${pinned}) OK"
}

# =============================================================================
# 0. PREFLIGHT — hard-fail unless the pinned toolchain + AADR data are present.
# =============================================================================
note "preflight: asserting the pinned toolchain (admixtools ${PIN_ADMIXTOOLS} / R ${PIN_R} / convertf v${PIN_CONVERTF})"

command -v Rscript  >/dev/null 2>&1 || fail "Rscript not on PATH"
command -v convertf >/dev/null 2>&1 || fail "convertf (EIGENSOFT) not on PATH"

# R version: Rscript reports e.g. "R scripting front-end version 4.3.3 (2024-02-29)".
R_VER="$(Rscript -e 'cat(as.character(getRversion()))' 2>/dev/null || true)"
[[ -n "${R_VER}" ]] || fail "could not query R version via Rscript"
assert_eq_version "R" "${R_VER}" "${PIN_R}"

# admixtools package version (the parity-defining dependency).
AT_VER="$(Rscript -e 'cat(as.character(packageVersion("admixtools")))' 2>/dev/null || true)"
[[ -n "${AT_VER}" ]] || fail "admixtools R package not installed / not loadable"
assert_eq_version "admixtools" "${AT_VER}" "${PIN_ADMIXTOOLS}"

# convertf version: EIGENSOFT tools print "version: NNNN" in their banner. Run it
# once on an empty parfile and scrape the banner. HARD-FAIL on a detected
# mismatch; only the undetectable case is overridable (documented env hatch).
CF_VER="$(convertf -p /dev/null 2>&1 | grep -oiE 'version[: ]+[0-9]+' | grep -oE '[0-9]+' | head -n1 || true)"
if [[ -n "${CF_VER}" ]]; then
    assert_eq_version "convertf" "${CF_VER}" "${PIN_CONVERTF}"
elif [[ "${STEPPE_REGEN_ALLOW_CONVERTF_UNKNOWN_VERSION:-0}" -eq 1 ]]; then
    note "WARNING: convertf did not report a version; proceeding (override set). Confirm it is v${PIN_CONVERTF} by hand."
else
    fail "convertf did not report a version banner to assert against v${PIN_CONVERTF}. \
Set STEPPE_REGEN_ALLOW_CONVERTF_UNKNOWN_VERSION=1 only after confirming the binary is v${PIN_CONVERTF}."
fi

# AADR raw prefix present (the .geno/.snp/.ind triple).
for ext in geno snp ind; do
    [[ -f "${RAW_PREFIX}.${ext}" ]] || fail "AADR raw input missing: ${RAW_PREFIX}.${ext} (set AADR_ROOT)"
done
note "AADR raw prefix OK: ${RAW_PREFIX}"

# R generators present.
[[ -d "${R_SCRIPT_DIR}" ]] || fail "R generator dir missing: ${R_SCRIPT_DIR}"
note "preflight OK"

if [[ "${APPLY}" -ne 1 ]]; then
    echo
    note "DRY RUN — preflight passed. Re-run with --apply to regenerate into:"
    note "    OUT = ${OUT}"
    note "The commands that WOULD run follow (echoed, not executed):"
    echo
fi

# =============================================================================
# 1. convertf TGENO -> PACKEDANCESTRYMAP (the documented correction).
#    AT2 2.0.10 cannot read the raw v66 TGENO; it must run on the PA conversion.
# =============================================================================
note "stage 1: convertf v${PIN_CONVERTF} TGENO -> PACKEDANCESTRYMAP"
if [[ -f "${PA_PREFIX}.geno" && "${APPLY}" -eq 1 ]]; then
    note "PA dataset already present (${PA_PREFIX}.geno) — skipping convertf (delete to force)"
else
    CF_PAR="$(mktemp /tmp/steppe_convertf.XXXXXX.par)"
    cat > "${CF_PAR}" <<EOF
genotypename:    ${RAW_PREFIX}.geno
snpname:         ${RAW_PREFIX}.snp
indivname:       ${RAW_PREFIX}.ind
outputformat:    PACKEDANCESTRYMAP
genotypeoutname: ${PA_PREFIX}.geno
snpoutname:      ${PA_PREFIX}.snp
indivoutname:    ${PA_PREFIX}.ind
EOF
    run mkdir -p "$(dirname "${PA_PREFIX}")"
    run convertf -p "${CF_PAR}"
    note "PA dataset -> ${PA_PREFIX}.{geno,snp,ind}"
    note "NOTE: the AT2 R generators read their dataset prefix internally. To"
    note "      reproduce the CORRECTED goldens they MUST point at the convertf-PA"
    note "      prefix above (${PA_PREFIX}), NOT the raw v66 TGENO — see"
    note "      tests/reference/goldens/at2/README.md §12 (Dataset prefix CORRECTED)."
fi

# =============================================================================
# 2. AT2 R generators, in dependency order (thin orchestrator: reuse as-is).
#    fit0 first (it is the acceptance oracle the others cross-reference), then
#    its fixup + the independent bit-exact verdict, then the remaining goldens.
# =============================================================================
note "stage 2: AT2 R generators (in order)"
R_GENERATORS=(
    "golden_fit0_generate.R"   # the M(fit-0) acceptance oracle (extract_f2 + qpadm + X/Q)
    "golden_fit0_fixup.R"      # re-capture: labelled X, all shas, the final csv/json
    "verify_bitexact.R"        # independent verdict — reproduces every value bit-exact
    "golden_fit1_generate.R"   # the nr>32 gesvd-fallback fixture (golden_fit1_NRBIG)
    "golden_fitNA_generate.R"  # the F1/OQ-12 missing-block (NA) golden
    "golden_rot_generate.R"    # the M(fit-6) S8 model-space rotation oracle
    "golden_qpwave_generate.R" # the run_qpwave (no-target) golden
    "golden_qpgraph_generate.R"# the qpGraph single-graph fit golden
)
for g in "${R_GENERATORS[@]}"; do
    script="${R_SCRIPT_DIR}/${g}"
    [[ -f "${script}" ]] || fail "generator missing: ${script}"
    run Rscript "${script}"
done

# =============================================================================
# 3. DATES goldens (frozen-by-provenance).
#    These are DATES Version 750 (github.com/priyamoorjani/DATES) outputs, built
#    on box5090, reproduced from the committed par.dates files. The DATES binary
#    is not part of the steppe toolchain preflight; if it is present on the box,
#    reproduce each curve with `DATES -p par.dates` from inside its golden dir and
#    confirm the .out covariance curve is BIT-IDENTICAL to the committed file.
# =============================================================================
note "stage 3: DATES goldens (frozen-by-provenance; see ${DATES_DIR}/README.md)"
for d in "${DATES_DIR}"/*/; do
    [[ -f "${d}/par.dates" ]] || continue
    note "  DATES dir: ${d}"
    note "    reproduce (if the DATES v750 binary is on the box):"
    note "      ( cd '${d}' && DATES -p par.dates )   # compare *.out to the committed curve"
done

# =============================================================================
# 4. MANDATORY parity re-run reminder — the script regenerates INPUTS; the
#    committed parity tests are the acceptance GATE and must still pass.
# =============================================================================
cat <<'REMINDER'

================================================================================
 MANDATORY: re-run the committed parity tests after ANY regeneration.
 This script regenerates the golden INPUTS; the tests are the acceptance gate.
 They MUST still pass (bit/tolerance-identical per architecture.md §12/§13):

   ctest --test-dir build-rel -R \
     'qpadm_parity|qpwave_parity|qpgraph_parity|qpgraph_search_parity|dates_parity|qpadm_rotation|qpadm_domain|qpadm_missing_block|fstat_sweep_parity' \
     --output-on-failure

 If any value moved, `git diff tests/reference/goldens` must be empty for a
 same-toolchain re-run; a non-empty diff means a toolchain/version drift slipped
 the preflight — investigate, do NOT commit a moved golden.
================================================================================
REMINDER

if [[ "${APPLY}" -ne 1 ]]; then
    note "DRY RUN complete (no files written). Re-run with --apply on box5090 to regenerate."
fi
