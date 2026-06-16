#!/usr/bin/env bash
# build_run.sh -- compile f2_emu_spike and run the sigma sweep that gates the
# steppe S2 policy question (ADR-0010 / architecture.md section 12): do the f2
# GEMMs stay native FP64, or can they run on cuBLAS FP64 fixed-point (Ozaki)
# emulation without losing the est tolerance tier under cancellation?
#
# Target: sm_120 (RTX 5090), CUDA 13, cuBLAS FP64 fixed-point emulation.
#
# The single binary prints one results-table row per (P, M, sigma, missing)
# cell, with a PASS / MARGINAL / FAIL / FAIL_IGNORED verdict, plus two
# emulation-engagement signals:
#   * emuEqNat = YES  -> the emulated f2 was BIT-IDENTICAL to native f2, i.e.
#                        emulation was almost certainly ignored / fell back to
#                        native (Ozaki output is not bit-identical to native);
#                        the verdict is forced to FAIL_IGNORED.
#   * mBits >= 0       -> (only when built with -DSTEPPE_HAVE_EMU_TUNING=1) the
#                        mantissa-bit verification hook positively confirms
#                        emulation engaged; -1 means it did not.
# The binary exits non-zero on FAIL / FAIL_IGNORED; this script tolerates that
# per-cell (so the whole grid still runs) and exits non-zero at the end if ANY
# cell failed. The header row prints once per process invocation, so each cell
# reprints the header -- that is intentional (one process per cell).
set -euo pipefail

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_120}"
SRC="f2_emu_spike.cu"
BIN="./f2_emu_spike"

# Emulation tuning symbols (strategy / mantissa control / verification pointer)
# are gated behind STEPPE_HAVE_EMU_TUNING in the .cu because their exact names
# could not be verified against the live CUDA 13 cuBLAS headers. Default OFF so
# the file always compiles using only the confirmed compute-type + math-mode
# mechanisms; set EMU_TUNING=1 once the symbol names are confirmed.
#   EMU_TUNING=1 ./build_run.sh
EMU_TUNING="${EMU_TUNING:-0}"
EXTRA_DEFS=()
if [[ "${EMU_TUNING}" == "1" ]]; then
    EXTRA_DEFS+=("-DSTEPPE_HAVE_EMU_TUNING=1")
fi

# Resolve paths relative to this script so it runs from anywhere.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

echo "==> Building ${SRC} for ${ARCH} (CUDA 13, -lcublas; EMU_TUNING=${EMU_TUNING})"
"${NVCC}" -O3 -std=c++17 -arch="${ARCH}" "${EXTRA_DEFS[@]}" "${SRC}" -lcublas -o f2_emu_spike
echo "==> Build OK: ${BIN}"
echo

# Track whether any cell failed without letting set -e abort the whole sweep.
ANY_FAIL=0
run_cell() {
    # Run one cell; tolerate a non-zero exit (FAIL / FAIL_IGNORED) so the sweep
    # continues, but remember it so the script's final exit code reflects it.
    if ! "${BIN}" "$@"; then
        ANY_FAIL=1
        echo "  [cell-fail] ${BIN} $* exited non-zero (FAIL / FAIL_IGNORED)" >&2
    fi
}

# Cancellation dial. Small sigma => severe cancellation (kappa grows ~ 1/sigma^2);
# the verdict is expected to flip somewhere in [1e-4, 1e-3] if it flips at all.
SIGMAS=(1e-1 1e-2 1e-3 3e-4 1e-4)

# Core sweep: (P, M) pairs. P scales the # scored pairs (max gate); M is the
# summation-length stress (worst at 1e6). A subset of the full P x M x sigma
# cross-product to keep the demo run bounded; widen these arrays for the full grid.
declare -a PM_PAIRS=(
    "10  100000"
    "30  1000000"
    "100 1000000"
)

for pm in "${PM_PAIRS[@]}"; do
    read -r P M <<< "${pm}"
    echo "==================================================================="
    echo "  Sigma sweep at P=${P}, M=${M}  (no missing data)"
    echo "==================================================================="
    for sig in "${SIGMAS[@]}"; do
        run_cell "${P}" "${M}" "${sig}"
    done
    echo
done

# Secondary arm B: missingness / masked-GEMM at the severe corner. Exercises
# Vpair < M, the @V^T masking, and the Vpair==0 => 0 branch under cancellation.
echo "==================================================================="
echo "  Missing-data arm (P=30, M=1000000, sigma=1e-4): miss in {0.1, 0.5}"
echo "==================================================================="
run_cell 30 1000000 1e-4 0.1
run_cell 30 1000000 1e-4 0.5
echo

# Secondary arm C: multi-seed worst corner. max-over-seeds of max_relerr is the
# worst-case gate; rerun the severe cell across several recorded seeds.
echo "==================================================================="
echo "  Multi-seed worst corner (P=100, M=1000000, sigma=1e-4)"
echo "==================================================================="
for s in 1 2 3 4 5; do
    run_cell 100 1000000 1e-4 0.0 "${s}"
done
echo

if [[ "${ANY_FAIL}" -ne 0 ]]; then
    echo "==> SWEEP RESULT: at least one cell was FAIL / FAIL_IGNORED (see rows above)." >&2
    exit 1
fi
echo "==> SWEEP RESULT: all cells PASS / MARGINAL."
