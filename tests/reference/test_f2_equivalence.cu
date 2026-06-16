// tests/reference/test_f2_equivalence.cu
//
// REFERENCE-EQUIVALENCE TEST for the M0 f2 path (ROADMAP M0 gate; architecture
// .md §13 "Golden / reference-equivalence", §5 S2, §12; ROADMAP §5).
//
// This is the central TRUST SEAM at f2: the GPU 3-GEMM reformulation diffed
// against the obviously-correct scalar/long-double CPU oracle, on REAL Q/V/N
// input, through the PRODUCTION ComputeBackend seam. It is a CORRECTNESS test
// (real genotype-derived input), NOT a precision or throughput benchmark — there
// is no synthetic data here (ROADMAP §0 cautionary tale: precision/throughput
// claims are benchmarked on real data only; this test merely confirms the two
// backends AGREE on real data).
//
// PRODUCTION SEAM (cleanup X-2/B4): this test drives the PRODUCTION backends —
// `steppe::core::make_cpu_backend()` (the long-double oracle, CpuBackend::compute_f2)
// and `steppe::device::make_cuda_backend()` (the 3-GEMM GPU path,
// CudaBackend::compute_f2) — through the CUDA-free `ComputeBackend::compute_f2`
// interface, NOT an inline re-implementation. It previously used an INLINE oracle
// and an INLINE GEMM driver, which (a) duplicated the formula the production code
// owns and (b) hid the M0 diagonal: the inline oracle walked j=i+1 (zero diagonal)
// while the GPU filled it, so the diagonal divergence was invisible. The production
// CpuBackend::compute_f2 now loops j=i (matching the GPU assemble_f2_kernel and the
// per-block oracle — the pinned F2Result diagonal convention in backend.hpp), so we
// diff the FULL [P × P] matrix INCLUDING the diagonal. A backend split on the
// diagonal would now FAIL this gate (the §13 oracle≡GPU seam, diagonal included).
//
// What it does:
//   1. Loads a small REAL Q/V/N matrix from a directory (SHARED BINARY FORMAT:
//      shape.txt + Q.f64 / V.f64 / N.f64, column-major [P × M], ld = P) — reusing
//      the validated spike loader. Default dir /workspace/data/aadr/derived_acc
//      (P=50, M=100000 on the box); overridable as argv[1].
//   2. Computes the f2 matrix THREE ways from the IDENTICAL Q/V/N input, all
//      through the production contract surfaces (steppe::core::MatView Q/V/N,
//      steppe::Precision, steppe::F2Result, and the SHARED per-element primitives
//      in core/internal/f2_estimator.hpp so the formula cannot diverge):
//        * CPU reference backend                  (scalar long-double oracle)
//        * CUDA backend, Precision::Fp64          (native FP64 GEMMs)
//        * CUDA backend, Precision::EmulatedFp64  (fixed-slice Ozaki, 40-bit)
//   3. Asserts max relative error over the FULL matrix (diagonal INCLUDED) for
//      entries with |ref| above the floor, in the TIGHT tier:
//        * emu  vs ref  <  1e-6   (architecture.md §12 est tier; ROADMAP §0
//                                  40-bit worst-case f2 error ≈ 2.2e-11 ≪ 1e-6)
//        * native vs ref  <  1e-9 (near bit-stable; §12)
//      plus an EXACT bit-for-bit diagonal check (native vs oracle): the diagonal
//      is −2·mean_het in both, computed from the same integer-counted/long-double
//      path, so it must agree to a tight floor and NEVER be the spurious 0/nonzero
//      split B4 fixed.
//   4. Prints a small pass/fail table and exits NONZERO on any failure (so CTest
//      / a sweep harness detects it).
//
// PRECISION POLICY IS THE LAW (architecture.md §12; ROADMAP §0): the f2 GEMMs use
// FIXED-slice Ozaki emulation at the default mantissa_bits (kDefaultMantissaBits
// == 40 ≈ native). DYNAMIC mantissa control is the rejected parity trap and is
// NEVER selected — the PRODUCTION `emulation_honorable` predicate (X-6/B2) decides
// whether the EmulatedFp64 request engages the FIXED-slice path or is downgraded
// to native Fp64; this test asserts the numeric consequence on BOTH lanes. The
// small numerator/divide stays native FP64 in BOTH GPU arms.
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Built by CMake/CTest as the
// `f2_equivalence` test (tests/CMakeLists.txt) linking steppe::core +
// steppe::device, with -DSTEPPE_HAVE_EMU_TUNING=1 on the box build.
// Run:
//   ./test_f2_equivalence [dir]      (default /workspace/data/aadr/derived_acc)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

// Project contract surfaces (architecture.md §4 real paths). The PRODUCTION
// backends are reached through the CUDA-free ComputeBackend seam; the device-
// private kernel header is included only for the honorability predicate (the same
// pattern test_f2_blocks_equivalence.cu uses for X-6/B2).
#include "steppe/config.hpp"            // steppe::Precision, kRelFloor, kAbsFloor, kDefaultMantissaBits
#include "core/internal/views.hpp"      // steppe::core::MatView (the Q/V/N contract)
#include "device/backend.hpp"           // steppe::ComputeBackend, steppe::F2Result (the f2/Vpair seam)
#include "device/cuda/f2_block_kernel.cuh"  // steppe::device::emulation_honorable (X-6/B2 predicate)

using steppe::Precision;
using steppe::core::MatView;
using steppe::F2Result;

// Backend factories (declared where defined: CPU in steppe::core, CUDA in
// steppe::device — mirrors tests/reference/test_f2_blocks_equivalence.cu and
// test_decode_equivalence.cu). Declared in no header today (cleanup X-9/B8); the
// reference tests hand-declare the one prototype each they call.
namespace steppe::core   { std::unique_ptr<steppe::ComputeBackend> make_cpu_backend(); }
namespace steppe::device { std::unique_ptr<steppe::ComputeBackend> make_cuda_backend(); }

namespace {

// Default real-data directory on the remote box (exists there per the task; the
// local box has no copy — that is expected, this test only runs on the box).
constexpr const char* kDefaultDataDir = "/workspace/data/aadr/derived_acc";

// Tight-tier relative-error thresholds (architecture.md §12 est tier; ROADMAP
// §0). These are TEST tolerances, the single home for the verdict thresholds.
constexpr double kTolEmuVsRef    = 1e-6;   // EmulatedFp64{40} vs oracle
constexpr double kTolNativeVsRef = 1e-9;   // native Fp64 vs oracle
// The diagonal (−2·mean_het) is computed from the same het-correction +
// integer-counted divide on both backends, so native vs oracle agrees to a tight
// absolute floor (well below the O(1e-2) entry magnitude). A spurious 0/nonzero
// split (the B4 bug) would blow past this by ~the diagonal's magnitude.
constexpr double kTolDiagNativeAbs = 1e-9;

// ---------------------------------------------------------------------------
// SHARED BINARY FORMAT loader (reused from the validated spike: f2_emu_spike.cu
// read_f64 / load_real_data). shape.txt holds "P M"; Q/V/N.f64 each hold P*M
// little-endian doubles, column-major [P × M] (element (pop i, snp s) at i+P*s).
// Loud on any shape/format mismatch.
// ---------------------------------------------------------------------------
void read_f64(const std::string& path, std::vector<double>& out, std::size_t count) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    out.resize(count);
    const std::size_t got = std::fread(out.data(), sizeof(double), count, f);
    std::size_t extra = 0;
    if (got == count) {  // probe one more double to catch a too-large file (wrong P*M)
        double probe = 0.0;
        extra = std::fread(&probe, sizeof(double), 1, f);
    }
    std::fclose(f);
    if (got != count) {
        std::fprintf(stderr,
            "ERROR: %s has %zu doubles, expected %zu (P*M). Shape/format mismatch.\n",
            path.c_str(), got, count);
        std::exit(EXIT_FAILURE);
    }
    if (extra != 0) {
        std::fprintf(stderr,
            "ERROR: %s has MORE than %zu doubles (trailing data). Shape/format mismatch.\n",
            path.c_str(), count);
        std::exit(EXIT_FAILURE);
    }
}

// Load shape.txt + Q/V/N.f64 from `dir`. Fills raw arrays and P, M.
void load_real_data(const std::string& dir, int& P_out, long& M_out,
                    std::vector<double>& Q_raw,
                    std::vector<double>& V_raw,
                    std::vector<double>& N_raw) {
    const std::string shapePath = dir + "/shape.txt";
    FILE* sf = std::fopen(shapePath.c_str(), "r");
    if (!sf) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", shapePath.c_str());
        std::exit(EXIT_FAILURE);
    }
    int P = 0;
    long M = 0;
    if (std::fscanf(sf, "%d %ld", &P, &M) != 2) {
        std::fprintf(stderr, "ERROR: %s must contain 'P M' (two ints)\n", shapePath.c_str());
        std::fclose(sf);
        std::exit(EXIT_FAILURE);
    }
    std::fclose(sf);
    if (P <= 1 || M <= 0) {
        std::fprintf(stderr, "ERROR: bad shape from %s: P=%d M=%ld (need P>1, M>0)\n",
                     shapePath.c_str(), P, M);
        std::exit(EXIT_FAILURE);
    }

    const std::size_t count = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    read_f64(dir + "/Q.f64", Q_raw, count);
    read_f64(dir + "/V.f64", V_raw, count);
    read_f64(dir + "/N.f64", N_raw, count);

    P_out = P;
    M_out = M;
}

// ---------------------------------------------------------------------------
// Max relative error of a candidate vs the reference over the FULL [P × P] matrix
// (the DIAGONAL is INCLUDED — cleanup X-2/B4) for entries whose |ref| is above the
// relative floor (kRelFloor): below that, an entry is at the noise floor and would
// blow up the relative metric, so it is skipped (the combined-form guard,
// architecture.md §12). Also counts sign flips (a FAIL signature).
// ---------------------------------------------------------------------------
struct ErrStats {
    double maxRel = 0.0;
    int    signFlips = 0;
    std::size_t scored = 0;
};

ErrStats rel_error_stats(const F2Result& cand, const F2Result& ref) {
    const int P = ref.P;
    ErrStats es;
    for (int j = 0; j < P; ++j) {
        for (int i = 0; i < P; ++i) {  // i==j INCLUDED — the diagonal is part of the gate now
            const std::size_t off = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * j;
            const double r = ref.f2[off];
            const double c = cand.f2[off];
            if (std::fabs(r) < steppe::kRelFloor) continue;  // near-zero: skip (noise floor)
            const double denom = std::max(std::fabs(r), steppe::kAbsFloor);
            const double rel = std::fabs(c - r) / denom;
            if (rel > es.maxRel) es.maxRel = rel;
            ++es.scored;
            if ((r > 0.0) != (c > 0.0) && c != 0.0) ++es.signFlips;
        }
    }
    return es;
}

// Worst absolute difference on the DIAGONAL between two backends. The B4 bug was a
// 0/nonzero split here; we assert agreement explicitly so it cannot reappear (the
// rel_error_stats pass already covers it for above-floor entries, but a 0-vs-(near-
// floor nonzero) split on a near-zero diagonal entry would slip the relative
// metric, so this absolute diagonal check is the belt-and-suspenders gate).
double max_diag_abs_diff(const F2Result& a, const F2Result& b) {
    const int P = a.P;
    double mx = 0.0;
    for (int i = 0; i < P; ++i) {
        const std::size_t d = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * i;
        mx = std::max(mx, std::fabs(a.f2[d] - b.f2[d]));
    }
    return mx;
}

}  // namespace

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    const std::string dir = (argc >= 2) ? argv[1] : kDefaultDataDir;

    // ---- Load real Q/V/N (reused spike loader) ------------------------------
    int  P = 0;
    long M = 0;
    std::vector<double> Q_raw, V_raw, N_raw;
    load_real_data(dir, P, M, Q_raw, V_raw, N_raw);
    std::fprintf(stderr,
        "[load] dir=%s  P=%d  M=%ld  (%zu doubles per matrix) — REAL data, not synthetic\n",
        dir.c_str(), P, M, static_cast<std::size_t>(P) * M);

    // Wrap the raw arrays as the Q/V/N contract views (column-major [P × M]).
    const MatView Q{Q_raw.data(), P, M};
    const MatView V{V_raw.data(), P, M};
    const MatView N{N_raw.data(), P, M};

    // ---- PRODUCTION backends through the CUDA-free seam ---------------------
    auto cpu = steppe::core::make_cpu_backend();
    auto gpu = steppe::device::make_cuda_backend();

    const Precision precNat{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};
    const Precision precEmu{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};

    // ---- (1) CPU reference oracle (the ground truth) ------------------------
    const F2Result ref = cpu->compute_f2(Q, V, N, precNat);

    // ---- (2) CUDA backend, native Fp64 --------------------------------------
    const F2Result f2_native = gpu->compute_f2(Q, V, N, precNat);

    // ---- (3) CUDA backend, EmulatedFp64{mantissa_bits=40} -------------------
    const F2Result f2_emu = gpu->compute_f2(Q, V, N, precEmu);

    // X-6/B2 — the EmulatedFp64 honorability gate, asserted OBJECTIVELY against the
    // PRODUCTION predicate compiled INTO steppe_device (reflects steppe_device's own
    // STEPPE_HAVE_EMU_TUNING, not this test target's). Ozaki output is NOT bit-
    // identical to native FP64, so on the honorable lane emu MUST differ from native
    // bit-for-bit (else it silently fell back — the C-1 trap); on the unhonorable
    // lane emu MUST be bit-identical to native (the X-6/B2 downgrade — never silent
    // DYNAMIC). This mirrors test_f2_blocks_equivalence.cu's gate.
    const bool honorable = steppe::device::emulation_honorable(precEmu);
    const bool emu_differs_from_native =
        (f2_emu.f2.size() == f2_native.f2.size()) &&
        (std::memcmp(f2_emu.f2.data(), f2_native.f2.data(),
                     f2_emu.f2.size() * sizeof(double)) != 0);

    // ---- Vpair cross-check (the retained S4 jackknife weight) ---------------
    // The CPU oracle and the GPU GEMM both produce the pairwise-valid count; they
    // are exact integer counts and must match EXACTLY over the FULL matrix —
    // diagonal INCLUDED (Vpair(i,i) is i's valid-SNP count, also pinned by B4).
    bool vpair_match = (f2_native.vpair.size() == ref.vpair.size());
    if (vpair_match) {
        for (std::size_t k = 0; k < ref.vpair.size() && vpair_match; ++k) {
            if (f2_native.vpair[k] != ref.vpair[k]) vpair_match = false;
        }
    }

    // ---- Accuracy stats vs the oracle (FULL matrix, diagonal included) ------
    const ErrStats esNative = rel_error_stats(f2_native, ref);
    const ErrStats esEmu    = rel_error_stats(f2_emu, ref);

    // ---- Diagonal agreement (the B4 close-out): native vs oracle ------------
    // Explicit absolute check that the diagonal (−2·mean_het) agrees across the
    // production CPU and GPU M0 paths — the exact 0/nonzero split B4 fixed.
    const double diag_native_abs = max_diag_abs_diff(f2_native, ref);
    const bool diag_native_ok = diag_native_abs < kTolDiagNativeAbs;

    // ---- Verdicts -----------------------------------------------------------
    // Native: tight-tier (< 1e-9) over the FULL matrix, no sign flips, Vpair exact,
    // diagonal agrees with the oracle.
    const bool native_pass = vpair_match &&
                             diag_native_ok &&
                             esNative.signFlips == 0 &&
                             esNative.maxRel < kTolNativeVsRef;

    // EmulatedFp64-arm verdict, branched on the PRODUCTION honorability state — a
    // HARD PASS/FAIL on BOTH lanes (no silent skip; X-6/B2).
    bool emu_pass = false;
    const char* emu_mode = "";
    if (honorable) {
        // Tuning ON: emulation must engage (output differs from native) AND meet
        // the EmulatedFp64-vs-oracle tight tolerance over the FULL matrix.
        emu_pass = emu_differs_from_native &&
                   esEmu.signFlips == 0 &&
                   esEmu.maxRel < kTolEmuVsRef;
        emu_mode = "engaged";
    } else {
        // Tuning OFF: the path must be DOWNGRADED to native Fp64 — emu bit-identical
        // to native (and therefore also within the native tolerance vs the oracle).
        const bool downgraded_to_native = !emu_differs_from_native;
        emu_pass = downgraded_to_native &&
                   esEmu.signFlips == 0 &&
                   esEmu.maxRel < kTolNativeVsRef;
        emu_mode = "downgraded->Fp64";
    }

    // ---- Pass/fail table ----------------------------------------------------
    std::printf("\n");
    std::printf("f2 reference-equivalence (REAL data, P=%d M=%ld) — tight tier, FULL matrix\n", P, M);
    std::printf("EmulatedFp64 honorable (production STEPPE_HAVE_EMU_TUNING): %s [%s]\n",
                honorable ? "YES" : "NO", emu_mode);
    std::printf("%-26s %12s %10s %9s %8s %8s\n",
                "arm", "maxRel", "tol", "signFlip", "scored", "verdict");
    std::printf("%-26s %12.3e %10.1e %9d %8zu %8s\n",
                "cuda Fp64 vs ref", esNative.maxRel, kTolNativeVsRef,
                esNative.signFlips, esNative.scored,
                native_pass ? "PASS" : "FAIL");
    std::printf("%-26s %12.3e %10.1e %9d %8zu %8s\n",
                honorable ? "cuda EmuFp64{40} vs ref" : "cuda EmuFp64->Fp64 vs ref",
                esEmu.maxRel, honorable ? kTolEmuVsRef : kTolNativeVsRef,
                esEmu.signFlips, esEmu.scored,
                emu_pass ? "PASS" : "FAIL");
    std::printf("%-26s %12.3e %10.1e %9s %8s %8s\n",
                "f2 diagonal Fp64 vs ref", diag_native_abs, kTolDiagNativeAbs,
                "-", "-", diag_native_ok ? "PASS" : "FAIL");
    std::printf("%-26s %12s %10s %9s %8s %8s\n",
                "Vpair native==ref", vpair_match ? "exact" : "MISMATCH",
                "-", "-", "-", vpair_match ? "PASS" : "FAIL");
    std::printf("\n");

    // ---- Diagnostics on failure ---------------------------------------------
    if (honorable && !emu_differs_from_native)
        std::fprintf(stderr,
            "  [FAIL] EmulatedFp64 honorable but emulation did NOT engage (emu == native "
            "bit-for-bit): the FIXED-slice Ozaki path silently fell back (X-6/B2 C-1).\n");
    if (!honorable && emu_differs_from_native)
        std::fprintf(stderr,
            "  [FAIL] EmulatedFp64 NOT honorable yet the emu output DIFFERS from native: a "
            "non-native mode engaged without the FIXED-slice pin — the DYNAMIC trap (X-6/B2).\n");
    if (!honorable && emu_pass)
        std::fprintf(stderr,
            "  [info] EmulatedFp64 NOT honorable (built without -DSTEPPE_HAVE_EMU_TUNING) -> "
            "path correctly DOWNGRADED to native Fp64 (X-6/B2). Rebuild with the tuning on the "
            "CUDA-13/sm_120 box to exercise the FIXED-slice Ozaki arm.\n");
    if (!diag_native_ok)
        std::fprintf(stderr,
            "  [FAIL] f2 DIAGONAL differs between the GPU and the CPU oracle (worst %.3e > %.1e): "
            "the M0 diagonal convention is not consistent across backends (cleanup X-2/B4).\n",
            diag_native_abs, kTolDiagNativeAbs);
    if (!vpair_match)
        std::fprintf(stderr,
            "  [FAIL] Vpair (pairwise-valid SNP count) differs between the GPU GEMM and the CPU "
            "oracle — the masking is wrong (architecture.md §5 S2).\n");

    const bool overall = native_pass && emu_pass;
    if (!overall) {
        std::fprintf(stderr, "RESULT: FAIL\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "RESULT: PASS (EmulatedFp64 arm %s)\n", emu_mode);
    return EXIT_SUCCESS;
}
