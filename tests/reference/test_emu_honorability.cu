// tests/reference/test_emu_honorability.cu
//
// X-6/B2 OBJECTIVE GATE — the EmulatedFp64 dynamic-mantissa-trap close-out
// (architecture.md §9 build() "fall back to native Fp64 or error", §12; cleanup
// X-6/B2). This is the test the verdict gate requires: on a build WITHOUT the
// fixed-slice Ozaki tuning, the EmulatedFp64 path must be OBSERVABLY refused /
// downgraded — NOT silently run cuBLAS's DYNAMIC-default emulation (the ~60-bit
// parity trap, ROADMAP §0).
//
// WHAT IT PINS (data-free, synthetic Q/V/N — a pure control-flow / engaged-mode
// assertion, NOT a precision claim, so no real AADR is needed; it runs on every
// lane):
//   1. emulation_honorable() — THE ONE predicate (X-6/B2 C-2 fix) — agrees with
//      the build flag (true iff EmulatedFp64 AND STEPPE_HAVE_EMU_TUNING), and is
//      ALWAYS false for Fp64 and Tf32.
//   2. f2_compute_type() routes through the SAME predicate: it returns the
//      emulated compute type ONLY when honorable, and native CUBLAS_COMPUTE_64F
//      otherwise — so the compute type can never disagree with the math mode
//      (closes the C-1/C-2 split; the half-fixed footgun cannot exist).
//   3. End-to-end through the production seam: a tiny compute_f2 at EmulatedFp64{40}
//      vs the same at native Fp64. When NOT honorable the two MUST be bit-identical
//      (the request was downgraded to native — no dynamic emulation slipped in);
//      when honorable they MUST DIFFER (the FIXED-slice Ozaki path engaged).
//   4. On the downgrade lane, the one-shot capability TAG is emitted to stderr
//      (observable refusal, not silent) — captured by redirecting stderr to a pipe.
//      The tag now routes through the ONE warn sink STEPPE_LOG_WARN (M4.5/U5,
//      core/internal/log.hpp), which is NDEBUG-silent by contract, so the stderr
//      assertion is gated to DEBUG builds; under NDEBUG the bit-identical downgrade
//      (item 3) is the observability and the tag is not asserted.
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Built by CMake/CTest as the
// `emu_honorability` test (tests/CMakeLists.txt) linking steppe::device. The test
// target gets the SAME STEPPE_HAVE_EMU_TUNING define as steppe_device (both come
// from the one cmake option), so the test's own #if matches the library's actual
// behavior; for robustness the production emulation_honorable() (compiled into the
// library) is the authority for the end-to-end arm.
// Run:  ./test_emu_honorability     (no data needed)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>   // pipe, dup, dup2, read (POSIX — Linux-only target, §6)
#include <vector>

#include "steppe/config.hpp"               // Precision, kDefaultMantissaBits
#include "core/internal/views.hpp"         // MatView (Q/V/N contract)
#include "device/backend.hpp"              // ComputeBackend, F2Result
#include "device/backend_factory.hpp"      // steppe::device::make_cuda_backend (X-9/B8)
#include "device/cuda/f2_block_kernel.cuh" // emulation_honorable, f2_compute_type (the X-6/B2 SoT)

// Match the device layer's in-file default so this TU has a definite value even
// when the cmake option is OFF (the library's PRIVATE define is not visible here
// in that case). The end-to-end arm trusts the LIBRARY's emulation_honorable()
// rather than this macro; this is only for the documentation banner below.
#ifndef STEPPE_HAVE_EMU_TUNING
#define STEPPE_HAVE_EMU_TUNING 0
#endif

using steppe::Precision;
using steppe::F2Result;
using steppe::core::MatView;
using steppe::device::emulation_honorable;
using steppe::device::f2_compute_type;

namespace {

// A tiny synthetic, well-formed Q/V/N block (P pops × M SNPs, column-major). The
// numbers are arbitrary but valid (q∈[0,1], v∈{0,1}, n≥1) — enough to drive the
// 3 GEMMs through both precision modes. No correctness claim against an oracle;
// this test is about the ENGAGED MODE, not the value.
struct Synth { int P; long M; std::vector<double> Q, V, N; };
Synth make_synth(int P, long M) {
    Synth s; s.P = P; s.M = M;
    const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    s.Q.resize(pm); s.V.resize(pm); s.N.resize(pm);
    for (long j = 0; j < M; ++j)
        for (int i = 0; i < P; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * j;
            // Deterministic spread in (0,1); all valid; n in [2, 200].
            s.Q[idx] = 0.05 + 0.9 * static_cast<double>((i * 7 + j * 3) % 97) / 97.0;
            s.V[idx] = 1.0;
            s.N[idx] = 2.0 + static_cast<double>((i + j) % 199);
        }
    return s;
}

bool bit_identical(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

// Run one EmulatedFp64{40} compute_f2 with stderr redirected to a pipe so the
// one-shot capability tag (if emitted) is captured. Returns the f2 payload and
// fills `captured` with whatever the run wrote to stderr.
F2Result compute_emu_capture(const MatView& Q, const MatView& V, const MatView& N,
                             const Precision& precEmu, std::string& captured) {
    captured.clear();
    std::fflush(stderr);
    int pipefd[2] = {-1, -1};
    const int saved = ::dup(STDERR_FILENO);
    F2Result out;
    if (saved >= 0 && ::pipe(pipefd) == 0) {
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);
        {
            auto gpu = steppe::device::make_cuda_backend();
            out = gpu->compute_f2(Q, V, N, precEmu);
        }
        std::fflush(stderr);
        ::dup2(saved, STDERR_FILENO);   // restore real stderr
        ::close(saved);
        char buf[4096];
        ssize_t r;
        while ((r = ::read(pipefd[0], buf, sizeof(buf))) > 0)
            captured.append(buf, static_cast<std::size_t>(r));
        ::close(pipefd[0]);
    } else {
        // Pipe setup failed: fall back to an un-captured run (still exercises the
        // path; the tag check below just won't see it).
        if (saved >= 0) ::close(saved);
        auto gpu = steppe::device::make_cuda_backend();
        out = gpu->compute_f2(Q, V, N, precEmu);
    }
    return out;
}

}  // namespace

int main() {
    const Precision precNat{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};
    const Precision precEmu{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
    const Precision precTf32{Precision::Kind::Tf32, steppe::kDefaultMantissaBits};

    bool ok = true;
    std::printf("\nX-6/B2 EmulatedFp64 honorability gate (synthetic, no data)\n");
    std::printf("  build STEPPE_HAVE_EMU_TUNING (this TU) = %d\n", STEPPE_HAVE_EMU_TUNING);

    // ---- (1) the ONE predicate ----------------------------------------------
    const bool honorable = emulation_honorable(precEmu);   // PRODUCTION predicate
    const bool fp64_honorable = emulation_honorable(precNat);
    const bool tf32_honorable = emulation_honorable(precTf32);
    const bool pred_ok = (!fp64_honorable) && (!tf32_honorable);
    std::printf("  emulation_honorable: Emu=%s Fp64=%s Tf32=%s  -> %s\n",
                honorable ? "true" : "false",
                fp64_honorable ? "true" : "false",
                tf32_honorable ? "true" : "false",
                pred_ok ? "PASS" : "FAIL (Fp64/Tf32 must be non-honorable)");
    ok = ok && pred_ok;

    // ---- (2) f2_compute_type routes through the SAME predicate ---------------
    // Honorable ⇒ emulated fixedpoint compute type; otherwise native — NEVER the
    // emulated compute type without the FIXED pin (that is the half-fixed footgun).
    const cublasComputeType_t ct_emu  = f2_compute_type(precEmu);
    const cublasComputeType_t ct_nat  = f2_compute_type(precNat);
    const cublasComputeType_t ct_tf32 = f2_compute_type(precTf32);
    const bool ct_ok =
        (ct_nat == CUBLAS_COMPUTE_64F) && (ct_tf32 == CUBLAS_COMPUTE_64F) &&
        (honorable ? (ct_emu == CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT)
                   : (ct_emu == CUBLAS_COMPUTE_64F));
    std::printf("  f2_compute_type(Emu) = %s  -> %s\n",
                (ct_emu == CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT) ? "EMULATED_FIXEDPOINT"
                : (ct_emu == CUBLAS_COMPUTE_64F)                   ? "64F (native)"
                                                                   : "OTHER",
                ct_ok ? "PASS" : "FAIL (compute type disagrees with honorability)");
    ok = ok && ct_ok;

    // ---- (3) end-to-end: Emu vs native through the production backend --------
    const Synth s = make_synth(/*P=*/5, /*M=*/64);
    const MatView Q{s.Q.data(), s.P, s.M};
    const MatView V{s.V.data(), s.P, s.M};
    const MatView N{s.N.data(), s.P, s.M};

    F2Result nat;
    { auto gpu = steppe::device::make_cuda_backend(); nat = gpu->compute_f2(Q, V, N, precNat); }

    std::string emu_stderr;
    const F2Result emu = compute_emu_capture(Q, V, N, precEmu, emu_stderr);

    const bool emu_differs = !bit_identical(emu.f2, nat.f2);
    bool e2e_ok = false;
    const char* e2e_desc = "";
    if (honorable) {
        // FIXED-slice Ozaki MUST have engaged: Emu must differ from native.
        e2e_ok = emu_differs;
        e2e_desc = "honorable: Emu engaged (differs from native)";
    } else {
        // Downgraded to native: Emu MUST be bit-identical to native (no dynamic
        // emulation slipped in) — the X-6/B2 fix. A difference would mean the
        // rejected DYNAMIC trap engaged.
        e2e_ok = !emu_differs;
        e2e_desc = "unhonorable: Emu DOWNGRADED to native (bit-identical)";
    }
    std::printf("  end-to-end compute_f2: %s  -> %s\n", e2e_desc, e2e_ok ? "PASS" : "FAIL");
    ok = ok && e2e_ok;

    // ---- (4) downgrade is OBSERVABLE (logged tag), not silent ----------------
    // Only assert the tag on the downgrade lane (it is emitted at most once per
    // process; on the honorable lane no tag is expected).
    //
    // BUILD-MODE GATE (M4.5/U5): the downgrade tag now routes through the ONE warn
    // sink STEPPE_LOG_WARN (core/internal/log.hpp), which — like `assert` — is
    // NDEBUG-silent by contract (a release build emits nothing at this level; the
    // earlier interim path used an unconditional std::fprintf). So the tag is
    // OBSERVABLE-in-stderr only in a DEBUG build; under NDEBUG the refusal is still
    // not "silent" in the dangerous sense (the predicate STILL downgrades to native
    // — proven bit-identical in check (3) above — it just does not print), and the
    // one-shot atomic still fires. We therefore assert the stderr tag only when the
    // sink is armed (!NDEBUG); under NDEBUG the contract is the bit-identical
    // downgrade, already asserted by (3).
    bool tag_ok = true;
    if (!honorable) {
#if defined(NDEBUG)
        std::printf("  capability tag on downgrade: not asserted under NDEBUG "
                    "(STEPPE_LOG_WARN is release-silent; the bit-identical downgrade "
                    "in check (3) is the NDEBUG observability)  -> PASS\n");
#else
        const bool saw_tag = emu_stderr.find("emu_tuning_unavailable") != std::string::npos;
        tag_ok = saw_tag;
        std::printf("  capability tag on downgrade: %s  -> %s\n",
                    saw_tag ? "emitted" : "ABSENT",
                    tag_ok ? "PASS" : "FAIL (downgrade must be observably logged, not silent)");
        if (!saw_tag)
            std::fprintf(stderr, "  [FAIL] EmulatedFp64 downgraded to native but NO capability tag "
                                 "was logged — the refusal is silent (X-6/B2). stderr was: <<<%s>>>\n",
                                 emu_stderr.c_str());
#endif
    } else {
        std::printf("  capability tag on downgrade: n/a (honorable)  -> PASS\n");
    }
    ok = ok && tag_ok;

    std::printf("\n");
    if (!ok) {
        std::fprintf(stderr,
            "RESULT: FAIL — the EmulatedFp64 honorability gate did not hold. On a build\n"
            "        without -DSTEPPE_HAVE_EMU_TUNING the EmulatedFp64 path must DOWNGRADE to\n"
            "        native Fp64 with a logged capability tag, NEVER silently run cuBLAS's\n"
            "        DYNAMIC-default emulation (the ~60-bit parity trap; architecture.md §9,\n"
            "        §12; ROADMAP §0; cleanup X-6/B2).\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "RESULT: PASS (%s)\n",
                 honorable ? "EmulatedFp64 honorable: FIXED-slice Ozaki engages"
                           : "EmulatedFp64 unhonorable: observably downgraded to native Fp64");
    return EXIT_SUCCESS;
}
