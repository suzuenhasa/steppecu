// tests/reference/test_handles.cu
//
// U4 / L10 OBJECTIVE GATE — the M4.5 capability-tier scaffold additions to the
// RAII cuBLAS handle wrapper (cleanup device-cuda-handles 2.3/11.x, L10; overview
// §(2).1; architecture.md §7 RAII / record-and-assert, §9 PerGpuResources, §11.4
// SPMG, §12 determinism; TODO M4.5). This is the test the verify gate requires:
//
//   (a) MathModeScope RESTORES the prior cuBLAS math mode after scope exit, and
//   (b) (debug) the recorded device ordinal MATCHES the creation device and the
//       handle's record-and-assert precondition holds when used on that device.
//
// WHY (the hazards these pin closed):
//   * MathModeScope (L10 / f2_block_kernel N-5): `cublasSetMathMode` is STICKY
//     handle state — it stays set until changed again (unlike the workspace, it
//     is NOT reset by `cublasSetStream`; cuBLAS §2.4.7 resets only the workspace).
//     `engage_f2_precision` engages a math mode once per compute call and never
//     restores it: benign while one precision owns the handle, but the §12
//     mandatory gate recomputes a sample of jackknife blocks in native `Fp64`
//     (PEDANTIC) on the SAME shared handle an `EmulatedFp64` run is using — and
//     whichever ran last would silently leak its math mode into the next. The
//     scope makes a math-mode change OBSERVABLY SCOPED (capture on ctor, restore
//     on dtor), so the Fp64 parity-recompute leaves the handle exactly as it
//     found it. Parity-NEUTRAL: it only restores state already set imperatively.
//   * Device-ordinal record-and-assert (2.3 / 11.x): "A cuBLAS library context is
//     tightly coupled with the CUDA context that is current at the time of the
//     `cublasCreate()` call" (cuBLAS §2.1.2). The §9 `PerGpuResources` holds one
//     handle per device and §11.4 switches devices with `cudaSetDevice`; a handle
//     used while a DIFFERENT device is current runs on the wrong GPU or fails
//     `CUBLAS_STATUS_ARCH_MISMATCH`. The ctor records the creation ordinal
//     (`cudaGetDevice`, query-only — NEVER `cudaSetDevice` inside the wrapper) and
//     the context-mutating methods debug-assert the current device still matches.
//
// WHAT IT PINS (data-free, synthetic — pure handle-state control flow, NOT a
// precision/numeric claim, so no real AADR is needed; it needs a real GPU because
// `cublasCreate`/`cublasGetMathMode` require a live cuBLAS context):
//   1. MATH-MODE RESTORE (the L10 gate): set a known mode (PEDANTIC) on the
//      handle; enter a MathModeScope requesting a DIFFERENT mode (DEFAULT); assert
//      the handle reports the requested mode INSIDE the scope; on scope exit
//      assert the handle is back to the captured PEDANTIC mode — exactly.
//   2. NESTED RESTORE: scopes nest correctly — an inner scope restores to the
//      OUTER scope's mode, not to the original, proving the ctor captures the
//      live current mode (not a hardcoded default).
//   3. MOVE LEAVES THE MOVED-FROM INERT: a moved-from MathModeScope restores
//      NOTHING at its dtor (the moved-into one owns the single restore), so the
//      mode is restored exactly ONCE.
//   4. DEVICE ORDINAL (2.3/11.x): a handle's device_id() equals the device that
//      was current at construction, and (debug) using it on that device passes
//      the record-and-assert precondition (set_stream does not abort). On a
//      single-GPU box this is device 0; the test sets the current device to the
//      first device explicitly so the recorded ordinal is deterministic.
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Built by CMake/CTest as the
// `handles_unit` test (tests/CMakeLists.txt) linking steppe::device's device-
// private headers (handles.hpp is a CUDA header — PRIVATE to steppe_device,
// architecture.md §4 — so this gate is a CUDA TU, not a host-only unit test). No
// data. Run:  ./test_handles   (needs a CUDA device; no data)
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <utility>

#include "device/cuda/check.cuh"    // STEPPE_CUDA_CHECK, CUBLAS_CHECK, CublasError
#include "device/cuda/handles.hpp"  // CublasHandle, MathModeScope (the units under test)

using steppe::device::CublasHandle;
using steppe::device::MathModeScope;

namespace {

// Read the handle's current math mode through the production accessor cuBLAS
// exposes (cublasGetMathMode, cuBLAS §2.4.x). This is the same query MathModeScope
// uses to capture/restore, so the assertions observe the SAME state the scope acts
// on.
cublasMath_t current_math_mode(const CublasHandle& h) {
    cublasMath_t mode = CUBLAS_DEFAULT_MATH;
    CUBLAS_CHECK(cublasGetMathMode(h.get(), &mode));
    return mode;
}

const char* math_mode_name(cublasMath_t m) {
    switch (m) {
        case CUBLAS_DEFAULT_MATH:  return "CUBLAS_DEFAULT_MATH";
        case CUBLAS_PEDANTIC_MATH: return "CUBLAS_PEDANTIC_MATH";
        default:                   return "other";
    }
}

// (1) + (2) + (3): MathModeScope captures the live mode, applies a requested one
// for its scope, and restores the captured mode at scope exit — including nesting
// and a moved-from scope that restores nothing. Returns true on PASS.
bool expect_math_mode_restore() {
    bool ok = true;
    try {
        CublasHandle h;

        // Establish a KNOWN, non-default starting mode so "restore" is observable
        // as a distinct value (not merely the default the handle starts in).
        CUBLAS_CHECK(cublasSetMathMode(h.get(), CUBLAS_PEDANTIC_MATH));
        const cublasMath_t base = current_math_mode(h);
        const bool base_ok = (base == CUBLAS_PEDANTIC_MATH);
        std::printf("  %-40s mode=%s -> %s\n", "baseline set PEDANTIC",
                    math_mode_name(base), base_ok ? "PASS" : "FAIL");
        ok = base_ok && ok;

        // (1) Inside a scope requesting a DIFFERENT mode, the handle reports the
        //     requested mode; on exit it is back to the captured baseline.
        {
            MathModeScope scope(h.get(), CUBLAS_DEFAULT_MATH);
            const cublasMath_t inside = current_math_mode(h);
            const bool inside_ok = (inside == CUBLAS_DEFAULT_MATH);
            std::printf("  %-40s mode=%s -> %s\n", "inside scope = requested (DEFAULT)",
                        math_mode_name(inside), inside_ok ? "PASS" : "FAIL");
            ok = inside_ok && ok;
        }
        const cublasMath_t after = current_math_mode(h);
        const bool after_ok = (after == base);
        std::printf("  %-40s mode=%s -> %s\n", "after scope = restored (PEDANTIC)",
                    math_mode_name(after), after_ok ? "PASS" : "FAIL");
        ok = after_ok && ok;

        // (2) NESTED: the inner scope restores to the OUTER scope's mode, proving
        //     the ctor captures the live current mode (not a hardcoded default).
        {
            MathModeScope outer(h.get(), CUBLAS_DEFAULT_MATH);  // base PEDANTIC -> DEFAULT
            {
                MathModeScope inner(h.get(), CUBLAS_PEDANTIC_MATH);  // DEFAULT -> PEDANTIC
            }
            const cublasMath_t mid = current_math_mode(h);
            const bool mid_ok = (mid == CUBLAS_DEFAULT_MATH);  // back to OUTER's mode
            std::printf("  %-40s mode=%s -> %s\n", "nested inner restores to outer",
                        math_mode_name(mid), mid_ok ? "PASS" : "FAIL");
            ok = mid_ok && ok;
        }
        const cublasMath_t nested_after = current_math_mode(h);
        const bool nested_after_ok = (nested_after == base);
        std::printf("  %-40s mode=%s -> %s\n", "after nested = original (PEDANTIC)",
                    math_mode_name(nested_after), nested_after_ok ? "PASS" : "FAIL");
        ok = nested_after_ok && ok;

        // (3) MOVE: a moved-from scope restores NOTHING (the moved-into one owns
        //     the single restore). Construct a scope, move it, let the SOURCE die
        //     first (it must not restore), then end the moved-into scope (the one
        //     real restore). The mode must change exactly once, at the move target's
        //     dtor — never twice, never zero times.
        {
            MathModeScope moved_into = [&] {
                MathModeScope src(h.get(), CUBLAS_DEFAULT_MATH);  // base PEDANTIC -> DEFAULT
                return src;  // move out; `src` (now moved-from) dies here, must NOT restore
            }();
            const cublasMath_t held = current_math_mode(h);
            const bool held_ok = (held == CUBLAS_DEFAULT_MATH);  // moved-from did NOT restore
            std::printf("  %-40s mode=%s -> %s\n", "moved-from inert (mode held)",
                        math_mode_name(held), held_ok ? "PASS" : "FAIL");
            ok = held_ok && ok;
        }
        const cublasMath_t move_after = current_math_mode(h);
        const bool move_after_ok = (move_after == base);  // the one restore fired, exactly once
        std::printf("  %-40s mode=%s -> %s\n", "after move-scope = restored once",
                    math_mode_name(move_after), move_after_ok ? "PASS" : "FAIL");
        ok = move_after_ok && ok;
    } catch (const std::exception& e) {
        std::printf("  %-40s -> THREW -> FAIL\n", "math-mode restore");
        std::fprintf(stderr, "  [FAIL] math-mode restore THREW: %s\n", e.what());
        ok = false;
    }
    return ok;
}

// (4) Device-ordinal record-and-assert (2.3/11.x): device_id() equals the device
// current at construction, and (debug) using the handle on that device passes the
// record-and-assert precondition (set_stream does not abort). Returns true on PASS.
bool expect_device_ordinal() {
    bool ok = true;
    try {
        // Pin the current device explicitly so the recorded ordinal is
        // deterministic (device 0 on a single-GPU box). NEVER assume the wrapper
        // set it — that is the caller's job (architecture.md §9/§11.4); the wrapper
        // only RECORDS it.
        STEPPE_CUDA_CHECK(cudaSetDevice(0));

        CublasHandle h;
        const bool id_ok = (h.device_id() == 0);
        std::printf("  %-40s device_id=%d -> %s\n", "device_id() == creation device (0)",
                    h.device_id(), id_ok ? "PASS" : "FAIL");
        ok = id_ok && ok;

        // Using the handle while its creation device is current passes the
        // debug-only record-and-assert (it would SIGABRT on a mismatch). On the
        // single-GPU box this is the always-true case; the assertion is the
        // multi-GPU scaffold. The default (NULL) stream is a legitimate argument.
        h.set_stream(nullptr);
        std::printf("  %-40s -> PASS\n", "set_stream on creation device (no abort)");
    } catch (const std::exception& e) {
        std::printf("  %-40s -> THREW -> FAIL\n", "device ordinal");
        std::fprintf(stderr, "  [FAIL] device ordinal THREW: %s\n", e.what());
        ok = false;
    }
    return ok;
}

}  // namespace

int main() {
    // Fail fast (not "no GPU PASS") if there is no usable device: constructing a
    // CublasHandle calls cublasCreate, which needs a live cuBLAS context. A silent
    // skip would hide a regression. The box always has a device.
    int dev_count = 0;
    const cudaError_t derr = cudaGetDeviceCount(&dev_count);
    if (derr != cudaSuccess || dev_count < 1) {
        std::fprintf(stderr, "RESULT: FAIL — no CUDA device available (%s); this is a GPU gate.\n",
                     cudaGetErrorString(derr));
        return EXIT_FAILURE;
    }

    std::printf("\nU4/L10 CublasHandle MathModeScope + device-ordinal scaffold (synthetic, no data)\n");

    bool ok = true;
    ok = expect_math_mode_restore() && ok;
    ok = expect_device_ordinal() && ok;

    std::printf("\n");
    if (!ok) {
        std::fprintf(stderr,
            "RESULT: FAIL — MathModeScope did not capture/restore the cuBLAS math mode across\n"
            "        scope exit (incl. nesting / moved-from inertness), or CublasHandle did not\n"
            "        record the creation device ordinal / honor the record-and-assert precondition.\n"
            "        architecture.md §7, §9, §11.4, §12; cleanup device-cuda-handles 2.3/11.x, L10.\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr,
                 "RESULT: PASS (MathModeScope restores the prior math mode at scope exit — incl.\n"
                 "        nested and moved-from cases — and CublasHandle records + asserts its\n"
                 "        creation device ordinal)\n");
    return EXIT_SUCCESS;
}
