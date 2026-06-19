// tests/unit/test_f2_from_blocks.cpp
//
// Host-only unit test of the f2 composition root — core::compute_f2_block (M0)
// and core::compute_f2_blocks (M4) — driven through a CUDA-FREE MockBackend
// (architecture.md §2 fail-fast, §8 DRY, §13 GPU-free-testability; cleanup B11 /
// f2_from_blocks F-12). Pure C++ TU, NO GPU, NO io, NO real data.
//
// The f2 orchestration is the ONE host point that sees all three Q/V/N views
// together with the full BlockPartition. Its entire job is (1) to forward those
// VERBATIM through the injected ComputeBackend seam, and (2) to enforce the
// documented preconditions (B11) so a malformed contract fails fast instead of
// degrading to a silent out-of-bounds read/write deep inside a backend (or a
// null/short block_id.data() deref the backend's block_ranges cannot catch — it
// only sees the raw `(const int*, int)` pair). The seam was DESIGNED to be
// GPU-free-testable (a stub ComputeBackend is all it needs) yet had no such test;
// this is it. The recording mock links against steppe_core with NO steppe_device
// runtime dependency at the test level — the proof the seam is what it claims.
//
// This test pins:
//   1. VALID forwarding — compute_f2_block / compute_f2_blocks pass Q/V/N, the
//      precision, AND (for M4) block_id.data()/n_block to the backend BYTE-FOR-
//      BYTE, and return the backend's result unchanged;
//   2. MALFORMED inputs FAIL FAST — when STEPPE_ASSERT is active (the default,
//      assert-enabled build), a mismatched Q/V/N P/M, a short/null block_id, an
//      n_block out of range, or a non-decreasing block_id ABORTS (SIGABRT) BEFORE
//      reaching the backend, rather than silently corrupting memory. Under NDEBUG
//      the guard is compiled out by contract, so those death cases are skipped
//      (the valid-forwarding cases still run).
//
// Dual harness (identical to tests/unit/test_block_ranges.cpp): with
// -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest (gtest death tests for the abort
// cases); otherwise it is a self-checking main() that fork()s each death case and
// checks the child died via SIGABRT, returning non-zero on the first failure.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/fstats/f2_from_blocks.hpp"          // compute_f2_block, compute_f2_blocks
#include "core/domain/block_partition_rule.hpp"    // BlockPartition
#include "core/internal/views.hpp"                 // MatView
#include "device/backend.hpp"                      // ComputeBackend, F2Result
#include "steppe/config.hpp"                        // Precision (Precision::Kind)
#include "steppe/fstats.hpp"                        // F2BlockTensor

namespace {

using steppe::ComputeBackend;
using steppe::DecodeResult;
using steppe::DecodeTileView;
using steppe::F2BlockTensor;
using steppe::F2Result;
using steppe::Precision;
using steppe::core::BlockPartition;
using steppe::core::compute_f2_block;
using steppe::core::compute_f2_blocks;
using steppe::core::MatView;

// ---------------------------------------------------------------------------
// A CUDA-FREE recording ComputeBackend. It performs NO arithmetic — it captures
// exactly what the orchestration forwarded so the test can assert verbatim
// dispatch, and returns a sentinel-tagged result the test can recognize. This is
// the GPU-free stub the §13 seam was designed to admit.
// ---------------------------------------------------------------------------
class MockBackend final : public ComputeBackend {
public:
    // Captured compute_f2 arguments (the last call).
    const double* seen_q_data = nullptr;
    const double* seen_v_data = nullptr;
    const double* seen_n_data = nullptr;
    int seen_qP = -1;
    long seen_qM = -1;
    int seen_precision_mantissa = -1;
    int compute_f2_calls = 0;

    // Captured compute_f2_blocks arguments (the last call).
    const int* seen_block_id = nullptr;
    int seen_n_block = -1;
    int compute_f2_blocks_calls = 0;

    [[nodiscard]] F2Result compute_f2(const MatView& Q, const MatView& V, const MatView& N,
                                      const Precision& precision) override {
        ++compute_f2_calls;
        seen_q_data = Q.data;
        seen_v_data = V.data;
        seen_n_data = N.data;
        seen_qP = Q.P;
        seen_qM = Q.M;
        seen_precision_mantissa = precision.mantissa_bits;
        F2Result out;
        out.P = Q.P;
        out.f2.assign(1, kSentinel);  // recognizable, non-empty
        return out;
    }

    [[nodiscard]] F2BlockTensor compute_f2_blocks(const MatView& Q, const MatView& V,
                                                  const MatView& N, const int* block_id,
                                                  int n_block, const Precision& precision) override {
        ++compute_f2_blocks_calls;
        seen_q_data = Q.data;
        seen_v_data = V.data;
        seen_n_data = N.data;
        seen_qP = Q.P;
        seen_qM = Q.M;
        seen_block_id = block_id;
        seen_n_block = n_block;
        seen_precision_mantissa = precision.mantissa_bits;
        F2BlockTensor out;
        out.P = Q.P;
        out.n_block = n_block;
        out.f2.assign(1, kSentinel);  // recognizable, non-empty
        return out;
    }

    // Not exercised by this unit (the f2 orchestration never decodes); a trivial
    // override keeps MockBackend concrete.
    [[nodiscard]] DecodeResult decode_af(const DecodeTileView&) override { return {}; }

    static constexpr double kSentinel = 42.0;
};

// A well-formed [P × M] view over caller-owned storage. The values are
// irrelevant — the orchestration touches none of them; only the pointers/P/M are.
[[nodiscard]] MatView make_view(const std::vector<double>& storage, int P, long M) {
    MatView mv;
    mv.data = storage.data();
    mv.P = P;
    mv.M = M;
    return mv;
}

// A consistent partition for M columns: block_id is dense, non-decreasing, in
// [0, n_block). Here a simple 2-block split (first half / second half).
[[nodiscard]] BlockPartition make_partition(long M) {
    BlockPartition bp;
    bp.block_id.resize(static_cast<std::size_t>(M));
    const long half = M / 2;
    for (long s = 0; s < M; ++s) {
        bp.block_id[static_cast<std::size_t>(s)] = (s < half) ? 0 : 1;
    }
    bp.n_block = (M >= 2) ? 2 : (M == 1 ? 1 : 0);
    // For M==1 collapse to a single block.
    if (M == 1) bp.block_id[0] = 0;
    return bp;
}

// =====================  VALID FORWARDING  ==================================

// compute_f2_block forwards Q/V/N + precision verbatim and returns the backend's
// result unchanged.
[[nodiscard]] bool test_block_forwards_verbatim() {
    constexpr int P = 3;
    constexpr long M = 4;
    const std::vector<double> qd(static_cast<std::size_t>(P) * M, 0.0);
    const std::vector<double> vd(static_cast<std::size_t>(P) * M, 0.0);
    const std::vector<double> nd(static_cast<std::size_t>(P) * M, 0.0);
    const MatView Q = make_view(qd, P, M);
    const MatView V = make_view(vd, P, M);
    const MatView N = make_view(nd, P, M);
    const Precision prec{Precision::Kind::EmulatedFp64, 40};

    MockBackend mock;
    const F2Result r = compute_f2_block(mock, Q, V, N, prec);

    return mock.compute_f2_calls == 1 &&
           mock.seen_q_data == qd.data() &&
           mock.seen_v_data == vd.data() &&
           mock.seen_n_data == nd.data() &&
           mock.seen_qP == P && mock.seen_qM == M &&
           mock.seen_precision_mantissa == 40 &&
           r.P == P && r.f2.size() == 1 && r.f2[0] == MockBackend::kSentinel;
}

// compute_f2_blocks forwards Q/V/N + block_id.data()/n_block + precision verbatim
// and returns the backend's result unchanged.
[[nodiscard]] bool test_blocks_forwards_verbatim() {
    constexpr int P = 2;
    constexpr long M = 6;
    const std::vector<double> qd(static_cast<std::size_t>(P) * M, 0.0);
    const std::vector<double> vd(static_cast<std::size_t>(P) * M, 0.0);
    const std::vector<double> nd(static_cast<std::size_t>(P) * M, 0.0);
    const MatView Q = make_view(qd, P, M);
    const MatView V = make_view(vd, P, M);
    const MatView N = make_view(nd, P, M);
    const BlockPartition bp = make_partition(M);
    const Precision prec{Precision::Kind::Fp64, 0};

    MockBackend mock;
    const F2BlockTensor t = compute_f2_blocks(mock, Q, V, N, bp, prec);

    return mock.compute_f2_blocks_calls == 1 &&
           mock.seen_q_data == qd.data() &&
           mock.seen_v_data == vd.data() &&
           mock.seen_n_data == nd.data() &&
           mock.seen_qP == P && mock.seen_qM == M &&
           // the raw pointer handed to the seam IS the partition's own storage
           mock.seen_block_id == bp.block_id.data() &&
           mock.seen_n_block == bp.n_block &&
           mock.seen_precision_mantissa == 0 &&
           t.P == P && t.n_block == bp.n_block &&
           t.f2.size() == 1 && t.f2[0] == MockBackend::kSentinel;
}

// =====================  MALFORMED -> FAIL FAST  =============================
// Each of these, when STEPPE_ASSERT is active, must abort BEFORE the backend is
// touched. Each lives in its own function so the death harness can invoke it.

void death_block_mismatched_M() {
    constexpr int P = 2;
    const std::vector<double> qd(static_cast<std::size_t>(P) * 4, 0.0);
    const std::vector<double> vd(static_cast<std::size_t>(P) * 3, 0.0);  // M differs
    const std::vector<double> nd(static_cast<std::size_t>(P) * 4, 0.0);
    const MatView Q = make_view(qd, P, 4);
    const MatView V = make_view(vd, P, 3);  // <-- V.M != Q.M
    const MatView N = make_view(nd, P, 4);
    MockBackend mock;
    (void)compute_f2_block(mock, Q, V, N, Precision{Precision::Kind::Fp64, 0});
}

void death_block_mismatched_P() {
    const std::vector<double> qd(2 * 4, 0.0);
    const std::vector<double> vd(3 * 4, 0.0);  // P differs
    const std::vector<double> nd(2 * 4, 0.0);
    const MatView Q = make_view(qd, 2, 4);
    const MatView V = make_view(vd, 3, 4);  // <-- V.P != Q.P
    const MatView N = make_view(nd, 2, 4);
    MockBackend mock;
    (void)compute_f2_block(mock, Q, V, N, Precision{Precision::Kind::Fp64, 0});
}

void death_blocks_short_block_id() {
    constexpr int P = 2;
    constexpr long M = 4;
    const std::vector<double> d(static_cast<std::size_t>(P) * M, 0.0);
    const MatView Q = make_view(d, P, M);
    const MatView V = make_view(d, P, M);
    const MatView N = make_view(d, P, M);
    BlockPartition bp;
    bp.block_id = {0, 0};  // length 2 != M=4
    bp.n_block = 1;
    MockBackend mock;
    (void)compute_f2_blocks(mock, Q, V, N, bp, Precision{Precision::Kind::Fp64, 0});
}

void death_blocks_null_block_id() {
    // The corrected F-1 null-deref sub-case: an EMPTY block_id (data() may legally
    // be null) with n_block > 0 over M > 0 columns. The size check (size != M)
    // fires before any null deref.
    constexpr int P = 2;
    constexpr long M = 4;
    const std::vector<double> d(static_cast<std::size_t>(P) * M, 0.0);
    const MatView Q = make_view(d, P, M);
    const MatView V = make_view(d, P, M);
    const MatView N = make_view(d, P, M);
    BlockPartition bp;  // block_id empty (.data() may be null)
    bp.n_block = 1;     // but n_block > 0 — the inconsistency
    MockBackend mock;
    (void)compute_f2_blocks(mock, Q, V, N, bp, Precision{Precision::Kind::Fp64, 0});
}

void death_blocks_n_block_too_large() {
    constexpr int P = 2;
    constexpr long M = 3;
    const std::vector<double> d(static_cast<std::size_t>(P) * M, 0.0);
    const MatView Q = make_view(d, P, M);
    const MatView V = make_view(d, P, M);
    const MatView N = make_view(d, P, M);
    BlockPartition bp;
    bp.block_id = {0, 1, 2};
    bp.n_block = 4;  // > M=3 (more blocks than SNPs)
    MockBackend mock;
    (void)compute_f2_blocks(mock, Q, V, N, bp, Precision{Precision::Kind::Fp64, 0});
}

void death_blocks_non_monotonic() {
    constexpr int P = 2;
    constexpr long M = 3;
    const std::vector<double> d(static_cast<std::size_t>(P) * M, 0.0);
    const MatView Q = make_view(d, P, M);
    const MatView V = make_view(d, P, M);
    const MatView N = make_view(d, P, M);
    BlockPartition bp;
    bp.block_id = {0, 1, 0};  // in range but NOT non-decreasing
    bp.n_block = 2;
    MockBackend mock;
    (void)compute_f2_blocks(mock, Q, V, N, bp, Precision{Precision::Kind::Fp64, 0});
}

struct ValidCase {
    const char* name;
    bool (*fn)();
};

struct DeathCase {
    const char* name;
    void (*fn)();
};

constexpr ValidCase kValidCases[] = {
    {"compute_f2_block forwards Q/V/N + precision verbatim", test_block_forwards_verbatim},
    {"compute_f2_blocks forwards Q/V/N + block_id.data()/n_block + precision verbatim",
     test_blocks_forwards_verbatim},
};

constexpr DeathCase kDeathCases[] = {
    {"MALFORMED: Q/V/N disagree on M -> abort", death_block_mismatched_M},
    {"MALFORMED: Q/V/N disagree on P -> abort", death_block_mismatched_P},
    {"MALFORMED: block_id shorter than M -> abort", death_blocks_short_block_id},
    {"MALFORMED: empty/null block_id with n_block>0 -> abort", death_blocks_null_block_id},
    {"MALFORMED: n_block > M -> abort", death_blocks_n_block_too_large},
    {"MALFORMED: non-monotonic block_id -> abort", death_blocks_non_monotonic},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(F2FromBlocks, ValidForwarding) {
    for (const auto& c : kValidCases) {
        EXPECT_TRUE(c.fn()) << "failed: " << c.name;
    }
}

#ifndef NDEBUG
// STEPPE_ASSERT is active: each malformed input must abort before the backend.
// "threadsafe" death-test style re-exec's the child (robust against the unit's
// allocations); the regex is "" so any abort message matches — we assert the
// fact of death, not its text. The whole TEST is compiled out under NDEBUG (the
// guard is debug-only by contract), matching the self-checking arm's #else SKIP.
TEST(F2FromBlocksDeathTest, MalformedAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    for (const auto& c : kDeathCases) {
        EXPECT_DEATH(c.fn(), "") << "did not abort: " << c.name;
    }
}
#endif  // !NDEBUG

#else  // self-checking main()

#include <sys/wait.h>
#include <unistd.h>

namespace {

// Run `fn` in a forked child and report whether the child died via SIGABRT (what
// a fired STEPPE_ASSERT/assert does). Returns true iff the child aborted.
//
// `[[maybe_unused]]` (C++17, [dcl.attr.unused]): the death-case loop in main() that
// calls this is itself `#ifndef NDEBUG`, so under NDEBUG the helper is defined but
// never called — that is the contract (the asserts compile out), not dead code, so
// the attribute keeps the TU clean under -Werror=unused-function in BOTH builds
// rather than #ifdef'ing the definition away from its <sys/wait.h>/<unistd.h> home.
[[maybe_unused]] [[nodiscard]] bool child_aborts(void (*fn)()) {
    const pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "fork() failed\n");
        return false;
    }
    if (pid == 0) {
        // Child: silence the assert message to stderr is fine; just run and let
        // the assert abort. If it does NOT abort, exit 0 (a test failure).
        fn();
        _exit(0);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) { /* retry on EINTR */ }
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

}  // namespace

int main() {
    int failures = 0;

    for (const auto& c : kValidCases) {
        const bool ok = c.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }

#ifndef NDEBUG
    // STEPPE_ASSERT active: each malformed input must abort (SIGABRT) before the
    // backend is reached.
    for (const auto& c : kDeathCases) {
        const bool ok = child_aborts(c.fn);
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }
#else
    std::printf("[SKIP] NDEBUG build: STEPPE_ASSERT compiled out, "
                "%zu malformed-input death cases skipped by contract\n",
                sizeof(kDeathCases) / sizeof(kDeathCases[0]));
#endif

    if (failures != 0) {
        std::fprintf(stderr, "test_f2_from_blocks: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_f2_from_blocks: all checks PASS\n");
    return 0;
}
#endif
