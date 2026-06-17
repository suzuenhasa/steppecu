// tests/reference/test_f2_determinism.cu
//
// B1 DETERMINISM GATE — the §12 emulated-FP64 run-to-run reproducibility test
// (architecture.md §12 "Results Reproducibility / bit-stable goldens", §13;
// cleanup X-1/B1).
//
// WHAT IT PINS. The §12 contract is that an EmulatedFp64 run is bit-reproducible
// on the single statistic stream when the FIXED-slice Ozaki path has its cuBLAS
// workspace pinned via `cublasSetWorkspace` (cuBLAS §2.1.4 Results
// Reproducibility). The X-1/B1 defect was that BOTH GEMM routines called
// `cublasSetStream` on every invocation, which "unconditionally resets the cuBLAS
// library workspace back to the default workspace pool" (cuBLAS §2.4.7) —
// discarding the determinism workspace before every GEMM batch (per-CHUNK on the
// M4 path). With that void open the emulated-FP64 GEMMs run without their pinned
// workspace and are NOT guaranteed bit-stable run to run.
//
// THIS TEST asserts BIT-IDENTICAL (exact equality, byte-for-byte memcmp) outputs
// when the SAME real-AADR Q/V/N is run TWICE through CudaBackend at
// EmulatedFp64{40} — once for M0 `compute_f2` and once for the M4
// `compute_f2_blocks` path. The two runs are bit-identical ONLY IF the workspace
// persists across the (now single, ctor-bound) stream binding. Before B1 the
// per-call `cublasSetStream` reset made this a determinism void; after B1 the
// CublasHandle owns the (stream, workspace) invariant and re-applies the
// workspace, so the run-to-run output is exact.
//
// This is a CORRECTNESS / determinism test (real genotype-derived input), NOT a
// precision or throughput benchmark — it does not compare against an oracle (the
// equivalence tests do that); it compares a run against ITSELF. If the build is
// WITHOUT -DSTEPPE_HAVE_EMU_TUNING, the fixed-slice Ozaki tuning is absent and
// cuBLAS declines to emulate these GEMMs (the emulated path collapses to native);
// run-to-run determinism STILL holds in that case (native FP64 single-stream is
// also deterministic), so the test asserts bit-identity unconditionally and
// additionally requires it on the second (M4) path which is the per-chunk worst
// case. The goldens (§13) rest on this bit-stability holding.
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Built by CMake/CTest as the
// `f2_determinism` test (tests/CMakeLists.txt) linking steppe::io + steppe::core
// + steppe::device, with -DSTEPPE_HAVE_EMU_TUNING=1.
// Run:  ./test_f2_determinism [data_root]   (default /workspace/data/aadr)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "steppe/config.hpp"                     // Precision, kDefaultMantissaBits
#include "steppe/fstats.hpp"                     // F2BlockTensor
#include "core/internal/views.hpp"              // MatView (Q/V/N contract)
#include "core/domain/block_partition_rule.hpp" // assign_blocks, BlockPartition, block_size_cm_to_morgans
#include "core/fstats/f2_from_blocks.hpp"       // compute_f2_block, compute_f2_blocks (the seam)
#include "device/backend.hpp"                   // ComputeBackend, F2Result
#include "device/backend_factory.hpp"           // steppe::device::make_cuda_backend (X-9/B8)
#include "io/snp_reader.hpp"                    // io::read_snp (SHARED .snp parse)

using steppe::Precision;
using steppe::F2BlockTensor;
using steppe::F2Result;
using steppe::core::BlockPartition;
using steppe::core::MatView;

namespace {

constexpr const char* kDefaultDataRoot = "/workspace/data/aadr";
constexpr const char* kGenoBase = "v66.p1_HO.aadr.patch.PUB";  // raw/<base>.snp

// shape.txt "P M" + Q/V/N.f64 (P*M little-endian doubles, column-major [P×M]).
void read_f64(const std::string& path, std::vector<double>& out, std::size_t count) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str()); std::exit(EXIT_FAILURE); }
    out.resize(count);
    const std::size_t got = std::fread(out.data(), sizeof(double), count, f);
    std::fclose(f);
    if (got != count) {
        std::fprintf(stderr, "ERROR: %s has %zu doubles, expected %zu\n", path.c_str(), got, count);
        std::exit(EXIT_FAILURE);
    }
}

void load_qvn(const std::string& dir, int& P, long& M,
              std::vector<double>& Q, std::vector<double>& V, std::vector<double>& N) {
    const std::string shapePath = dir + "/shape.txt";
    FILE* sf = std::fopen(shapePath.c_str(), "r");
    if (!sf) { std::fprintf(stderr, "ERROR: cannot open %s\n", shapePath.c_str()); std::exit(EXIT_FAILURE); }
    if (std::fscanf(sf, "%d %ld", &P, &M) != 2) {
        std::fprintf(stderr, "ERROR: %s must contain 'P M'\n", shapePath.c_str());
        std::fclose(sf); std::exit(EXIT_FAILURE);
    }
    std::fclose(sf);
    const std::size_t count = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    read_f64(dir + "/Q.f64", Q, count);
    read_f64(dir + "/V.f64", V, count);
    read_f64(dir + "/N.f64", N, count);
}

// Exact byte-for-byte equality of two double payloads. NaN-tolerant: a NaN in the
// SAME bit position on both runs still compares equal (memcmp on the bytes), which
// is the run-to-run reproducibility property we want (a deterministic run repeats
// even non-finite specials bit-for-bit).
bool bit_identical(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

// First differing element index (for a diagnosable failure), or -1 if identical.
long first_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        if (std::memcmp(&a[k], &b[k], sizeof(double)) != 0) return static_cast<long>(k);
    }
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = (argc >= 2) ? argv[1] : kDefaultDataRoot;
    const std::string derived_dir = root + "/derived_acc";
    const std::string snp = root + "/raw/" + kGenoBase + ".snp";

    // ---- Load real Q/V/N + the SNP prefix; SHARED io::read_snp + assign_blocks --
    int P = 0; long M = 0;
    std::vector<double> Qd, Vd, Nd;
    load_qvn(derived_dir, P, M, Qd, Vd, Nd);
    const MatView Q{Qd.data(), P, M};
    const MatView V{Vd.data(), P, M};
    const MatView N{Nd.data(), P, M};

    steppe::io::SnpTable snptab;
    try { snptab = steppe::io::read_snp(snp, static_cast<std::size_t>(M)); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: read_snp(%s) failed: %s\n", snp.c_str(), e.what());
        return EXIT_FAILURE;
    }
    if (static_cast<long>(snptab.count) != M) {
        std::fprintf(stderr, "ERROR: .snp rows %zu != M %ld\n", snptab.count, M);
        return EXIT_FAILURE;
    }
    const double bs_morgans = steppe::core::block_size_cm_to_morgans(steppe::kDefaultBlockSizeCm);
    const BlockPartition part = steppe::core::assign_blocks(snptab.chrom, snptab.genpos_morgans, bs_morgans);
    std::fprintf(stderr, "[load] derived_acc P=%d M=%ld -> n_block=%d — REAL AADR\n",
                 P, M, part.n_block);

    // The single statistic stream + the emulated-FP64 workspace are bound ONCE in
    // the backend ctor; a fresh backend per run proves the determinism is a
    // property of the (stream, workspace) invariant, not of warm cuBLAS state.
    const Precision precEmu{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};

    // ---- (1) M0 compute_f2 TWICE at EmulatedFp64{40} ------------------------
    F2Result m0_a, m0_b;
    {
        auto gpu = steppe::device::make_cuda_backend();
        m0_a = steppe::core::compute_f2_block(*gpu, Q, V, N, precEmu);
    }
    {
        auto gpu = steppe::device::make_cuda_backend();
        m0_b = steppe::core::compute_f2_block(*gpu, Q, V, N, precEmu);
    }
    const bool m0_f2_bit    = bit_identical(m0_a.f2, m0_b.f2);
    const bool m0_vpair_bit = bit_identical(m0_a.vpair, m0_b.vpair);

    // ---- (2) M4 compute_f2_blocks TWICE at EmulatedFp64{40} -----------------
    // The per-chunk worst case: pre-B1 the workspace was reset before EVERY
    // chunk's strided-batched GEMMs.
    F2BlockTensor m4_a, m4_b;
    {
        auto gpu = steppe::device::make_cuda_backend();
        m4_a = steppe::core::compute_f2_blocks(*gpu, Q, V, N, part, precEmu);
    }
    {
        auto gpu = steppe::device::make_cuda_backend();
        m4_b = steppe::core::compute_f2_blocks(*gpu, Q, V, N, part, precEmu);
    }
    const bool m4_f2_bit    = bit_identical(m4_a.f2, m4_b.f2);
    const bool m4_vpair_bit = bit_identical(m4_a.vpair, m4_b.vpair);

    // ---- Verdicts -----------------------------------------------------------
    const long m0_f2_diff = first_diff(m0_a.f2, m0_b.f2);
    const long m4_f2_diff = first_diff(m4_a.f2, m4_b.f2);

    std::printf("\nB1 emulated-FP64 run-to-run determinism (REAL data P=%d M=%ld n_block=%d)\n",
                P, M, part.n_block);
    std::printf("%-38s %10s %12s\n", "check", "verdict", "firstDiff");
    std::printf("%-38s %10s %12ld\n", "M0 compute_f2 f2 bit-identical",
                m0_f2_bit ? "PASS" : "FAIL", m0_f2_diff);
    std::printf("%-38s %10s %12s\n", "M0 compute_f2 Vpair bit-identical",
                m0_vpair_bit ? "PASS" : "FAIL", "-");
    std::printf("%-38s %10s %12ld\n", "M4 compute_f2_blocks f2 bit-identical",
                m4_f2_bit ? "PASS" : "FAIL", m4_f2_diff);
    std::printf("%-38s %10s %12s\n", "M4 compute_f2_blocks Vpair bit-identical",
                m4_vpair_bit ? "PASS" : "FAIL", "-");
    std::printf("\n");

    const bool overall = m0_f2_bit && m0_vpair_bit && m4_f2_bit && m4_vpair_bit;
    if (!overall) {
        std::fprintf(stderr,
            "RESULT: FAIL — emulated-FP64 output is NOT bit-identical run-to-run. The §12\n"
            "        determinism workspace did not persist across the stream binding (cleanup\n"
            "        X-1/B1: a per-call cublasSetStream resets the cuBLAS workspace, cuBLAS\n"
            "        §2.4.7). Bind (stream, workspace) once in the CublasHandle and re-apply the\n"
            "        workspace in set_stream.\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "RESULT: PASS\n");
    return EXIT_SUCCESS;
}
